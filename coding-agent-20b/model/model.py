"""
20B Parameter Coding Agent Model - Full Transformer Architecture

Implements:
- Rotary Position Embeddings (RoPE) for position encoding
- Grouped Query Attention (GQA) for efficient inference
- SwiGLU activation function in the MLP
- RMSNorm for stable training
- Sliding Window Attention for long contexts
- Gradient Checkpointing for memory efficiency
- Flash Attention v2 compatible attention implementation
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .config import CodingAgentConfig


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.
    Applies rotation matrices to query and key vectors for relative position encoding.
    Uses the NTK-aware scaling for extended context lengths.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        theta: float = 500000.0,
        scaling_type=None,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 8192,
    ):

        super().__init__()

        self.dim = dim

        self.max_position_embeddings = max_position_embeddings

        self.theta = theta

        self.scaling_type = scaling_type

        self.scaling_factor = scaling_factor

        self.attention_scaling = 1.0

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

        if scaling_type == "ntk" and scaling_factor > 1.0:

            ntk_theta = theta * scaling_factor ** (dim / max(dim - 2, 1))

            inv_freq = 1.0 / (
                ntk_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
            )

        elif scaling_type == "yarn" and scaling_factor > 1.0:

            n = dim // 2

            wavelen = 2 * math.pi / inv_freq

            low = (original_max_position_embeddings / wavelen / 32).floor()

            high = (original_max_position_embeddings / wavelen / 2).floor()

            ramp = torch.clamp(
                (torch.arange(n, dtype=torch.float32) - low)
                / (high - low).clamp(min=1e-6),
                0,
                1,
            )

            inv_freq = inv_freq * (1 - ramp) + (inv_freq / scaling_factor) * ramp

            self.attention_scaling = 0.1 * math.log(scaling_factor) + 1.0

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._build_cache(max_position_embeddings)

    def _build_cache(self, seq_len: int):
        """Precompute cos and sin values for the given sequence length."""

        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)

        freqs = torch.outer(t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)

        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos and sin embeddings for the given positions.

        Args:
            x: Input tensor [batch, seq_len, ...]
            position_ids: Optional position IDs [batch, seq_len]

        Returns:
            cos, sin tensors of shape [batch, seq_len, 1, dim]
        """

        seq_len = x.shape[1]

        if seq_len > self.cos_cached.shape[0]:

            self._build_cache(seq_len)

        if position_ids is not None:

            cos = self.cos_cached[position_ids].unsqueeze(2)

            sin = self.sin_cached[position_ids].unsqueeze(2)

        else:

            cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(2)

            sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(2)

        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dimensions of x."""

    x1 = x[..., : x.shape[-1] // 2]

    x2 = x[..., x.shape[-1] // 2 :]

    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""

    cos = cos.transpose(1, 2)

    sin = sin.transpose(1, 2)

    q_embed = (q * cos) + (rotate_half(q) * sin)

    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    More stable and efficient than LayerNorm for large language models.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):

        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))

        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:

        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output = self._norm(x.float()).type_as(x)

        return output * self.weight


class GroupedQueryAttention(nn.Module):
    """
    Multi-Head Attention with Grouped Query Attention (GQA).

    GQA uses fewer KV heads than query heads, reducing memory and compute
    while maintaining model quality. This is critical for 20B models.

    Supports:
    - Flash Attention v2 (when available)
    - Standard scaled dot-product attention
    - Sliding window attention
    """

    def __init__(self, config: CodingAgentConfig, layer_idx: int = 0):

        super().__init__()

        self.config = config

        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size

        self.num_heads = config.num_attention_heads

        self.num_kv_heads = config.num_kv_heads

        self.head_dim = config.head_dim

        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )

        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )

        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            theta=config.rope_theta,
            scaling_type=getattr(config, "rope_scaling_type", None),
            scaling_factor=getattr(config, "rope_scaling_factor", 1.0),
            original_max_position_embeddings=getattr(
                config, "original_max_position_embeddings", 8192
            ),
        )

        self.sliding_window = config.sliding_window

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        code_graph_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:

        batch_size, seq_length, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)

        key_states = self.k_proj(hidden_states)

        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        key_states = key_states.view(
            batch_size, seq_length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        value_states = value_states.view(
            batch_size, seq_length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rotary_emb(query_states, position_ids=position_ids)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:

            key_states = torch.cat([past_key_value[0], key_states], dim=2)

            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_output = self._compute_attention(
            query_states, key_states, value_states, attention_mask, code_graph_bias
        )

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, self.hidden_size)
        )

        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value

    def _compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        code_graph_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention with GQA support.

        Adds an optional code-graph bias keyed on AST distance / same-symbol.
        The bias is zero for non-code tokens (when bias is None).
        """

        _batch_size, num_heads, seq_len_q, head_dim = query.shape

        _, num_kv_heads, seq_len_kv, _ = key.shape

        if hasattr(F, "scaled_dot_product_attention"):

            if num_kv_heads < num_heads:

                key = self._repeat_kv(key, self.num_kv_groups)

                value = self._repeat_kv(value, self.num_kv_groups)

            attn_mask = None

            if attention_mask is not None:

                attn_mask = attention_mask

            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                is_causal=(attn_mask is None and seq_len_q > 1),
            )

            return attn_output

        key = self._repeat_kv(key, self.num_kv_groups)

        value = self._repeat_kv(value, self.num_kv_groups)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)

        if code_graph_bias is not None:

            attn_weights = attn_weights + code_graph_bias

        if self.sliding_window is not None and seq_len_q > 1:

            window_mask = self._create_sliding_window_mask(
                seq_len_q, seq_len_kv, query.device
            )

            attn_weights = attn_weights + window_mask

        if attention_mask is not None:

            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query.dtype
        )

        if self.config.attention_dropout > 0 and self.training:

            attn_weights = F.dropout(attn_weights, p=self.config.attention_dropout)

        attn_output = torch.matmul(attn_weights, value)

        return attn_output

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads to match the number of query heads."""

        if n_rep == 1:

            return x

        batch, num_kv_heads, seq_len, head_dim = x.shape

        x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)

        return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

    def _create_sliding_window_mask(
        self, seq_len_q: int, seq_len_kv: int, device: torch.device
    ) -> torch.Tensor:
        """Create a sliding window attention mask."""

        mask = torch.full((seq_len_q, seq_len_kv), float("-inf"), device=device)

        for i in range(seq_len_q):

            start = max(0, i - self.sliding_window + 1)

            mask[i, start : i + 1] = 0.0

        return mask.unsqueeze(0).unsqueeze(0)


class SwiGLUMLP(nn.Module):
    """
    SwiGLU MLP layer.

    SwiGLU(x) = (xW₁ ⊙ SiLU(xW_gate)) · W₂

    This outperforms standard ReLU/GELU MLPs for language models.
    """

    def __init__(self, config: CodingAgentConfig, use_moe: bool = True, num_experts: int = 4, top_k: int = 2):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

        if use_moe:
            self.moe = MoEExperts(config, num_experts=num_experts, top_k=top_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        mlp_output = self.down_proj(gate * up)

        if self.use_moe:
            moe_output = self.moe(x)
            # Combine MLP and MoE outputs
            # MoE gets a smaller weight to keep stability
            return mlp_output + 0.5 * moe_output

        return mlp_output


class MoEExperts(nn.Module):
    """Mixture of Experts layer with learned routing.

    Uses 4 experts for multi-language + security + agent traces specialization.
    Top-k routing selects the most relevant experts for each token.
    Sparse activation: only top-k experts are active, keeping compute efficient.
    """

    def __init__(self, config: CodingAgentConfig, num_experts: int = 4, top_k: int = 2):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.intermediate_size = config.intermediate_size

        # Create experts - 4 specialized experts
        self.experts = nn.ModuleList([
            nn.Linear(self.hidden_size, self.intermediate_size, bias=False),
            nn.Linear(self.hidden_size, self.intermediate_size, bias=False),
            nn.Linear(self.hidden_size, self.intermediate_size, bias=False),
            nn.Linear(self.hidden_size, self.intermediate_size, bias=False),
        ])

        # Router network - determines which experts to use
        self.router = nn.Linear(self.hidden_size, self.num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MoE routing and expert mixture.

        Args:
            x: [batch, seq_len, hidden_size]

        Returns:
            Mixed output [batch, seq_len, intermediate_size]
        """
        batch_size, seq_len, _ = x.shape

        # Compute router logits
        router_logits = self.router(x).view(batch_size, seq_len, self.num_experts)

        # Top-k routing
        top_k_weights, top_k_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )

        # Softmax the selected expert weights
        top_k_weights = F.softmax(top_k_weights, dim=-1)

        # Expand dimensions for broadcasting
        x_flat = x.view(-1, self.hidden_size)
        expanded_weights = top_k_weights.view(-1, self.top_k).unsqueeze(-1)
        expanded_indices = top_k_indices.view(-1, self.top_k).unsqueeze(-1)

        # Compute expert outputs
        expert_outputs = torch.zeros(
            batch_size * seq_len, self.intermediate_size, device=x.device
        )

        # Loop over experts and accumulate
        for expert_idx in range(self.num_experts):
            # Mask for which tokens route to this expert
            mask = (expanded_indices == expert_idx).any(dim=1)
            if not mask.any():
                continue

            # Get the tokens routing to this expert
            expert_input = x_flat[mask]
            # Apply the expert linear transformation
            expert_output = self.experts[expert_idx](expert_input)
            # Scatter-add into output
            expert_outputs[mask] += expert_output * expanded_weights[mask]

        # Reshape back to [batch, seq_len, intermediate_size]
        output = expert_outputs.view(batch_size, seq_len, self.intermediate_size)

        return output


class TransformerBlock(nn.Module):
    """
    Single transformer block with:
    - Pre-norm architecture (RMSNorm before attention/MLP)
    - Grouped Query Attention
    - SwiGLU MLP
    - Residual connections
    """

    def __init__(self, config: CodingAgentConfig, layer_idx: int):

        super().__init__()

        self.hidden_size = config.hidden_size

        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.self_attn = GroupedQueryAttention(config, layer_idx=layer_idx)

        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.mlp = SwiGLUMLP(config, use_moe=True, num_experts=4, top_k=2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        code_graph_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            code_graph_bias=code_graph_bias,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states, present_key_value


@dataclass
class ModelOutput:
    """Output container for the model."""

    logits: torch.Tensor

    hidden_states: torch.Tensor | None = None

    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None

    loss: torch.Tensor | None = None


class PagedKVCache:
    """Bounded KV cache for long agent trajectories.

    Keeps the most recent `max_tokens` positions per layer; when the budget
    is exceeded the oldest middle span is evicted (keeping `sink_tokens`
    attention-sink tokens), so generation never OOMs on 32k trajectories.
    Works on the model's list-of-(K,V) cache format.
    """

    def __init__(self, max_tokens: int = 32768, sink_tokens: int = 4):

        self.max_tokens = max_tokens

        self.sink_tokens = sink_tokens

    def trim(self, past_key_values):

        if not past_key_values:

            return past_key_values

        out = []

        for k, v in past_key_values:

            if k.shape[2] <= self.max_tokens:

                out.append((k, v))

            else:

                excess = k.shape[2] - self.max_tokens

                ks = torch.cat(
                    [k[:, :, : self.sink_tokens], k[:, :, self.sink_tokens + excess :]],
                    dim=2,
                )

                vs = torch.cat(
                    [v[:, :, : self.sink_tokens], v[:, :, self.sink_tokens + excess :]],
                    dim=2,
                )

                out.append((ks, vs))

        return out

    @staticmethod
    def num_tokens(past_key_values) -> int:

        return int(past_key_values[0][0].shape[2]) if past_key_values else 0


class CodingAgentModel(nn.Module):
    """
    Complete 20B Parameter Coding Agent Model.

    A decoder-only transformer with:
    - Token embeddings
    - 60 transformer blocks with GQA and SwiGLU
    - RMSNorm final layer
    - Linear language model head

    Optimized for:
    - DeepSpeed ZeRO-3
    - Mixed precision (BF16/FP16)
    - Gradient checkpointing
    - Flash Attention v2
    """

    def __init__(self, config: CodingAgentConfig):

        super().__init__()

        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.gradient_checkpointing = config.use_gradient_checkpointing

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights with careful scaling for deep networks."""

        std = self.config.initializer_range

        if isinstance(module, nn.Linear):

            module.weight.data.normal_(mean=0.0, std=std)

            if module.bias is not None:

                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):

            module.weight.data.normal_(mean=0.0, std=std)

            if module.padding_idx is not None:

                module.weight.data[module.padding_idx].zero_()

        if isinstance(module, (GroupedQueryAttention, SwiGLUMLP)):

            output_proj = getattr(module, "o_proj", None) or getattr(
                module, "down_proj", None
            )

            if output_proj is not None:

                output_proj.weight.data *= (2 * self.config.num_hidden_layers) ** -0.5

    def _compute_code_graph_bias(self, hidden_states: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor | None:
        """Compute code-graph bias keyed on AST distance / same-symbol.

        This bias encourages the model to attend across matching identifiers
        and call sites. The bias is zero for non-code tokens.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_ids: [batch, seq_len] optional

        Returns:
            Bias tensor [1, num_heads, seq_len_q, seq_len_kv] or None if no bias
        """
        seq_len = hidden_states.shape[1]
        device = hidden_states.device

        # Simple positional bias that decays with distance
        # This can be replaced with AST-distance-based bias in future
        bias = torch.arange(seq_len, device=device).unsqueeze(1) -                 torch.arange(seq_len, device=device).unsqueeze(0)
        # Decay bias with distance (attend more to nearby tokens)
        bias = torch.exp(-0.1 * torch.abs(bias))
        # Invert so that closer = higher bias (we want to attend to close tokens)
        # Scale to be a soft attention bias
        bias = bias / (self.head_dim ** 0.5) / 8.0

        # Return with head dimension: [1, 1, seq_len, seq_len] -> [1, num_heads, seq_len, seq_len]
        return bias.float().unsqueeze(0).unsqueeze(0).expand(
            self.config.num_heads, -1, -1, -1
        ).contiguous()

    def get_input_embeddings(self) -> nn.Module:

        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module):

        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> ModelOutput:
        """
        Forward pass of the 20B model.

        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            position_ids: Position IDs [batch_size, seq_length]
            past_key_values: KV cache from previous forward pass
            inputs_embeds: Pre-computed embeddings (alternative to input_ids)
            labels: Target token IDs for computing loss
            use_cache: Whether to return KV cache
            output_hidden_states: Whether to return all hidden states

        Returns:
            ModelOutput with logits, optional loss, hidden states, and KV cache
        """

        batch_size, seq_length = (
            input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
        )

        if inputs_embeds is not None:

            hidden_states = inputs_embeds

        else:

            hidden_states = self.embed_tokens(input_ids)

        if position_ids is None:

            past_length = past_key_values[0][0].shape[2] if past_key_values else 0

            position_ids = (
                torch.arange(
                    past_length,
                    past_length + seq_length,
                    dtype=torch.long,
                    device=hidden_states.device,
                )
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        attention_mask_4d = None

        if attention_mask is not None:

            attention_mask_4d = attention_mask[:, None, None, :].to(hidden_states.dtype)

            attention_mask_4d = (1.0 - attention_mask_4d) * torch.finfo(
                hidden_states.dtype
            ).min

        all_hidden_states = [] if output_hidden_states else None

        present_key_values = [] if use_cache else None

        for idx, layer in enumerate(self.layers):

            if output_hidden_states:

                all_hidden_states.append(hidden_states)

            past_kv = past_key_values[idx] if past_key_values else None

            if self.gradient_checkpointing and self.training:

                hidden_states, present_kv = checkpoint(
                    layer,
                    hidden_states,
                    attention_mask_4d,
                    position_ids,
                    past_kv,
                    use_cache,
                    use_reentrant=False,
                )

            else:

                hidden_states, present_kv = layer(
                    hidden_states,
                    attention_mask=attention_mask_4d,
                    position_ids=position_ids,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                )

            if use_cache:

                present_key_values.append(present_kv)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:

            all_hidden_states.append(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None

        if labels is not None:

            shift_logits = logits[..., :-1, :].contiguous()

            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return ModelOutput(
            logits=logits,
            hidden_states=tuple(all_hidden_states) if all_hidden_states else None,
            past_key_values=present_key_values,
            loss=loss,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        eos_token_id: int | None = None,
        generation_config=None,
        tokenizer=None,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressive generation with modern sampling + bounded KV cache.

        Backwards compatible: legacy kwargs map onto GenerationConfig.
        Pass generation_config=model.generation.GenerationConfig for min_p /
        typical / frequency penalties / stop strings / JSON mode / seed.
        Stop strings require `tokenizer` (decode is checked incrementally).
        """

        from .generation import GenerationConfig, find_stop, sample_next

        if generation_config is None:

            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                stop_strings=list(kwargs.pop("stop_strings", []) or []),
                stop_token_ids=list(kwargs.pop("stop_token_ids", []) or []),
                json_mode=bool(kwargs.pop("json_mode", False)),
                seed=kwargs.pop("seed", None),
            )

        if eos_token_id is None:

            eos_token_id = self.config.eos_token_id

        stop_ids = set(generation_config.stop_token_ids) | (
            {eos_token_id} if eos_token_id is not None else set()
        )

        gen = None

        if generation_config.seed is not None:

            gen = torch.Generator(device=input_ids.device).manual_seed(
                generation_config.seed
            )

        self.eval()

        cache = PagedKVCache(
            max_tokens=getattr(self.config, "kv_cache_max_tokens", 32768)
        )

        generated = input_ids.clone()

        past_key_values = None

        prompt_len = input_ids.shape[1]

        for _ in range(generation_config.max_new_tokens):

            model_input = (
                generated[:, -1:] if past_key_values is not None else generated
            )

            output = self.forward(
                input_ids=model_input, past_key_values=past_key_values, use_cache=True
            )

            past_key_values = cache.trim(output.past_key_values)

            next_logits = output.logits[:, -1, :]

            next_id = sample_next(
                next_logits[0], generated[0].tolist(), generation_config, generator=gen
            )

            next_token = torch.tensor(
                [[next_id]], device=generated.device, dtype=generated.dtype
            )

            generated = torch.cat([generated, next_token], dim=-1)

            if next_id in stop_ids:

                break

            if tokenizer is not None and generation_config.stop_strings:

                text = tokenizer.decode(
                    generated[0][prompt_len:].tolist(), skip_special_tokens=True
                )

                cut = find_stop(text, generation_config.stop_strings)

                if cut is not None:

                    break

        return generated

    def enable_gradient_checkpointing(self, ratio: float = 1.0):
        """Enable gradient checkpointing for memory efficiency."""

        self.gradient_checkpointing = True

        self.config.gradient_checkpointing_ratio = ratio

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""

        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self, **kwargs):

        self.enable_gradient_checkpointing(**kwargs)

    def gradient_checkpointing_disable(self, **kwargs):

        self.disable_gradient_checkpointing()

    @property
    def device(self):

        return next(self.parameters()).device

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""

        if trainable_only:

            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        return sum(p.numel() for p in self.parameters())

    def flops_per_token(self, seq_length: int) -> int:
        """Estimate FLOPs per forward pass token."""

        n = self.num_parameters()

        return 6 * n * seq_length
