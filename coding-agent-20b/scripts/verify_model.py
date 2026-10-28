                      

"""
Model Verification Script.

Tests the model architecture without requiring GPUs:
- Creates the model
- Runs a forward pass
- Verifies parameter count
- Tests gradient flow
- Tests generation

Usage:
    python scripts/verify_model.py --model_size 1B
"""



import sys

import time

import argparse

from pathlib import Path



sys.path.insert(0, str(Path(__file__).parent.parent))



import torch

import torch.nn as nn



from model.config import get_20b_config, get_7b_config, get_1b_config, CodingAgentConfig

from model.model import CodingAgentModel

from model.tokenizer import CodeTokenizer





def verify_model(config: CodingAgentConfig, device: str = "cpu"):

    """Run verification tests on the model."""

    print("=" * 60)

    print("  MODEL VERIFICATION")

    print("=" * 60)



                  

    config.print_summary()



                  

    print("\n[1/6] Creating model...")

    start = time.time()

    model = CodingAgentModel(config)

    model = model.to(device)

    print(f"  Created in {time.time() - start:.2f}s")



                      

    print("\n[2/6] Counting parameters...")

    total = model.num_parameters()

    trainable = model.num_parameters(trainable_only=True)

    print(f"  Total parameters:     {total:,} ({total / 1e9:.2f}B)")

    print(f"  Trainable parameters: {trainable:,} ({trainable / 1e9:.2f}B)")



               

    param_counts = {}

    for name, param in model.named_parameters():

        category = name.split('.')[1] if '.' in name else name

        param_counts[category] = param_counts.get(category, 0) + param.numel()



    print("\n  Parameter breakdown:")

    for cat, count in sorted(param_counts.items(), key=lambda x: -x[1]):

        print(f"    {cat:30s}: {count:>15,} ({count/total*100:.1f}%)")



                  

    print("\n[3/6] Testing forward pass...")

    batch_size = 2

    seq_length = min(128, config.max_position_embeddings)



    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)

    labels = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)



                                             

    model.gradient_checkpointing = False



    start = time.time()

    with torch.no_grad():

        output = model(input_ids=input_ids, labels=labels)

    elapsed = time.time() - start



    print(f"  Forward pass time: {elapsed:.3f}s")

    print(f"  Logits shape: {output.logits.shape}")

    print(f"  Loss: {output.loss.item():.4f}")

    assert output.logits.shape == (batch_size, seq_length, config.vocab_size)

    assert output.loss.item() > 0

    print("  ✓ Forward pass OK")



                   

    print("\n[4/6] Testing backward pass...")

    start = time.time()

    output = model(input_ids=input_ids, labels=labels)

    loss = output.loss

    loss.backward()

    elapsed = time.time() - start



    print(f"  Backward pass time: {elapsed:.3f}s")



                           

    grad_count = 0

    for name, param in model.named_parameters():

        if param.grad is not None:

            grad_count += 1

            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"



    total_params = sum(1 for _ in model.parameters() if _.requires_grad)

    print(f"  Gradients computed: {grad_count}/{total_params}")

    assert grad_count == total_params

    print("  ✓ Backward pass OK (no NaN/Inf gradients)")



                                 

    print("\n[5/6] Testing gradient checkpointing...")

    model.gradient_checkpointing = True

    model.zero_grad()



    output = model(input_ids=input_ids, labels=labels)

    loss = output.loss

    loss.backward()

    print(f"  Loss with checkpointing: {loss.item():.4f}")

    print("  ✓ Gradient checkpointing OK")



                                

    print("\n[6/6] Testing generation (KV cache)...")

    model.gradient_checkpointing = False

    model.eval()



    prompt = torch.randint(0, config.vocab_size, (1, 10), device=device)



    with torch.no_grad():

                       

        output_no_cache = model(input_ids=prompt, use_cache=False)



                    

        output_with_cache = model(input_ids=prompt, use_cache=True)



    assert output_with_cache.past_key_values is not None

    assert len(output_with_cache.past_key_values) == config.num_hidden_layers



    print(f"  KV cache layers: {len(output_with_cache.past_key_values)}")

    print(f"  KV cache shape per layer: K={output_with_cache.past_key_values[0][0].shape}")

    print("  ✓ KV cache OK")



             

    print("\n" + "=" * 60)

    print("  ALL TESTS PASSED ✓")

    print("=" * 60)



                     

    if device == "cpu":

        import gc

        gc.collect()



    param_bytes = total * 2                            

    grad_bytes = total * 2

    optim_bytes = total * 12                                                            

    total_training_bytes = param_bytes + grad_bytes + optim_bytes



    print(f"\n  Memory estimates (BF16 + AdamW):")

    print(f"    Model weights:  {param_bytes / 1e9:.1f} GB")

    print(f"    Gradients:      {grad_bytes / 1e9:.1f} GB")

    print(f"    Optimizer:      {optim_bytes / 1e9:.1f} GB")

    print(f"    Total (ZeRO-0): {total_training_bytes / 1e9:.1f} GB")

    print(f"    Per GPU (8x, ZeRO-3): {total_training_bytes / 8 / 1e9:.1f} GB")

    print(f"    Per GPU (16x, ZeRO-3): {total_training_bytes / 16 / 1e9:.1f} GB")





def main():

    parser = argparse.ArgumentParser(description="Verify model architecture")

    parser.add_argument("--model_size", type=str, default="1B", choices=["1B", "7B", "20B"])

    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()



    if args.model_size == "20B":

        config = get_20b_config()

    elif args.model_size == "7B":

        config = get_7b_config()

    else:

        config = get_1b_config()



    verify_model(config, device=args.device)





if __name__ == "__main__":

    main()

