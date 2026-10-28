"""Attention-aware document packing for 8k/32k training.

Packing rules (no cross-document attention leakage):
- Documents are concatenated with EOS separators to fill each sequence.
- `position_ids` restart at 0 for every document.
- `labels` are -100 on pad positions (pads never contribute to loss).
- Optionally emits a `doc_ids` array so attention can be blocked across
  documents (consumed by training collators / FlashAttention varlen).
"""

from __future__ import annotations



from dataclasses import dataclass, field

from typing import Dict, Iterable, List, Optional



import numpy as np





@dataclass

class PackedSequence:

    input_ids: np.ndarray                  

    labels: np.ndarray                                   

    position_ids: np.ndarray                                    

    doc_ids: np.ndarray                                

    num_docs: int = 0

    num_pad: int = 0





@dataclass

class PackingStats:

    sequences: int = 0

    documents: int = 0

    tokens: int = 0

    pad_tokens: int = 0



    @property

    def efficiency(self) -> float:

        return 1.0 - (self.pad_tokens / max(self.tokens, 1))





class SequencePacker:

    """First-fit-decreasing-free streaming packer: greedy fill, no document
    is split across sequences unless it alone exceeds seq_length (then it is
    chunked with continuation positions)."""



    def __init__(self, tokenizer, seq_length: int = 8192, pad_id: Optional[int] = None):

        self.tokenizer = tokenizer

        self.seq_length = seq_length

        self.pad_id = tokenizer.pad_token_id if pad_id is None else pad_id

        self.eos_id = tokenizer.eos_token_id

        self.stats = PackingStats()

        self._reset()



    def _reset(self):

        self._ids: List[int] = []

        self._pos: List[int] = []

        self._doc: List[int] = []

        self._ndocs = 0



    def _flush(self) -> PackedSequence:

        n = len(self._ids)

        pad = self.seq_length - n

        input_ids = np.full(self.seq_length, self.pad_id, dtype=np.int64)

        labels = np.full(self.seq_length, -100, dtype=np.int64)

        position_ids = np.zeros(self.seq_length, dtype=np.int64)

        doc_ids = np.full(self.seq_length, -1, dtype=np.int64)

        input_ids[:n] = self._ids

        labels[:n] = self._ids                                              

        position_ids[:n] = self._pos

        doc_ids[:n] = self._doc

        seq = PackedSequence(input_ids, labels, position_ids, doc_ids, self._ndocs, pad)

        self.stats.sequences += 1

        self.stats.tokens += self.seq_length

        self.stats.pad_tokens += pad

        self._reset()

        return seq



    def add_tokens(self, tokens: List[int]) -> List[PackedSequence]:

        """Add one document's tokens (EOS appended if missing)."""

        if not tokens:

            return []

        if tokens[-1] != self.eos_id:

            tokens = list(tokens) + [self.eos_id]

        done: List[PackedSequence] = []

                                              

        while len(tokens) > self.seq_length:

            if self._ids:

                done.append(self._flush())

            chunk, tokens = tokens[: self.seq_length], tokens[self.seq_length:]

            self._ids, self._pos = list(chunk), list(range(len(chunk)))

            self._doc = [self._ndocs] * len(chunk)

            self._ndocs += 1

            self.stats.documents += 1

            done.append(self._flush())

        if len(self._ids) + len(tokens) > self.seq_length:

            done.append(self._flush())

        start = len(self._ids)

        self._ids.extend(tokens)

        self._pos.extend(range(len(tokens)))

        self._doc.extend([self._ndocs] * len(tokens))

        self._ndocs += 1

        self.stats.documents += 1

        if len(self._ids) == self.seq_length:

            done.append(self._flush())

        return done



    def add_document(self, text: str) -> List[PackedSequence]:

        return self.add_tokens(self.tokenizer.encode(text, add_special_tokens=False))



    def flush(self) -> List[PackedSequence]:

        return [self._flush()] if self._ids else []





def pack_documents(tokenizer, texts: Iterable[str], seq_length: int = 8192) -> List[PackedSequence]:

    packer = SequencePacker(tokenizer, seq_length=seq_length)

    out: List[PackedSequence] = []

    for t in texts:

        out.extend(packer.add_document(t))

    out.extend(packer.flush())

    return out





def collate_packed(sequences: List[PackedSequence]) -> Dict[str, np.ndarray]:

    """Stack packed sequences into a training batch dict."""

    return {

        "input_ids": np.stack([s.input_ids for s in sequences]),

        "labels": np.stack([s.labels for s in sequences]),

        "position_ids": np.stack([s.position_ids for s in sequences]),

        "doc_ids": np.stack([s.doc_ids for s in sequences]),

    }

