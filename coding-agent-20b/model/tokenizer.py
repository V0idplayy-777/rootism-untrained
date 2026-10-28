"""
Code-Optimized Tokenizer for the Coding Agent Model.

This tokenizer is specifically designed for code generation tasks. It:
1. Handles code-specific tokens (indentation, brackets, operators)
2. Preserves whitespace significance (critical for Python)
3. Has a vocabulary optimized for programming languages
4. Supports special tokens for chat/instruction format
5. Uses BPE with code-aware pre-tokenization

For production use, this builds on top of sentencepiece/tiktoken patterns
but with code-specific extensions.
"""







import json



import os



import re



import unicodedata



from collections import Counter, defaultdict



from typing import List, Optional, Dict, Tuple, Union



import heapq











                                                              



                



                                                              







SPECIAL_TOKENS = {



    "<pad>": 0,



    "<bos>": 1,



    "<eos>": 2,



    "<unk>": 3,



                                  



    "<|system|>": 4,



    "<|user|>": 5,



    "<|assistant|>": 6,



    "<|code_start|>": 7,



    "<|code_end|>": 8,



    "<|comment|>": 9,



    "<|string|>": 10,



    "<|function|>": 11,



    "<|class|>": 12,



    "<|import|>": 13,



    "<|return|>": 14,



    "<|indent|>": 15,



    "<|dedent|>": 16,



    "<|newline|>": 17,



    "<|tab|>": 18,



    "<|space|>": 19,



                   



    "<|python|>": 20,



    "<|javascript|>": 21,



    "<|typescript|>": 22,



    "<|java|>": 23,



    "<|cpp|>": 24,



    "<|rust|>": 25,



    "<|go|>": 26,



    "<|ruby|>": 27,



    "<|php|>": 28,



    "<|swift|>": 29,



    "<|kotlin|>": 30,



    "<|csharp|>": 31,



    "<|scala|>": 32,



    "<|shell|>": 33,



    "<|sql|>": 34,



    "<|html|>": 35,



    "<|css|>": 36,



    "<|json|>": 37,



    "<|yaml|>": 38,



    "<|xml|>": 39,



    "<|markdown|>": 40,



    "<|latex|>": 41,



    "<|r|>": 42,



    "<|lua|>": 43,



    "<|perl|>": 44,



    "<|haskell|>": 45,



    "<|elixir|>": 46,



    "<|clojure|>": 47,



                           



    "<|tool_call|>": 48,



    "<|tool_result|>": 49,



    "<|function_call|>": 50,



    "<|function_result|>": 51,



               



    "<|think|>": 52,



    "<|answer|>": 53,



    "<|search|>": 54,



    "<|execute|>": 55,



                                              



    "<|fim_prefix|>": 56,



    "<|fim_middle|>": 57,



    "<|fim_suffix|>": 58,



                        



    "<|repo_name|>": 59,



    "<|file_sep|>": 60,



    "<|file_path|>": 61,



                    



    "<|reserved_62|>": 62,



    "<|reserved_63|>": 63,



}







NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)      











                                                              



                          



                                                              







class CodePreTokenizer:



    """
    Pre-tokenizer optimized for source code.
    
    Splits code into meaningful units while preserving:
    - Indentation (spaces and tabs)
    - String literals
    - Comments
    - Operators
    - Identifiers and keywords
    """







                                         



    CODE_PATTERN = re.compile(



        r"""
        (?:'''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\") |
        (?:'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*") |
        (?:`(?:[^`\\]|\\.)*`) |
        (?://[^\n]*|[\x23][^\n]*) |
        (?:/\*[\s\S]*?\*/) |
        (?:0[xX][0-9a-fA-F]+|0[oO][0-7]+|0[bB][01]+|\d+\.?\d*(?:[eE][+-]?\d+)?) |
        (?:[a-zA-Z_$][\w$]*) |
        (?:::=|>>>|<<=|>>=|===|!==|\?\?|\?\.) |
        (?:[+\-*/%&|^~<>!]=?|[<>&|]{2}|->|=>|\.\.\.|\*\*) |
        (?:[{}()\[\];:,.?@&~^]) |
        (?:[ \t]+) |
        (?:\n+) |
        (?:\S)
        """,



        re.VERBOSE,



    )







    def pre_tokenize(self, text: str) -> List[str]:



        """Split text into pre-tokens."""



        return self.CODE_PATTERN.findall(text)











                                                              



                                                   



                                                              







class BPETrainer:



    """
    BPE tokenizer trainer optimized for code.
    
    Trains a BPE vocabulary from raw code text, with special handling for:
    - Preserving common code patterns
    - Including all ASCII printable characters as base tokens
    - Adding common multi-character operators
    """







    def __init__(



        self,



        vocab_size: int = 128256,



        min_frequency: int = 2,



        special_tokens: Optional[Dict[str, int]] = None,



    ):



        self.vocab_size = vocab_size



        self.min_frequency = min_frequency



        self.special_tokens = special_tokens or SPECIAL_TOKENS



        self.num_special = len(self.special_tokens)







                                                   



        self.num_merges = vocab_size - self.num_special







    def train(self, texts: List[str], verbose: bool = True) -> 'CodeTokenizer':



        """
        Train BPE on a corpus of texts.
        
        Args:
            texts: List of text strings to train on
            verbose: Whether to print training progress
            
        Returns:
            Trained CodeTokenizer
        """



        pre_tokenizer = CodePreTokenizer()







        if verbose:



            print("Step 1: Pre-tokenizing corpus...")







                                



        word_freqs = Counter()



        for text in texts:



            pre_tokens = pre_tokenizer.pre_tokenize(text)



            for token in pre_tokens:



                                                      



                word = self._to_byte_level(token)



                word_freqs[word] += 1







        if verbose:



            print(f"  Found {len(word_freqs)} unique pre-tokens")







                                                     



        base_vocab = set()



        for i in range(256):



            base_vocab.add(self._byte_to_char(i))







                                                  



        for word in word_freqs:



            for char in word:



                base_vocab.add(char)







        if verbose:



            print(f"  Base vocabulary size: {len(base_vocab)}")



            print("Step 2: Training BPE merges...")







                                         



        splits = {word: list(word) for word in word_freqs}







                              



        merges = []



        vocab = base_vocab.copy()







        for i in range(self.num_merges):



            if verbose and (i + 1) % 5000 == 0:



                print(f"  Merge {i + 1}/{self.num_merges} | Vocab size: {len(vocab) + self.num_special}")







                                         



            pair_freqs = Counter()



            for word, freq in word_freqs.items():



                chars = splits[word]



                for j in range(len(chars) - 1):



                    pair = (chars[j], chars[j + 1])



                    pair_freqs[pair] += freq







            if not pair_freqs:



                if verbose:



                    print("  No more pairs to merge. Stopping early.")



                break







            best_pair = pair_freqs.most_common(1)[0][0]







            if pair_freqs[best_pair] < self.min_frequency:



                if verbose:



                    print(f"  Best pair frequency ({pair_freqs[best_pair]}) below minimum. Stopping.")



                break







                         



            merges.append(best_pair)



            merged = best_pair[0] + best_pair[1]



            vocab.add(merged)







                           



            for word in splits:



                chars = splits[word]



                new_chars = []



                j = 0



                while j < len(chars):



                    if j < len(chars) - 1 and chars[j] == best_pair[0] and chars[j + 1] == best_pair[1]:



                        new_chars.append(merged)



                        j += 2



                    else:



                        new_chars.append(chars[j])



                        j += 1



                splits[word] = new_chars







        if verbose:



            print(f"  Training complete. Learned {len(merges)} merges.")



            print(f"  Final vocabulary size: {len(vocab) + self.num_special}")







                             



        return CodeTokenizer(



            vocab=sorted(vocab),



            merges=merges,



            special_tokens=self.special_tokens,



        )







    def _to_byte_level(self, text: str) -> str:



        """Convert text to byte-level representation."""



        return ''.join(self._byte_to_char(b) for b in text.encode('utf-8'))







    def _byte_to_char(self, byte_val: int) -> str:



        """Map a byte value to a unique unicode character."""



                                                            



        if 33 <= byte_val <= 126:



            return chr(byte_val)



        elif byte_val == 32:



            return 'Ġ'         



        elif byte_val == 10:



            return 'Ċ'           



        elif byte_val == 9:



            return 'ĉ'       



        else:



            return chr(byte_val + 256)











                                                              



           



                                                              







class CodeTokenizer:



    """
    Code-optimized BPE Tokenizer.
    
    Features:
    - Byte-level BPE (handles any unicode)
    - Code-aware pre-tokenization
    - Special tokens for chat/instruction format
    - Fill-in-the-middle support for code completion
    - Efficient batch encoding
    """







    def __init__(



        self,



        vocab: Optional[List[str]] = None,



        merges: Optional[List[Tuple[str, str]]] = None,



        special_tokens: Optional[Dict[str, int]] = None,



    ):



        self.special_tokens = special_tokens or SPECIAL_TOKENS



        self.num_special = len(self.special_tokens)







                          



        if vocab is not None:



            self._build_vocab(vocab, merges or [])



        else:



                                            



            self._build_default_vocab()







                       



        self.pre_tokenizer = CodePreTokenizer()







    def _build_default_vocab(self):



        """Build a default byte-level vocabulary."""



        self.token_to_id = dict(self.special_tokens)



        self.id_to_token = {v: k for k, v in self.special_tokens.items()}







                                           



        for i in range(256):



            char = self._byte_to_char(i)



            if char not in self.token_to_id:



                idx = len(self.token_to_id)



                self.token_to_id[char] = idx



                self.id_to_token[idx] = char







        self.merges = []



        self.merge_ranks = {}







    def _build_vocab(self, vocab: List[str], merges: List[Tuple[str, str]]):



        """Build vocabulary from trained BPE data.

        Byte-fallback guarantee: all 256 byte-level symbols are always
        present, so ANY unicode string encodes without <unk>.
        """



        self.token_to_id = dict(self.special_tokens)



        self.id_to_token = {v: k for k, v in self.special_tokens.items()}







                                                                                



        for i in range(256):



            char = self._byte_to_char(i)



            if char not in self.token_to_id:



                idx = len(self.token_to_id)



                self.token_to_id[char] = idx



                self.id_to_token[idx] = char







                            



        for token in vocab:



            if token not in self.token_to_id:



                idx = len(self.token_to_id)



                self.token_to_id[token] = idx



                self.id_to_token[idx] = token







        self.merges = merges



        self.merge_ranks = {merge: i for i, merge in enumerate(merges)}







    def _byte_to_char(self, byte_val: int) -> str:



        """Map a byte value to a unique unicode character."""



        if 33 <= byte_val <= 126:



            return chr(byte_val)



        elif byte_val == 32:



            return 'Ġ'



        elif byte_val == 10:



            return 'Ċ'



        elif byte_val == 9:



            return 'ĉ'



        else:



            return chr(byte_val + 256)







    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:



        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS token
            
        Returns:
            List of token IDs
        """



        tokens = []







        if add_special_tokens:



            tokens.append(self.special_tokens["<bos>"])







                      



        pre_tokens = self.pre_tokenizer.pre_tokenize(text)







        for pre_token in pre_tokens:



                                   



            bytes_str = ''.join(self._byte_to_char(b) for b in pre_token.encode('utf-8'))







                                                                        



                                                                            



            bpe_tokens = self._apply_bpe(bytes_str)



            for token in bpe_tokens:



                token_id = self.token_to_id.get(token)



                if token_id is not None:



                    tokens.append(token_id)



                else:



                    for ch in token:



                        tokens.append(self.token_to_id.get(ch, self.special_tokens["<unk>"]))







        return tokens







    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:



        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text string
        """



        text_parts = []







        for token_id in token_ids:



            if skip_special_tokens and token_id < self.num_special:



                continue







            token = self.id_to_token.get(token_id, "")



            text_parts.append(token)







                                                  



        text = ''.join(text_parts)



        return self._byte_level_decode(text)







    def _apply_bpe(self, word: str) -> List[str]:



        """Apply BPE merges to a word."""



        if not self.merges:



            return list(word)







        chars = list(word)







        while len(chars) > 1:



                                 



            best_rank = float('inf')



            best_idx = -1







            for i in range(len(chars) - 1):



                pair = (chars[i], chars[i + 1])



                rank = self.merge_ranks.get(pair, float('inf'))



                if rank < best_rank:



                    best_rank = rank



                    best_idx = i







            if best_rank == float('inf'):



                break







                         



            chars[best_idx] = chars[best_idx] + chars[best_idx + 1]



            chars.pop(best_idx + 1)







        return chars







    def _byte_level_decode(self, text: str) -> str:



        """Convert byte-level representation back to text."""



                                          



        byte_map = {}



        for i in range(256):



            char = self._byte_to_char(i)



            byte_map[char] = i







        bytes_list = []



        for char in text:



            if char in byte_map:



                bytes_list.append(byte_map[char])



            elif ord(char) >= 256:



                bytes_list.append(ord(char) - 256)



            else:



                bytes_list.append(ord(char))







        try:



            return bytes(bytes_list).decode('utf-8', errors='replace')



        except Exception:



            return text







    def encode_batch(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:



        """Encode a batch of texts."""



        return [self.encode(text, add_special_tokens) for text in texts]







    def decode_batch(self, token_ids_batch: List[List[int]], skip_special_tokens: bool = True) -> List[str]:



        """Decode a batch of token ID lists."""



        return [self.decode(ids, skip_special_tokens) for ids in token_ids_batch]







    def __len__(self) -> int:



        return len(self.token_to_id)







    @property



    def vocab_size(self) -> int:



        return len(self.token_to_id)







    @property



    def bos_token_id(self) -> int:



        return self.special_tokens["<bos>"]







    @property



    def eos_token_id(self) -> int:



        return self.special_tokens["<eos>"]







    @property



    def pad_token_id(self) -> int:



        return self.special_tokens["<pad>"]







    def get_token_id(self, token: str) -> int:



        """Get the ID for a token."""



        return self.token_to_id.get(token, self.special_tokens["<unk>"])







    def get_token(self, token_id: int) -> str:



        """Get the token for an ID."""



        return self.id_to_token.get(token_id, "<unk>")







    def save(self, directory: str):



        """Save tokenizer to directory."""



        os.makedirs(directory, exist_ok=True)







                         



        vocab_data = {



            "token_to_id": self.token_to_id,



            "merges": [(m[0], m[1]) for m in self.merges],



            "special_tokens": self.special_tokens,



        }



        with open(os.path.join(directory, "tokenizer.json"), "w") as f:



            json.dump(vocab_data, f, indent=2, ensure_ascii=False)







    @classmethod



    def load(cls, directory: str) -> 'CodeTokenizer':



        """Load tokenizer from directory."""



        with open(os.path.join(directory, "tokenizer.json"), "r") as f:



            data = json.load(f)







        tokenizer = cls(special_tokens=data["special_tokens"])



        tokenizer.token_to_id = data["token_to_id"]



        tokenizer.id_to_token = {v: k for k, v in data["token_to_id"].items()}



        tokenizer.merges = [tuple(m) for m in data["merges"]]



        tokenizer.merge_ranks = {tuple(m): i for i, m in enumerate(data["merges"])}







        return tokenizer











                                                              



                                                                



                                                              







class EfficientBPETrainer:



    """Scalable byte-level BPE trainer.

    Fixes vs. the legacy BPETrainer:
    - Streams the corpus (iterator of strings / files); no corpus-size cap
      and the corpus is never materialised in RAM.
    - Incremental pair counting with an inverted word->pairs index and a
      max-heap, so each merge only recounts words containing the merged
      pair (near O(corpus) total work instead of O(vocab x corpus)).
    - Byte fallback: the 256 byte symbols are always in the vocab.
    - Deterministic tie-breaking (pair frequency, then lexicographic) so
      training is reproducible for a fixed seed/corpus order.
    """







    def __init__(self, vocab_size: int = 128256, min_frequency: int = 2,



                 special_tokens: Optional[Dict[str, int]] = None):



        self.vocab_size = vocab_size



        self.min_frequency = min_frequency



        self.special_tokens = special_tokens or SPECIAL_TOKENS



        self.pre_tokenizer = CodePreTokenizer()







    @staticmethod



    def _byte_to_char(byte_val: int) -> str:



        if 33 <= byte_val <= 126:



            return chr(byte_val)



        if byte_val == 32:



            return "Ġ"



        if byte_val == 10:



            return "Ċ"



        if byte_val == 9:



            return "ĉ"



        return chr(byte_val + 256)







    def _count_word_freqs(self, texts) -> Counter:



        word_freqs: Counter = Counter()



        for text in texts:                                     



            if not text:



                continue



            for pre in self.pre_tokenizer.pre_tokenize(text):



                word = "".join(self._byte_to_char(b) for b in pre.encode("utf-8"))



                word_freqs[word] += 1



        return word_freqs







    def train_from_iterator(self, texts, verbose: bool = True) -> "CodeTokenizer":



        word_freqs = self._count_word_freqs(texts)



        if verbose:



            print(f"Unique pre-tokens: {len(word_freqs):,}")



        return self._train_from_word_freqs(word_freqs, verbose=verbose)







    def train(self, texts: List[str], verbose: bool = True) -> "CodeTokenizer":



        """Backwards-compatible list API (delegates to the streaming path)."""



        return self.train_from_iterator(iter(texts), verbose=verbose)







    def _train_from_word_freqs(self, word_freqs: Counter, verbose: bool = True) -> "CodeTokenizer":



        num_merges = max(0, self.vocab_size - len(self.special_tokens) - 256)



        splits: Dict[str, List[str]] = {w: list(w) for w in word_freqs}



                                                         



        pair_counts: Counter = Counter()



        pair_to_words: Dict[Tuple[str, str], set] = defaultdict(set)



        for word, freq in word_freqs.items():



            chars = splits[word]



            for j in range(len(chars) - 1):



                pair = (chars[j], chars[j + 1])



                pair_counts[pair] += freq



                pair_to_words[pair].add(word)







        def heap_key(pair, count):



            return (-count, pair)                                                      







        heap = [heap_key(p, c) for p, c in pair_counts.items()]



        heapq.heapify(heap)



        merges: List[Tuple[str, str]] = []



        vocab = {self._byte_to_char(i) for i in range(256)}







        def push(pair):



            c = pair_counts.get(pair, 0)



            if c >= self.min_frequency:



                heapq.heappush(heap, heap_key(pair, c))







        while heap and len(merges) < num_merges:



            neg_count, pair = heapq.heappop(heap)



            count = -neg_count



            if pair_counts.get(pair, 0) != count or count < self.min_frequency:



                continue                    



            merged = pair[0] + pair[1]



            merges.append(pair)



            vocab.add(merged)



            affected = list(pair_to_words.get(pair, ()))



            for word in affected:



                freq = word_freqs[word]



                chars = splits[word]



                                                 



                for j in range(len(chars) - 1):



                    old = (chars[j], chars[j + 1])



                    pair_counts[old] -= freq



                    if pair_counts[old] <= 0:



                        del pair_counts[old]



                new_chars, j = [], 0



                while j < len(chars):



                    if j < len(chars) - 1 and chars[j] == pair[0] and chars[j + 1] == pair[1]:



                        new_chars.append(merged)



                        j += 2



                    else:



                        new_chars.append(chars[j])



                        j += 1



                splits[word] = new_chars



                for j in range(len(new_chars) - 1):



                    new_pair = (new_chars[j], new_chars[j + 1])



                    pair_counts[new_pair] = pair_counts.get(new_pair, 0) + freq



                    pair_to_words[new_pair].add(word)



                    push(new_pair)



            pair_to_words.pop(pair, None)



            if verbose and len(merges) % 5000 == 0:



                print(f"  Merge {len(merges)}/{num_merges}")



        if verbose:



            print(f"Learned {len(merges)} merges; vocab={len(vocab) + len(self.special_tokens):,}")



        return CodeTokenizer(vocab=sorted(vocab), merges=merges, special_tokens=self.special_tokens)











class TokenizerValidator:



    """Compression / round-trip / fertility validation for a trained tokenizer."""







    @staticmethod



    def validate(tokenizer: "CodeTokenizer", samples: List[str]) -> Dict:



        total_chars = total_bytes = total_tokens = unk = 0



        roundtrip_ok = True



        per_sample = []



        for s in samples:



            ids = tokenizer.encode(s, add_special_tokens=False)



            back = tokenizer.decode(ids, skip_special_tokens=True)



            ok = (back == s)



            roundtrip_ok = roundtrip_ok and ok



            total_chars += len(s)



            total_bytes += len(s.encode("utf-8"))



            total_tokens += max(len(ids), 1)



            unk += sum(1 for i in ids if i == tokenizer.special_tokens["<unk>"])



            per_sample.append({"chars": len(s), "tokens": len(ids),



                               "chars_per_token": len(s) / max(len(ids), 1), "roundtrip": ok})



        return {



            "num_samples": len(samples),



            "chars_per_token": total_chars / max(total_tokens, 1),



            "bytes_per_token": total_bytes / max(total_tokens, 1),



            "unk_rate": unk / max(total_tokens, 1),



            "roundtrip_ok": roundtrip_ok,



            "per_sample": per_sample,



        }







    @staticmethod



    def assert_production_ready(report: Dict, min_chars_per_token: float = 2.5,



                                  max_unk_rate: float = 0.001) -> None:



        assert report["roundtrip_ok"], "tokenizer round-trip failure"



        assert report["unk_rate"] <= max_unk_rate, f"unk_rate {report['unk_rate']} too high (byte-fallback broken?)"



        assert report["chars_per_token"] >= min_chars_per_token, (



            f"chars/token {report['chars_per_token']:.2f} < {min_chars_per_token}: merges are not helping")











                                                              



               



                                                              







class CodeChatTemplate:



    """
    Chat template for formatting conversations in the coding agent format.
    Supports multi-turn conversations with system prompts and tool use.
    """







    def __init__(self, tokenizer: CodeTokenizer):



        self.tokenizer = tokenizer







    def format_message(self, role: str, content: str, language: Optional[str] = None) -> str:



        """Format a single message."""



        role_tokens = {



            "system": "<|system|>",



            "user": "<|user|>",



            "assistant": "<|assistant|>",



            "tool": "<|tool_result|>",



        }



        role_token = role_tokens.get(role, "<|user|>")







        parts = [role_token]







        if language:



            lang_tokens = {



                "python": "<|python|>",



                "javascript": "<|javascript|>",



                "typescript": "<|typescript|>",



                "java": "<|java|>",



                "cpp": "<|cpp|>",



                "rust": "<|rust|>",



                "go": "<|go|>",



            }



            if language in lang_tokens:



                parts.append(lang_tokens[language])







        parts.append(content)



        parts.append("<eos>")







        return "\n".join(parts)







    def format_conversation(self, messages: List[Dict[str, str]]) -> str:



        """Format a full conversation."""



        formatted = []



        for msg in messages:



            formatted.append(self.format_message(



                role=msg["role"],



                content=msg["content"],



                language=msg.get("language"),



            ))



        return "\n".join(formatted)







    def encode_conversation(self, messages: List[Dict[str, str]]) -> List[int]:



        """Encode a conversation to token IDs."""



        text = self.format_conversation(messages)



        return self.tokenizer.encode(text, add_special_tokens=True)







                                                                           



                                                                



                                                                               



                                                            



                                                                         



    @staticmethod



    def format_tool_call(tool: str, arguments: Dict) -> str:



        import json as _json



        return "<|tool_call|>\n" + _json.dumps({"tool": tool, "arguments": arguments}) + "\n<eos>"







    @staticmethod



    def format_tool_result(tool: str, result: Dict) -> str:



        import json as _json



        return "<|tool_result|>\n" + _json.dumps({"tool": tool, "result": result}) + "\n<eos>"



