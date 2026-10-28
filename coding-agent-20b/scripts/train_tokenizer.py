                      

"""
Train a production-grade tokenizer on real code data.

This script trains a BPE tokenizer using tiktoken (same as GPT-4) or SentencePiece
on a large corpus of code. For best results, use with The Stack or similar datasets.

Usage:
    # Train on local code files
    python scripts/train_tokenizer.py --data_dir /path/to/code --output tokenizer/
    
    # Train on The Stack (requires huggingface datasets)
    python scripts/train_tokenizer.py --dataset bigcode/the-stack --output tokenizer/
    
    # Train on StarCoderData
    python scripts/train_tokenizer.py --dataset bigcode/starcoderdata --output tokenizer/
"""



import os

import sys

import json

import argparse

import logging

from pathlib import Path

from typing import List, Iterator, Optional

import tempfile

import shutil



logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

logger = logging.getLogger(__name__)





                                                              

              

                                                              



def load_code_from_directory(data_dir: str, extensions: List[str] = None) -> Iterator[str]:

    """Load code files from a directory."""

    if extensions is None:

        extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', 

                     '.rs', '.go', '.rb', '.php', '.swift', '.kt', '.scala', '.cs',

                     '.html', '.css', '.sql', '.sh', '.bash', '.yml', '.yaml', '.json',

                     '.md', '.txt', '.xml']

    

    data_path = Path(data_dir)

    files = []

    for ext in extensions:

        files.extend(data_path.rglob(f'*{ext}'))

    

    logger.info(f"Found {len(files)} code files")

    

    for filepath in files:

        try:

            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:

                content = f.read()

                if len(content) > 100:                   

                    yield content

        except Exception as e:

            logger.warning(f"Error reading {filepath}: {e}")





def load_from_huggingface(dataset_name: str, split: str = "train", 

                          streaming: bool = True, max_samples: int = None) -> Iterator[str]:

    """Load code from HuggingFace datasets."""

    try:

        from datasets import load_dataset

    except ImportError:

        logger.error("Install datasets: pip install datasets")

        sys.exit(1)

    

    logger.info(f"Loading dataset: {dataset_name}")

    

    if streaming:

        dataset = load_dataset(dataset_name, split=split, streaming=True)

        count = 0

        for sample in dataset:

                                              

            if 'content' in sample:

                yield sample['content']

            elif 'text' in sample:

                yield sample['text']

            elif 'code' in sample:

                yield sample['code']

            

            count += 1

            if max_samples and count >= max_samples:

                break

            if count % 10000 == 0:

                logger.info(f"  Loaded {count} samples")

    else:

        dataset = load_dataset(dataset_name, split=split)

        for sample in dataset:

            if 'content' in sample:

                yield sample['content']

            elif 'text' in sample:

                yield sample['text']

            elif 'code' in sample:

                yield sample['code']





                                                              

                   

                                                              



def train_tiktoken(text_iterator: Iterator[str], vocab_size: int, output_dir: str):

    """Train a tiktoken BPE tokenizer."""

    try:

        import tiktoken

        from tiktoken.load import load_tiktoken_bpe

    except ImportError:

        logger.error("Install tiktoken: pip install tiktoken")

        sys.exit(1)

    

    logger.info(f"Training tiktoken BPE (vocab_size={vocab_size})")

    

                           

    logger.info("Collecting training text...")

    training_text = []

    total_chars = 0

    

    for text in text_iterator:

        training_text.append(text)

        total_chars += len(text)

        if total_chars > 10_000_000:                       

            break

    

    logger.info(f"Collected {len(training_text)} documents, {total_chars:,} characters")

    

               

                                                                                     

                                                     

    logger.info("Training BPE merges...")

    

                                                   

                                                         

    from model.tokenizer import BPETrainer, CodeTokenizer

    

    trainer = BPETrainer(vocab_size=vocab_size, min_frequency=2)

    tokenizer = trainer.train(training_text, verbose=True)

    

                    

    os.makedirs(output_dir, exist_ok=True)

    tokenizer.save(output_dir)

    

    logger.info(f"Tokenizer saved to {output_dir}")

    logger.info(f"Vocabulary size: {tokenizer.vocab_size}")

    

    return tokenizer





                                                              

                        

                                                              



def train_sentencepiece(text_iterator: Iterator[str], vocab_size: int, output_dir: str):

    """Train a SentencePiece tokenizer."""

    try:

        import sentencepiece as spm

    except ImportError:

        logger.error("Install sentencepiece: pip install sentencepiece")

        sys.exit(1)

    

    logger.info(f"Training SentencePiece (vocab_size={vocab_size})")

    

                                      

    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:

        temp_path = f.name

        total_chars = 0

        

        for text in text_iterator:

            f.write(text + '\n')

            total_chars += len(text)

            if total_chars > 10_000_000:             

                break

    

    logger.info(f"Wrote training data to {temp_path}")

    

                               

    os.makedirs(output_dir, exist_ok=True)

    model_prefix = os.path.join(output_dir, "tokenizer")

    

    spm.SentencePieceTrainer.train(

        input=temp_path,

        model_prefix=model_prefix,

        vocab_size=vocab_size,

        model_type='bpe',

        character_coverage=1.0,

        num_threads=os.cpu_count(),

        byte_fallback=True,                            

        split_digits=True,

        split_by_whitespace=False,                      

        treat_whitespace_as_suffix=False,

        add_dummy_prefix=False,                                

        max_sentence_length=16384,

        shuffle_input_sentence=True,

    )

    

                        

    os.unlink(temp_path)

    

    logger.info(f"SentencePiece model saved to {model_prefix}.model")

    

                                                    

    create_sentencepiece_wrapper(model_prefix, output_dir)

    

    return model_prefix





def create_sentencepiece_wrapper(model_prefix: str, output_dir: str):

    """Create a wrapper to use SentencePiece with our CodeTokenizer interface."""

    import sentencepiece as spm

    

    sp = spm.SentencePieceProcessor()

    sp.load(f"{model_prefix}.model")

    

                        

    vocab = {}

    for i in range(sp.get_piece_size()):

        piece = sp.id_to_piece(i)

        vocab[piece] = i

    

                     

    vocab_path = os.path.join(output_dir, "vocab.json")

    with open(vocab_path, 'w') as f:

        json.dump(vocab, f, indent=2)

    

    logger.info(f"Vocabulary saved to {vocab_path}")





                                                              

      

                                                              



def main():

    parser = argparse.ArgumentParser(description="Train tokenizer on code data")

    parser.add_argument("--data_dir", type=str, help="Local directory with code files")

    parser.add_argument("--dataset", type=str, help="HuggingFace dataset name")

    parser.add_argument("--output", type=str, default="tokenizer", help="Output directory")

    parser.add_argument("--vocab_size", type=int, default=128256, help="Vocabulary size")

    parser.add_argument("--backend", type=str, choices=["tiktoken", "sentencepiece", "custom"],

                       default="custom", help="Tokenizer backend")

    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to use")

    args = parser.parse_args()

    

                       

    if args.data_dir:

        logger.info(f"Loading code from {args.data_dir}")

        text_iterator = load_code_from_directory(args.data_dir)

    elif args.dataset:

        logger.info(f"Loading from HuggingFace: {args.dataset}")

        text_iterator = load_from_huggingface(args.dataset, max_samples=args.max_samples)

    else:

        logger.error("Specify --data_dir or --dataset")

        sys.exit(1)

    

                     

    if args.backend == "tiktoken":

        train_tiktoken(text_iterator, args.vocab_size, args.output)

    elif args.backend == "sentencepiece":

        train_sentencepiece(text_iterator, args.vocab_size, args.output)

    else:          

        from model.tokenizer import BPETrainer, CodeTokenizer

        

        logger.info(f"Training custom BPE (vocab_size={args.vocab_size})")

        trainer = BPETrainer(vocab_size=args.vocab_size, min_frequency=2)

        

                      

        texts = []

        for text in text_iterator:

            texts.append(text)

            if len(texts) >= 100000:                  

                break

        

        tokenizer = trainer.train(texts, verbose=True)

        

        os.makedirs(args.output, exist_ok=True)

        tokenizer.save(args.output)

        

        logger.info(f"Tokenizer saved to {args.output}")

    

    logger.info("Tokenizer training complete!")





if __name__ == "__main__":

    main()

