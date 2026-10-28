                      



"""
Production Dataset Builder - Integrates Real Code Datasets.

This script downloads and processes real-world code datasets for training:
- The Stack (BigCode): 300B+ tokens from GitHub
- StarCoderData (BigCode): 1T+ tokens
- OpenCoder: High-quality filtered code
- CodeAlpaca: Instruction-tuning data
- Custom GitHub repositories

Usage:
    # Build dataset from The Stack
    python scripts/build_real_dataset.py --dataset the-stack --output data/processed
    
    # Build from multiple sources
    python scripts/build_real_dataset.py --dataset the-stack,starcoderdata --output data/processed
    
    # Build from local GitHub repos
    python scripts/build_real_dataset.py --local_repos /path/to/repos --output data/processed
"""







import os



import sys



import json



import argparse



import logging



import hashlib



from pathlib import Path



from typing import List, Dict, Any, Iterator, Optional



from dataclasses import dataclass



import tempfile



import shutil







import numpy as np







                    



sys.path.insert(0, str(Path(__file__).parent.parent))







logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")



logger = logging.getLogger(__name__)











                                                              



                        



                                                              







@dataclass



class DatasetConfig:



    """Configuration for a dataset source."""



    name: str



    huggingface_id: Optional[str]



    description: str



    languages: List[str]



    estimated_tokens: str



    license: str



    url: str











DATASETS = {



    "the-stack": DatasetConfig(



        name="The Stack",



        huggingface_id="bigcode/the-stack",



        description="300B+ tokens of permissively licensed code from GitHub",



        languages=["python", "javascript", "typescript", "java", "cpp", "go", "rust", "ruby", "php"],



        estimated_tokens="300B+",



        license="Various (filtered for permissive)",



        url="https://huggingface.co/datasets/bigcode/the-stack",



    ),



    "starcoderdata": DatasetConfig(



        name="StarCoderData",



        huggingface_id="bigcode/starcoderdata",



        description="1T+ tokens of code and GitHub issues/PRs",



        languages=["python", "javascript", "typescript", "java", "cpp", "go", "rust", "c", "csharp"],



        estimated_tokens="1T+",



        license="Various",



        url="https://huggingface.co/datasets/bigcode/starcoderdata",



    ),



    "opencoder": DatasetConfig(



        name="OpenCoder",



        huggingface_id="open-code-interpreter/Code-Feedback-66K",



        description="High-quality instruction-tuning code data",



        languages=["python", "javascript", "typescript", "java", "cpp"],



        estimated_tokens="5B",



        license="Apache 2.0",



        url="https://huggingface.co/datasets/open-code-interpreter/Code-Feedback-66K",



    ),



    "codealpaca": DatasetConfig(



        name="CodeAlpaca",



        huggingface_id="sahil2801/CodeAlpaca-20k",



        description="20K instruction-following code examples",



        languages=["python"],



        estimated_tokens="100M",



        license="Apache 2.0",



        url="https://huggingface.co/datasets/sahil2801/Code-Feedback-66K",



    ),



}











                                                              



                  



                                                              







def download_huggingface_dataset(



    dataset_id: str,



    languages: List[str] = None,



    max_samples: int = None,



    streaming: bool = True,



) -> Iterator[Dict[str, Any]]:



    """
    Download dataset from HuggingFace.
    
    Args:
        dataset_id: HuggingFace dataset ID
        languages: Filter by programming languages
        max_samples: Maximum number of samples
        streaming: Use streaming mode (no full download)
        
    Yields:
        Dict with 'content', 'language', 'metadata'
    """



    try:



        from datasets import load_dataset



    except ImportError:



        logger.error("Install datasets: pip install datasets")



        sys.exit(1)



    



    logger.info(f"Loading {dataset_id} (streaming={streaming})")



    



                  



    if streaming:



        dataset = load_dataset(dataset_id, split="train", streaming=True)



    else:



        dataset = load_dataset(dataset_id, split="train")



    



    count = 0



    for sample in dataset:



                                         



        if languages:



            lang = sample.get("lang", sample.get("language", ""))



            if lang and lang not in languages:



                continue



        



                         



        content = None



        if "content" in sample:



            content = sample["content"]



        elif "text" in sample:



            content = sample["text"]



        elif "code" in sample:



            content = sample["code"]



        elif "instruction" in sample:



                                



            content = format_instruction(sample)



        



        if content and len(content) > 100:



            yield {



                "content": content,



                "language": sample.get("lang", sample.get("language", "unknown")),



                "metadata": {k: v for k, v in sample.items() if k not in ["content", "text", "code"]},



            }



            



            count += 1



            if max_samples and count >= max_samples:



                break



            



            if count % 10000 == 0:



                logger.info(f"  Downloaded {count:,} samples")











def format_instruction(sample: Dict) -> str:



    """Format instruction-tuning sample."""



    parts = []



    



    if "instruction" in sample:



        parts.append(f"<|user|>\n{sample['instruction']}\n<eos>")



    



    if "input" in sample and sample["input"]:



        parts.append(f"```\n{sample['input']}\n```\n<eos>")



    



    if "output" in sample:



        parts.append(f"<|assistant|>\n{sample['output']}\n<eos>")



    



    return "\n".join(parts)











def load_local_repos(repos_dir: str, extensions: List[str] = None) -> Iterator[Dict[str, Any]]:



    """
    Load code from local GitHub repositories.
    
    Args:
        repos_dir: Directory containing git repositories
        extensions: File extensions to include
        
    Yields:
        Dict with 'content', 'language', 'metadata'
    """



    if extensions is None:



        extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.rb']



    



    repos_path = Path(repos_dir)



    



                           



    repos = [d for d in repos_path.iterdir() if d.is_dir() and (d / '.git').exists()]



    logger.info(f"Found {len(repos)} repositories")



    



    for repo in repos:



        logger.info(f"  Processing {repo.name}")



        



        for ext in extensions:



            for filepath in repo.rglob(f'*{ext}'):



                try:



                                                        



                    if any(skip in str(filepath) for skip in ['node_modules', '.git', '__pycache__', 'venv', 'build', 'dist']):



                        continue



                    



                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:



                        content = f.read()



                    



                    if len(content) > 100:



                                                        



                        lang_map = {



                            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',



                            '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.go': 'go',



                            '.rs': 'rust', '.rb': 'ruby', '.php': 'php',



                        }



                        language = lang_map.get(ext, 'unknown')



                        



                        yield {



                            "content": content,



                            "language": language,



                            "metadata": {



                                "repo": repo.name,



                                "file": str(filepath.relative_to(repo)),



                            },



                        }



                except Exception as e:



                    logger.warning(f"Error reading {filepath}: {e}")











                                                              



                 



                                                              







def deduplicate_documents(documents: Iterator[Dict[str, Any]], threshold: float = 0.85) -> Iterator[Dict[str, Any]]:



    """
    Remove near-duplicate documents using MinHash.
    
    Args:
        documents: Iterator of documents
        threshold: Jaccard similarity threshold
        
    Yields:
        Unique documents
    """



    from dataset.processor import TextDeduplicator



    



    dedup = TextDeduplicator(num_hashes=128, num_bands=16)



    



    count = 0



    duplicates = 0



    



    for doc in documents:



        doc_id = hashlib.md5(doc["content"][:500].encode()).hexdigest()



        



        if dedup.add(doc_id, doc["content"]):



            yield doc



            count += 1



        else:



            duplicates += 1



        



        if (count + duplicates) % 10000 == 0:



            logger.info(f"  Dedup: {count:,} unique, {duplicates:,} duplicates ({duplicates/(count+duplicates)*100:.1f}%)")











def quality_filter(documents: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:



    """Filter low-quality documents."""



    from dataset.processor import QualityFilter



    



    qf = QualityFilter()



    



    count = 0



    filtered = 0



    



    for doc in documents:



        passes, reason = qf.is_quality_code(doc["content"])



        



        if passes:



            yield doc



            count += 1



        else:



            filtered += 1



        



        if (count + filtered) % 10000 == 0:



            logger.info(f"  Quality: {count:,} passed, {filtered:,} filtered")











                                                              



                 



                                                              







class DatasetBuilder:



    """Build production training dataset from multiple sources."""



    



    def __init__(self, output_dir: str, tokenizer):



        self.output_dir = Path(output_dir)



        self.output_dir.mkdir(parents=True, exist_ok=True)



        self.tokenizer = tokenizer



    



    def build(



        self,



        dataset_names: List[str],



        local_repos: Optional[str] = None,



        max_samples_per_dataset: Optional[int] = None,



        languages: List[str] = None,



    ) -> Dict[str, Any]:



        """
        Build dataset from multiple sources.
        
        Args:
            dataset_names: List of dataset names from DATASETS
            local_repos: Path to local repositories
            max_samples_per_dataset: Max samples per dataset
            languages: Filter by languages
            
        Returns:
            Build statistics
        """



        stats = {



            "datasets": {},



            "total_documents": 0,



            "total_tokens": 0,



        }



        



        all_documents = []



        



                                       



        for name in dataset_names:



            if name not in DATASETS:



                logger.warning(f"Unknown dataset: {name}")



                continue



            



            config = DATASETS[name]



            logger.info(f"\n{'='*60}")



            logger.info(f"Processing {config.name}")



            logger.info(f"{'='*60}")



            logger.info(f"Description: {config.description}")



            logger.info(f"Estimated tokens: {config.estimated_tokens}")



            



                      



            documents = download_huggingface_dataset(



                config.huggingface_id,



                languages=languages or config.languages,



                max_samples=max_samples_per_dataset,



                streaming=True,



            )



            



                                    



            documents = quality_filter(documents)



            documents = deduplicate_documents(documents)



            



                     



            count = 0



            for doc in documents:



                all_documents.append(doc)



                count += 1



            



            stats["datasets"][name] = {"documents": count}



            stats["total_documents"] += count



            



            logger.info(f"  Added {count:,} documents from {config.name}")



        



                                



        if local_repos:



            logger.info(f"\n{'='*60}")



            logger.info(f"Processing local repositories")



            logger.info(f"{'='*60}")



            



            documents = load_local_repos(local_repos)



            documents = quality_filter(documents)



            documents = deduplicate_documents(documents)



            



            count = 0



            for doc in documents:



                all_documents.append(doc)



                count += 1



            



            stats["datasets"]["local_repos"] = {"documents": count}



            stats["total_documents"] += count



            



            logger.info(f"  Added {count:,} documents from local repos")



        



                            



        logger.info(f"\n{'='*60}")



        logger.info(f"Tokenizing and sharding")



        logger.info(f"{'='*60}")



        



        from dataset.processor import DatasetProcessor, DatasetConfig



        



        processor_config = DatasetConfig(



            max_seq_length=8192,



            train_split=0.95,



            val_split=0.04,



            test_split=0.01,



        )



        



        processor = DatasetProcessor(processor_config, self.tokenizer)



        



                                                 



        raw_texts = [



            {"text": doc["content"], "category": "code", "language": doc["language"]}



            for doc in all_documents



        ]



        



        proc_stats = processor.process(raw_texts, str(self.output_dir))



        



        stats.update(proc_stats)



        stats["total_tokens"] = proc_stats.get("total_tokens", 0)



        



                             



        manifest = {



            "datasets": [DATASETS[name].__dict__ for name in dataset_names if name in DATASETS],



            "local_repos": local_repos,



            "languages": languages,



            "statistics": stats,



        }



        



        with open(self.output_dir / "build_manifest.json", 'w') as f:



            json.dump(manifest, f, indent=2)



        



        return stats











                                                              



      



                                                              







def main():



    parser = argparse.ArgumentParser(description="Build production dataset from real sources")



    parser.add_argument("--dataset", type=str, default="the-stack",



                       help="Comma-separated dataset names (the-stack,starcoderdata,opencoder,codealpaca)")



    parser.add_argument("--local_repos", type=str, default=None,



                       help="Path to directory with local git repositories")



    parser.add_argument("--output", type=str, default="data/processed",



                       help="Output directory")



    parser.add_argument("--tokenizer", type=str, default="data/tokenizer",



                       help="Path to trained tokenizer")



    parser.add_argument("--languages", type=str, default=None,



                       help="Comma-separated languages to filter (e.g., python,javascript)")



    parser.add_argument("--max_samples", type=int, default=None,



                       help="Max samples per dataset")



    args = parser.parse_args()



    



                     



    dataset_names = [d.strip() for d in args.dataset.split(",")]



    languages = [l.strip() for l in args.languages.split(",")] if args.languages else None



    



                    



    logger.info(f"Loading tokenizer from {args.tokenizer}")



    from model.tokenizer import CodeTokenizer



    tokenizer = CodeTokenizer.load(args.tokenizer)



    



                   



    builder = DatasetBuilder(args.output, tokenizer)



    



    stats = builder.build(



        dataset_names=dataset_names,



        local_repos=args.local_repos,



        max_samples_per_dataset=args.max_samples,



        languages=languages,



    )



    



                   



    logger.info(f"\n{'='*60}")



    logger.info(f"DATASET BUILD COMPLETE")



    logger.info(f"{'='*60}")



    logger.info(f"Total documents: {stats['total_documents']:,}")



    logger.info(f"Total tokens: {stats['total_tokens']:,}")



    logger.info(f"Output directory: {args.output}")



    



    print(f"\nDataset breakdown:")



    for dataset, dataset_stats in stats["datasets"].items():



        print(f"  {dataset}: {dataset_stats['documents']:,} documents")



    



    print(f"\nReady for training!")



    print(f"  bash scripts/launch_train.sh --num_gpus 8")











if __name__ == "__main__":



    main()



