"""Corpus v1 Dataset Pipeline for CodingAgent (3B & 20B).

Corpus v1 combines a **real permissively licensed code backbone** (73% of
pretraining tokens across source files, issues, pull requests/diffs, unit and
integration test suites, and technical documentation from The Stack v2 /
StarCoderData and pinned GitHub, PyPI, and npm releases gated fail-closed by
SPDX license) with **minority, verified synthetic and induced slices** (27%
across AST/multi-language instruction induction, CWE-tagged security pairs,
grounded fail->diagnose->minimal-diff->green repair traces, pytest-verified
full-stack applications, taste/correctness preference pairs, tool-failure
recovery, and multi-file project scaffolds).

Primary production entry points:
- ``scripts/fetch_real_corpus.py`` & ``scripts/build_real_dataset.py`` — fetch
  and filter real permissively licensed code with held-out split isolation.
- ``dataset/production_processor.py`` & ``scripts/build_corpus_v1.py`` — apply
  SPDX license gates, quality & complexity floors, exact + MinHash near-dedup,
  mixture sampling (``dataset.mixture.DEFAULT_MIXTURE``), sequence packing, and
  shard + SHA-256 manifest emission.
- ``dataset/quality_gates.py`` — pre-training coverage, mixture, license,
  diversity, held-out isolation, real-backbone and synthetic-budget gates over
  the built corpus manifest; the 3B trainer refuses to start when any gate
  fails (``verify_corpus_manifest_for_3b``).

Pure template generators (``dataset/code_templates.py``, ``dataset/generator.py``)
are supplementary minority sources only and are never used as a standalone
pretraining corpus.
"""

from .code_templates import PythonCodeGenerator
from .generator import DatasetGenerator
from .instruction_data import InstructionGenerator
from .mixture import (
    CORPUS_V1_MIXTURE,
    DEFAULT_MIXTURE,
    REAL_BACKBONE_SOURCES,
    SYNTHETIC_SLICE_SOURCES,
    MixtureSampler,
    SourceSpec,
    curriculum_order,
    mixture_upsampling,
    mixture_weights,
    real_backbone_weight,
    validate_mixture,
)
from .processor import DatasetProcessor, DatasetSharder, MemoryMappedDataset, StreamingDataset
from .production_processor import DataProcessor, Deduplicator, LanguageDetector, LicenseFilter, QualityScorer

__all__ = [
    "CORPUS_V1_MIXTURE",
    "DEFAULT_MIXTURE",
    "DataProcessor",
    "DatasetGenerator",
    "DatasetProcessor",
    "DatasetSharder",
    "Deduplicator",
    "InstructionGenerator",
    "LanguageDetector",
    "LicenseFilter",
    "MemoryMappedDataset",
    "MixtureSampler",
    "PythonCodeGenerator",
    "QualityScorer",
    "REAL_BACKBONE_SOURCES",
    "SYNTHETIC_SLICE_SOURCES",
    "SourceSpec",
    "StreamingDataset",
    "curriculum_order",
    "mixture_upsampling",
    "mixture_weights",
    "real_backbone_weight",
    "validate_mixture",
]
