"""
Massive Code Dataset Generator for 20B Parameter Coding Agent.

This module generates a comprehensive, high-quality code training dataset
entirely from scratch - no external datasets are used. The dataset includes:

1. Multi-language code samples with varying complexity
2. Code documentation and comments
3. Bug fixing examples
4. Code refactoring pairs
5. Instruction-following conversations
6. Multi-turn coding conversations
7. Code review examples
8. Algorithm implementations
9. API usage patterns
10. Test generation examples
11. Fill-in-the-middle completion data
12. Repository-level context data

The generator is designed to produce millions of training examples
through combinatorial expansion and template-based generation.
"""















from .generator import DatasetGenerator







from .code_templates import PythonCodeGenerator







from .instruction_data import InstructionGenerator







from .processor import DatasetProcessor















__all__ = ["DatasetGenerator", "PythonCodeGenerator", "InstructionGenerator", "DatasetProcessor"]







