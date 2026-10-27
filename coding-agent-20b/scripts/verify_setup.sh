#!/bin/bash
# Quick verification script
set -e

echo "=== Coding Agent 20B - Quick Verification ==="
echo ""

# Check Python
echo "[1/4] Checking Python..."
python3 --version

# Check PyTorch
echo "[2/4] Checking PyTorch..."
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()}'); print(f'  GPUs: {torch.cuda.device_count()}' if torch.cuda.is_available() else '')"

# Check model can be created
echo "[3/4] Testing model creation (1B)..."
python3 -c "
import sys; sys.path.insert(0, '.')
from model.config import get_1b_config
from model.model import CodingAgentModel
config = get_1b_config()
model = CodingAgentModel(config)
print(f'  Parameters: {model.num_parameters():,} ({model.num_parameters()/1e9:.2f}B)')
print('  Model created successfully ✓')
"

# Check dataset generator
echo "[4/4] Testing dataset generator..."
python3 -c "
import sys; sys.path.insert(0, '.')
from dataset.code_templates import PythonCodeGenerator
gen = PythonCodeGenerator()
sample = gen.generate('beginner', 'algorithms')
print(f'  Generated: {sample.language} | {sample.complexity} | {sample.domain}')
print(f'  Code length: {len(sample.code)} chars')
print('  Dataset generator works ✓')
"

echo ""
echo "=== All checks passed ✓ ==="
