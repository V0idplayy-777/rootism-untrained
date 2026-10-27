#!/bin/bash
# ============================================================
# Training Launch Script for Coding Agent 20B
# ============================================================
# 
# This script launches distributed training with DeepSpeed.
#
# Usage:
#   # 8 GPUs on single node:
#   bash scripts/launch_train.sh --num_gpus 8
#
#   # Multi-node (create hostfile first):
#   bash scripts/launch_train.sh --num_gpus 8 --num_nodes 4 --hostfile hostfile.txt
#
#   # Quick smoke test (1B model, 100 steps):
#   bash scripts/launch_train.sh --num_gpus 1 --model_size 1B --max_steps 100
# ============================================================

set -e

# Defaults
NUM_GPUS=8
NUM_NODES=1
HOSTFILE=""
MODEL_SIZE="20B"
CONFIG="configs/training_config.json"
MAX_STEPS=""
RESUME=""
MASTER_PORT=29500

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus) NUM_GPUS=$2; shift 2;;
        --num_nodes) NUM_NODES=$2; shift 2;;
        --hostfile) HOSTFILE=$2; shift 2;;
        --model_size) MODEL_SIZE=$2; shift 2;;
        --config) CONFIG=$2; shift 2;;
        --max_steps) MAX_STEPS="--max_steps $2"; shift 2;;
        --resume) RESUME="--resume"; shift;;
        --master_port) MASTER_PORT=$2; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

echo "============================================================"
echo "  CODING AGENT MODEL - TRAINING LAUNCH"
echo "============================================================"
echo "  Model Size:  $MODEL_SIZE"
echo "  GPUs:        $NUM_GPUS per node"
echo "  Nodes:       $NUM_NODES"
echo "  Config:      $CONFIG"
echo "  Master Port: $MASTER_PORT"
echo "============================================================"

# Check if dataset exists
if [ ! -d "data/processed" ]; then
    echo "ERROR: Processed dataset not found at data/processed"
    echo "Run: python scripts/generate_dataset.py first"
    exit 1
fi

# Build DeepSpeed command
DS_CMD="deepspeed"
DS_ARGS="--num_gpus=$NUM_GPUS"
DS_ARGS="$DS_ARGS --master_port=$MASTER_PORT"

if [ -n "$HOSTFILE" ]; then
    DS_ARGS="$DS_ARGS --hostfile=$HOSTFILE"
    DS_ARGS="$DS_ARGS --num_nodes=$NUM_NODES"
fi

TRAIN_ARGS="training/train.py"
TRAIN_ARGS="$TRAIN_ARGS --config $CONFIG"
TRAIN_ARGS="$TRAIN_ARGS --model_size $MODEL_SIZE"
TRAIN_ARGS="$TRAIN_ARGS --num_gpus $NUM_GPUS"

if [ -n "$MAX_STEPS" ]; then
    TRAIN_ARGS="$TRAIN_ARGS $MAX_STEPS"
fi

if [ -n "$RESUME" ]; then
    TRAIN_ARGS="$TRAIN_ARGS $RESUME"
fi

# Create directories
mkdir -p checkpoints logs

# Launch
echo ""
echo "Launching: $DS_CMD $DS_ARGS $TRAIN_ARGS"
echo ""

$DS_CMD $DS_ARGS $TRAIN_ARGS

echo ""
echo "Training complete!"
