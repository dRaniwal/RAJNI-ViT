#!/bin/bash
#
# RAJNI ImageNet Benchmark
#
# Standard benchmark script for reproducing paper results.
# Runs RAJNI with default parameters on ImageNet validation set.
#
# Usage:
#   ./benchmark_imagenet.sh /path/to/imagenet
#
# For multi-GPU:
#   ./benchmark_imagenet.sh /path/to/imagenet --multi_gpu

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <imagenet_path> [additional_args]"
    echo "Example: $0 /datasets/imagenet --multi_gpu"
    exit 1
fi

DATA_PATH=$1
shift
EXTRA_ARGS="$@"

# Default configuration (matches paper)
MODEL="vit_base_patch16_224"
GAMMA="0.01"
MIN_TOKENS="16"
BATCH_SIZE="256"

echo "=============================================="
echo "RAJNI ImageNet Benchmark"
echo "=============================================="
echo "Model: ${MODEL}"
echo "Gamma: ${GAMMA}"
echo "Min tokens: ${MIN_TOKENS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Data path: ${DATA_PATH}"
echo "Extra args: ${EXTRA_ARGS}"
echo "=============================================="

python examples/run_imagenet.py \
    --data_path ${DATA_PATH} \
    --model ${MODEL} \
    --gamma ${GAMMA} \
    --min_tokens ${MIN_TOKENS} \
    --batch_size ${BATCH_SIZE} \
    --baseline \
    ${EXTRA_ARGS}
