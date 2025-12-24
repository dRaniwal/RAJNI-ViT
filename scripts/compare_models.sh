#!/bin/bash
#
# RAJNI Multi-Model Comparison
#
# Evaluates RAJNI across different ViT model sizes to generate
# the accuracy vs FLOPs plot for the paper.
#
# Usage:
#   ./compare_models.sh /path/to/imagenet

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <imagenet_path>"
    exit 1
fi

DATA_PATH=$1
GAMMA="0.01"

# Models to evaluate
MODELS=(
    "vit_tiny_patch16_224"
    "vit_small_patch16_224"
    "vit_base_patch16_224"
    "vit_large_patch16_224"
    "deit_small_patch16_224"
    "deit_base_patch16_224"
)

RESULTS_DIR="results"
mkdir -p ${RESULTS_DIR}

OUTPUT_FILE="${RESULTS_DIR}/model_comparison.txt"
echo "RAJNI Multi-Model Comparison" > ${OUTPUT_FILE}
echo "Date: $(date)" >> ${OUTPUT_FILE}
echo "Gamma: ${GAMMA}" >> ${OUTPUT_FILE}
echo "======================================" >> ${OUTPUT_FILE}

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "Evaluating ${MODEL}..."
    echo "" >> ${OUTPUT_FILE}
    echo "Model: ${MODEL}" >> ${OUTPUT_FILE}
    
    python examples/run_imagenet.py \
        --data_path ${DATA_PATH} \
        --model ${MODEL} \
        --gamma ${GAMMA} \
        --baseline \
        --batch_size 128 \
        2>&1 | tee -a ${OUTPUT_FILE}
    
    echo "--------------------------------------" >> ${OUTPUT_FILE}
done

echo ""
echo "Comparison complete! Results saved to ${OUTPUT_FILE}"
