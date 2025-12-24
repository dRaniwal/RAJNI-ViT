#!/bin/bash
#
# RAJNI Gamma Hyperparameter Sweep
#
# Runs evaluation across different gamma values to find optimal
# accuracy-efficiency tradeoff for a given model.
#
# Usage:
#   ./sweep_gamma.sh /path/to/imagenet vit_base_patch16_224
#
# Results are saved to results/gamma_sweep_<model>.txt

set -e

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <data_path> <model_name> [gpu_id]"
    echo "Example: $0 /datasets/imagenet vit_base_patch16_224 0"
    exit 1
fi

DATA_PATH=$1
MODEL=$2
GPU=${3:-0}

# Gamma values to sweep (adjust based on your needs)
GAMMAS="0.001 0.005 0.01 0.02 0.05 0.1"

# Output directory
RESULTS_DIR="results"
mkdir -p ${RESULTS_DIR}

OUTPUT_FILE="${RESULTS_DIR}/gamma_sweep_${MODEL}.txt"
echo "RAJNI Gamma Sweep Results" > ${OUTPUT_FILE}
echo "Model: ${MODEL}" >> ${OUTPUT_FILE}
echo "Date: $(date)" >> ${OUTPUT_FILE}
echo "======================================" >> ${OUTPUT_FILE}
echo "" >> ${OUTPUT_FILE}

echo "Starting gamma sweep for ${MODEL}"
echo "Results will be saved to ${OUTPUT_FILE}"
echo ""

for GAMMA in ${GAMMAS}; do
    echo "Running gamma=${GAMMA}..."
    
    CUDA_VISIBLE_DEVICES=${GPU} python examples/run_imagenet.py \
        --data_path ${DATA_PATH} \
        --model ${MODEL} \
        --gamma ${GAMMA} \
        --baseline \
        --batch_size 256 \
        2>&1 | tee -a ${OUTPUT_FILE}
    
    echo "" >> ${OUTPUT_FILE}
    echo "--------------------------------------" >> ${OUTPUT_FILE}
    echo "" >> ${OUTPUT_FILE}
done

echo ""
echo "Sweep complete! Results saved to ${OUTPUT_FILE}"
