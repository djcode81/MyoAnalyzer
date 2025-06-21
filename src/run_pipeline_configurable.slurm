#!/bin/bash
#SBATCH --job-name=muscle_pipeline
#SBATCH --partition=dept_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=muscle_pipeline_%j.log

# Configuration - Edit these variables as needed
SUBJECT_ID=${SUBJECT_ID:-"HV003_1"}  # Default to HV003_1, can be overridden
STACK=${STACK:-"both"}              # Options: stack1, stack2, both
DATA_ROOT="/net/dali/home/mscbio/dhp72/muscle/muscle_health_quant/data/raw"
PIPELINE_ROOT="/net/dali/home/mscbio/dhp72/muscle/muscle_health_quant/muscle_pipeline"

echo "================================="
echo "Starting Muscle Pipeline at $(date)"
echo "Subject: $SUBJECT_ID"
echo "Stack(s): $STACK"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "================================="

module purge
module load cuda/11.8
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Navigate to pipeline src directory
cd $PIPELINE_ROOT/src

# Activate conda environment
source /net/dali/home/mscbio/dhp72/miniconda3/etc/profile.d/conda.sh
conda activate muscle_seg

# Set TotalSegmentator cache
export TOTALSEG_WEIGHTS_PATH="$HOME/.totalsegmentator"
mkdir -p $TOTALSEG_WEIGHTS_PATH

echo "================================="
echo "Running Muscle Analysis Pipeline..."
echo "Models will be cached to: $TOTALSEG_WEIGHTS_PATH"
echo "================================="

# Check if subject exists
SUBJECT_PATH="$DATA_ROOT/$SUBJECT_ID"
if [ ! -d "$SUBJECT_PATH" ]; then
    echo "ERROR: Subject $SUBJECT_ID not found at $SUBJECT_PATH"
    echo "Available subjects:"
    ls -1 $DATA_ROOT | grep -E "^(HV|P)" | head -10
    exit 1
fi

# Run analysis based on stack configuration
if [ "$STACK" = "both" ]; then
    echo "Analyzing both stack1 and stack2..."
    python muscle_analyzer.py --subject "$SUBJECT_PATH" --both-stacks
elif [ "$STACK" = "stack1" ] || [ "$STACK" = "stack2" ]; then
    echo "Analyzing $STACK..."
    python muscle_analyzer.py --subject "$SUBJECT_PATH" --stack "$STACK"
else
    echo "ERROR: Invalid stack option: $STACK"
    echo "Valid options: stack1, stack2, both"
    exit 1
fi

echo "================================="
echo "Checking output files..."
echo "================================="
ls -la ../outputs/ 2>/dev/null || echo "No outputs directory found"

echo "================================="
echo "Job completed at $(date)"
echo "================================="
