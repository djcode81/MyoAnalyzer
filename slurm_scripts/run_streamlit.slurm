#!/bin/bash
#SBATCH --job-name=streamlit_myoanalyzer
#SBATCH --partition=dept_cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=streamlit_%j.log

echo "================================="
echo "Starting MyoAnalyzer Streamlit App at $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Job ID: $SLURM_JOB_ID"
echo "================================="

module purge
export PYTHONUNBUFFERED=1

# Navigate to project directory
cd /net/dali/home/mscbio/dhp72/muscle/muscle_health_quant

# Activate conda environment
source /net/dali/home/mscbio/dhp72/miniconda3/etc/profile.d/conda.sh
conda activate muscle_seg

# Get tunneling info
XDG_RUNTIME_DIR=""
PORT=$(shuf -i8000-9999 -n1)
NODE=$(hostname)
NODE_IP=$(hostname -i)

echo ""
echo "================================="
echo "STREAMLIT SERVER INFORMATION"
echo "================================="
echo "Node: $NODE"
echo "Node IP: $NODE_IP"
echo "Port: $PORT"
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "================================="
echo "PORT FORWARDING INSTRUCTIONS"
echo "================================="
echo ""
echo "COPY AND RUN THIS COMMAND IN YOUR LOCAL TERMINAL:"
echo ""
echo "ssh -N -L $PORT:$NODE_IP:$PORT dhp72@cluster.csb.pitt.edu"
echo ""
echo "================================="
echo ""
echo "THEN OPEN THIS URL IN YOUR BROWSER:"
echo ""
echo "http://localhost:$PORT"
echo ""
echo "================================="
echo ""
echo "IMPORTANT NOTES:"
echo "   - Keep the SSH tunnel running in a separate terminal"
echo "   - The Streamlit app will run until the job times out (4 hours)"
echo "   - Check this log file for any errors: streamlit_$SLURM_JOB_ID.log"
echo "   - To stop early: scancel $SLURM_JOB_ID"
echo ""
echo "================================="

# Check if streamlit_app.py exists
if [ ! -f "streamlit_app.py" ]; then
    echo "ERROR: streamlit_app.py not found in current directory"
    echo "Current directory contents:"
    ls -la
    exit 1
fi

echo "Found streamlit_app.py, starting server..."
echo ""

# Start Streamlit server
streamlit run streamlit_app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.maxUploadSize=1024
echo ""
echo "================================="
echo "Streamlit server stopped at $(date)"
echo "================================="
