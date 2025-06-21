# MRI Muscle Health Quantifier - Production Docker Image
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA support (matching your cluster environment)
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install core scientific computing packages
RUN pip install \
    numpy==1.24.3 \
    scipy==1.10.1 \
    pandas==2.0.3 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    scikit-learn==1.3.0 \
    nibabel==5.1.0 \
    SimpleITK==2.2.1

# Install TotalSegmentator and dependencies
RUN pip install totalsegmentator==2.0.5

# Install additional analysis packages
RUN pip install \
    reportlab==4.0.4 \
    streamlit==1.25.0 \
    plotly==5.15.0 \
    openpyxl==3.1.2

# Set permanent model cache location
ENV TOTALSEG_WEIGHTS_PATH=/app/models
RUN mkdir -p $TOTALSEG_WEIGHTS_PATH

# Pre-download TotalSegmentator models to reduce startup time
RUN python -c "from totalsegmentator.python_api import totalsegmentator; \
    import tempfile; \
    import numpy as np; \
    import nibabel as nib; \
    temp_dir = tempfile.mkdtemp(); \
    dummy_data = np.random.randint(0, 100, (64, 64, 32), dtype=np.uint16); \
    dummy_img = nib.Nifti1Image(dummy_data, np.eye(4)); \
    dummy_path = f'{temp_dir}/dummy.nii.gz'; \
    nib.save(dummy_img, dummy_path); \
    print('Pre-downloading TotalSegmentator weights...'); \
    try: totalsegmentator(dummy_path, f'{temp_dir}/out', task='thigh_shoulder_muscles_mr', fast=False, quiet=True); \
    except: print('Model download initiated')"

# Copy application code
COPY src/ /app/src/
COPY requirements.txt /app/
COPY setup.py /app/

# Install the package in development mode
RUN pip install -e .

# Create directories for data processing
RUN mkdir -p /app/data/input \
             /app/data/output \
             /app/data/temp \
             /app/logs

# Set permissions
RUN chmod -R 755 /app

# Create non-root user for security
RUN useradd -m -u 1000 muscle_user && \
    chown -R muscle_user:muscle_user /app
USER muscle_user

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import totalsegmentator; print('TotalSegmentator OK')" || exit 1

# Expose port for Streamlit
EXPOSE 8501

# Default command
CMD ["python", "src/main.py", "--help"]
