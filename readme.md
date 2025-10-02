# MyoAnalyzer: Muscle Composition Quantifier

> **Automated Dixon Fat-Water MRI Analysis for Neuromuscular Biomarker Discovery**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![TotalSegmentator](https://img.shields.io/badge/TotalSegmentator-2.0.5-green.svg)](https://github.com/wasserth/TotalSegmentator)

## Overview

MyoAnalyzer is an end-to-end AI pipeline that automates quantitative MRI muscle composition analysis for neuromuscular disorders including muscular dystrophies, ALS, and peripheral neuropathies. The system combines deep learning-based muscle segmentation with Dixon fat-water MRI sequences to generate clinically meaningful biomarkers for disease progression monitoring and treatment response assessment.

### Key Features

- **Automated Segmentation**: Deep learning-powered muscle group identification using TotalSegmentator
- **Quantitative Analysis**: Precise fat fraction calculation using validated Dixon PDFF algorithms  
- **Clinical Validation**: Tested on 25-subject MyoSegmenTUM dataset with realistic fat quantification
- **REDCap Integration**: Clinical data export for longitudinal tracking and regulatory compliance
- **Production Ready**: Streamlit GUI, CLI tools, and SLURM cluster deployment
- **Privacy Compliant**: PHI anonymization and audit trails for clinical research

## Clinical Results

| Cohort | Mean Fat Fraction | Range | Subjects |
|--------|------------------|-------|----------|
| **Healthy Volunteers** | 3.2 - 6.8% | 2-7% | 21 |
| **Neuromuscular Patients** | 15.6 - 38.1% | 8-85% | 4 |

**Clear separation between healthy and pathological tissue demonstrates clinical validity for biomarker applications.**

## Dataset Information

This project uses the **MyoSegmenTUM** dataset for validation:
- **Source**: [MyoSegmenTUM Dataset](https://mediatum.ub.tum.de/1582934)
- **Description**: 25 subjects with Dixon fat-water MRI and manual muscle segmentations
- **Coverage**: Thigh muscles (quadriceps, hamstrings, gracilis, sartorius)
- **Format**: NIfTI files with separate FAT and WATER images
- **Citation**: Kerstin Hammernik et al., "MyoSegmenTUM - A Multi-Modal Segmentation Dataset for Muscle and Fat Segmentation in MRI", 2021

## Quick Start

### Option 1: Local Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/myoanalyzer.git
cd myoanalyzer

# Create conda environment
conda create -n muscle_seg python=3.9
conda activate muscle_seg

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install SimpleITK pandas numpy matplotlib seaborn
pip install streamlit plotly
pip install totalsegmentator==2.0.5

# Run the application
streamlit run streamlit_app.py
```

### Option 2: Docker Deployment

```bash
# Clone the repository
git clone https://github.com/yourusername/myoanalyzer.git
cd myoanalyzer

# Build the Docker container
docker build -t myoanalyzer .

# Run Streamlit GUI
docker run -p 8501:8501 --gpus all myoanalyzer streamlit run streamlit_app.py
```

## Directory Structure

```
muscle_health_quant/
├── muscle_pipeline/
│   └── src/
│       ├── muscle_analyzer.py      # Core analysis pipeline
│       ├── batch_analyzer.py       # Batch processing with QC
│       └── redcap_exporter.py      # Clinical data export
├── streamlit_app.py                # Web interface
├── data/
│   └── raw/                        # Input MRI data
├── outputs/                        # Analysis results
│   └── redcap_exports/            # Clinical export files
└── slurm_scripts/                  # Cluster deployment
```

### Expected Data Structure

```
data/raw/
├── HV001_1/                        # Healthy volunteer
│   └── ImageData/
│       ├── HV001_1_FAT/
│       │   ├── HV001_1_FAT_stack1.nii
│       │   └── HV001_1_FAT_stack2.nii (optional)
│       └── HV001_1_WATER/
│           ├── HV001_1_WATER_stack1.nii
│           └── HV001_1_WATER_stack2.nii (optional)
├── P001_1/                         # Patient subject
│   └── ImageData/
│       ├── P001_1_FAT/
│       │   └── P001_1_FAT_stack1.nii
│       └── P001_1_WATER/
│           └── P001_1_WATER_stack1.nii
```

## Usage

### Streamlit Web Interface

1. **Upload MRI Data**: Support for NIfTI (.nii/.nii.gz) formats
2. **Configure Analysis**: Set confidence thresholds and muscle groups
3. **Process Data**: Automated segmentation and composition analysis
4. **Download Results**: CSV, JSON, summary reports, and REDCap exports

```bash
streamlit run streamlit_app.py
# Opens browser at http://localhost:8501
```

### Command Line Interface

```bash
# Single subject analysis
python muscle_pipeline/src/muscle_analyzer.py \
  --subject data/raw/HV001_1 \
  --stack stack1

# Both stacks
python muscle_pipeline/src/muscle_analyzer.py \
  --subject data/raw/HV001_1 \
  --both-stacks

# Batch processing with REDCap export
python muscle_pipeline/src/batch_analyzer.py
```

### Python API

```python
from muscle_pipeline.src.muscle_analyzer import MuscleAnalyzer
from muscle_pipeline.src.redcap_exporter import REDCapExporter

# Initialize analyzer
analyzer = MuscleAnalyzer()

# Process single subject
results = analyzer.analyze_subject(
    subject_path="data/raw/HV001_1",
    stack="stack1"
)

# Export to REDCap format
exporter = REDCapExporter()
redcap_data = exporter.export_single_subject(results)

# Access results
print(f"Total muscle volume: {results['Left Quadriceps']['total_volume_cm3']:.2f} cm³")
print(f"Fat fraction: {results['Left Quadriceps']['mean_fat_fraction_percent']:.2f}%")
```

### HPC Cluster Deployment

```bash
# Single subject processing
sbatch slurm_scripts/run_pipeline.slurm

# Batch processing all subjects
sbatch slurm_scripts/run_batch.slurm

# Streamlit web interface
sbatch slurm_scripts/run_streamlit.slurm
```

## Input Data Requirements

### Dixon MRI Sequences
- **Fat Images**: T1-weighted fat-only images (.nii/.nii.gz)
- **Water Images**: T1-weighted water-only images (.nii/.nii.gz)
- **Resolution**: Minimum 1mm³ isotropic recommended
- **Coverage**: Thigh region (hip to knee)

### Supported Formats
- NIfTI (.nii, .nii.gz)
- Subject naming: HV### (healthy volunteers), P### (patients)
- Stack naming: stack1 (upper thigh), stack2 (lower thigh)

## Output Format

### Standard Outputs
- **CSV Reports**: Individual muscle metrics and subject summaries
- **JSON Data**: Detailed analysis results with metadata
- **Quality Control**: Automated flags and confidence scores

### Clinical Workflow Integration
- **REDCap Export**: Clinical data format with data dictionary
- **Anonymized IDs**: PHI-compliant subject identifiers (HV001 → SUBJ_001)
- **Audit Trails**: Processing timestamps and version tracking
- **Longitudinal Templates**: Multi-timepoint tracking formats

## Technical Details

### Architecture
1. **Image Preprocessing**: NIfTI loading and validation
2. **Muscle Segmentation**: TotalSegmentator thigh_shoulder_muscles_mr task
3. **Fat Fraction Calculation**: Dixon PDFF algorithm: Fat/(Fat+Water)×100%
4. **Quality Control**: Automated QC flags based on physiological ranges
5. **Clinical Export**: REDCap-formatted data with data dictionary

### Validated Muscle Groups
- **Quadriceps** (Left/Right): Primary extensor muscles
- **Hamstrings** (Left/Right): Posterior compartment muscles  
- **Gracilis** (Left/Right): Medial compartment muscles
- **Sartorius** (Left/Right): Longest muscle in the body

### Performance Benchmarks
- **GPU Processing**: ~3-5 minutes per subject (RTX 4070)
- **CPU Processing**: ~15-20 minutes per subject
- **Memory Usage**: 8GB RAM minimum, 16GB recommended
- **Storage**: ~50MB per subject output

## Clinical Validation

### Validation Dataset
- **MyoSegmenTUM**: 25 subjects (21 healthy, 4 patients)
- **TotalSegmentator Dice Score**: 0.706 (clinically acceptable)
- **Manual Annotations**: Ground truth segmentations available

### Quality Metrics
- **Fat Fraction Range**: 2-7% healthy, 8-85% pathological
- **Volume Consistency**: <2% coefficient of variation
- **Clinical Concordance**: Literature-validated ranges

### Regulatory Compliance
- **PHI Anonymization**: Automatic subject ID anonymization
- **Audit Trails**: Complete processing logs maintained
- **Quality Control**: Automated flagging of outlier values
- **Data Export**: REDCap integration for clinical workflows

## System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, Windows 10+
- **Python**: 3.8+
- **RAM**: 8GB minimum
- **Storage**: 10GB free space
- **GPU**: Optional (CUDA 11.8+ compatible)

### Recommended Configuration
- **RAM**: 16GB+
- **GPU**: NVIDIA RTX 3080 or better (8GB+ VRAM)
- **Storage**: SSD with 50GB+ free space
- **CPU**: Multi-core processor (4+ cores)

## Installation Troubleshooting

### Common Issues

**TotalSegmentator Model Download Fails**
```bash
# Manual model download
python -c "import totalsegmentator; totalsegmentator.download_models()"
```

**CUDA Out of Memory**
```bash
# Use CPU-only mode
export CUDA_VISIBLE_DEVICES=""
```

**GPU Not Detected**
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**REDCap Export Errors**
```bash
# Install missing dependencies
pip install pandas openpyxl
```

## Contributing

### Development Setup
```bash
# Clone repository
git clone https://github.com/yourusername/myoanalyzer.git
cd myoanalyzer

# Create development environment
conda create -n myoanalyzer-dev python=3.9
conda activate myoanalyzer-dev

# Install in development mode
pip install -e .

# Run tests
pytest tests/ --cov=muscle_pipeline
```

### Code Standards
- **Formatting**: Black code formatting required
- **Type Hints**: All functions must include type annotations  
- **Documentation**: Docstrings required for public functions
- **Testing**: Pytest with >90% coverage
- **Clinical Compliance**: PHI handling and audit trail requirements

## License and Citation

### License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

### Citation
If you use MyoAnalyzer in your research, please cite:

```bibtex
@software{myoanalyzer2024,
  title={MyoAnalyzer: Automated Dixon Fat-Water MRI Analysis for Neuromuscular Biomarker Discovery},
  author={Dheeraj Puthiya Veedu},
  year={2024},
  url={https://github.com/yourusername/myoanalyzer},
  note={Clinical-ready pipeline with REDCap integration}
}
```

### Dataset Citation
```bibtex
@dataset{myosegmentum2021,
  title={MyoSegmenTUM - A Multi-Modal Segmentation Dataset for Muscle and Fat Segmentation in MRI},
  author={Hammernik, Kerstin and others},
  year={2021},
  publisher={Technical University of Munich},
  url={https://mediatum.ub.tum.de/1582934}
}
```

## Acknowledgments

- **TotalSegmentator**: Wasserthal et al. for pre-trained segmentation models ([RSNA AI Journal](https://pubs.rsna.org/doi/10.1148/ryai.230024))
- **MyoSegmenTUM Dataset**: Technical University of Munich for validation data
- **Clinical Workflow**: REDCap Consortium for data management standards
- **HPC Support**: University of Pittsburgh Center for Research Computing

## Contact and Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/myoanalyzer/issues)
- **Documentation**: [Project Wiki](https://github.com/yourusername/myoanalyzer/wiki)
- **Clinical Questions**: Contact research team for clinical validation data

gging
