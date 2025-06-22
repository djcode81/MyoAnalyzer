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
- **Clinical Validation**: Tested on 21-subject neuromuscular cohort with realistic fat quantification
- **Production Ready**: Streamlit GUI, CLI tools, and Docker containerization for multi-site deployment
- **Clinical Reports**: Automated CSV/PDF generation for seamless clinical workflow integration

## Clinical Results

| Cohort | Mean Fat Fraction | Range | Subjects |
|--------|------------------|-------|----------|
| **Healthy Volunteers** | 3.2 - 6.8% | 2-7% | 17 |
| **Neuromuscular Patients** | 15.6 - 38.1% | 8-85% | 4 |

**Clear separation between healthy and pathological tissue demonstrates clinical validity for biomarker applications.**


This project uses the **MyoSegmenTUM** dataset for validation:

- **Source**: [MyoSegmenTUM Dataset](https://mediatum.ub.tum.de/1582934)
- **Description**: 25 subjects with Dixon fat-water MRI and manual muscle segmentations
- **Coverage**: Thigh muscles (quadriceps, hamstrings, gracilis, sartorius)
- **Format**: NIfTI files with separate FAT and WATER images
- **Citation**: Kerstin Hammernik et al., "MyoSegmenTUM - A Multi-Modal Segmentation Dataset for Muscle and Fat Segmentation in MRI", 2021

### Data Structure Expected:
```
data/raw/
├── HV001_1/
│   ├── ImageData/
│   │   ├── HV001_1_FAT/
│   │   │   ├── HV001_1_FAT_stack1.nii
│   │   │   └── HV001_1_FAT_stack2.nii
│   │   └── HV001_1_WATER/
│   │       ├── HV001_1_WATER_stack1.nii
│   │       └── HV001_1_WATER_stack2.nii
│   └── SegmentationMasks/
└── P001_1/ (patient subjects)
```


## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/myoanalyzer.git
cd myoanalyzer

# Build the Docker container
./build_docker.sh

# Run Streamlit GUI
docker-compose up

# Or run CLI batch processing
docker-compose --profile cli up
```

### Option 2: Local Installation

```bash
# Create conda environment
conda create -n myoanalyzer python=3.9
conda activate myoanalyzer

# Install dependencies
pip install -r requirements.txt

# Install TotalSegmentator
pip install totalsegmentator==2.0.5

# Run the application
streamlit run src/streamlit_app.py
```

## Usage

### Streamlit Web Interface

1. **Upload MRI Data**: Support for NIfTI (.nii/.nii.gz) and DICOM formats
2. **Configure Analysis**: Set confidence thresholds and muscle groups
3. **Process Data**: Automated segmentation and composition analysis
4. **Download Results**: CSV reports and PDF summaries

```bash
streamlit run src/streamlit_app.py
```

### Command Line Interface

```bash
# Single subject analysis
python src/main.py --input /path/to/subject --output /path/to/results

# Batch processing
python src/batch_processor.py --input /path/to/dataset --output /path/to/results --n-jobs 4
```

### Python API

```python
from myoanalyzer import MuscleAnalyzer

# Initialize analyzer
analyzer = MuscleAnalyzer()

# Process single subject
results = analyzer.analyze_subject(
    fat_image_path="subject_FAT.nii",
    water_image_path="subject_WATER.nii"
)

# Access results
print(f"Total muscle volume: {results['total_volume_cm3']:.2f} cm³")
print(f"Mean fat fraction: {results['mean_fat_fraction_percent']:.2f}%")
```

## Input Data Requirements

### Dixon MRI Sequences
- **Fat Images**: T1-weighted fat-only images (.nii/.nii.gz)
- **Water Images**: T1-weighted water-only images (.nii/.nii.gz)
- **Resolution**: Minimum 1mm³ isotropic recommended
- **Coverage**: Thigh region (hip to knee)

### File Structure
```
subject_data/
├── subject_FAT_stack1.nii
├── subject_WATER_stack1.nii
├── subject_FAT_stack2.nii (optional)
└── subject_WATER_stack2.nii (optional)
```

## Output Format

### CSV Reports
- **Individual muscle metrics** (volume, fat fraction, statistics)
- **Subject-level summaries** (total volumes, mean values)
- **Batch processing results** (all subjects combined)

### JSON Data
- **Detailed analysis results** with metadata
- **Processing timestamps** and version info
- **Quality control metrics** and confidence scores

## Technical Details

### Architecture
1. **Image Preprocessing**: NIfTI loading, resampling, normalization
2. **Muscle Segmentation**: TotalSegmentator deep learning models
3. **Fat Fraction Calculation**: Dixon PDFF algorithm: Fat/(Fat+Water)×100%
4. **Quality Control**: Entropy-based confidence scoring
5. **Report Generation**: Automated CSV/PDF output

### Validated Muscle Groups
- Quadriceps (Left/Right)
- Hamstrings (Left/Right)
- Gracilis (Left/Right)
- Sartorius (Left/Right)

### Performance
- **Processing Time**: ~3-5 minutes per subject (GPU)
- **Memory Requirements**: 8GB RAM minimum, 16GB recommended
- **GPU Support**: CUDA 11.8+ compatible devices

## Clinical Validation

### Dataset
- **MyoSegmenTUM**: 21 subjects (17 healthy, 4 patients)
- **Coverage**: Bilateral thigh muscles with dual-stack imaging
- **Manual Annotations**: Ground truth segmentations for validation

### Validation Metrics
- **Segmentation Accuracy**: Dice coefficient vs manual segmentations
- **Fat Fraction Reliability**: Bland-Altman analysis
- **Clinical Concordance**: Literature comparison with established ranges

### Results Summary
- **Healthy Subjects**: 2-7% muscle fat fraction (literature concordant)
- **Pathological Cases**: 15-85% fat fraction (expected disease progression)
- **Reproducibility**: <2% coefficient of variation across repeated analyses

## System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, Windows 10+
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum
- **Storage**: 10GB free space
- **GPU**: Optional but recommended (CUDA compatible)

### Recommended Configuration
- **RAM**: 16GB or higher
- **GPU**: NVIDIA with 8GB+ VRAM
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
# Reduce batch size or use CPU-only mode
export CUDA_VISIBLE_DEVICES=""
```

**Permission Denied (Docker)**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Logout and login again
```

## Contributing

### Development Setup
```bash
# Clone repository
git clone https://github.com/yourusername/myoanalyzer.git
cd myoanalyzer

# Create development environment
conda env create -f environment.yml
conda activate myoanalyzer-dev

# Install in development mode
pip install -e .

# Run tests
pytest tests/
```

### Code Standards
- **Black** formatting: `black src/`
- **Type hints**: All functions must include type annotations
- **Documentation**: Docstrings required for all public functions
- **Testing**: Pytest with >90% coverage required

## License and Citation

### License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

### Citation
If you use MyoAnalyzer in your research, please cite:

```bibtex
@software{myoanalyzer2024,
  title={MyoAnalyzer: Automated Dixon Fat-Water MRI Analysis for Neuromuscular Biomarker Discovery},
  author={DPV},
  year={2024},
  url={https://github.com/djcode81/myoanalyzer}
}
```

## Acknowledgments

- **TotalSegmentator**: Wasserthal et al. for providing pre-trained segmentation models
- **MyoSegmenTUM Dataset**: Technical University of Munich for validation data


