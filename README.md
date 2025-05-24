# WMH-Dual Tasker

## Overview

**WMH-Dual Tasker** performs both white matter hyperintensity (WMH) visual rating and segmentation from FLAIR MRI volumes using a single deep learning model.

## Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yilei-wu/WMH-DualTasker.git
   cd WMH-DualTasker
   ```

2. **Install Dependencies:**
   ```bash
   pip install torch torchvision nibabel numpy scipy scikit-learn matplotlib tqdm
   ```

## Model Download

Download the pre-trained model weights:
- **Model weights**: [Download here](./for_test/model.pth)
- Place the downloaded `.pth` file in your desired location

## Inference

Use the `inference.py` script to get both visual rating score and WMH segmentation from FLAIR volumes:

### Basic Usage
```bash
python inference.py --input /path/to/flair_volume.nii.gz \
                   --model_path /path/to/model_weights.pth
```

### With Brain Mask
```bash
python inference.py --input /path/to/flair_volume.nii.gz \
                   --model_path /path/to/model_weights.pth \
                   --brain_mask /path/to/brain_mask.nii.gz \
                   --normalize
```

### Parameters
- `--input`: Path to FLAIR volume (.nii/.nii.gz/.npy)
- `--model_path`: Path to downloaded model weights (.pth)
- `--brain_mask`: Brain mask (optional, improves accuracy)
- `--output_dir`: Output directory (default: ./wmh_output)
- `--normalize`: Apply intensity normalization (recommended)
- `--model_type`: Model architecture (default: sfcn_rep1)

### Outputs
- `visual_rating.txt`: Visual rating score (0-30 scale)
- `wmh_segmentation.nii.gz`: Binary WMH mask

### Requirements
- **Input**: FLAIR T2-weighted MRI volume
- **Format**: NIfTI (.nii/.nii.gz) or NumPy (.npy)
- **Hardware**: GPU recommended

## Contact

For questions or issues, contact [Yilei Wu](yilei.wu@u.nus.edu).