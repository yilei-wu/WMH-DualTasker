Based on the files you've provided, I will draft a `README.md` file that outlines the purpose, setup, and usage of your project.

---

# WMH-Dual Tasker

## Overview

This project, titled **WMH-Dual Tasker**, is designed to tackle the challenges of jointly weakly-supervised white matter hyperintensity (WMH) segmentation and visual rating with self-supervised consistency. The framework incorporates several components for training, evaluation, and analysis, such as generating Class Activation Maps (CAMs) and specialized loss functions.

## Project Structure

- **train.py**: Script for training the model on the dataset.
- **evaluate_single.py**: Script for evaluating the model on a single subject.
- **evaluate_seg.py**: Script for evaluating segmentation results.
- **evaluate_study.py**: Script for evaluating the model performance across multiple subjects.
- **generate_CAM.py**: Script to generate Class Activation Maps (CAMs) for model interpretability.
- **utils.py**: Utility functions used across different modules.
- **mypath.py**: Configurations for dataset paths and other settings.
- **loss.py**: Custom loss functions used in the training process.
- **trainer.py**: Main training loop and logic for model training.

## Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yilei-wu/WMH-DualTasker.git
   cd WMH-DualTasker
   ```

2. **Install Required Dependencies:**
   It's recommended to create a virtual environment and install the required Python packages using the provided `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Configuration:**
   Update `mypath.py` to include the paths to your datasets and other configuration details.

## Training

To train the model, use the `train.py` script. You can customize the training parameters directly in the script or pass them as arguments.

```bash
python train.py --epochs 50 --batch_size 8
```

## Evaluation

### 1. Evaluate a Single Subject
To evaluate the model on a single subject, run the following command:

```bash
python evaluate_single.py --subject_id <SUBJECT_ID>
```

### 2. Evaluate Segmentation Results
For evaluating segmentation performance:

```bash
python evaluate_seg.py --data_dir <DATA_DIRECTORY>
```

### 3. Evaluate Across Multiple Subjects
For comprehensive evaluation across multiple subjects:

```bash
python evaluate_study.py --study_dir <STUDY_DIRECTORY>
```

## Generate CAMs

To generate Class Activation Maps (CAMs) for model interpretability, use:

```bash
python generate_CAM.py --subject_id <SUBJECT_ID> --layer <LAYER_NAME>
```

## Custom Loss Functions

The custom loss functions are defined in `loss.py`. You can modify them as needed to experiment with different loss strategies.

## Utilities

Additional utility functions for data preprocessing, augmentation, and evaluation metrics are available in `utils.py`.

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any inquiries or issues, please contact [Yilei Wu](mailto:ucs@nus.edu.sg).

---

Let me know if you'd like to add or modify any section of this `README.md`!