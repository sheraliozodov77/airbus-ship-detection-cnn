
# Airbus Ship Detection CNN

This repository presents a full deep learning pipeline for the [Airbus Ship Detection Challenge](https://www.kaggle.com/competitions/airbus-ship-detection), hosted on Kaggle. The goal is to accurately detect ships in satellite imagery using modern CNN-based segmentation architectures.

**Project Author:** Sherali Ozodov  
**Instructor:** Jason Pacheco  
**Course:** CSC 480 – Principles of Machine Learning  
**University of Arizona**

## Project Summary

The task involves identifying all ships in satellite images—even under clouds, haze, or varied sea conditions. We approach this as a **supervised segmentation problem** and explore multiple deep learning models:

- **ResUNet** – a residual version of U-Net with skip connections
- **U-Net++** – a nested U-Net architecture for better multi-scale feature fusion
- **Attention U-Net** – integrates attention gates to focus on ship-relevant regions

Each model was trained on a processed version of the dataset using TFRecords for performance and reproducibility.

## Competition Details

- **Organizer**: Airbus via Kaggle
- **Dataset**: Satellite images with masks provided via Run-Length Encoding (RLE)
- **Evaluation Metric**: Average **F2 Score** over IoU thresholds `0.5 → 0.95`
- **Submission Format**: RLE masks in CSV

[Kaggle Competition Page](https://www.kaggle.com/competitions/airbus-ship-detection)

## What This Repository Covers

### Preprocessing & TFRecord Generation
- Converts RLE masks into binary masks
- Applies augmentations with `Albumentations`
- Serializes data into efficient `TFRecord` shards

### Model Architectures
- Lightweight and standard versions of:
  - `ResUNet`
  - `U-Net++`
  - `Attention U-Net`

### Training Pipeline
- Mixed precision for speed
- Combo loss: Dice + Focal Tversky
- ModelCheckpoint, EarlyStopping, ReduceLRO callbacks
- Loss and accuracy visualizations

### Evaluation & Submission
- Runs predictions on test set
- Converts masks to RLE
- Saves `submission.csv`
- Generates side-by-side image + predicted mask visualizations

## Folder Structure

```bash
airbus-ship-detection-cnn/
├── data/                  # Raw and processed data (not uploaded here)
├── models/                # Trained models
├── notebooks/             # Jupyter notebooks for exploration and training
├── submission/            # Final submission CSVs
├── utils/                 # Helper functions (preprocessing, RLE, etc.)
├── README.md              # This file
└── requirements.txt       # All required packages (TensorFlow, OpenCV, etc.)
```

## Results & Highlights

| Model           | Val Dice | Val IoU  | LB Public | LB Private |
|----------------|----------|----------|-----------|------------|
| ResUNet        | ~0.64    | ~0.50    | 0.517     | 0.755 ✅   |
| U-Net++        | ~0.66    | ~0.52    | 0.477     | 0.715      |
| Attention U-Net| ~0.63    | ~0.49    | 0.505     | 0.728      |

**ResUNet++** performed best on the **private leaderboard**, which is the official metric for the competition ranking.


## Dependencies

- Python 3.10+
- TensorFlow 2.x
- Albumentations
- OpenCV
- scikit-image
- pandas, numpy, matplotlib, seaborn

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Inference Sample

```python
from utils.prediction import predict_mask
from utils.visuals import save_visualization

image = imread('sample_test.jpg')
pred = predict_mask(image)
save_visualization("sample_test", image, pred)
```

## References

- 📘 [Kaggle Competition Page](https://www.kaggle.com/competitions/airbus-ship-detection)


## Author

Sherali Ozodov  
[GitHub Profile](https://github.com/sheraliozodov)  


## License

This repository is for educational and research purposes only.
