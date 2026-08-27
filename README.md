# Video Action Recognition with 3D CNN

3D Convolutional Neural Network for temporal action recognition in videos.
Classifies 5 action categories: Static, Slow Motion, Fast Motion, Rotation, Vibration.

## Architecture
- **3D Convolutions**: Process both spatial (H×W) and temporal (T) dimensions
- **Multi-stage encoding**: 4 conv blocks with max pooling (spatial + temporal)
- **Global average pooling**: Aggregate temporal information
- **Classification head**: Fully connected layers

## Key Features
- **Permutation-invariant**: Learns temporal dynamics
- **GPU-optimized**: Efficient 3D convolution kernels
- **Dropout regularization**: Prevents overfitting on video sequences
- **Multi-scale temporal sampling**: Captures motion at different speeds

## Applications
- Robotics action understanding (robot arm motion classification)
- Autonomous systems (behavior monitoring)
- Video surveillance (action detection)
- Human activity recognition

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/train.py
python src/evaluate.py
```

## Results
- `results/training_curves.png` — loss and accuracy curves
- `results/confusion_matrix.png` — per-action classification performance
- `results/per_class_accuracy.png` — individual action recognition rates

## Author
Hessam Kaveh — Research Fellow, Italian Institute of Technology
