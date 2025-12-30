# Blue Iris Custom AI Models

![Blue Iris](blue_iris.jpg)

Custom trained AI models optimized for use with Blue Iris video surveillance software.

## Overview

This repository contains custom ONNX models trained specifically for detection tasks in Blue Iris. These models are optimized for accuracy and performance in video surveillance scenarios.

## Models Included

### License Plate Detection Model
- **Path**: `custom-license-plates-model/plates.onnx`
- **Framework**: YOLO11n (nano variant)
- **Format**: ONNX (Open Neural Network Exchange)
- **Task**: Object Detection
- **Classes**: 1 (license_plate)

## Requirements

- Blue Iris version 6.0.1.2 or higher with AI support
- Compatible AI processing backend (CPU or GPU with DirectML support)

## Installation

1. Clone this repository or download the model files
2. Copy the desired `.onnx` model file to your Blue Iris AI models directory

## Model Performance

The license plate detection model achieves:
- **Precision**: 98.9%
- **Recall**: 97.3%
- **mAP50**: 99.3%
- **mAP50-95**: 85.9%

Trained on 42,024 training images and validated on 11,988 validation images over 300 epochs.

## Model Details

### License Plate Model Specifications
- **Architecture**: YOLO11n (lightweight nano variant)
- **Input Size**: 640×640 pixels
- **Parameters**: 2,590,035
- **Training Device**: NVIDIA GeForce RTX 5090
- **Training Framework**: Ultralytics YOLO 8.3.195
- **Inference Format**: TorchScript ONNX

### Training Configuration
- Epochs: 300
- Batch Size: 128
- Optimizer: SGD with automatic tuning
- Image Size: 640×640
- Augmentation: Mosaic, mixup, HSV adjustments, flip, scale, translate

## Repository Structure

```
Blue-Iris-Custom-AI-Models/
├── LICENSE                           # MIT License
├── README.md                         # This file
└── custom-license-plates-model/      # License plate detection model
    ├── plates.onnx                   # ONNX model file
    └── plates.onnx_training_results/ # Training metrics and logs
        ├── args.yaml                 # Training arguments
        ├── results.csv               # Training results data
        └── training-results.txt      # Complete training log
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! If you have improvements or additional models to share:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues or questions:
- Open an issue in this repository
- Consult Blue Iris documentation for integration help
- Check the model training results for performance metrics

## Acknowledgments

- Built with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- Designed for [Blue Iris](https://blueirissoftware.com/) video surveillance software
- Trained using PyTorch and exported to ONNX format

## Version History

- **Initial Release**: License plate detection model based on YOLO11n
  - High accuracy detection (99.3% mAP50)
  - Optimized for real-time inference
  - Lightweight architecture suitable for various hardware configurations

## Technical Notes

### Model Export
The models are exported using TorchScript format with simplification enabled for optimal inference performance.

### Hardware Requirements
- **Minimum**: Modern CPU with AVX2 support
- **Recommended**: NVIDIA GPU with DirectML support for real-time processing
- **Memory**: At least 4GB RAM for model loading and inference

### Performance Tips
- Use GPU acceleration when available
- Adjust confidence thresholds based on your environment
- Configure appropriate detection zones to reduce false positives
- Monitor system resources during high camera count deployments
