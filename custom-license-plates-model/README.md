# License Plate Detection Model

Custom YOLO11n model trained specifically for license plate detection in video surveillance applications.

## Model Information

- **Model Name**: plates.onnx
- **Architecture**: YOLO11n (nano - lightweight variant)
- **Task**: Object Detection
- **Classes**: 1 (license_plate)
- **Format**: ONNX (TorchScript)
- **Input Size**: 640×640 pixels
- **Parameters**: 2,590,035 (~2.6M)
- **GFLOPs**: 6.4

## Performance Metrics

Final model performance after 300 epochs of training:

| Metric | Value |
|--------|-------|
| Precision | 98.9% |
| Recall | 97.3% |
| mAP50 | 99.3% |
| mAP50-95 | 85.9% |

## Dataset

- **Training Images**: 42,024 images (684 background)
- **Validation Images**: 11,988 images (176 background)
- **Data Augmentation**: Mosaic, HSV adjustments, flipping, scaling, translation

## Training Details

### Hardware
- **GPU**: NVIDIA GeForce RTX 5090 (32GB VRAM)
- **CUDA**: Enabled with automatic mixed precision (AMP)

### Training Configuration

```yaml
Model: YOLO11n.pt (pretrained)
Epochs: 300
Batch Size: 128
Image Size: 640×640
Optimizer: SGD (auto-tuned)
  - Learning Rate: 0.01
  - Momentum: 0.9
Workers: 8
Device: CUDA (GPU 0)
```

### Loss Weights
- Box Loss: 7.5
- Classification Loss: 0.5
- DFL Loss: 1.5

### Data Augmentation
- **HSV**: H=0.015, S=0.7, V=0.4
- **Flip LR**: 0.5 probability
- **Mosaic**: 0.7 probability (disabled last 10 epochs)
- **Translate**: 0.1
- **Scale**: 0.5
- **Erasing**: 0.2
- **Albumentations**: Blur, MedianBlur, ToGray, CLAHE

## Model Architecture

The YOLO11n architecture consists of 181 layers:

- **Backbone**: Efficient feature extraction with Conv and C3k2 blocks
- **Neck**: SPPF and C2PSA modules for multi-scale feature fusion
- **Head**: Detect module for bounding box and classification outputs

Key architectural features:
- 3 detection scales for multi-scale object detection
- Upsampling and concatenation for feature pyramid
- Optimized for speed and accuracy balance

## Training Progress

The model showed consistent improvement throughout training:

- **Early Training (Epochs 1-10)**: Rapid initial learning
  - Box Loss: 0.89 → 0.75
  - mAP50: 97.7% → 98.9%
  
- **Mid Training (Epochs 11-100)**: Steady refinement
  - Box Loss: 0.74 → 0.66
  - mAP50-95: 80.5% → 84.7%

- **Late Training (Epochs 101-290)**: Fine-tuning with patience
  - Box Loss: 0.66 → 0.50
  - mAP50-95: 84.7% → 85.9%

- **Final Phase (Epochs 291-300)**: Mosaic disabled for stability
  - Final convergence with consistent high performance

Training utilized early stopping with patience=30, but continued to improve through all 300 epochs.

## Files Included

- **plates.onnx** - The trained model file (ready for inference)
- **plates.onnx_training_results/** - Complete training documentation
  - **args.yaml** - Full training configuration
  - **results.csv** - Per-epoch metrics in CSV format
  - **training-results.txt** - Detailed training log with epoch-by-epoch results

## Usage

### With Blue Iris

1. Copy `plates.onnx` to your Blue Iris AI models directory
2. Set confidence threshold (recommended: Day:0.80 Night:0.85)
3. Configure detection zones as needed

### General ONNX Inference

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load the model
session = ort.InferenceSession("plates.onnx")

# Prepare input (640x640, normalized)
image = Image.open("your_image.jpg").resize((640, 640))
input_array = np.array(image).astype(np.float32) / 255.0
input_array = np.transpose(input_array, (2, 0, 1))
input_array = np.expand_dims(input_array, axis=0)

# Run inference
outputs = session.run(None, {session.get_inputs()[0].name: input_array})

# Process outputs (boxes, scores, classes)
# ... post-processing code here ...
```

## Inference Performance

Expected inference speeds (may vary by hardware):

- **RTX 5090**: ~2-3ms per image
- **RTX 3080**: ~5-8ms per image  
- **RTX 2060**: ~10-15ms per image
- **CPU (modern)**: ~100-200ms per image

## Optimization Tips

1. **Confidence Threshold**: Start with 0.5 and adjust based on your environment
2. **NMS Threshold**: Default 0.7 works well for most scenarios
3. **Input Resolution**: 640×640 is optimal; smaller sizes reduce accuracy
4. **Batch Processing**: Use batch size >1 when processing multiple images
5. **Hardware Acceleration**: Always use GPU when available for real-time processing

## Limitations

- Trained primarily on standard vehicle license plates
- Performance may vary with:
  - Extreme angles or distances
  - Heavily obscured or damaged plates
  - Non-standard plate formats
  - Very low light conditions

## Retraining

To retrain or fine-tune this model:

```bash
# Install ultralytics
pip install ultralytics

# Train with your own data
yolo detect train model=yolo11n.pt data=your_data.yaml epochs=300 imgsz=640 batch=128

# Export to ONNX
yolo export model=best.pt format=onnx simplify=True
```

Refer to `args.yaml` for the exact training configuration used.

## Version

- **Model Version**: 1.0
- **Training Date**: 2025
- **Ultralytics Version**: 8.3.195
- **PyTorch Version**: 2.7.1+cu128
- **Python Version**: 3.11.10

## License

This model is released under the MIT License. See the repository root LICENSE file for details.

## Credits

Trained using the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) framework.
