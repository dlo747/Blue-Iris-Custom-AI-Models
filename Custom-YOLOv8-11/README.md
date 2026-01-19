# License Plate Detection Model

Custom YOLO11n model trained specifically for license plate detection in video surveillance applications.

## Finetuned model MikeLud initial model. 
## Dataset was finetuned on night time USA license plate photos for more accurate night time detection

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
