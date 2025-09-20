# BinAI

BinAI is an API service for detecting bins in images using a YOLO11 ONNX model. It provides a simple HTTP interface for image inference, returning confidence scores and detection results. The API is designed for easy integration into automation, robotics, or smart waste management systems where bin detection is required.

## Features

- **YOLO11 ONNX Model:** Fast and efficient bin detection using a pre-trained YOLO11 model.
- **REST API Endpoints:**
  - `/` and `/health`: Service status and health checks.
  - `/detect`: Main detection endpoint. Accepts base64-encoded images and returns detection confidence and bin presence.
- **Configurable Confidence Thresholds:** Customizable detection sensitivity via API parameters.
- **Production-ready:** Built with Flask and ONNX Runtime, supports CORS and deployment on platforms like Render.

## How It Works

1. Send a POST request to `/detect` with a base64-encoded image.
2. The API processes the image, runs inference using YOLO11, and returns:
   - Whether a bin is detected (`isBin`)
   - Confidence score for the detection
   - Contextual message based on confidence
   - Total detections found in the image

## Example Request

```bash
curl -X POST https://your-api-url/detect \
     -H "Content-Type: application/json" \
     -d '{"image": "<base64_image_data>", "threshold": 0.75}'
```

## Requirements

- Python
- Flask
- ONNX Runtime
- OpenCV
- Pillow (PIL)

## Usage

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
python app.py
```
