"""
Run a rest API exposing the yolov5s object detection model
"""
import argparse
import io

import torch
from PIL import Image
from flask import Flask, request
import base64

app = Flask(__name__)
DETECTION_URL = "/inspix-models/test"


@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return
    base64img = request.json['base64image']
    if True:
        base64_bytes = base64img.encode('ascii')
        img_data = base64.b64decode(base64_bytes)

        img = Image.open(io.BytesIO(img_data))

        results = model(img, size=640)  # reduce size=320 for faster inference
        return results.pandas().xyxy[0].to_json(orient="records")
    return "request invalid"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=8001, type=int, help="port number")
    args = parser.parse_args()
    print('downloadings')
    model = torch.hub.load("ultralytics/yolov5", "yolov5s",
                           force_reload=True)  # force_reload to recache
    app.run(host="0.0.0.0", port=args.port, debug=True)
