"""
Run a rest API exposing the yolov5s object detection model
"""
import argparse
import io

import torch
from PIL import Image
from flask import Flask, request
import base64
from flask_socketio import SocketIO, send, emit
from flask_cors import CORS, cross_origin


app = Flask(__name__)
DETECTION_URL = "/inspix-models/test"
socket_io = SocketIO(app, cors_allowed_origins="*")


@app.route(DETECTION_URL, methods=["POST"])
@cross_origin()
def predict():
    # if not request.method == "POST":
    #     return
    base64img = request.json['base64image']

    base64_bytes = base64img.encode('ascii')
    img_data = base64.b64decode(base64_bytes)

    img = Image.open(io.BytesIO(img_data))

    results = model(img, size=640)  # reduce size=320 for faster inference
    return results.pandas().xyxy[0].to_json(orient="records")


@socket_io.on('predict')
def handle_predict(json):
    base64img = json['base64image']
    filtered_base64 = base64img.split(',')[1]
    base64_bytes = filtered_base64.encode('ascii')
    img_data = base64.b64decode(base64_bytes)

    img = Image.open(io.BytesIO(img_data))

    results = model(img, size=640)
    emit('predict-result', results.pandas().xyxy[0].to_json(orient="records"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=8001, type=int, help="port number")
    args = parser.parse_args()
    print('downloadings')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/mask-best.pt')  # force_reload to recache
    socket_io.run(app, host="0.0.0.0", port=args.port, debug=True)
    # app.run(host="0.0.0.0", port=args.port, debug=True)
