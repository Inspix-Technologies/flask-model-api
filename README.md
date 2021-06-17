# Flask REST API
[REST](https://en.wikipedia.org/wiki/Representational_state_transfer) [API](https://en.wikipedia.org/wiki/API)s are commonly used to expose Machine Learning (ML)  models to other services. This folder contains an example REST API created using Flask to expose the YOLOv5s model from [PyTorch Hub](https://pytorch.org/hub/ultralytics_yolov5/).

## Requirements

[Flask](https://palletsprojects.com/p/flask/) is required. Install with:
```shell
$ pip install Flask
```

## Run

After Flask installation run:

```shell
$ python3 restapi.py --port 5000
