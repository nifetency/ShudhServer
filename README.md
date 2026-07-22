# ShudhServer

A [Gabriel](https://github.com/cmusatyalab/gabriel) cognitive engine server that performs real-time object detection on video frames streamed over websocket.

Frames are received from a Gabriel client (e.g. a wearable or mobile camera), run through a YOLOv3 model trained on the COCO dataset, and detection results are returned to the client. This is an early edge-computing / wearable cognitive assistance experiment, related to [picamcommunication](https://github.com/nifetency/picamcommunication) and [yolotrial](https://github.com/nifetency/yolotrial) in this org.

## How it works

- Loads a YOLOv3 model (`yolov3.cfg` + `yolov3.weights`, not checked into this repo) via OpenCV's DNN module
- Classifies detected objects against the standard COCO class list (`coco.names`)
- Runs as a Gabriel `local_engine` server, listening on port `9099` (see `common.py`)
- Logs request latency to `Loglatency.log`

## Requirements

See `requirements.txt`:
- `gabriel-client` / `gabriel-server` (2.0.1)
- `opencv-python`
- `torchvision`
- `PyQt5`
- `py-cpuinfo`

You'll also need YOLOv3 weights (`yolov3.weights`) alongside the included `yolov3.cfg`, which aren't included in this repo due to file size.

## Running

```bash
pip install -r requirements.txt
python server.py
```
