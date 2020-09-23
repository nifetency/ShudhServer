import argparse
import common
import cv2
import numpy as np
import time
import logging

from gabriel_protocol import gabriel_pb2
from gabriel_server import local_engine
from gabriel_server import cognitive_engine


logging.basicConfig(filename="Loglatency.log", level=logging.INFO)
net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
classes = []
with open('coco.names','r') as f:
    classes = f.read().splitlines()

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))


def getresults(img_str):
    t = time.time()
    nparr = np.fromstring(img_str, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR
   
    height, width,_ =  img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

    t1 = time.time()
    tf = t1 - t
    logging.info('{}'.format(tf))
    return cv2.imencode('.jpg', img)[1].tostring()


class DisplayEngine(cognitive_engine.Engine):
    def handle(self, input_frame):
        status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        result_wrapper = cognitive_engine.create_result_wrapper(status)

        image = getresults(input_frame.payloads[0])
        result = gabriel_pb2.ResultWrapper.Result()
        result.payload_type = gabriel_pb2.PayloadType.IMAGE
        #result.payload = input_frame.payloads[0]
        result.payload = image

        result_wrapper.results.append(result)

        return result_wrapper



def main():
    common.configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'source_name', nargs='?', default=common.DEFAULT_SOURCE_NAME)
    args = parser.parse_args()

    def engine_factory():
        return DisplayEngine()
    
    local_engine.run(engine_factory, args.source_name, input_queue_maxsize=60,
                     port=common.WEBSOCKET_PORT, num_tokens=2)


if __name__ == '__main__':
    main()
