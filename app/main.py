import os
import cv2
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

dir_name = "images_uploaded"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

YOLO_DIR = ".cvlib/object_detection/yolo/yolov3"
weights_path = f"{YOLO_DIR}/yolov3-tiny.weights"
cfg_path = f"{YOLO_DIR}/yolov3-tiny.cfg"
names_path = f"{YOLO_DIR}/yolov3_classes.txt"

net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

with open(names_path, "r") as f:
    CLASSES = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def detect_objects(image, conf_threshold=0.5, nms_threshold=0.3):
    height, width = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)
    
    boxes, confidences, class_ids = [], [], []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    return boxes, confidences, class_ids, indices


def draw_boxes(image, boxes, confidences, class_ids, indices):
    for i in indices:
        x, y, w, h = boxes[i]
        label = CLASSES[class_ids[i]]
        confidence = confidences[i]
        color = COLORS[class_ids[i]]
        
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = f"{label}: {confidence:.2f}"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

# Assign an instance of the FastAPI class to the variable "app".
# You will interact with your api using this instance.
app = FastAPI(title='Deploying a ML Model with FastAPI')

# By using @app.get("/") you are allowing the GET method to work for the / endpoint.
@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to <your_server>/docs"


@app.post("/predict") 
def prediction(file: UploadFile = File(...)):
    filename = str(file.filename)
    if filename.split(".")[-1].lower() not in ("jpg", "jpeg", "png"):
        raise HTTPException(status_code=415, detail="Unsupported file provided.")
    image_stream = io.BytesIO(file.file.read())
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    boxes, confidences, class_ids, indices = detect_objects(image)
    output_image = draw_boxes(image, boxes, confidences, class_ids, indices)
    
    cv2.imwrite(f'{dir_name}/{filename}', output_image)
    file_image = open(f'{dir_name}/{filename}', mode="rb")
    
    # Return the image as a stream specifying media type
    return StreamingResponse(file_image, media_type="image/jpeg")