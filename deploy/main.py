import io
from PIL import Image, ImageDraw
from flask import Flask, request, send_file
from ultralytics import YOLO,RTDETR
import cv2
import base64
import os

# from flask_swagger_ui import get_swaggerui_blueprint

# dependency
# pip install nest-asyncio pyngrok
# pip install gunicorn    -> WSGI for deploy

# run this project
# flask --app main.py --debug run
# sudo docker compose up -d --scale flask-yolo=2

app = Flask(__name__)

# increse file input size
app.config["MAX_CONTENT_LENGTH"] = 16 * 1000 * 1000

# model_yolo = YOLO('./models/yolov8/best_32batch_rect_yolov8.pt')
model_path = "./models"
if not os.path.exists(model_path):
    os.makedirs(model_path, exist_ok=True)

# model_yolo = YOLO("./models/yolomod/best.pt")
# model_yolo = YOLO('./models/yolo11/best.pt')
# model_yolo = YOLO('./models/yolov10/best.pt')
# model_yolo = YOLO('./models/yolov9/best.pt')
# model_yolo = YOLO('./models/yolov8/best.pt')
# model_yolo = YOLO('./models/yolov-worldv2/best.pt')
model_yolo = RTDETR('./models/rt-detr/best.pt')

# Display model information (optional)
# model_yolo.info()


@app.route("/")
def home():
    with open("index.html") as file:
        return file.read()


@app.route("/hello", methods=["GET"])
def hello():
    return {"result": "Hello World!!"}


@app.route("/objectdetection", methods=["POST"])
def predict():
    if not request.method == "POST":
        return {"result": "wrong method"}

    if request.files.get("image"):
        # import file
        image_file = request.files["image"]
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))

        # model_yolo predict confidence %
        results = model_yolo.predict(img, conf=0.1)

        # recreate array that return number of larva
        result = results[0]
        output = []
        larva_state_1 = []
        larva_state_2 = []
        larva_state_3 = []
        larva_state_4 = []
        larva_state_5 = []
        larva_state_6 = []
        for box in result.boxes:
            x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
            class_id = box.cls[0].item()
            prob = round(box.conf[0].item(), 2)
            output.append([x1, y1, x2, y2, result.names[class_id], prob])

            # classify larva_stage
            if (
                result.names[class_id] == "larva1"
                or result.names[class_id] == "larva_1"
            ):
                larva_state_1.append(result.names[class_id])
            if (
                result.names[class_id] == "larva2"
                or result.names[class_id] == "larva_2"
            ):
                larva_state_2.append(result.names[class_id])
            if (
                result.names[class_id] == "larva3"
                or result.names[class_id] == "larva_3"
            ):
                larva_state_3.append(result.names[class_id])
            if (
                result.names[class_id] == "larva4"
                or result.names[class_id] == "larva_4"
            ):
                larva_state_4.append(result.names[class_id])
            if (
                result.names[class_id] == "larva5"
                or result.names[class_id] == "larva_5"
            ):
                larva_state_5.append(result.names[class_id])
            if (
                result.names[class_id] == "larva6"
                or result.names[class_id] == "larva_6"
            ):
                larva_state_6.append(result.names[class_id])

    return {
        "result": {
            "larva_stage": {
                "larva1": len(larva_state_1),
                "larva2": len(larva_state_2),
                "larva3": len(larva_state_3),
                "larva4": len(larva_state_4),
                "larva5": len(larva_state_5),
                "larva6": len(larva_state_6),
            },
            "larva_total": len(output),
        }
    }


@app.route("/objectdetection_img", methods=["POST"])
def predict_img():
    if not request.method == "POST":
        return {"result": "wrong method"}

    if request.files.get("image"):
        print("processing....")

        # import file
        image_file = request.files["image"]
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))

        # model_yolo predict confidence %
        results = model_yolo.predict(img, conf=0.1)

        # input must be byte type
        res_plotted = results[0].plot(conf=False)

        # Save the image to a temporary file
        cv2.imwrite("temp_image.jpg", res_plotted)

        # Return the image file as a response
        return send_file("temp_image.jpg", mimetype="image/jpeg")


@app.route("/objectdetection_img_base64", methods=["POST"])
def predict_img_base64():
    if not request.method == "POST":
        return {"result": "wrong method"}

    if request.files.get("image"):
        # import file
        image_file = request.files["image"]
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))

        # model_yolo predict confidence %
        results = model_yolo.predict(img, conf=0.1)

        # input must be byte type
        res_plotted = results[0].plot(conf=False)

        # Save the image to a temporary file
        cv2.imwrite("temp_image.jpg", res_plotted)

        with open("temp_image.jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())

        # Return the image file as a response
        return encoded_string

# Use this when want to debug on phone
# app.run(debug=True, host='0.0.0.0', port=5000)
# app.run(debug=True)
