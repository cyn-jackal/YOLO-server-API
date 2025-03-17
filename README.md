# Flask YOLO Object Detection

A simple Flask application using [YOLO11-DSConv](https://github.com/cyn-jackal/YOLO11-DSConv) to detect objects (e.g., larva stages) in images and build on Open Source [YOLO11 models](https://github.com/ultralytics/ultralytics). You can run it locally or via Docker/Docker Compose.

## Quick Start

1. **Clone/Download** the project and place your model at `./models/YOLO11-DSConv-model/best.pt`.
2. **Install dependencies**:
    ```
    pip install -r requirements.txt
    ```
3. **Run Project**:
    ```
    flask --app main.py --debug run
    ```

## Quick Start with docker compose (Windowns) recommend

1. **Clone/Download** the project and place your model at `./models/YOLO11-DSConv-model/best.pt`.
2. **Doecker compose**:
    ```
    docker compose up -d --scale flask-yolo=2
    ```
    flask-yolo mean the number of load-balance that you want in server-side

## Quick Start with docker (Linux) recommend

1. **Clone/Download** the project and place your model at `./models/YOLO11-DSConv-model/best.pt`.
2. **Doecker compose**:
    ```bash
    sudo docker compose up -d --build --scale flask-yolo=2
    ```
    flask-yolo mean the number of load-balance that you want in server-side
