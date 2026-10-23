from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(
        data="data/raw/ds4/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        workers=2,
        device=0,
        project="runs",
        name="baseline_ds4",
    )