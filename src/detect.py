from ultralytics import YOLO

model = YOLO("yolov8n.pt")   # downloads the small pretrained model automatically
results = model("https://ultralytics.com/images/bus.jpg", device=0)  # device=0 means GPU
results[0].save("data/test_result.jpg")
for box in results[0].boxes:
    print(model.names[int(box.cls)], round(float(box.conf), 2))