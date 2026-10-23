from ultralytics import YOLO

if __name__ == "__main__":
    for tag in ("A", "B"):
        m = YOLO("yolov8n-cls.pt")
        m.train(data=f"data/cls_{tag}", epochs=30, imgsz=64, batch=64,
                device=0, project="runs", name=f"cls_{tag}", seed=0)
        r = m.val(data=f"data/cls_{tag}", split="test", imgsz=64, device=0)
        print(f"MODEL {tag}  test top1 accuracy: {r.top1:.4f}")