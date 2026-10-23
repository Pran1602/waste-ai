from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/detect/runs/baseline_ds4/weights/best.pt")
    model.val(
        data="data/raw/ds4/data.yaml",
        split="test",
        device=0,
        project="runs",
        name="eval_baseline_test",
    )