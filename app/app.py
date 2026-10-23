import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# ====== YE DO CHEEZEN APNE HISAAB SE BADLO ======
MODEL_PATH = r"runs\detect\runs\baseline_ds4\weights\best.pt"  # apna best.pt ka path

# class_name: (estimated weight in grams, scrap rate in Rs/kg)
# Class ke naam bilkul wahi likho jo model.names mein hain (sidebar mein dikhenge)
ESTIMATES = {
    "Can": (15, 80),
    "HDPE": (40, 30),
    "PET_Bottle": (20, 12),
    "Plastic_wrapper": (5, 3),
    "Tetrapak": (25, 5),
}
DEFAULT = (20, 10)  # jo class dictionary mein nahi hai uske liye
# =================================================

st.set_page_config(page_title="Smart Waste Analyzer", layout="wide")
st.title("Smart Waste Analyzer")
st.caption("Detection model ka output. Weight aur rate abhi estimated values hain.")


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()

conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
st.sidebar.write("Model ki classes:")
st.sidebar.write(list(model.names.values()))

file = st.file_uploader("Waste ki photo upload karo", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    result = model.predict(img, conf=conf, verbose=False)[0]

    col1, col2 = st.columns(2)
    col1.subheader("Original")
    col1.image(img, use_container_width=True)
    col2.subheader("Detection")
    col2.image(result.plot()[..., ::-1], use_container_width=True)  # BGR -> RGB

    rows = []
    for box in result.boxes:
        name = model.names[int(box.cls)]
        w, rate = ESTIMATES.get(name, DEFAULT)
        rows.append({
            "Class": name,
            "Confidence": round(float(box.conf), 2),
            "Est. weight (g)": w,
            "Rate (Rs/kg)": rate,
            "Value (Rs)": round(w / 1000 * rate, 2),
        })

    if rows:
        df = pd.DataFrame(rows)
        st.subheader("Items")
        st.dataframe(df, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total items", len(df))
        c2.metric("Total weight (g)", int(df["Est. weight (g)"].sum()))
        c3.metric("Total value (Rs)", round(df["Value (Rs)"].sum(), 2))
        st.caption("Weight aur rate estimated hain, model ka output nahi.")
    else:
        st.warning("Kuch detect nahi hua. Confidence slider kam karke dekho.")