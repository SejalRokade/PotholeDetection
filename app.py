import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Road Damage Detection", page_icon="🛣️", layout="wide")

# =============================
# THEME TOGGLE (TOP RIGHT)
# =============================
col1, col2 = st.columns([8,1])
with col2:
    dark_mode = st.toggle("🌙 Dark Mode")

# =============================
# APPLY THEME
# =============================
if dark_mode:
    st.markdown("""
        <style>
        .stApp {background-color: #0e1117; color: white;}
        section[data-testid="stSidebar"] {background-color: #161b22 !important;}
        h1, h2, h3, h4 {color: white !important;}
        .stMarkdown, .stText {color: white !important;}
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp {background-color: #f7f9fc;}
        section[data-testid="stSidebar"] {background-color: #ffffff;}
        </style>
    """, unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.title("🛣️ Road Damage Detection System")
st.markdown("### YOLOv8 Segmentation – Pothole Detection")

# =============================
# SIDEBAR CONTROLS
# =============================
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3)
st.sidebar.markdown("---")
st.sidebar.info("Model: YOLOv8-Seg\nClass: Pothole")

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =============================
# FILE UPLOADER
# =============================
uploaded_files = st.file_uploader(
    "Upload Image(s) or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
    accept_multiple_files=True
)

# =============================
# PROCESS FILES
# =============================
if uploaded_files:

    for file in uploaded_files:

        file_type = file.name.split(".")[-1].lower()
        st.markdown("---")
        st.subheader(f"📁 {file.name}")

        # ================= IMAGE =================
        if file_type in ["jpg", "jpeg", "png"]:

            image = Image.open(file).convert("RGB")

            with st.spinner("Processing image..."):
                results = model(image, conf=confidence)
                result_img = results[0].plot()

            st.image(result_img, use_column_width=True)

            if results[0].masks is not None:
                count = len(results[0].boxes)
                st.success(f"Potholes Detected: {count}")
            else:
                st.warning("No potholes detected.")

            # Proper download (PNG format)
            _, buffer = cv2.imencode(".png", result_img)
            st.download_button(
                label="⬇ Download Processed Image",
                data=buffer.tobytes(),
                file_name=f"processed_{file.name}",
                mime="image/png"
            )

        # ================= VIDEO =================
        elif file_type in ["mp4", "avi", "mov"]:

            suffix = "." + file_type
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tfile.write(file.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            output_path = os.path.join(tempfile.gettempdir(), f"processed_{file.name}")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            total_potholes = 0
            frame_count = 0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            progress_bar = st.progress(0)
            frame_placeholder = st.empty()   # 🔥 live frame container
            unique_ids = set()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=confidence
                )

                annotated_frame = results[0].plot()

                if results[0].boxes.id is not None:
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    for id_ in ids:
                        unique_ids.add(id_)

                out.write(annotated_frame)

                frame_placeholder.image(
                    annotated_frame,
                    channels="BGR",
                    use_column_width=True
                )

                frame_count += 1
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            out.release()
            progress_bar.empty()

            st.success("Video processing completed ✅")
            st.success(f"Total Frames Processed: {frame_count}")
            st.success(f"Total Pothole Detections: {len(unique_ids)}")

            with open(output_path, "rb") as f:
                video_bytes = f.read()

            st.video(video_bytes)

            st.download_button(
                label="⬇ Download Processed Video",
                data=video_bytes,
                file_name=f"processed_{file.name}",
                mime="video/mp4"
            )