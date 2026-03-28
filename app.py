import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from PIL import Image
from collections import Counter

# Load model
model = YOLO("best.pt")

st.set_page_config(layout="wide")

# Title
st.title("🚗 Smart Traffic Monitoring System")

# Sidebar
st.sidebar.header("⚙️ Settings")
option = st.sidebar.selectbox("Choose Input Type", ["Image", "Video", "Live Camera"])
conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# -------- SESSION STATE INIT --------
if "prev_centroids" not in st.session_state:
    st.session_state.prev_centroids = []

if "in_count" not in st.session_state:
    st.session_state.in_count = 0

if "out_count" not in st.session_state:
    st.session_state.out_count = 0

# -------- VEHICLE COUNT FUNCTION --------
def count_vehicles(results):
    names = results[0].names
    classes = results[0].boxes.cls.tolist()

    labels = [names[int(c)] for c in classes]

    vehicle_classes = ["car", "truck", "bus", "motorcycle", "bicycle"]
    filtered = [l for l in labels if l in vehicle_classes]

    return Counter(filtered)

# -------- DIRECTION TRACKING --------
def get_direction(current_centroids, frame_height):
    prev_centroids = st.session_state.prev_centroids
    in_count = st.session_state.in_count
    out_count = st.session_state.out_count

    mid_line = frame_height // 2

    for (cx, cy) in current_centroids:
        for (px, py) in prev_centroids:
            if abs(cx - px) < 50 and abs(cy - py) < 50:
                if py < mid_line and cy > mid_line:
                    in_count += 1
                elif py > mid_line and cy < mid_line:
                    out_count += 1

    st.session_state.prev_centroids = current_centroids
    st.session_state.in_count = in_count
    st.session_state.out_count = out_count

# ---------------- IMAGE ----------------
if option == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="📷 Original Image", width=400)

        if st.button("🚀 Detect Vehicles"):
            results = model(image, conf=conf)
            plotted = results[0].plot()

            counts = count_vehicles(results)

            with col2:
                st.image(plotted, caption="✅ Detection Output", width=400)

                st.subheader("📊 Vehicle Count")
                for k, v in counts.items():
                    st.metric(label=k.upper(), value=v)

# ---------------- VIDEO ----------------
elif option == "Video":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        metric_placeholder = st.empty()

        if st.button("▶️ Start Detection"):
            # Reset counters
            st.session_state.in_count = 0
            st.session_state.out_count = 0
            st.session_state.prev_centroids = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                h, w, _ = frame.shape

                results = model(frame, conf=conf)
                annotated = results[0].plot()

                # ---- Get centroids ----
                current_centroids = []
                if results[0].boxes is not None:
                    for box in results[0].boxes.xyxy:
                        x1, y1, x2, y2 = map(int, box)
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        current_centroids.append((cx, cy))

                get_direction(current_centroids, h)

                counts = count_vehicles(results)

                # Draw middle line
                cv2.line(annotated, (0, h//2), (w, h//2), (0, 255, 255), 2)

                stframe.image(annotated, channels="BGR", width=500)

                # ---- Metrics ----
                with metric_placeholder.container():
                    cols = st.columns(3)
                    cols[0].metric("⬇️ Inward", st.session_state.in_count)
                    cols[1].metric("⬆️ Outward", st.session_state.out_count)

                if sum(counts.values()) > 15:
                    st.warning("🚨 High Traffic Detected!")

        cap.release()

# ---------------- LIVE CAMERA ----------------
elif option == "Live Camera":

    cap = cv2.VideoCapture(0)

    stframe = st.empty()
    metric_placeholder = st.empty()

    if st.button("📷 Start Camera"):
        # Reset counters
        st.session_state.in_count = 0
        st.session_state.out_count = 0
        st.session_state.prev_centroids = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape

            results = model(frame, conf=conf)
            annotated = results[0].plot()

            # ---- Get centroids ----
            current_centroids = []
            if results[0].boxes is not None:
                for box in results[0].boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    current_centroids.append((cx, cy))

            get_direction(current_centroids, h)

            # Draw middle line
            cv2.line(annotated, (0, h//2), (w, h//2), (0, 255, 255), 2)

            stframe.image(annotated, channels="BGR", width=500)

            # ---- Metrics ----
            with metric_placeholder.container():
                cols = st.columns(3)
                cols[0].metric("⬇️ Inward", st.session_state.in_count)
                cols[1].metric("⬆️ Outward", st.session_state.out_count)

    cap.release()