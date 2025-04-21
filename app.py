import streamlit as st
import cv2
import numpy as np
import io
import traceback
import logging
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS

# Cấu hình log
logging.basicConfig(level=logging.DEBUG)

class InferenceApp:
    def __init__(self, model_path="yolov8n.pt"):
        check_requirements("streamlit>=1.29.0")
        self.st = st
        self.source = "webcam"
        self.enable_trk = False
        self.conf = 0.25
        self.iou = 0.45
        self.org_frame = None
        self.ann_frame = None
        self.vid_file_name = None
        self.selected_ind = []
        self.cap = None

        # Load model
        try:
            LOGGER.info("Loading YOLO model...")
            self.model = YOLO(model_path)
            self.model_path = model_path
            LOGGER.info("Model loaded successfully.")
        except Exception as e:
            LOGGER.error(f"Error loading YOLO model: {e}")
            st.error(f"Error loading YOLO model: {e}")
            self.model = None

    def setup_ui(self):
        self.st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide")

        self.st.markdown("<style>MainMenu {visibility: hidden;}</style>", unsafe_allow_html=True)
        self.st.markdown(
            "<h1 style='text-align: center; color: #FF64DA;'>Ultralytics YOLO Streamlit Application</h1>",
            unsafe_allow_html=True
        )
        self.st.markdown(
            "<h4 style='text-align: center; color: #042AFF;'>Real-time object detection with YOLO 🚀</h4>",
            unsafe_allow_html=True
        )

    def setup_sidebar(self):
        with self.st.sidebar:
            self.st.image("https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg", width=250)

            self.source = self.st.selectbox("Video Source", ["webcam", "video"])
            self.enable_trk = self.st.radio("Enable Tracking", ["Yes", "No"]) == "Yes"
            self.conf = self.st.slider("Confidence Threshold", 0.0, 1.0, self.conf, 0.01)
            self.iou = self.st.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.01)

            # Load models
            available_models = [x.replace("yolo", "YOLO") for x in GITHUB_ASSETS_STEMS if x.startswith("yolo8")]
            selected_model = self.st.selectbox("Select Model", [self.model_path.split(".pt")[0]] + available_models)
            if selected_model.lower() + ".pt" != self.model_path:
                try:
                    with self.st.spinner("Downloading and loading model..."):
                        self.model = YOLO(selected_model.lower() + ".pt")
                        self.model_path = selected_model.lower() + ".pt"
                    self.st.success("Model loaded successfully!")
                except Exception as e:
                    self.st.error(f"Could not load model: {e}")
                    LOGGER.error(traceback.format_exc())

            # Select classes
            class_names = list(self.model.names.values())
            selected_classes = self.st.multiselect("Select Classes", class_names, default=class_names[:3])
            self.selected_ind = [class_names.index(c) for c in selected_classes]

            # Frame placeholders
            col1, col2 = self.st.columns(2)
            self.org_frame = col1.empty()
            self.ann_frame = col2.empty()

            if self.source == "video":
                video_file = self.st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
                if video_file:
                    self.vid_file_name = "ultralytics_temp_video.mp4"
                    with open(self.vid_file_name, "wb") as f:
                        f.write(video_file.read())
                    st.success("Video uploaded successfully!")

            if "webcam_active" not in st.session_state:
                st.session_state.webcam_active = False

            if self.st.button("Start Webcam"):
                st.session_state.webcam_active = True

            if self.st.button("Stop Webcam"):
                st.session_state.webcam_active = False
                if self.cap:
                    self.cap.release()
                    self.cap = None

    def run_inference(self):
        if self.source == "webcam" and st.session_state.webcam_active:
            if self.cap is None:
                self.cap = cv2.VideoCapture(0)

            if not self.cap.isOpened():
                self.st.error("Could not open webcam. Please check your device.")
                st.session_state.webcam_active = False
                return

            ret, frame = self.cap.read()
            if not ret:
                self.st.error("Could not read frame from webcam.")
                return

            results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)
            annotated = results[0].plot()

            self.org_frame.image(frame, channels="BGR", caption="Original")
            self.ann_frame.image(annotated, channels="BGR", caption="Detected")

        elif self.source == "video" and self.vid_file_name:
            cap = cv2.VideoCapture(self.vid_file_name)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                annotated = results[0].plot()

                self.ann_frame.image(annotated, channels="BGR")
                cv2.waitKey(1)
            cap.release()


def main():
    app = InferenceApp()
    app.setup_ui()
    app.setup_sidebar()
    app.run_inference()

if __name__ == "__main__":
    main()
