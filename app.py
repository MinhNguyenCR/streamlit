import io
import cv2
import numpy as np
import streamlit as st
import traceback
import logging
from typing import Any

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class Inference:
    def __init__(self, **kwargs):
        check_requirements("streamlit>=1.29.0")
        self.st = st
        self.source = None
        self.enable_trk = False
        self.conf = 0.25
        self.iou = 0.45
        self.org_frame = None
        self.ann_frame = None
        self.vid_file_name = None
        self.selected_ind = []
        self.model = None
        self.cap = None

        # Default to yolov8n.pt
        self.temp_dict = {"model": kwargs.get("model", "yolov8n.pt"), **kwargs}
        self.model_path = self.temp_dict["model"]

        if __debug__:
            LOGGER.debug(f"Ultralytics Solutions: ✅ {self.temp_dict}")

        # Load YOLO model
        try:
            LOGGER.info("Loading YOLO model...")
            self.model = YOLO(self.model_path)
            LOGGER.info("YOLO model loaded successfully.")
        except Exception as e:
            LOGGER.error(f"Error loading YOLO model: {e}")
            self.st.error(f"Error loading YOLO model: {e}")

    def web_ui(self):
        menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""
        main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px; margin-top:-50px;
        font-family: 'Archivo', sans-serif; margin-bottom:20px;">Ultralytics YOLO Streamlit Application</h1></div>"""
        sub_title_cfg = """<div><h4 style="color:#042AFF; text-align:center; font-family: 'Archivo', sans-serif;
        margin-top:-15px; margin-bottom:50px;">Experience real-time object detection on your webcam or video with the power
        of Ultralytics YOLO! 🚀</h4></div>"""
        self.st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide")
        self.st.markdown(menu_style_cfg, unsafe_allow_html=True)
        self.st.markdown(main_title_cfg, unsafe_allow_html=True)
        self.st.markdown(sub_title_cfg, unsafe_allow_html=True)

    def sidebar(self):
        with self.st.sidebar:
            logo = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg"
            self.st.image(logo, width=250)

            self.st.sidebar.title("User Configuration")
            self.source = self.st.sidebar.selectbox(
                "Video Source", ("webcam", "video")
            )
            self.enable_trk = self.st.sidebar.radio("Enable Tracking", ("Yes", "No"))
            self.conf = float(self.st.sidebar.slider("Confidence Threshold", 0.0, 1.0, self.conf, 0.01))
            self.iou = float(self.st.sidebar.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.01))

            col1, col2 = self.st.columns(2)
            self.org_frame = col1.empty()
            self.ann_frame = col2.empty()

    def source_upload(self):
        self.vid_file_name = ""
        if self.source == "video":
            vid_file = self.st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
            if vid_file is not None:
                try:
                    LOGGER.info(f"File: {vid_file.name}, Size: {vid_file.size}")
                    if vid_file.size == 0:
                        self.st.error("Uploaded video file is empty. Please upload a valid file.")
                        return
                    with io.BytesIO(vid_file.read()) as g:
                        with open("ultralytics.mp4", "wb") as out:
                            out.write(g.read())
                    self.vid_file_name = "ultralytics.mp4"
                    self.st.success("Video file uploaded successfully!")
                except Exception as e:
                    self.st.error(f"Error processing video file: {e}")
                    LOGGER.error(f"Error processing video file: {traceback.format_exc()}")

    def configure(self):
        available_models = [x.replace("yolo", "YOLO") for x in GITHUB_ASSETS_STEMS if x.startswith("yolo11")]
        if self.model_path:
            available_models.insert(0, self.model_path.split(".pt")[0])
        selected_model = self.st.sidebar.selectbox("Model", available_models, index=0)

        # Load a different model if selected
        if selected_model != self.model_path.split(".pt")[0]:
            try:
                with self.st.spinner("Downloading and loading model..."):
                    self.model = YOLO(f"{selected_model.lower()}.pt")
                    self.model_path = f"{selected_model.lower()}.pt"
                    class_names = list(self.model.names.values())
                self.st.success("Model loaded successfully!")
            except Exception as e:
                self.st.error(f"Error loading model: {e}")
                LOGGER.error(f"Error loading model: {traceback.format_exc()}")
                return
        else:
            class_names = list(self.model.names.values())

        selected_classes = self.st.sidebar.multiselect("Classes", class_names, default=class_names[:3])
        self.selected_ind = [class_names.index(option) for option in selected_classes]

        if not isinstance(self.selected_ind, list):
            self.selected_ind = list(self.selected_ind)

    def inference(self):
        self.web_ui()
        self.sidebar()
        self.source_upload()
        self.configure()

        # Manage webcam state
        if 'webcam_active' not in st.session_state:
            st.session_state.webcam_active = False

        if self.st.sidebar.button("Start Webcam"):
            st.session_state.webcam_active = True
            LOGGER.info("Start Webcam button pressed.")

        if self.st.sidebar.button("Stop Webcam"):
            st.session_state.webcam_active = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            LOGGER.info("Stop Webcam button pressed.")

        if self.source == "webcam" and st.session_state.webcam_active:
            try:
                if self.cap is None:
                    self.cap = cv2.VideoCapture(0)  # Open default webcam
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                # Check if webcam can be opened
                if not self.cap.isOpened():
                    self.st.error("Could not open webcam. Please check your device.")
                    st.session_state.webcam_active = False
                    return
                LOGGER.info("Webcam feed started. Press Stop Webcam to stop.")

                while st.session_state.webcam_active:
                    ret, frame = self.cap.read()
                    if not ret:
                        self.st.error("Could not read frame from webcam.")
                        break

                    # Perform inference
                    if self.model:
                        results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                        annotated_frame = results[0].plot()
                    else:
                        annotated_frame = frame

                    # Display frames
                    self.org_frame.image(frame, channels="BGR", caption="Original Frame")
                    self.ann_frame.image(annotated_frame, channels="BGR", caption="Annotated Frame")

                    # Add a small delay to reduce processing load
                    cv2.waitKey(1)

            except Exception as e:
                self.st.error(f"Error accessing or processing webcam feed: {e}")
                LOGGER.error(f"Error accessing or processing webcam feed: {traceback.format_exc()}")
                st.session_state.webcam_active = False
            finally:
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None

        elif self.source == "video" and self.vid_file_name:
            try:
                cap = cv2.VideoCapture(self.vid_file_name)
                if not cap.isOpened():
                    self.st.error("Could not open the uploaded video file.")
                    return
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if self.model:
                        results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                        annotated_frame = results[0].plot()
                        self.ann_frame.image(annotated_frame, channels="BGR")
                cap.release()
            except Exception as e:
                self.st.error(f"Error processing video: {e}")
                LOGGER.error(f"Error processing video: {traceback.format_exc()}")

def main():
    import sys
    args = len(sys.argv)
    model = sys.argv[1] if args > 1 else "yolov8n.pt"

    if 'inference' not in st.session_state:
        st.session_state.inference = Inference(model=model)

    st.session_state.inference.inference()

if __name__ == "__main__":
    main()
