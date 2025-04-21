import io
from typing import Any
import cv2
import asyncio
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import streamlit as st
import traceback

asyncio.set_event_loop(asyncio.new_event_loop())

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        tryRosenthal try:
            self.model = YOLO("yolov8.pt")
            self.selected_ind = [0, 1]
            self.conf = 0.25
            self.iou = 0.45
            LOGGER.info("YOLO model loaded successfully.")
        except Exception as e:
            LOGGER.error(f"Error loading YOLO model: {e}")
            st.error(f"Error loading YOLO model: {e}")
            self.model = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if img is not None and self.model:
            try:
                results = self.model(img, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                annotated_frame = results[0].plot()
                return annotated_frame
            except Exception as e:
                LOGGER.error(f"Error processing frame with YOLO model: {e}")
                st.error(f"Error processing frame with YOLO model: {e}")
        return img

class Inference:
    def __init__(self, **kwargs: Any):
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

        self.temp_dict = {"model": None, **kwargs}
        self.model_path = None
        if self.temp_dict["model"] is not None:
            self.model_path = self.temp_dict["model"]

        # Chỉ ghi log khi debug
        if __debug__:
            LOGGER.info(f"Ultralytics Solutions: ✅ {self.temp_dict}")

    def web_ui(self):
        menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""
        main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px; margin-top:-50px;
        font-family: 'Archivo', sans-serif; margin-bottom:20px;">Ultralytics YOLO Streamlit Application</h1></div>"""
        sub_title_cfg = """<div><h4 style="color:#042AFF; text-align:center; font-family: 'Archivo', sans-serif; 
        margin-top:-15px; margin-bottom:50px;">Experience real-time object detection on your webcam with the power 
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
            "Video", ("webcam", "video")
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
                    LOGGER.info(f"Tệp: {vid_file.name}, Kích thước: {vid_file.size}")
                    if vid_file.size == 0:
                        self.st.error("Tệp video rỗng. Vui lòng tải lên tệp hợp lệ.")
                        return
                    with io.BytesIO(vid_file.read()) as g:
                        LOGGER.info("Đã đọc tệp vào BytesIO")
                        with open("ultralytics.mp4", "wb") as out:
                            out.write(g.read())
                            LOGGER.info("Đã ghi tệp ra ultralytics.mp4")
                    self.vid_file_name = "ultralytics.mp4"
                    self.st.success("Tệp video đã được tải lên thành công!")
                except Exception as e:
                    self.st.error(f"Lỗi khi xử lý tệp video: {e}")
                    LOGGER.error(f"Lỗi khi xử lý tệp video: {traceback.format_exc()}")

    def configure(self):
        available_models = [x.replace("yolo", "YOLO") for x in GITHUB_ASSETS_STEMS if x.startswith("yolo11")]
        if self.model_path:
            available_models.insert(0, self.model_path.split(".pt")[0])
        selected_model = self.st.sidebar.selectbox("Model", available_models)

        with self.st.spinner("Model is downloading..."):
            self.model = YOLO(f"{selected_model.lower()}.pt")
            class_names = list(self.model.names.values())
        self.st.success("Model loaded successfully!")

        selected_classes = self.st.sidebar.multiselect("Classes", class_names, default=class_names[:3])
        self.selected_ind = [class_names.index(option) for option in selected_classes]

        if not isinstance(self.selected_ind, list):
            self.selected_ind = list(self.selected_ind)

    def inference(self):
        self.web_ui()
        self.sidebar()
        self.source_upload()
        self.configure()

        if self.st.sidebar.button("Start"):
            try:
                if self.source == "webcam":
                    webrtc_streamer(
                        key="example",
                        video_processor_factory=VideoProcessor,
                        media_stream_constraints={"video": True},
                        async_processing=True,
                    )
                elif self.source == "video" and self.vid_file_name:
                    cap = cv2.VideoCapture(self.vid_file_name)
                    if not cap.isOpened():
                        self.st.error("Không thể mở tệp video.")
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
                self.st.error(f"Lỗi khi chạy xử lý video/webcam: {e}")
                LOGGER.error(f"Lỗi khi chạy xử lý video/webcam: {traceback.format_exc()}")

def main():
    import sys
    args = len(sys.argv)
    model = sys.argv[1] if args > 1 else None

    if 'inference' not in st.session_state:
        st.session_state.inference = Inference(model=model)

    st.session_state.inference.inference()

if __name__ == "__main__":
    main()