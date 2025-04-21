import io
from typing import Any
import cv2
import asyncio
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import streamlit as st
import traceback
import uuid
import logging

asyncio.set_event_loop(asyncio.new_event_loop())

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG)

# Cấu hình WebRTC với STUN và TURN
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            },
            {
                "urls": ["turn:openrelay.metered.ca:443"],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            }
        ],
        "iceTransportPolicy": "all",
        "rtcpMuxPolicy": "require",
        "bundlePolicy": "balanced"
    }
)

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        try:
            LOGGER.info("Đang tải mô hình YOLO trong VideoProcessor...")
            self.model = YOLO("yolov8n.pt")  # Mô hình nhẹ
            self.selected_ind = [0, 1]
            self.conf = 0.25
            self.iou = 0.45
            LOGGER.info("YOLO model loaded successfully in VideoProcessor.")
        except Exception as e:
            LOGGER.error(f"Error loading YOLO model in VideoProcessor: {e}")
            st.error(f"Error loading YOLO model in VideoProcessor: {e}")

    def transform(self, frame):
        LOGGER.debug("Nhận khung hình mới trong transform")
        try:
            img = frame.to_ndarray(format="bgr24")
            if img is None:
                LOGGER.warning("Khung hình rỗng nhận được")
                return img

            if self.model:
                LOGGER.debug("Xử lý khung hình với mô hình YOLO")
                results = self.model(img, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                annotated_frame = results[0].plot()
                LOGGER.debug("Khung hình đã được xử lý thành công")
                return annotated_frame
            else:
                LOGGER.warning("Không có mô hình YOLO để xử lý khung hình")
                return img
        except Exception as e:
            LOGGER.error(f"Error processing frame with YOLO model: {e}")
            st.error(f"Error processing frame: {e}")
            return img  # Trả về khung hình gốc nếu lỗi

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

        # Mặc định yolov8n.pt
        self.temp_dict = {"model": kwargs.get("model", "yolov8n.pt"), **kwargs}
        self.model_path = self.temp_dict["model"]

        # Chỉ log khi debug
        if __debug__:
            LOGGER.debug(f"Ultralytics Solutions: ✅ {self.temp_dict}")

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
        selected_model = self.st.sidebar.selectbox("Model", available_models, index=0)

        try:
            with self.st.spinner("Model is downloading..."):
                self.model = YOLO(f"{selected_model.lower()}.pt")
                class_names = list(self.model.names.values())
            self.st.success("Model loaded successfully!")
        except Exception as e:
            self.st.error(f"Lỗi khi tải mô hình: {e}")
            LOGGER.error(f"Lỗi khi tải mô hình: {traceback.format_exc()}")
            return

        selected_classes = self.st.sidebar.multiselect("Classes", class_names, default=class_names[:3])
        self.selected_ind = [class_names.index(option) for option in selected_classes]

        if not isinstance(self.selected_ind, list):
            self.selected_ind = list(self.selected_ind)

    def inference(self):
        self.web_ui()
        self.sidebar()
        self.source_upload()
        self.configure()

        # Quản lý trạng thái webcam
        if 'webcam_active' not in st.session_state:
            st.session_state.webcam_active = False
        if 'webcam_key' not in st.session_state:
            st.session_state.webcam_key = str(uuid.uuid4())

        if self.st.sidebar.button("Start"):
            st.session_state.webcam_active = True
            st.session_state.webcam_key = str(uuid.uuid4())  # Tạo key mới khi nhấn Start
            LOGGER.info("Nút Start được nhấn, khởi động webcam")

        if self.st.sidebar.button("Stop"):
            st.session_state.webcam_active = False
            LOGGER.info("Nút Stop được nhấn, dừng webcam")

        if self.source == "webcam" and st.session_state.webcam_active:
            try:
                self.st.info("Đang khởi động webcam... Vui lòng đảm bảo webcam hoạt động và mạng ổn định.")
                LOGGER.debug(f"Khởi tạo webrtc_streamer với key: {st.session_state.webcam_key}")
                webrtc_streamer(
                    key=st.session_state.webcam_key,
                    video_processor_factory=VideoProcessor,
                    media_stream_constraints={
                        "video": {"width": {"ideal": 320}, "height": {"ideal": 240}},
                        "audio": False
                    },
                    async_processing=True,
                    rtc_configuration=RTC_CONFIGURATION,
                    timeout=30  # Tăng thời gian chờ lên 30 giây
                )
                LOGGER.debug("webrtc_streamer đã khởi tạo thành công")
            except Exception as e:
                self.st.error(f"Lỗi khi chạy webcam: {e}")
                LOGGER.error(f"Lỗi khi chạy webcam: {traceback.format_exc()}")
                st.session_state.webcam_active = False
        elif self.source == "video" and self.vid_file_name:
            try:
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
                self.st.error(f"Lỗi khi xử lý video: {e}")
                LOGGER.error(f"Lỗi khi xử lý video: {traceback.format_exc()}")

def main():
    import sys
    args = len(sys.argv)
    model = sys.argv[1] if args > 1 else "yolov8n.pt"  # Mặc định yolov8n.pt

    if 'inference' not in st.session_state:
        st.session_state.inference = Inference(model=model)

    st.session_state.inference.inference()

if __name__ == "__main__":
    main()
