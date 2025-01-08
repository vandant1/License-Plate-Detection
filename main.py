import os
import sys
import platform
import pathlib
from pathlib import Path
import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.models import resnet18
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.plots import Annotator

# Set Streamlit page configuration
st.set_page_config(
    page_title="YOLOv5 License Plate Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Redirect PosixPath to WindowsPath if running on Windows
if platform.system() == "Windows":
    original_PosixPath = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

try:
    # Ensure the working directory is correct
    project_path = Path("C:/Mid2k24_Projects/License Detection Model").resolve()
    os.chdir(project_path)
    sys.path.append(str(project_path / "yolov5"))

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.sidebar.info(f"Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")

    # Load YOLOv5 model
    weights_path = project_path / "yolov5/weights/best.pt"
    try:
        yolo_model = DetectMultiBackend(str(weights_path), device=device)
        yolo_model.model.float()  # Ensure model is in eval mode
        st.sidebar.success("YOLOv5 Model Loaded Successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to load YOLOv5 model: {e}")
        st.stop()

    # Load ResNet model for visualization
    try:
        resnet_model = resnet18(pretrained=True)
        resnet_model.eval()
        st.sidebar.success("ResNet18 Model Loaded Successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to load ResNet18 model: {e}")
        st.stop()

    # Streamlit UI Layout
    st.title("🚗 License Plate Recognition & Visualization Dashboard")

    # Sidebar for navigation
    module = st.sidebar.radio(
        "Choose Visualization Module:",
        (
            "Input Pipeline Visualization",
            "Convolutional Layers & Filters",
            "Model Metrics",
            "Node & Connection",
        ),
    )

    # Input Pipeline Visualization Module
    if module == "Input Pipeline Visualization":
        st.header("📹 Input Pipeline Visualization")
        st.write("Real-time YOLOv5 detection with bounding boxes and OCR confidence.")

        input_source = st.radio("Input Source", ("Camera Feed", "Upload Video"))
        video_file = (
            st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
            if input_source == "Upload Video"
            else None
        )

        run_detection = st.button("Run Detection")
        stop_detection = st.button("Stop Detection")

        if run_detection:
            try:
                # Open video source
                cap = (
                    cv2.VideoCapture(0)
                    if input_source == "Camera Feed"
                    else cv2.VideoCapture(video_file.name)
                )

                if not cap.isOpened():
                    st.error("Error: Could not open video source.")
                    st.stop()

                stframe = st.empty()
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Video stream ended.")
                        break

                    # YOLOv5 detection
                    img = cv2.resize(frame, (640, 640))
                    img = torch.from_numpy(img).permute(2, 0, 1).float().to(device) / 255.0
                    img = img.unsqueeze(0)

                    pred = yolo_model(img)
                    pred = non_max_suppression(pred)

                    annotator = Annotator(frame, line_width=2)
                    if pred:
                        for det in pred:
                            if det is not None and len(det):
                                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                                for *xyxy, conf, cls in det:
                                    color = (0, 255, 0) if conf > 0.75 else (0, 0, 255)
                                    label = f"{yolo_model.names[int(cls)]} {conf:.2f}"
                                    annotator.box_label(xyxy, label, color=color)

                    # Display video
                    stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

                    # Stop detection when button pressed
                    if stop_detection:
                        st.info("Detection stopped by user.")
                        break

            except Exception as e:
                st.error(f"Error during detection: {e}")
            finally:
                cap.release()

except Exception as e:
    st.error(f"An unexpected error occurred: {e}")

finally:
    # Restore PosixPath to its original state if overridden
    if platform.system() == "Windows":
        pathlib.PosixPath = original_PosixPath
