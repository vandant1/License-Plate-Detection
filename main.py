import os
import sys
import platform
import pathlib
from pathlib import Path
import gradio as gr
import cv2
import torch
import numpy as np
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.plots import Annotator

# Redirect PosixPath to WindowsPath if running on Windows
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# Set project path
project_path = Path("C:/Mid2k24_Projects/License Detection Model").resolve()
os.chdir(project_path)
sys.path.append(str(project_path / "yolov5"))

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLOv5 model
weights_path = project_path / "yolov5/weights/best.pt"
try:
    yolo_model = DetectMultiBackend(str(weights_path), device=device)
    yolo_model.model.float()  # Ensure model is in eval mode
except Exception as e:
    raise RuntimeError(f"Failed to load YOLOv5 model: {e}")

def detect_license_plate(video_source):
    try:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            return "Error: Could not open video source.", None
        
        success_message = "Detection Successful!"
        frames_with_detections = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess frame
            img = cv2.resize(frame, (640, 640))
            img = torch.from_numpy(img).permute(2, 0, 1).float().to(device) / 255.0
            img = img.unsqueeze(0)

            # YOLOv5 detection
            pred = yolo_model(img)
            pred = non_max_suppression(pred)

            annotator = Annotator(frame, line_width=2)
            detection_made = False

            if pred:
                for det in pred:
                    if det is not None and len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                        for *xyxy, conf, cls in det:
                            # Filter for license plate detection (assuming class 0 is 'license plate')
                            if cls == 0:
                                detection_made = True
                                color = (0, 255, 0) if conf > 0.75 else (0, 0, 255)
                                label = f"License Plate {conf:.2f}"
                                annotator.box_label(xyxy, label, color=color)

            # Annotate the frame
            annotated_frame = annotator.result()
            frames_with_detections.append(annotated_frame)
            
            if detection_made:
                success_message = "Detection Successful!"

        cap.release()
        return success_message, frames_with_detections
    except Exception as e:
        return f"Error during detection: {e}", None

# Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("## 🚗 YOLOv5 License Plate Detection Dashboard")
    video_input = gr.Video(label="Upload a Video or Provide Camera Feed URL")
    success_message = gr.Textbox(label="Detection Status", interactive=False)
    output_video = gr.Video(label="Annotated Video")
    
    detect_button = gr.Button("Run Detection")
    
    detect_button.click(
        detect_license_plate,
        inputs=video_input,
        outputs=[success_message, output_video]
    )

interface.launch()
