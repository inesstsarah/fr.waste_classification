import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO 
import supervision as sv

model_path = "MODEL_PATH"
model_path = "yolo11n-seg.pt"
# Define YOLO model being used
model = YOLO(model_path)

st.title("Trash Detection and Classification App")

img_file_buffer = st.camera_input("Take a picture")
if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    results = model(cv2_img)
    detections = sv.Detections.from_ultralytics(results[0])

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(
        scene=cv2_img.copy(),
        detections=detections
    )
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame.copy(),
        detections=detections
    )


    # Check the type of cv2_img:
    st.header("Trash Classification", divider="gray")
    st.image(annotated_frame, caption="Segmentation and Classification of Trash")
