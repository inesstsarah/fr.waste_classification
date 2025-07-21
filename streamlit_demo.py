import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO 
import supervision as sv

model_path = "MODEL_PATH"
model_path = "yolo11n-seg.pt"
model_path = "models/garbage_obj_det.pt"
# Define YOLO model being used
model = YOLO(model_path)

st.title("Sampah Kita AI Waste Classification")

img_file_buffer = st.camera_input("Take a picture")
if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    results = model(cv2_img)

    detections = sv.Detections.from_ultralytics(results[0])

    bbox_annotator = sv.BoxAnnotator()

    # Convert to RGB
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    gbr_img = cv2.COLOR_BGR2RGB
    annotated_frame = bbox_annotator.annotate(
        scene=rgb_img.copy(),
        detections=detections
    )

    label_mapping = dict()
    label_mapping[0] = "ORGANIK (Sampah Mudah Terurai)"
    label_mapping[1] = "KERTAS (Sampah Daur Ulang - Kertas)"
    label_mapping[2] = "LAINNYA (Sampah Residu)"
    label_mapping[3] = "LAINNYA (Sampah Residu)"
    label_mapping[4] = "KERTAS (Sampah Daur Ulang - Kertas)"
    label_mapping[5] = "PLASTIK (Sampah Daur Ulang - Plastik)"
    
    labels = []
    for classes in results[0].boxes.cls:
        class_int = int(classes.cpu().numpy())
        label = label_mapping[class_int]
        labels.append(label)

    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame.copy(),
        detections=detections,
        labels = labels
    )


    # Check the type of cv2_img:
    st.header("Trash Detection and Classification", divider="gray")
    caption = ""
    unique_counts_labels = np.unique_counts(results[0].boxes.cls.cpu().numpy())[0]
    unique_counts_amts = np.unique_counts(results[0].boxes.cls.cpu().numpy())[1]

    class_mapping = dict()
    class_mapping[0] = "organik"
    class_mapping[1] = "kertas"
    class_mapping[2] = "lainnya"
    class_mapping[3] = "lainnya"
    class_mapping[4] = "kertas"
    class_mapping[5] = "plastik"
    caption = ""
    for i in range (0, len(unique_counts_labels)):
        print(f"{unique_counts_amts[i]} {class_mapping[unique_counts_labels[i]]}")
        temp = f"{unique_counts_amts[i]} {class_mapping[unique_counts_labels[i]]}"
        caption += temp

    st.image(annotated_frame, caption=f"{caption}")
