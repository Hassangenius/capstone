
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2



st.set_page_config(
    page_title="Pavement Crack Detector",
    page_icon="🛣️",
    layout="wide"
)

st.title("🛣️ Pavement Crack Detection System")
st.write(
    "Upload a pavement or road image to detect cracks "
    
)




@st.cache_resource
def load_model():
    return YOLO(
        r"C:\Users\user\Downloads\runs\detect\train-26\weights\best.pt"
    )


try:
    model = load_model()

except Exception as e:
    st.error(f"Could not load YOLO model: {e}")
    st.stop()




st.subheader("📸 Upload Pavement Image")

file = st.file_uploader(
    "Upload a road or pavement image here",
    type=["jpg", "jpeg", "png", "webp"]
)


if file is not None:

    image = Image.open(file).convert("RGB")

    st.subheader("🖼️ Uploaded Pavement Image")

    st.image(
        image,
        caption="Original Pavement Image",
        width=500
    )


    

    if st.button("🔍 Detect Cracks", type="primary"):

        with st.spinner("YOLO26 is detecting pavement cracks..."):

            image_array = np.array(image)

            results = model.predict(
                source=image_array,
                conf=0.25,
                verbose=False
            )

            result = results[0]

            annotated_image = image_array.copy()

            boxes = result.boxes


            

            if boxes is not None and len(boxes) > 0:

                for box in boxes:

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    crack_name = model.names[class_id]


                    
                    cv2.rectangle(
                        annotated_image,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        2
                    )


                    
                    label = f"{crack_name} {confidence:.2f}"

                    cv2.putText(
                        annotated_image,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2
                    )


                

                st.success("Pavement crack detection completed!")

                st.subheader("🔎 Crack Detection Results")

                st.image(
                    annotated_image,
                    caption="Detected Pavement Cracks",
                    width=700
                )


                

                st.subheader("📋 Detected Crack Details")

                st.write(f"**Total detections:** {len(boxes)}")

                for i, box in enumerate(boxes):

                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    crack_name = model.names[class_id]

                    st.write(f"**Detection {i + 1}**")

                    st.write(f"🛣️ **Crack Type:** {crack_name}")

                    st.write(
                        f"🎯 **Confidence:** {confidence * 100:.2f}%"
                    )

                    st.divider()


            else:

                st.warning(
                    "No pavement cracks detected. "
                    "Please try another image."
                )

                st.info(
                    "The absence of detections does not guarantee "
                    "that the pavement is crack-free."
                )
