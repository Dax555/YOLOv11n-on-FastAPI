import streamlit as st
import requests
from PIL import Image
import io
import base64
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.title("YOLOv11 Object Detection")

# Image upload section
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Log and display file details
    logger.info(f"Uploaded file: {uploaded_file.name}, type: {uploaded_file.type}, size: {uploaded_file.size} bytes")
    st.write(f"**File Details**:")
    st.write(f"- Name: {uploaded_file.name}")
    st.write(f"- Type: {uploaded_file.type}")
    st.write(f"- Size: {uploaded_file.size} bytes")

    # Validate file type
    if uploaded_file.type not in ["image/jpeg", "image/png"]:
        st.error("Invalid file type. Please upload a JPEG or PNG image.")
        logger.error(f"Invalid file type: {uploaded_file.type}")
    else:
        try:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Send image to FastAPI backend
            with st.spinner("Processing..."):
                # Reset file pointer to start
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
                logger.info("Sending request to FastAPI")
                response = requests.post("http://localhost:8000/predict", files=files)

            if response.status_code == 200:
                data = response.json()
                if "error" not in data:
                    # Display detection results
                    st.subheader("Detection Results")
                    for det in data["detections"]:
                        st.write(f"**Class**: {det['class']}")
                        st.write(f"**Confidence**: {(det['confidence'] * 100):.2f}%")
                        st.write(
                            f"**Bounding Box**: [{det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, {det['bbox'][2]:.0f}, {det['bbox'][3]:.0f}]")
                        st.markdown("---")

                    # Display image with bounding boxes
                    st.subheader("Image with Detections")
                    img_data = base64.b64decode(data["image"].split(",")[1])
                    img = Image.open(io.BytesIO(img_data))

                    # Draw bounding boxes
                    from PIL import ImageDraw, ImageFont

                    draw = ImageDraw.Draw(img)
                    try:
                        font = ImageFont.truetype("arial.ttf", 16)
                    except:
                        font = ImageFont.load_default()

                    for det in data["detections"]:
                        bbox = det["bbox"]
                        draw.rectangle(bbox, outline="red", width=2)
                        draw.text((bbox[0], bbox[1] - 20), f"{det['class']} ({(det['confidence'] * 100):.2f}%)",
                                  fill="red", font=font)

                    st.image(img, caption="Image with Detections", use_column_width=True)
                    logger.info("Successfully processed and displayed results")
                else:
                    st.error(f"Error from server: {data['error']}")
                    logger.error(f"Server returned error: {data['error']}")
            else:
                st.error(f"Failed to process image: HTTP {response.status_code} - {response.text}")
                logger.error(f"HTTP error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")
            logger.error(f"Client error: {str(e)}")