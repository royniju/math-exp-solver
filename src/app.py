import streamlit as st
from streamlit_drawable_canvas import st_canvas
from predict import predict
import numpy as np
import cv2


st.title("Handwritten Calculator")

canvas_result = st_canvas(
    fill_color= "white",
    stroke_width= 20,
    stroke_color= "black",
    background_color = "white",
    background_image = None,
    update_streamlit= True,
    height = 400,
    width = 1600,
    drawing_mode= "freedraw",
    initial_drawing = None,
    display_toolbar = True,
    point_display_radius= 3,
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        img= canvas_result.image_data.astype(np.uint8)
        gray= cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        exp = predict(gray)

        st.subheader(f"predicted exp: {exp}")

    else: 
        st.subheader("No image in canvas")
