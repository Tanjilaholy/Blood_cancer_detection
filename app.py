import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# --- LOAD MODEL ---
# Use raw string for Windows path
model_path = r"C:\Users\Holy\Downloads\Blood_Cancer\weights (2).keras"
model = load_model(model_path)

# PAGE CONFIG
st.set_page_config(
    page_title="Blood Cancer Detection",
    page_icon="🩸",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* Background */
    .stApp {
        background-color: #f0f0f0;
    }
    /* Title */
    .title {
        color: #8B0000;  /* Deep red */
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    /* Subtitle / Instructions below title */
    .subtitle {
        color: #555555;   /* Halko gray */
        font-size: 18px;
        text-align: center;
        margin-bottom: 20px;
    }
    /* Prediction Text */
    .result-normal {
        font-size: 28px;
        font-weight: bold;
        color: #006400; /* Deep green */
        text-align: center;
        margin-top: 20px;
    }
    .result-cancer {
        font-size: 28px;
        font-weight: bold;
        color: #8B0000; /* Deep red */
        text-align: center;
        margin-top: 20px;
    }
    /* File uploader label */
    div[data-baseweb="file-uploader"] > div > label {
        color: #555555;   /* Halko gray */
        font-size: 16px;
        font-weight: normal;
    }
    /* File uploader button */
    .stButton>button {
        background-color: #8B0000;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# --- TITLE ---
st.markdown("<div class='title'>🩸 Blood Cancer Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a blood cell image and check prediction instantly.</div>", unsafe_allow_html=True)

# --- SIDEBAR INFO ---
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Click the **Browse files** button below.  
2. Select your blood cell image (jpg, png).  
3. Wait for prediction.  
4. See result below.
""")

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # --- PROCESS IMAGE FOR MODEL ---
    img_array = np.array(img.resize((150,150))) / 255.0  # Adjust size to match model input
    img_array = np.expand_dims(img_array, axis=0)

    # --- MODEL PREDICTION ---
    pred = model.predict(img_array)
    
    # Binary classification (sigmoid output assumed)
    if pred[0][0] > 0.5:
        result = "Cancer Positive"
    else:
        result = "Normal"
    
    # --- DISPLAY RESULT ---
    if result == "Cancer Positive":
        st.markdown(f"<div class='result-cancer'>Prediction: {result}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result-normal'>Prediction: {result}</div>", unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("<hr><p style='text-align:center;color:gray;'>Designed by Holy | Blood Cancer Detection Project</p>", unsafe_allow_html=True)
