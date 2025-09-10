import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# ===============================
# Load TFLite model
# ===============================
tflite_model_path = "weights (2) (1).tflite"  # Replace with your uploaded tflite file in same folder
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Blood Cancer Detection",
    page_icon="🩸",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #f0f0f0; }
    .title { color: #8B0000; font-size:48px; font-weight:bold; text-align:center; margin-bottom:10px; }
    .subtitle { color:#555555; font-size:18px; text-align:center; margin-bottom:20px; }
    .result-normal { font-size:28px; font-weight:bold; color:#006400; text-align:center; margin-top:20px; }
    .result-cancer { font-size:28px; font-weight:bold; color:#8B0000; text-align:center; margin-top:20px; }
    div[data-baseweb="file-uploader"] > div > label { color: #555555; font-size:16px; font-weight:normal; }
    .stButton>button { background-color:#8B0000; color:white; font-weight:bold; padding:10px 20px; border-radius:8px; border:none; }
    </style>
""", unsafe_allow_html=True)

# --- TITLE ---
st.markdown("<div class='title'>🩸 Blood Cancer Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a blood cell image and check prediction instantly.</div>", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Click the **Browse files** button below.  
2. Select your blood cell image (jpg, png).  
3. Wait for prediction.  
4. See result below.
""")

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess
    img_array = np.array(img.resize((150,150)))/255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    # Predict using TFLite
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    
    result = "Cancer Positive" if pred[0][0] > 0.5 else "Normal"
    
    # Display result
    if result == "Cancer Positive":
        st.markdown(f"<div class='result-cancer'>Prediction: {result}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result-normal'>Prediction: {result}</div>", unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("<hr><p style='text-align:center;color:gray;'>Designed by Holy | Blood Cancer Detection Project</p>", unsafe_allow_html=True)
