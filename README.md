# Blood Cancer Detection Using CNN 
## 1. Project Overview
This project presents a **Supervised Deep Learning** approach for detecting **Blood Cancer (Acute Lymphoblastic Leukemia - ALL)** from microscopic cell images.
  
The model is built using **Convolutional Neural Networks (CNN)** and deployed through **Streamlit**, with the trained model converted to a **TensorFlow Lite (.tflite)** format for efficient performance and lightweight deployment.
## 2. Key Objectives
- Detect Blood Cancer (Normal vs Cancer classification)
- Apply CNN for image-based diagnosis
- Use TensorFlow Lite for optimized model inference
- Develop an interactive frontend using Streamlit
- Evaluate model performance with standard ML metrics

## 3. Dataset
- Dataset contains microscopic blood cell images labeled as **Normal** or **Cancer**.  
- Data has been preprocessed and split into training and testing sets (80%-20%).  
- Class imbalance handled using computed class weights.  

**Dataset Link:** https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class
## 4.Install Dependencies
 
```bash
pip install tensorflow streamlit opencv-python numpy matplotlib scikit-learn

```
## 4. Technologies Used

| Category | Tools / Frameworks |
|-----------|-------------------|
| Programming Language | Python |
| Deep Learning Framework | TensorFlow, Keras |
| Model Format | TensorFlow Lite (.tflite) |
| Frontend Framework | Streamlit |
| Libraries | NumPy, Pandas, OpenCV, Matplotlib, Scikit-learn |
| Environment | Google Colab, VS Code |                 




## 5.Implementation Steps
## 5.1 Model Training
- Preprocess and resize input images (150x150 pixels)
- Train CNN using labeled data
- Apply class weighting to handle imbalance
- Evaluate model using validation and test data
- Convert trained model to `.tflite` format for deployment
## 5.2  Deployment with Streamlit
- Load `.tflite` model using TensorFlow Lite Interpreter
- Design interactive interface using Streamlit
- Allow image upload and perform real-time prediction
- Display prediction result with visualization


## 6. Live Demo
You can try the live version of the app here: [Blood Cancer Detection App](https://bloodcancerdetection-tndxzu5tfy7whzgosxs5tn.streamlit.app/)

## Authors

- [@Tanjilaholy](https://github.com/Tanjilaholy)


