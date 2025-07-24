import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
import matplotlib.cm as cm

# --- Page Configuration ---
st.set_page_config(
    page_title="ASL Interpreter Comparison",
    layout="wide"
)

# --- Class Names ---
CLASS_NAMES = sorted(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                      'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 
                      'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 
                      'v', 'w', 'x', 'y', 'z'])

# --- Model Loading ---
@st.cache_resource
def load_models():
    """Loads both Keras models and creates their visualization counterparts."""
    models = {}
    
    # --- Load CNN from Scratch ---
    try:
        cnn_path = os.path.join('models', 'best_cnn_from_scratch.keras')
        full_cnn_model = tf.keras.models.load_model(cnn_path)
        
        # Create a visualization model for Grad-CAM (stripping the Rescaling layer)
        vis_input_cnn = tf.keras.Input(shape=(128, 128, 3))
        x = vis_input_cnn
        for layer in full_cnn_model.layers[1:]:
            x = layer(x)
        vis_cnn_model = tf.keras.Model(inputs=vis_input_cnn, outputs=x)
        
        models['cnn_from_scratch'] = {'full': full_cnn_model, 'vis': vis_cnn_model}
        st.success("CNN from Scratch model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading CNN from Scratch model: {e}")

    # --- Load Transfer Learning Model ---
    try:
        transfer_path = os.path.join('models', 'best_transfer_learning_mobilenet.keras')
        full_transfer_model = tf.keras.models.load_model(transfer_path)
        
        # Create a visualization model for Grad-CAM (rebuilding from the base)
        base_layer = full_transfer_model.get_layer('mobilenetv2_1.00_128')
        vis_input_transfer = base_layer.input
        x = base_layer.output
        x = full_transfer_model.get_layer('global_average_pooling2d')(x)
        x = full_transfer_model.get_layer('dropout_1')(x)
        vis_output_transfer = full_transfer_model.get_layer('dense_2')(x)
        vis_transfer_model = tf.keras.Model(vis_input_transfer, vis_output_transfer)
        
        models['transfer_learning'] = {'full': full_transfer_model, 'vis': vis_transfer_model}
        st.success("Transfer Learning model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading Transfer Learning model: {e}")
        
    return models

# --- Grad-CAM Generation Function ---
@st.cache_data
def generate_gradcam_overlay(model_dict, model_name, img_array):
    """Generates a Grad-CAM overlay for a given model and image."""
    full_model = model_dict['full']
    vis_model = model_dict['vis']
    
    # Get prediction
    prediction = full_model.predict(np.expand_dims(img_array, axis=0))
    pred_index = np.argmax(prediction[0])
    
    # Preprocess for the specific visualization model
    if model_name == 'cnn_from_scratch':
        # Manually rescale for the vis model that skips the Rescaling layer
        processed_img = img_array / 255.0
    else: # Transfer Learning
        # Use the specific preprocessing function for MobileNetV2
        processed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array.copy())

    # Generate Grad-CAM
    score = CategoricalScore([pred_index])
    gradcam = Gradcam(vis_model, model_modifier=ReplaceToLinear(), clone=True)
    cam = gradcam(score, np.expand_dims(processed_img, axis=0), penultimate_layer=-1)
    
    # Create and overlay heatmap
    heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
    overlay = np.uint8(img_array * 0.5 + heatmap * 0.5)
    
    return overlay, prediction

# --- Main App UI ---
st.title("ASL Interpreter: Model Comparison")
st.write("Upload an image to see a side-by-side comparison of a custom CNN and a Transfer Learning model (MobileNetV2).")

models = load_models()
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and models:
    image = Image.open(uploaded_file)
    
    # --- Display Uploaded Image (smaller) ---
    st.subheader("Original Uploaded Image")
    st.image(image, width=256)
    
    # Preprocess the image once
    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized)
    if img_array.ndim == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    if img_array.shape[2] == 4:
        img_array = img_array[..., :3]
    
    st.divider()
    
    col1, col2 = st.columns(2)

    # --- Column 1: CNN from Scratch ---
    with col1:
        st.header("CNN from Scratch")
        if 'cnn_from_scratch' in models:
            with st.spinner("Analyzing with CNN from Scratch..."):
                overlay, prediction = generate_gradcam_overlay(models['cnn_from_scratch'], 'cnn_from_scratch', img_array)
                
                # --- CHANGE: Set a fixed width for the Grad-CAM image ---
                st.image(overlay, caption="Grad-CAM: Model Attention", width=300)
                
                pred_index = np.argmax(prediction[0])
                confidence = prediction[0][pred_index]
                
                st.success(f"**Prediction:** {CLASS_NAMES[pred_index].upper()}")
                st.info(f"**Confidence:** {confidence:.2%}")
        else:
            st.warning("CNN from Scratch model not available.")

    # --- Column 2: Transfer Learning (MobileNetV2) ---
    with col2:
        st.header("Transfer Learning (MobileNetV2)")
        if 'transfer_learning' in models:
            with st.spinner("Analyzing with Transfer Learning model..."):
                overlay, prediction = generate_gradcam_overlay(models['transfer_learning'], 'transfer_learning', img_array)
                
                # --- CHANGE: Set a fixed width for the Grad-CAM image ---
                st.image(overlay, caption="Grad-CAM: Model Attention", width=300)

                pred_index = np.argmax(prediction[0])
                confidence = prediction[0][pred_index]

                st.success(f"**Prediction:** {CLASS_NAMES[pred_index].upper()}")
                st.info(f"**Confidence:** {confidence:.2%}")
        else:
            st.warning("Transfer Learning model not available.")
