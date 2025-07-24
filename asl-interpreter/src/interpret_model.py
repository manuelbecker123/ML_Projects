import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

# --- 1. Configuration ---
MODEL_PATH = 'models/best_transfer_learning_mobilenet.keras'
DATA_DIR = 'data/asl_dataset'
IMG_SIZE = (128, 128)

# --- 2. Load Full Model and Prepare for Visualization ---
print("--- Loading Full Model ---")
full_model = tf.keras.models.load_model(MODEL_PATH)
class_names = sorted([d for d in os.listdir(DATA_DIR) if not d.startswith('.') and os.path.isdir(os.path.join(DATA_DIR, d))])

if len(class_names) != 36:
    raise ValueError(f"Expected 36 class folders, but found {len(class_names)}.")

print(f"Found {len(class_names)} classes to visualize.")

# --- Create a new model for visualization that is Grad-CAM compatible ---
base_model_layer = full_model.get_layer('mobilenetv2_1.00_128')
vis_input = base_model_layer.input
x = base_model_layer.output
x = full_model.get_layer('global_average_pooling2d')(x)
x = full_model.get_layer('dropout_1')(x)
vis_output = full_model.get_layer('dense_2')(x)
vis_model = tf.keras.Model(vis_input, vis_output)
print("--- Visualization Model Created Successfully ---")


# --- 3. Grad-CAM Implementation ---
gradcam = Gradcam(vis_model, model_modifier=ReplaceToLinear(), clone=True)

# Get one sample image path from each class folder
sample_images = []
for class_name in class_names:
    class_dir = os.path.join(DATA_DIR, class_name)
    image_files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
    if image_files:
        sample_images.append(os.path.join(class_dir, image_files[0]))

print(f"--- Generating Grad-CAM visualizations for {len(sample_images)} classes ---")
# --- CHANGE: Create a 6x6 subplot grid ---
fig, axes = plt.subplots(6, 6, figsize=(22, 22))
# --- CHANGE: Flatten the 2D array of axes for easy iteration ---
axes = axes.flatten()

for i, img_path in enumerate(sample_images):
    # Load and preprocess the image
    img = Image.open(img_path).resize(IMG_SIZE)
    img_array = np.array(img)
    img_array_float = img_array.astype(np.float32)

    # Get prediction using the ORIGINAL full model
    prediction = full_model.predict(np.expand_dims(img_array_float, axis=0), verbose=0)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    actual_class_name = os.path.basename(os.path.dirname(img_path))

    # Manually preprocess the image for the visualization model
    preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array_float)
    
    # Define the score for Grad-CAM
    score = CategoricalScore([predicted_class_index])
    
    # Generate heatmap
    cam = gradcam(score, np.expand_dims(preprocessed_img, axis=0), penultimate_layer=-1)
    heatmap = np.uint8(plt.cm.jet(cam[0])[..., :3] * 255)
    
    # Overlay heatmap
    overlayed_img = np.uint8(img_array * 0.5 + heatmap * 0.5)
    
    # --- CHANGE: Plot on the correct subplot axis ---
    ax = axes[i]
    ax.imshow(overlayed_img)
    ax.set_title(f"Actual: {actual_class_name}\nPredicted: {predicted_class_name}", fontsize=10)
    ax.axis('off')

plt.suptitle("Grad-CAM: Visualizing Model Attention (Transfer Learning)", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.98])
gradcam_filename = 'reports/figures/grad_cam_visualization_transfer_learning.png'
plt.savefig(gradcam_filename)
print(f"Saved Grad-CAM plot to {gradcam_filename}")
plt.show()

