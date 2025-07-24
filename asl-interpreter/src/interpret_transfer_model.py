import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

# --- 1. Configuration ---
MODEL_PATH = 'models/best_cnn_from_scratch.keras'
DATA_DIR = 'data/asl_dataset'
IMG_SIZE = (128, 128)

# --- 2. Load Model and Data ---
print("--- Loading Model ---")
# Load the full sequential model
full_model = tf.keras.models.load_model(MODEL_PATH)

# --- FIX: Create a new, compatible Functional model for visualization ---
# This new model will have a new Input layer and will reuse all the trained layers
# from the original model, *except* for the problematic Rescaling layer at the beginning.
vis_input = tf.keras.Input(shape=(128, 128, 3))
# Start the chain of layers from the second layer of the original model (skipping Rescaling)
# and connect them sequentially.
x = vis_input
for layer in full_model.layers[1:]:
    x = layer(x)
vis_model = tf.keras.Model(inputs=vis_input, outputs=x, name="visualization_model")
print("--- Visualization Model Created Successfully ---")


class_names = sorted([d for d in os.listdir(DATA_DIR) if not d.startswith('.') and os.path.isdir(os.path.join(DATA_DIR, d))])

if len(class_names) != 36:
    print(f"Warning: Found {len(class_names)} classes, but expected 36 for a 6x6 grid. Adjusting plot size.")
    grid_size = int(np.ceil(np.sqrt(len(class_names))))
    rows, cols = grid_size, grid_size
else:
    rows, cols = 6, 6

print(f"Found {len(class_names)} classes to visualize.")


# --- 3. Grad-CAM Implementation ---
# Create a Gradcam instance. It will now work correctly with the new Functional visualization model.
gradcam = Gradcam(vis_model, model_modifier=ReplaceToLinear(), clone=True)

# Get one sample image path from each class folder
sample_images = []
for class_name in class_names:
    class_dir = os.path.join(DATA_DIR, class_name)
    image_files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
    if image_files:
        sample_images.append(os.path.join(class_dir, image_files[0]))

print(f"--- Generating Grad-CAM visualizations for {len(sample_images)} classes ---")
fig, axes = plt.subplots(rows, cols, figsize=(22, 22))
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

    # Define the score for Grad-CAM
    score = CategoricalScore([predicted_class_index])
    
    # Manually apply the rescaling to the image data for the visualization model
    rescaled_img = img_array_float / 255.0
    
    # Generate heatmap using the rescaled image and the visualization model
    cam = gradcam(score, np.expand_dims(rescaled_img, axis=0), penultimate_layer=-1)
    heatmap = np.uint8(plt.cm.jet(cam[0])[..., :3] * 255)
    
    # Overlay heatmap on the original integer image for correct display
    overlayed_img = np.uint8(img_array * 0.5 + heatmap * 0.5)
    
    # Plotting
    ax = axes[i]
    ax.imshow(overlayed_img)
    ax.set_title(f"Actual: {actual_class_name}\nPredicted: {predicted_class_name}", fontsize=10)
    ax.axis('off')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.suptitle("Grad-CAM: Visualizing Model Attention (CNN from Scratch)", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.98])
gradcam_filename = 'reports/figures/grad_cam_visualization_cnn_scratch.png'
plt.savefig(gradcam_filename)
print(f"Saved Grad-CAM plot to {gradcam_filename}")
plt.show()

