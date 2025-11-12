import tensorflow as tf
from tensorflow.keras.utils import plot_model

# Load your trained model
model = tf.keras.models.load_model(r"C:\Users\Tinsae Tesfamichael\Desktop\continousPhase.h5")

# Print a text summary
model.summary()

# (Optional) visualize model as an image (requires pydot & graphviz)
plot_model(
    model,
    to_file="model_structure.png",
    show_shapes=True,
    show_layer_names=True,
    dpi=96
)

print("\n✅ Model diagram saved as 'model_structure.png'")
