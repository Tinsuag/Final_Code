import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM as BaseLSTM
from tensorflow.keras.utils import register_keras_serializable

# === Path to your model ===
MODEL_PATH = r"C:\Users\Tinsae Tesfamichael\Desktop\Thesis\[_Final_code]\Final_Code\Final\src\continousPhase.h5"

# --------------------------------------------------------------------
# 1. Define a patched LSTM that ignores the `time_major` argument
# --------------------------------------------------------------------
@register_keras_serializable()
class PatchedLSTM(BaseLSTM):
    def __init__(self, *args, time_major=False, **kwargs):
        # We IGNORE time_major and just call the parent LSTM
        super().__init__(*args, **kwargs)

# --------------------------------------------------------------------
# 2. Check model file exists
# --------------------------------------------------------------------
print("Checking model file...")
if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Model file not found at: {MODEL_PATH}")
    raise SystemExit

print(f"✔ Model file found: {MODEL_PATH}")

# --------------------------------------------------------------------
# 3. Try loading with custom_objects
# --------------------------------------------------------------------
print("Loading model with patched LSTM...")
try:
    model = load_model(
        MODEL_PATH,
        custom_objects={"LSTM": PatchedLSTM},
    )
    print("✔ Model loaded successfully!")
except Exception as e:
    print("❌ ERROR loading model:")
    print(e)
    raise SystemExit

# --------------------------------------------------------------------
# 4. Dummy inference to confirm it runs
# --------------------------------------------------------------------
dummy_input = np.zeros((1, 10, 7), dtype=np.float32)

print("Running dummy inference...")
try:
    output = model(dummy_input, training=False).numpy()
    print("✔ Inference successful!")
    print("Output:", output)
    print("Output shape:", output.shape)
    if output.shape[-1] == 2:
        print("✔ Model outputs 2 values (cos, sin)")
    else:
        print("⚠ Model does NOT output 2 values. Check architecture!")
except Exception as e:
    print("❌ ERROR in inference:")
    print(e)
