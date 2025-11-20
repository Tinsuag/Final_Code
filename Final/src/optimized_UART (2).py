import struct
import time
import traceback
import serial
import numpy as np
import tensorflow as tf
from keras.models import load_model
import threading

# Load the pre-trained model
myNN = load_model(r"C:\Users\Tinsae Tesfamichael\Desktop\Thesis\[_Final_code]\Final_Code\Final\src\continousPhase.h5")
BAUD_RATE = 115200
COMM_PORT = 'COM5'
BYTES_IN = 28
BYTES_OUT = 8
CONVERTER = 2 ** 16
USE_FILTER = False #added
ALPHA = 0.8 #added

data_in = np.zeros((10, 7))
current_reply = None
data_lock = threading.Lock()
new_data_available = threading.Event()

import numpy as np
import time

#added
def filter_sin_cos(cos_val, sin_val, alpha=0.1, enable_filter=False, reset=False):
    """
    Exponential Moving Average (EMA) smoother for sine and cosine outputs.

    Parameters
    ----------
    cos_val : float
        Current cosine value from NN output.
    sin_val : float
        Current sine value from NN output.
    alpha : float
        EMA smoothing factor in [0,1]. Higher = faster response, lower = smoother output.
        Typical values: 0.05–0.3 for gait data.
    enable_filter : bool
        If False, filtering is bypassed and raw values are returned.
    reset : bool
        If True, resets the internal state (useful when starting or reinitializing).

    Returns
    -------
    filtered_cos : float
        Filtered cosine value.
    filtered_sin : float
        Filtered sine value.
    """

    # --- initialize persistent state ---
    if not hasattr(filter_sin_cos, "z_prev"):
        filter_sin_cos.z_prev = 1 + 0j  # start on unit circle
        filter_sin_cos.last_time = time.perf_counter()

    if reset:
        filter_sin_cos.z_prev = complex(cos_val, sin_val)
        filter_sin_cos.last_time = time.perf_counter()
        return cos_val, sin_val

    # --- filtering disabled ---
    if not enable_filter:
        return cos_val, sin_val

    # --- EMA filtering on complex vector ---
    z_new = complex(cos_val, sin_val)
    z_prev = filter_sin_cos.z_prev

    # Exponential moving average
    z_filt = (1 - alpha) * z_prev + alpha * z_new

    # Normalize to maintain magnitude near 1
    mag = abs(z_filt)
    if mag > 1e-12:
        z_filt /= mag

    # Save state for next call
    filter_sin_cos.z_prev = z_filt

    # Return real and imaginary parts
    return z_filt.real, z_filt.imag

@tf.function # Optimize prediction with TensorFlow's graph execution
def optimized_predict(model, input_data):
    return model(input_data, training=False)


def stack_data(data_in_prev, data_in):
    return np.vstack((data_in_prev[1:], data_in))


def read_from_serial():
    global data_in, current_reply
    try:
        print("Listening for messages...")
        while True:
            if ser.in_waiting >= BYTES_IN:
                t1 = time.perf_counter()
                message = ser.read(BYTES_IN)
                t2 = time.perf_counter()
                read_time = t2 - t1
                print(f'READ TIME: {read_time * 1000:.3f} ms')

                if message:
                    t4 = time.perf_counter()
                    my_msg, stop = convert_msg(message, BYTES_IN)
                    if stop:
                        break

                    with data_lock:
                        data_in = stack_data(data_in, my_msg / CONVERTER)
                        NN = np.array(optimized_predict(myNN, data_in.reshape((1, 10, 7))))[0][0]

                        cos_val, sin_val = NN[0], NN[1]
                        cos_filt, sin_filt = filter_sin_cos(cos_val, sin_val, alpha=ALPHA, enable_filter=USE_FILTER)

                        # Scale back and pack filtered values for sending
                        cos_scaled = int(cos_filt * CONVERTER) #added
                        sin_scaled = int(sin_filt * CONVERTER) #added
                        current_reply = struct.pack('>ll', cos_scaled, sin_scaled)

                        #NN = (NN * CONVERTER).astype(int)
                        #current_reply = struct.pack('>ll', NN[0], NN[1])

                    new_data_available.set()

                    t3 = time.perf_counter()
                    process_time = t3 - t4
                    print(f'PROCESS TIME: {process_time * 1000:.3f} ms')
    except Exception as e:
        print("Error during message reception or processing.")
        traceback.print_exc()


def send_to_serial():
    global current_reply
    try:
        while True:
            new_data_available.wait()
            with data_lock:
                if current_reply:
                    ser.write(current_reply)
                    print(f'Sent: {current_reply}')
            new_data_available.clear()
    except Exception as e:
        print("Error during message sending.")
        traceback.print_exc()


def convert_msg(msg, my_length):
    try:
        # Assumes msg is a byte string as received from the serial port
        if len(msg) == my_length:
            converted_msg = np.array(struct.unpack('>l' + 'l' * ((len(msg) - 4) // 4), msg))
            stop = 0
        else:
            converted_msg = [0]
            stop = 1
        # converted_msg = converted_msg / 100000
    except Exception as e:
        print("Error during message conversion.")
        traceback.print_exc()
        # converted_msg = None
    return converted_msg, stop

try:
    ser = serial.Serial(COMM_PORT, BAUD_RATE)
    time.sleep(2)  # Allow the serial connection to establish
except Exception as e:
    print("Failed to open serial port.")
    traceback.print_exc()
    ser = None

if ser:
    try:
        read_thread = threading.Thread(target=read_from_serial, daemon=True)
        write_thread = threading.Thread(target=send_to_serial, daemon=True)

        read_thread.start()
        write_thread.start()

        read_thread.join()
        write_thread.join()
    finally:
        try:
            ser.close()
            print("Serial port closed.")
        except Exception:
            print("Failed to close serial port.")
            traceback.print_exc()
