import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import time
import math
import struct
import threading
from collections import deque
import os
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

Row=0

# ==== Keras / TF (safe import) ====
RUN_NN = False
try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None
    RUN_NN = False

# ==== PIL (safe import for logo) ====
try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

# ==== Serial (safe import) ====
try:
    import serial
    from serial.tools import list_ports
except Exception:
    serial = None
    list_ports = None

# ======================
# USER SERIAL CONFIG (defaults)
# ======================
DEFAULT_PORT = "COM3"
DEFAULT_BAUD = 115200

# ======================
# COMMAND IDs (sync with Arduino)
# ======================
CMD_IDLE            = 0x01
CMD_INITIALIZING    = 0x02
CMD_CALIBRATE       = 0x03
CMD_READING         = 0x04
CMD_OFFSETTING      = 0x05
CMD_CONTROL         = 0x06
CMD_STOP            = 0x07

# ======================
# MODEL / DATA SCAFFOLD
# ======================
BYTES_IN   = 28
BYTES_OUT  = 8
CONVERTER  = 2 ** 16  # scale cos/sin to int32
USE_FILTER = False
ALPHA      = 0.2      # EMA for cos/sin if enabled

# Global rolling window for NN: last 10 samples of 7 features
data_in = np.zeros((10, 7), dtype=np.float32)

# Two “reply” names as requested (big-endian)
response_bytes = None
current_reply  = None

# Lock + event for NN worker
data_lock = threading.Lock()
new_data_available = threading.Event()

# ======================
# PROTOCOL CONSTANTS
# ======================
START_BYTE   = 0xAA
PKT_CMD_ID   = 0x20
PKT_TELEM_ID = 0x30

# ======================
# SERIAL HANDLE (global)
# ======================
SER = None
MODEL_PATH = "continousPhase.h5"

# ======================
# STANDARDIZATION (MEAN 0, STD 1) — Online (Welford) per feature
# ======================
class OnlineFeatureScaler:
    def __init__(self, n_features=7, clip=8.0, eps=1e-6):
        self.n = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.M2 = np.zeros(n_features, dtype=np.float64)
        self.clip = float(clip)
        self.eps = float(eps)

    def partial_fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != self.mean.shape[0]:
            raise ValueError("x has wrong length")
        self.n += 1
        delta = x - self.mean
        self.mean += delta / max(self.n, 1)
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def _std(self):
        if self.n < 2:
            return np.ones_like(self.mean, dtype=np.float64)
        var = self.M2 / (self.n - 1)
        return np.sqrt(np.maximum(var, self.eps))

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        z = (x - self.mean) / self._std()
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        z = np.clip(z, -self.clip, self.clip)
        return z.astype(np.float32)

    def save(self, path="continousPhase_stats_live.npz"):
        np.savez(path, n=self.n, mean=self.mean, M2=self.M2)

    def load(self, path="continousPhase_stats_live.npz"):
        if not os.path.exists(path):
            return False
        try:
            d = np.load(path)
            self.n = int(d["n"])
            self.mean = d["mean"].astype(np.float64)
            self.M2 = d["M2"].astype(np.float64)
            return True
        except Exception:
            return False

SCALER = OnlineFeatureScaler(n_features=7, clip=8.0, eps=1e-6)
SCALER.load("continousPhase_stats_live.npz")
_last_scaler_save = time.time()

# ======================
# Serial helpers
# ======================
def safe_open_serial(port, baud):
    global SER
    if serial is None:
        return None
    try:
        ser = serial.Serial(port, baud, timeout=0.01)
        time.sleep(2.0)  # MCU reset delay
        SER = ser
        return SER
    except Exception:
        SER = None
        return None

def safe_close_serial():
    global SER
    try:
        if SER and SER.is_open:
            SER.close()
    except Exception:
        pass
    SER = None

# ======================
# Framing helpers
# ======================
def compute_checksum(pkt_id, payload_bytes: bytes) -> int:
    length = len(payload_bytes)
    calc = pkt_id + length + sum(payload_bytes)
    return calc % 256

def wrap_full_frame(pkt_id, payload_bytes: bytes) -> bytes:
    frame = bytearray()
    frame.append(START_BYTE & 0xFF)
    frame.append(pkt_id & 0xFF)
    frame.append(len(payload_bytes) & 0xFF)
    frame += payload_bytes
    frame.append(compute_checksum(pkt_id, payload_bytes) & 0xFF)
    return bytes(frame)

# ======================
# Payload builders (PC -> Arduino)
# ======================
def payload_idle():         return bytes([CMD_IDLE])
def payload_init():         return bytes([CMD_INITIALIZING])
def payload_calibrate():    return bytes([CMD_CALIBRATE])
def payload_reading():      return bytes([CMD_READING])
def payload_offsetting():   return bytes([CMD_OFFSETTING])
def payload_stop():         return bytes([CMD_STOP])

def payload_control(torque_val, speed_val, direction_flag):
    p = bytearray()
    p.append(CMD_CONTROL & 0xFF)
    p += struct.pack('<f', float(torque_val))
    p += struct.pack('<f', float(speed_val))
    p.append(int(direction_flag) & 0xFF)
    return bytes(p)

def build_cmd_idle():        return wrap_full_frame(PKT_CMD_ID, payload_idle())
def build_cmd_init():        return wrap_full_frame(PKT_CMD_ID, payload_init())
def build_cmd_calibrate():   return wrap_full_frame(PKT_CMD_ID, payload_calibrate())
def build_cmd_reading():     return wrap_full_frame(PKT_CMD_ID, payload_reading())
def build_cmd_offsetting():  return wrap_full_frame(PKT_CMD_ID, payload_offsetting())
def build_cmd_stop():        return wrap_full_frame(PKT_CMD_ID, payload_stop())
def build_cmd_control(torque_val, speed_val, direction_flag):
    return wrap_full_frame(PKT_CMD_ID, payload_control(torque_val, speed_val, direction_flag))

# ======================
# RX / Telemetry
# ======================
def read_one_packet_blocking(timeout=0.05):
    if SER is None or not getattr(SER, "is_open", False):
        return None
    start_t = time.time()
    while True:
        b = SER.read(1)
        if b and b[0] == START_BYTE:
            break
        if time.time() - start_t > timeout:
            return None
    hdr = SER.read(2)
    if len(hdr) < 2:
        return None
    pkt_id = hdr[0]
    length = hdr[1]
    payload = SER.read(length)
    if len(payload) < length:
        return None
    csum_raw = SER.read(1)
    if len(csum_raw) < 1:
        return None
    if (compute_checksum(pkt_id, payload) & 0xFF) != csum_raw[0]:
        return None
    return (pkt_id, payload)

def parse_telemetry_payload(pkt_id, payload):
    if len(payload) < 2:
        return None
    state_code = payload[0]
    num_floats = payload[1]
    float_section_len = num_floats * 4
    flags_start = 2 + float_section_len
    if len(payload) < flags_start + 2:
        return None

    float_values = []
    off = 2
    for _ in range(num_floats):
        chunk = payload[off:off+4]
        float_values.append(struct.unpack('<f', chunk)[0])
        off += 4

    imuFully = payload[flags_start + 0]
    encOK    = payload[flags_start + 1]

    parsed = {
        "state_code": state_code,
        "angleTheta": float_values[0] if len(float_values) > 0 else 0.0,
        "speed":      float_values[1] if len(float_values) > 1 else 0.0,
        "torque":     float_values[2] if len(float_values) > 2 else 0.0,
        "temp":       float_values[3] if len(float_values) > 3 else 0.0,
        "accelX":     float_values[4] if len(float_values) > 4 else 0.0,
        "accelY":     float_values[5] if len(float_values) > 5 else 0.0,
        "accelZ":     float_values[6] if len(float_values) > 6 else 0.0,
        "gyroX":      float_values[7] if len(float_values) > 7 else 0.0,
        "gyroZ":      float_values[8] if len(float_values) > 8 else 0.0,
        "quatW":      float_values[9] if len(float_values) > 9 else 0.0,
        "quatX":      float_values[10] if len(float_values) > 10 else 0.0,
        "quatY":      float_values[11] if len(float_values) > 11 else 0.0,
        "quatZ":      float_values[12] if len(float_values) > 12 else 0.0,
        "imuFully":   imuFully,
        "encOK":      encOK,
    }
    parsed["float_vector"] = float_values + [float(imuFully), float(encOK)]
    return parsed

def state_code_to_name(code: int) -> str:
    table = {
        CMD_IDLE: "IDLE",
        CMD_INITIALIZING: "INITIALIZING",
        CMD_CALIBRATE: "CALIBRATING",
        CMD_READING: "READING",
        CMD_OFFSETTING: "OFFSETTING",
        CMD_CONTROL: "CONTROL",
        CMD_STOP: "STOP",
    }
    return table.get(code, f"0x{code:02X}")

# ======================
# NN FILTER & FEATURE PIPE
# ======================
def filter_sin_cos(cos_val, sin_val, alpha=0.1, enable_filter=False, reset=False):
    if not hasattr(filter_sin_cos, "z_prev"):
        filter_sin_cos.z_prev = 1 + 0j
    if reset:
        filter_sin_cos.z_prev = complex(cos_val, sin_val)
        return cos_val, sin_val
    if not enable_filter:
        return cos_val, sin_val
    z_new = complex(cos_val, sin_val)
    z_prev = filter_sin_cos.z_prev
    z_filt = (1 - alpha) * z_prev + alpha * z_new
    mag = abs(z_filt)
    if mag > 1e-12:
        z_filt /= mag
    filter_sin_cos.z_prev = z_filt
    return z_filt.real, z_filt.imag

def stack_data(win: np.ndarray, row7: np.ndarray) -> np.ndarray:
    return np.vstack((win[1:], row7))

def optimized_predict(model, input_data):
    return model(input_data, training=False)

def make_feature_vector(parsed, angle_alpha_rad):
    angle_alpha_deg = math.degrees(angle_alpha_rad)
    return np.array([
        float(parsed["accelX"]),
        float(parsed["accelY"]),
        float(parsed["accelZ"]),
        float(angle_alpha_deg),         # α (deg)
        float(parsed["angleTheta"]),    # encoder θ (deg)
        float(parsed["gyroX"]),
        float(parsed["gyroZ"]),
    ], dtype=np.float32)

# ======================
# GUI (keeps your previous theme/format)
# ======================
class Mini_GUI(tk.Tk):
    """Ankle Exoskeleton Control GUI."""
    def __init__(self):
        super().__init__()
        self.title("Ankle Exoskeleton Control GUI By Tinsae Tesfamichael")
        self.geometry("1000x600")
        self.minsize(1000, 950)

        # ---------- mirrored telemetry ----------
        self.angle_theta = 0.0
        self.angle_alpha = 0.0
        self.speed_meas  = 0.0
        self.torque_meas = 0.0
        self.temp_meas   = 0.0
        self.accel_x     = 0.0
        self.accel_y     = 0.0
        self.accel_z     = 0.0
        self.gyro_x      = 0.0
        self.gyro_z      = 0.0
        self.quaternion_w = 0.0
        self.quaternion_x = 0.0
        self.quaternion_y = 0.0
        self.quaternion_z = 0.0
        self.yaw = self.pitch = self.roll = 0.0

        self.imu_fully = 0
        self.enc_ok    = 0

        # ---------- command-side ----------
        self.is_it_safe = False
        self.desired_cmd_id = CMD_IDLE
        self.requested_torque_gui = 0.0
        self.requested_speed_gui  = 0.0
        self.direction_flag       = 1  # 1 fwd, 0 rev

        # ---------- connection state ----------
        self.connected_var = tk.BooleanVar(value=False)
        self.current_port = tk.StringVar(value=DEFAULT_PORT)
        self.current_baud = tk.IntVar(value=DEFAULT_BAUD)

        # Calibration status vars
        self.encoder_cal_status = tk.StringVar(value="Not calibrated")
        self.sys_cal_status     = tk.StringVar(value="Not calibrated")
        self.gyro_cal_status    = tk.StringVar(value="Not calibrated")
        self.mag_cal_status     = tk.StringVar(value="Not calibrated")

        # ---------- styles (unchanged theme; same format) ----------
        self._init_styles()

        # ---------- layout grid ----------
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.main_frame = tk.Frame(self)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        for r in range(6):
            self.main_frame.grid_rowconfigure(r, weight=0) # 
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)

        # Build UI sections
        self.build_logobar()          # row 0 col 0..1
        self.build_toolbar()          # row 1 col 0..1
        self.build_calibration_bar()  # row 2 col 0..1
        self.build_controls()         # row 3 col 0
        self.build_inputs()           # row 3 col 1
        self.status_bar()             # row 4 col 0
        self.build_log()              # row 4 col 1
        self.graphs()                 # row 5 the whole thing

        # Buffers / threads / timers
        self.PLOT_DEPTH = 400
        self.t_axis = deque(maxlen=self.PLOT_DEPTH)
        self.th_axis = deque(maxlen=self.PLOT_DEPTH)
        self.sp_axis = deque(maxlen=self.PLOT_DEPTH)
        self.tq_axis = deque(maxlen=self.PLOT_DEPTH)
        self._plot_init_time = time.time()

        self._last_packet = None
        self.reader_thread = threading.Thread(target=self._reader_worker, daemon=True)
        self.reader_thread.start()

        self.nn_thread = threading.Thread(target=self._nn_worker, daemon=True)
        self.nn_thread.start()

        self.after(120, self.poll_serial_binary)
        self.after(250, self._update_status)
        self.after(500, self._refresh_ports_combo)

        self.protocol("WM_DELETE_WINDOW", self.on_exit)

    # -------- styles (same spirit as your earlier code) --------
    def _init_styles(self):
        s = ttk.Style(self)
        try:
            s.theme_use("aquativo")  # as you had earlier; if missing, it will fall back
        except tk.TclError:
            pass
        s.configure("Ok.TLabel",   foreground="#059669")  # green
        s.configure("Warn.TLabel", foreground="#dc2626")  # red
        s.configure("Info.TLabel", foreground="#374151")  # gray

    # -------- sections --------
    def build_logobar(self):
        logo_frame = tk.Frame(self.main_frame, bd=0, relief=tk.RIDGE )
        logo_frame.grid(row=Row, column=0,rowspan=2, columnspan=4, sticky="ew")
        if Image and ImageTk:
            try:
                img = Image.open(r"C:\Users\Tinsae Tesfamichael\Desktop\Thesis\[_Final_code]\Final\src\icon.png")
                try:
                    dimensions_w = img.width // 6
                    dimensions_h = img.height // 6
                    
                    img = img.resize((dimensions_w, dimensions_h), resample=Image.Resampling.LANCZOS)
                except Exception:
                    img = img.resize((dimensions_w, dimensions_h))
                self._logo_img = ImageTk.PhotoImage(img)
                tk.Label(logo_frame, image=self._logo_img).pack(padx=0, pady=0)
            except Exception:
                tk.Label(logo_frame, text="Ankle Exoskeleton Control").pack(padx=10, pady=10)
        else:
            tk.Label(logo_frame, text="Ankle Exoskeleton Control").pack(padx=10, pady=10)

    def build_toolbar(self):
        bar = ttk.Frame(self.main_frame, padding=(8, 6))
        bar.grid(row=Row+2, column=0, columnspan=2, sticky="ew")

        for c in range(8): bar.columnconfigure(c, weight=0)
        bar.columnconfigure(8, weight=1) # 

        ttk.Label(bar, text="Port").grid(row=0, column=0, sticky="w")
        self.port_combo = ttk.Combobox(bar, textvariable=self.current_port, width=15, values=[])
        self.port_combo.grid(row=0, column=1, padx=(4, 10), sticky="w")

        ttk.Label(bar, text="Baud").grid(row=0, column=2, sticky="w")
        self.baud_combo = ttk.Combobox(
            bar, textvariable=self.current_baud, width=10,
            values=[9600, 19200, 38400, 57600, 115200, 230400]
        )
        self.baud_combo.grid(row=0, column=3, padx=(4, 10), sticky="w")

        self.connect_btn = ttk.Button(bar, text="Connect", command=self._connect_clicked, state="normal")
        self.connect_btn.grid(row=0, column=4, padx=2)

        self.disconnect_btn = ttk.Button(bar, text="Disconnect", command=self._disconnect_clicked, state="disabled")
        self.disconnect_btn.grid(row=0, column=5, padx=2)

        self.safe_btn = tk.Button(bar, text="SAFE LATCH: OFF", command=self.on_isitsafe, width=18, bg="#ff0000")
        self.safe_btn.grid(row=0, column=6, padx=(14, 0))

        self.conn_label = ttk.Label(bar, text="Not connected")
        self.conn_label.grid(row=0, column=8, sticky="e")

    def build_calibration_bar(self):
        cal_bar = ttk.LabelFrame(self.main_frame, text="Calibration status", padding=(8, 6))
        cal_bar.grid(row=Row+3, column=0, columnspan=2, sticky="ew", padx=4, pady=0)
        for c in range(8): cal_bar.columnconfigure(c, weight=1)

        ttk.Label(cal_bar, text="Encoder:").grid(row=0, column=0, sticky="e", padx=(0, 6))
        self.encoder_cal_lbl = ttk.Label(cal_bar, textvariable=self.encoder_cal_status, style="Warn.TLabel")
        self.encoder_cal_lbl.grid(row=0, column=1, sticky="w")

        ttk.Label(cal_bar, text="System:").grid(row=0, column=2, sticky="e", padx=(10, 6))
        self.sys_cal_lbl = ttk.Label(cal_bar, textvariable=self.sys_cal_status, style="Warn.TLabel")
        self.sys_cal_lbl.grid(row=0, column=3, sticky="w")

        ttk.Label(cal_bar, text="Gyroscope:").grid(row=0, column=4, sticky="e", padx=(10, 6))
        self.gyro_cal_lbl = ttk.Label(cal_bar, textvariable=self.gyro_cal_status, style="Warn.TLabel")
        self.gyro_cal_lbl.grid(row=0, column=5, sticky="w")

        ttk.Label(cal_bar, text="Magnetometer:").grid(row=0, column=6, sticky="e", padx=(10, 6))
        self.mag_cal_lbl = ttk.Label(cal_bar, textvariable=self.mag_cal_status, style="Warn.TLabel")
        self.mag_cal_lbl.grid(row=0, column=7, sticky="w")

    def build_controls(self):
        btns = ttk.LabelFrame(self.main_frame, text="Controls", padding=(8, 6))
        btns.grid(row=Row+4, column=0, sticky="nsew", padx=0, pady=0)
        for c in range(3): btns.columnconfigure(c, weight=1)

        self.start_btn = ttk.Button(btns, text="Init", command=self.on_init, state="disabled")
        self.idle_btn  = ttk.Button(btns, text="Idle", command=self.on_idle, state="disabled")
        self.ctrl_btn  = ttk.Button(btns, text="Control", command=self.on_control, state="disabled")
        self.read_btn  = ttk.Button(btns, text="Request Status", command=self.on_request_status, state="disabled")
        self.cal_btn   = ttk.Button(btns, text="Calibrate", command=self.on_calibrate, state="disabled")
        self.zero_btn  = ttk.Button(btns, text="Off-set Encoder", command=self.on_zero, state="disabled")
        self.stop_btn  = ttk.Button(btns, text="STOP", command=self.on_stop, state="disabled")
        self.clear_btn = ttk.Button(btns, text="Clear Log", command=self.on_clear_log)
        self.exit_btn  = ttk.Button(btns, text="Exit", command=self.on_exit)

        self.start_btn.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        self.idle_btn.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        self.ctrl_btn.grid(row=0, column=2, padx=2, pady=2, sticky="ew")
        self.read_btn.grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        self.cal_btn.grid(row=1, column=1, padx=2, pady=2, sticky="ew")
        self.zero_btn.grid(row=1, column=2, padx=2, pady=2, sticky="ew")
        self.stop_btn.grid(row=2, column=0, padx=2, pady=2, sticky="ew")
        self.clear_btn.grid(row=2, column=1, padx=2, pady=2, sticky="ew")
        self.exit_btn.grid(row=2, column=2, padx=2, pady=2, sticky="ew")

    def build_inputs(self):
        inputs = ttk.LabelFrame(self.main_frame, text="Inputs", padding=(8, 6))
        inputs.grid(row=Row+4, column=1, sticky="nsew", padx=0, pady=0)
        for c in range(7): inputs.columnconfigure(c, weight=0)
        inputs.columnconfigure(6, weight=1)

        ttk.Label(inputs, text="Torque (raw)").grid(row=0, column=0, sticky="w")
        self.torque_in = tk.DoubleVar(value=self.requested_torque_gui)
        self.torque_spin = ttk.Spinbox(inputs, from_=-20000, to=20000, increment=1, width=10,
                                       textvariable=self.torque_in, state="disabled")
        self.torque_spin.grid(row=0, column=1, padx=6, sticky="w")
        self.torque_set_btn = ttk.Button(inputs, text="Set", command=self.on_set_torque, state="disabled")
        self.torque_set_btn.grid(row=0, column=2, padx=(0, 8))

        ttk.Label(inputs, text="Speed cmd").grid(row=1, column=0, sticky="w")
        self.speed_in = tk.DoubleVar(value=self.requested_speed_gui)
        self.speed_spin = ttk.Spinbox(inputs, from_=0, to=20000, increment=0.5, width=10,
                                      textvariable=self.speed_in, state="disabled")
        self.speed_spin.grid(row=1, column=1, padx=6, sticky="w")
        self.speed_set_btn = ttk.Button(inputs, text="Set", command=self.on_set_speed, state="disabled")
        self.speed_set_btn.grid(row=1, column=2, padx=(0, 8))

        self.dir_var = tk.IntVar(value=self.direction_flag)
        dir_frame = ttk.Frame(inputs)
        dir_frame.grid(row=2, column=0, columnspan=3, sticky="w")
        self.rb_fwd = ttk.Radiobutton(dir_frame, text="Forward", value=1, variable=self.dir_var,
                                      command=self.on_set_direction, state="disabled")
        self.rb_rev = ttk.Radiobutton(dir_frame, text="Reverse", value=0, variable=self.dir_var,
                                      command=self.on_set_direction, state="disabled")
        self.rb_fwd.pack(side="left")
        self.rb_rev.pack(side="left")

        self.inputs_status = ttk.Label(inputs, text="", style="Info.TLabel")
        self.inputs_status.grid(row=3, column=0, columnspan=7, sticky="w", pady=(6,0))

        self.inputs_widgets = [
            self.torque_spin, self.speed_spin, self.rb_fwd, self.rb_rev,
            self.torque_set_btn, self.speed_set_btn
        ]

    def status_bar(self):
        status = ttk.LabelFrame(self.main_frame, text="Status", padding=(8, 6))
        status.grid(row=Row + 5, column=0, sticky="nsew", padx=0, pady=0)
        for c in range(2): status.columnconfigure(c, weight=1)

        self.status_var = tk.StringVar(value="Ready")
        self.temp_var   = tk.StringVar(value="--")
        self.torque_var = tk.StringVar(value="--")
        self.speed_var  = tk.StringVar(value="--")
        self.angl_theta_var = tk.StringVar(value="--")
        self.angl_alpha_var = tk.StringVar(value="--")
        self.calib_tuple_var = tk.StringVar(value="(S,G,A,M) = 0,0,0,0")
        self.encoder_ok_var  = tk.StringVar(value="0")
        self.conn_var = tk.StringVar(value="Disconnected")

        self._row(status, 0, "State:", self.status_var)
        self._row(status, 1, "Temperature (°C):", self.temp_var)
        self._row(status, 2, "Torque (raw):", self.torque_var)
        self._row(status, 3, "Speed (deg/s):", self.speed_var)
        self._row(status, 4, "Angle θ (deg):", self.angl_theta_var)
        self._row(status, 5, "Euler α (deg):", self.angl_alpha_var)
        self._row(status, 6, "IMU Calib (S,G,A,M):", self.calib_tuple_var)
        self._row(status, 7, "Encoder OK (1/0):", self.encoder_ok_var)
        self._row(status, 8, "Connection:", self.conn_var)

    def build_log(self):
        log_frame = ttk.LabelFrame(self.main_frame, text="Log", padding=(8, 6))
        log_frame.grid(row=Row + 5, column=1, sticky="nsew", padx=0, pady=0)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log = tk.Text(log_frame, height=12, wrap="none", state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        vs = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        hs = ttk.Scrollbar(log_frame, orient="horizontal", command=self.log.xview)
        self.log.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")

        # bottom statusbar strip (same feel)
        sb = ttk.Frame(self, padding=(8, 4))
        sb.grid(row=4, column=0, sticky="ew")
        sb.columnconfigure(0, weight=1)
        self.statusbar_msg = tk.StringVar(value="Ready")
        ttk.Label(sb, textvariable=self.statusbar_msg).grid(row=0, column=0, sticky="w")
        
    def graphs(self):
        graphs_window = ttk.LabelFrame(self.main_frame, text= "Plots")
        graphs_window.grid(row=Row+3,column=3, columnspan=2, rowspan=3,sticky="nsew")
        Plots = ttk.Notebook(graphs_window)
        Plots.grid(row=0,column=1)

    # -------- small helpers --------
    def _row(self, parent, r, label, var):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="e", padx=(0,6))
        ttk.Label(parent, textvariable=var).grid(row=r, column=1, sticky="w")

    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", f"{msg}\n")
        self.log.see("end")
        self.log.configure(state="disabled")
        self.statusbar_msg.set(msg)

    def _refresh_ports_combo(self):
        if list_ports:
            try:
                ports = [p.device for p in list_ports.comports()]
            except Exception:
                ports = []
        else:
            ports = []
        cur = self.current_port.get()
        self.port_combo["values"] = ports
        if ports and cur not in ports:
            self.current_port.set(ports[0])
        self.after(3000, self._refresh_ports_combo)

    def _set_controls_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for b in (self.start_btn, self.idle_btn, self.ctrl_btn, self.read_btn,
                  self.cal_btn, self.zero_btn, self.stop_btn):
            b.configure(state=state)

    # -------- math --------
    def quaternion_to_euler(self):
        qw = self.quaternion_w; qx = self.quaternion_x; qy = self.quaternion_y; qz = self.quaternion_z
        t0 = +2.0*(qw*qx + qy*qz)
        t1 = +1.0 - 2.0*(qx*qx + qy*qy)
        roll_rad = math.atan2(t0, t1)
        t2 = +2.0*(qw*qy - qz*qx)
        t2 = max(-1.0, min(1.0, t2))
        pitch_rad = math.asin(t2)
        t3 = +2.0*(qw*qz + qx*qy)
        t4 = +1.0 - 2.0*(qy*qy + qz*qz)
        yaw_rad = math.atan2(t3, t4)
        self.roll = roll_rad; self.pitch = pitch_rad; self.yaw = yaw_rad
        self.angle_alpha = math.degrees(self.pitch)

    # -------- connection --------
    def _connect_clicked(self):
        port = self.current_port.get()
        baud = int(self.current_baud.get())
        safe_close_serial()
        ser = safe_open_serial(port, baud)
        if ser and ser.is_open:
            self.connected_var.set(True)
            self._log(f"Connected: {port} @ {baud}")
            self.conn_label.configure(text=f"Connected: {port}@{baud}")
            self.conn_var.set(f"Port:{port} @ {baud}")
            self._set_controls_enabled(True)
            self.connect_btn.configure(state="disabled")
            self.disconnect_btn.configure(state="normal")
            self.status_var.set("Idle")
            for w in getattr(self, "inputs_widgets", []):
                w.configure(state="normal")
            # auto-start NN flag
            global RUN_NN
            RUN_NN = True
            self._log("NN: ON (online standardization enabled)")
        else:
            self.connected_var.set(False)
            self._log(f"Failed to connect: {port} @ {baud}")
            self.conn_label.configure(text="Not connected")
            self.conn_var.set("Port: (disconnected)")
            self._set_controls_enabled(False)
            self.connect_btn.configure(state="normal")
            self.disconnect_btn.configure(state="disabled")
        self._update_safe_btn()

    def _disconnect_clicked(self):
        safe_close_serial()
        self.status_var.set("Disconnected")
        self.connected_var.set(False)
        self._log("Disconnected")
        self.conn_label.configure(text="Not connected")
        self.conn_var.set("Port: (disconnected)")
        self._set_controls_enabled(False)
        for w in getattr(self, "inputs_widgets", []):
            w.configure(state="disabled")
        self._update_safe_btn()
        self.connect_btn.configure(state="normal")
        self.disconnect_btn.configure(state="disabled")

        global RUN_NN
        RUN_NN = False
        try:
            SCALER.save("continousPhase_stats_live.npz")
            self._log("NN: OFF (stats saved)")
        except Exception:
            pass

    def _update_safe_btn(self):
        if self.is_it_safe:
            self.safe_btn.configure(text="SAFE LATCH: ON", bg="#22cc55")
        else:
            self.safe_btn.configure(text="SAFE LATCH: OFF", bg="#ee4444")

    # -------- UI callbacks --------
    def on_isitsafe(self):
        self.is_it_safe = not self.is_it_safe
        self._update_safe_btn()
        # purely visual tie-in for the labels (same as earlier demo behavior)
        if self.is_it_safe:
            self.encoder_cal_status.set("Calibrated")
            self.sys_cal_status.set("Calibrated")
            self.gyro_cal_status.set("Calibrated")
            self.mag_cal_status.set("Calibrated")
        else:
            self.encoder_cal_status.set("Not calibrated")
            self.sys_cal_status.set("Not calibrated")
            self.gyro_cal_status.set("Not calibrated")
            self.mag_cal_status.set("Not calibrated")
        self._log(f"Safety latch {'ON' if self.is_it_safe else 'OFF'}.")

    def on_set_torque(self):
        try:
            val = float(self.torque_in.get())*100
        except ValueError:
            messagebox.showerror("Invalid torque", "Enter a number for torque.")
            return
        self.requested_torque_gui = val
        self.inputs_status.config(text=f"Torque set to {val:.0f} (raw)")
        self._log(f"Requested torque = {val}")

    def on_set_speed(self):
        try:
            val = float(self.speed_in.get())
        except ValueError:
            messagebox.showerror("Invalid speed", "Enter a number for speed.")
            return
        self.requested_speed_gui = abs(val)
        self.inputs_status.config(text=f"Speed set to {self.requested_speed_gui:.1f}")
        self._log(f"Requested speed magnitude = {self.requested_speed_gui}")

    def on_set_direction(self):
        self.direction_flag = 1 if self.dir_var.get() == 1 else 0
        self.inputs_status.config(text=f"Direction set to {'Forward' if self.direction_flag else 'Reverse'}")
        self._log(f"Direction = {'Forward' if self.direction_flag else 'Reverse'}")

    def _tx_frame(self, frame_bytes, log_label):
        global SER
        if SER is None or not getattr(SER, "is_open", False):
            self._log("TX blocked: serial not open.")
            return
        try:
            SER.write(frame_bytes)
            self._log(log_label)
        except Exception as e:
            self._log(f"TX error: {e}")

    def on_init(self):
        self.desired_cmd_id = CMD_INITIALIZING
        self._tx_frame(build_cmd_init(), "TX INITIALIZING")
        self.status_var.set("Initializing...")

    def on_idle(self):
        self.desired_cmd_id = CMD_IDLE
        self._tx_frame(build_cmd_idle(), "TX IDLE")

    def on_control(self):
        if not self.is_it_safe:
            self._log("CONTROL blocked: safety is OFF")
            return
        self.desired_cmd_id = CMD_CONTROL
        frame = build_cmd_control(self.requested_torque_gui, self.requested_speed_gui, self.direction_flag)
        self._tx_frame(frame, f"TX CONTROL τ={self.requested_torque_gui:.1f} spd={self.requested_speed_gui:.1f} dir={self.direction_flag}")
        self.status_var.set("CONTROL mode")

    def on_stop(self):
        self.desired_cmd_id = CMD_STOP
        self._tx_frame(build_cmd_stop(), "TX STOP (EMERGENCY_STOP)")
        self.status_var.set("STOPPED")

    def on_zero(self):
        self.desired_cmd_id = CMD_OFFSETTING
        self._tx_frame(build_cmd_offsetting(), "TX OFFSETTING (encoder zero)")

    def on_calibrate(self):
        self.desired_cmd_id = CMD_CALIBRATE
        self._tx_frame(build_cmd_calibrate(), "TX CALIBRATE")
        self.status_var.set("Calibrating...")

    def on_request_status(self):
        self.desired_cmd_id = CMD_READING
        self._tx_frame(build_cmd_reading(), "TX READING / Status Request")
        self.status_var.set("Requesting Status...")
        at = getattr(self, "angle_theta_rads", 0.0)
        self._log(f"RX {self.status_var.get()} θ={at:.2f}rad α={self.angle_alpha:.2f}rad τ={self.torque_meas:.1f} T={self.temp_meas:.2f}°C spd={self.speed_meas:.2f}deg/s")

    def on_clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self._log("Log cleared.")

    def on_exit(self):
        try:
            safe_close_serial()
        except Exception:
            pass
        try:
            SCALER.save("continousPhase_stats_live.npz")
        except Exception:
            pass
        self.destroy()

    # -------- background RX --------
    def _reader_worker(self):
        while True:
            pkt = read_one_packet_blocking(timeout=0.05)
            if pkt is None:
                time.sleep(0.03)
                continue
            self._last_packet = pkt

    def poll_serial_binary(self):
        data = getattr(self, "_last_packet", None)
        self._last_packet = None

        if data:
            (pkt_id, payload) = data
            parsed = None
            try:
                parsed = parse_telemetry_payload(pkt_id, payload)
            except Exception as e:
                self._log(f"Parse error: {e}")

            if parsed:
                self.angle_theta = parsed["angleTheta"]
                self.speed_meas  = parsed["speed"]
                self.torque_meas = parsed["torque"]
                self.temp_meas   = parsed["temp"]
                self.angle_theta_rads = math.radians(self.angle_theta)

                self.accel_x = parsed["accelX"]; self.accel_y = parsed["accelY"]; self.accel_z = parsed["accelZ"]
                self.gyro_x  = parsed["gyroX"];  self.gyro_z  = parsed["gyroZ"]

                self.quaternion_w = parsed["quatW"]; self.quaternion_x = parsed["quatX"]
                self.quaternion_y = parsed["quatY"]; self.quaternion_z = parsed["quatZ"]

                self.imu_fully = int(parsed["imuFully"])
                self.enc_ok    = int(parsed["encOK"])

                self.quaternion_to_euler()
                self.status_var.set(state_code_to_name(parsed["state_code"]))

                # NN feed
                if RUN_NN:
                    fv = make_feature_vector(parsed, self.angle_alpha)
                    SCALER.partial_fit(fv)
                    fv_std = SCALER.transform(fv)
                    with data_lock:
                        global data_in
                        data_in = stack_data(data_in, fv_std)
                    new_data_available.set()

                    global _last_scaler_save
                    if time.time() - _last_scaler_save > 10.0:
                        try:
                            SCALER.save("continousPhase_stats_live.npz")
                        except Exception:
                            pass
                        _last_scaler_save = time.time()

        self.after(10, self.poll_serial_binary)

    # -------- periodic label refresh --------
    def _update_status(self):
        self.temp_var.set(f"{self.temp_meas:.2f}")
        self.torque_var.set(f"{self.torque_meas:.1f}")
        self.speed_var.set(f"{self.speed_meas:.2f}")
        self.angl_theta_var.set(f"{self.angle_theta:.2f}")
        self.angl_alpha_var.set(f"{self.angle_alpha:.2f}")

        if SER and getattr(SER, "is_open", False):
            self.conn_var.set(f"Port:{SER.port} @ {SER.baudrate}")
        else:
            self.conn_var.set("Port: (disconnected)")

        self.encoder_ok_var.set(str(self.enc_ok))
        self.after(250, self._update_status)

    # -------- NN worker --------
    def _nn_worker(self):
        global response_bytes, current_reply
        model = None
        if RUN_NN and load_model is not None:
            try:
                model = load_model(MODEL_PATH)
                self._log(f"NN loaded: {MODEL_PATH}")
            except Exception as e:
                self._log(f"NN load failed: {e}")
                model = None

        while True:
            new_data_available.wait()
            new_data_available.clear()
            if model is None:
                continue
            try:
                with data_lock:
                    window = data_in.copy()  # (10,7) standardized

                NN = model(window.reshape((1, 10, 7)), training=False).numpy()[0][0]
                cos_val, sin_val = float(NN[0]), float(NN[1])
                cos_filt, sin_filt = filter_sin_cos(cos_val, sin_val, alpha=ALPHA, enable_filter=USE_FILTER)
                cos_i = int(max(-1.0, min(1.0, cos_filt)) * CONVERTER)
                sin_i = int(max(-1.0, min(1.0, sin_filt)) * CONVERTER)
                packed = struct.pack('>ii', cos_i, sin_i)
                response_bytes = packed
                current_reply  = packed
                self._log(f"NN cos={cos_val:.3f} sin={sin_val:.3f} -> packed (BE int32)")
            except Exception as e:
                self._log(f"NN worker error: {e}")

if __name__ == "__main__":
    gui = Mini_GUI()
    gui.mainloop()

