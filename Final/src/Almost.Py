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
import json
import keras

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ==== Keras / TF (safe import) ====
RUN_NN = False
try:
    from keras.models import load_model
    from keras.layers import LSTM as BaseLSTM
except Exception:
    load_model = None
    BaseLSTM = None

# Patched LSTM that ignores `time_major` argument when loading old models
if BaseLSTM is not None:
    class PatchedLSTM(BaseLSTM):
        def __init__(self, *args, time_major=False, **kwargs):
            # ignore time_major, just call base LSTM
            super().__init__(*args, **kwargs)
else:
    PatchedLSTM = None

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
DEFAULT_PORT = "COM4"
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
MODEL_PATH = "C:\\Users\\Tinsae Tesfamichael\\Desktop\\Thesis\\[_Final_code]\\Final_Code\\Final\\src\\continousPhase.h5"

# for calibration saving
cSys_level = 0
cG_level = 0
cA_level = 0
cM_level = 0

# ======================
# Global hook for Plots-tab log
# ======================
LOG2_FUNC = None


class LowPassFilter:
    def __init__(self, n_features=7, cutoff_hz=5.0, fs=50.0):
        dt = 1.0 / fs
        RC = 1.0 / (2.0 * np.pi * cutoff_hz)
        self.alpha = dt / (RC + dt)
        self.y = np.zeros(n_features, dtype=float)
        self.initialized = False

    def update(self, x):
        x = np.asarray(x, dtype=float)
        if not self.initialized:
            self.y = x.copy()
            self.initialized = True
        else:
            self.y = self.y + self.alpha * (x - self.y)
        return self.y.copy()


class Preprocessor50Hz:
    def __init__(self, n_features=7, max_rows=10, cutoff_hz=5.0, fs=50.0):
        self.n_features = n_features
        self.max_rows = max_rows
        self.buffer = np.empty((0, n_features), dtype=float)
        self.lpf = LowPassFilter(n_features=n_features,
                                 cutoff_hz=cutoff_hz,
                                 fs=fs)

    def _add_and_stack(self, filtered_sample):
        filtered_sample = np.asarray(filtered_sample, dtype=float)

        if self.buffer.shape[0] >= 2:
            mean = self.buffer.mean(axis=0)
            std = self.buffer.std(axis=0)
            std[std == 0.0] = 1.0
            std_sample = (filtered_sample - mean) / std
        else:
            std_sample = filtered_sample

        self.buffer = np.vstack([self.buffer, std_sample])

        if self.buffer.shape[0] > self.max_rows:
            self.buffer = self.buffer[-self.max_rows:, :]

    def step(self, raw_sample):
        filtered = self.lpf.update(raw_sample)
        self._add_and_stack(filtered)
        return self.buffer

    def get_buffer(self):
        return self.buffer


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
        chunk = payload[off:off + 4]
        float_values.append(struct.unpack('<f', chunk)[0])
        off += 4

    imuFully = payload[flags_start]
    encOK    = payload[flags_start + 1]

    global cSys_level, cG_level, cA_level, cM_level
    if state_code == CMD_CALIBRATE and len(payload) >= flags_start + 7:
        cSys_level = payload[flags_start + 2]
        cG_level   = payload[flags_start + 3]
        cA_level   = payload[flags_start + 4]
        cM_level   = payload[flags_start + 5]

    parsed = {
        "state_code": state_code,
        "angleAlpha": float_values[0] if len(float_values) > 0 else 0.0,
        "speed":      float_values[1] if len(float_values) > 1 else 0.0,
        "torque":     float_values[2] if len(float_values) > 2 else 0.0,
        "temp":       float_values[3] if len(float_values) > 3 else 0.0,
        "accelX":     float_values[4] if len(float_values) > 4 else 0.0,
        "accelY":     float_values[5] if len(float_values) > 5 else 0.0,
        "accelZ":     float_values[6] if len(float_values) > 6 else 0.0,
        "gyroX":      float_values[7] if len(float_values) > 7 else 0.0,
        "gyroZ":      float_values[8] if len(float_values) > 8 else 0.0,
        "quatW":      float_values[9]  if len(float_values) > 9  else 0.0,
        "quatX":      float_values[10] if len(float_values) > 10 else 0.0,
        "quatY":      float_values[11] if len(float_values) > 11 else 0.0,
        "quatZ":      float_values[12] if len(float_values) > 12 else 0.0,
        "imuFully":   int(imuFully),
        "encOK":      int(encOK),
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


def optimized_predict(model, input_data):
    return model(input_data, training=False)


def GP_estimation(sin_val, cos_val):
    gp = (1.0 / (2.0 * np.pi)) * (np.arctan2(sin_val, cos_val) + np.pi)
    return float(np.clip(gp, 0.0, 1.0)) # ensure in [0, 1] range




class AssistiveTorque:
    def __init__(self, mu=0.5, sigma=0.17, c=0.6, d=0.02, p=0.8,
                 alpha0_deg=-10.0, alpha1_deg=10.0, K_rep=1.0, BW_gain=1.0):
        self.mu = mu
        self.sigma = sigma
        self.c = c
        self.d = d
        self.p = p
        self.alpha0 = np.deg2rad(alpha0_deg)
        self.alpha1 = np.deg2rad(alpha1_deg)
        self.K_rep = K_rep
        self.BW_gain = BW_gain

    def f(self, GP): return np.exp(-((GP - self.mu) ** 2) / (2 * self.sigma ** 2))
    def s(self, GP): return (self.p + 1) / (np.exp((GP - self.c) / self.d) + 1) - self.p
    def r(self, alpha): return 1 + self.K_rep * (np.exp(self.alpha0 - alpha) + np.exp(alpha - self.alpha1))
    def tau(self, GP, alpha): return self.BW_gain * self.f(GP) * self.s(GP) / self.r(alpha)


class TorqueGUI(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill=tk.BOTH, expand=True)
        self.model = AssistiveTorque()
        self.streaming = False
        self.prev_tau = None

        self.latest_gp = 0.5
        self.latest_tau = 0.0
        self.latest_time = 0.0
        self.gp_history = deque(maxlen=500)  # for static plot overlay
        self.torque_history = deque(maxlen=500)
        self.time_history = deque(maxlen=500)

        toolbar = ttk.Frame(self, padding=(8, 6))
        toolbar.pack(fill=tk.X, side=tk.TOP)

        self.presets = {
            "Default": dict(gp=50, alpha=0, bw=1, mu=0.5, sigma=0.17, c=0.6, d=0.02, p=0.8, a0=-10, a1=10, krep=1),
            "Heel-Strike Assist": dict(gp=5, alpha=-5, bw=1.2, mu=0.1, sigma=0.12, c=0.18, d=0.02, p=0.6, a0=-12, a1=8, krep=1.2),
            "Push-Off Boost": dict(gp=60, alpha=6, bw=1.5, mu=0.6, sigma=0.14, c=0.55, d=0.02, p=0.9, a0=-8, a1=12, krep=0.8),
        }
        preset_box = ttk.Frame(toolbar)
        preset_box.pack(side=tk.LEFT, padx=(12, 6))
        ttk.Label(preset_box, text="Preset:").pack(side=tk.LEFT)
        self.preset_var = tk.StringVar(value="Default")
        ttk.Combobox(preset_box, width=18, textvariable=self.preset_var,
                     values=list(self.presets.keys()), state="readonly").pack(side=tk.LEFT, padx=4)
        ttk.Button(preset_box, text="Apply", command=self.apply_preset).pack(side=tk.LEFT)

        btns = ttk.Frame(toolbar)
        btns.pack(side=tk.RIGHT)
        self.play_btn = ttk.Button(btns, text="⏸ Pause IMU", command=self.toggle_stream)
        self.play_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="⟳ Reset", command=self.reset_params).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="⬇ Save Figures", command=self.save_figs).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="💾 Save Params", command=self.save_params).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="📂 Load Params", command=self.load_params).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="📌 Freeze τ", command=self.freeze_curve).pack(side=tk.LEFT, padx=4)

        splitter = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        splitter.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        left_pane = ttk.Frame(splitter)
        splitter.add(left_pane, weight=1)
        ctrl = ttk.Frame(left_pane)
        ctrl.pack(fill=tk.BOTH, expand=True)
        ctrl.grid_rowconfigure(0, weight=1)
        ctrl.grid_rowconfigure(1, weight=6)
        ctrl.grid_columnconfigure(0, weight=1)

        params_card = ttk.LabelFrame(ctrl, text="Parameters")
        params_card.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6, 4))
        params_wrap = ttk.Frame(params_card)
        params_wrap.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(params_wrap, height=240, highlightthickness=0)
        vbar = ttk.Scrollbar(params_wrap, orient="vertical", command=canvas.yview)
        self.params_inner = ttk.Frame(canvas)
        self.params_inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        inner = canvas.create_window((0, 0), window=self.params_inner, anchor="nw")
        canvas.configure(yscrollcommand=vbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        vbar.pack(side="right", fill="y")
        params_wrap.bind("<Configure>", lambda e: canvas.itemconfig(inner, width=params_wrap.winfo_width()))

        Torq_Ctrl_plt = ttk.LabelFrame(ctrl, text="Assistive Torque Model")
        Torq_Ctrl_plt.grid(row=1, column=0, sticky="nsew", padx=6, pady=(4, 6))
        Torq_Ctrl_plt.grid_rowconfigure(0, weight=1)
        Torq_Ctrl_plt.grid_columnconfigure(0, weight=1)

        right_pane = ttk.Frame(splitter)
        splitter.add(right_pane, weight=2)
        right_pane.rowconfigure(0, weight=3)
        right_pane.rowconfigure(1, weight=2)
        right_pane.rowconfigure(2, weight=0)
        right_pane.columnconfigure(0, weight=1)

        plot = ttk.LabelFrame(right_pane, text="NN & Sensor Plots")
        plot.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6, 3))
        plot.grid_rowconfigure(0, weight=1)
        plot.grid_columnconfigure(0, weight=1)

        log_frame2 = ttk.LabelFrame(right_pane, text="Log")
        log_frame2.grid(row=1, column=0, sticky="nsew", padx=6, pady=(3, 6))
        log_frame2.rowconfigure(0, weight=1)
        log_frame2.columnconfigure(0, weight=1)
        self.log2 = tk.Text(log_frame2, height=8, wrap="none", state="disabled")
        self.log2.grid(row=0, column=0, sticky="nsew")
        vs2 = ttk.Scrollbar(log_frame2, orient="vertical", command=self.log2.yview)
        hs2 = ttk.Scrollbar(log_frame2, orient="horizontal", command=self.log2.xview)
        self.log2.configure(yscrollcommand=vs2.set, xscrollcommand=hs2.set)
        vs2.grid(row=0, column=1, sticky="ns")
        hs2.grid(row=1, column=0, sticky="ew")

        self.status = ttk.Label(right_pane, anchor="w")
        self.status.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 6))

        def add_slider(frame, label, frm, to, init, cmd, unit=""):
            row = ttk.Frame(frame, height=26)
            row.pack(fill=tk.X, pady=2, padx=6)
            row.pack_propagate(False)

            ttk.Label(row, text=label, width=14, anchor="w").pack(side=tk.LEFT)
            s = ttk.Scale(row, from_=frm, to=to, orient=tk.HORIZONTAL, length=170)
            val = ttk.Label(row, width=8, anchor="e")
            val.pack(side=tk.RIGHT)
            s.pack(side=tk.LEFT, padx=6)

            s.set(init)
            val.config(text=f"{init:.2f}{unit}")

            def _on_move(_=None):
                try:
                    val.config(text=f"{s.get():.2f}{unit}")
                    if cmd:
                        cmd()
                except Exception:
                    pass

            s.configure(command=_on_move)
            return s

        def collapsible(parent, title):
            f = ttk.Frame(parent)
            bar = ttk.Frame(f)
            bar.pack(fill=tk.X, pady=(3, 0))
            sv = tk.StringVar(value="▼ " + title)
            lbl = ttk.Label(bar, textvariable=sv)
            lbl.pack(side=tk.LEFT, padx=2)
            body = ttk.Frame(f)
            body.pack(fill=tk.X, padx=2, pady=(2, 0))

            def toggle(_=None):
                if body.winfo_manager():
                    body.pack_forget()
                    sv.set("► " + title)
                else:
                    body.pack(fill=tk.X, padx=2, pady=(2, 0))
                    sv.set("▼ " + title)

            bar.bind("<Button-1>", toggle)
            lbl.bind("<Button-1>", toggle)
            return f, body

        grp_core, core_body = collapsible(self.params_inner, "Core Controls")
        grp_core.pack(fill=tk.X)
        self.gp_percent = add_slider(core_body, "GP (%)", 0, 100, 50, self.update_static_plots, "%")
        self.alpha_deg  = add_slider(core_body, "α (deg)", -30, 30, 0, self.update_static_plots, "°")
        self.bw_gain    = add_slider(core_body, "BW gain", 0, 2.5, 1, self._update_model)

        grp_f, body_f = collapsible(self.params_inner, "f(GP)")
        grp_f.pack(fill=tk.X)
        self.mu    = add_slider(body_f, "μ", 0, 1, 0.5, self._update_model)
        self.sigma = add_slider(body_f, "σ", 0.05, 0.4, 0.17, self._update_model)

        grp_s, body_s = collapsible(self.params_inner, "s(GP)")
        grp_s.pack(fill=tk.X)
        self.c   = add_slider(body_s, "c", 0, 1, 0.6, self._update_model)
        self.d   = add_slider(body_s, "d", 0.005, 0.1, 0.02, self._update_model)
        self.p   = add_slider(body_s, "p", 0, 1.5, 0.8, self._update_model)

        grp_r, body_r = collapsible(self.params_inner, "r(α)")
        grp_r.pack(fill=tk.X)
        self.a0   = add_slider(body_r, "α0 (deg)", -40, 0, -10, self._update_model, "°")
        self.a1   = add_slider(body_r, "α1 (deg)", 0, 40, 10, self._update_model, "°")
        self.krep = add_slider(body_r, "Krep", 0, 5, 1, self._update_model)

        # --- Matplotlib plots for torque model ---
        self.fig = Figure(figsize=(6.6, 3.8), dpi=100, layout="constrained")
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[2.6, 1.2],
                                   hspace=0.28, wspace=0.28)
        self.ax_tau = self.fig.add_subplot(gs[0, 0])
        self.ax_phase = self.ax_tau.inset_axes([0.08, 0.03, 0.84, 0.12])
        self.ax_f = self.fig.add_subplot(gs[0, 1])
        self.ax_s = self.fig.add_subplot(gs[1, 1])

        self.canvas = FigureCanvasTkAgg(self.fig, master=Torq_Ctrl_plt)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # === IMU + NN figure (uses real data from Application) ===
        # 2x2 layout: NN, Accel, Speed, Angles
        self.fig_imu = Figure(figsize=(7.5, 4.5), dpi=100, layout="constrained")
        imu_gs = self.fig_imu.add_gridspec(2, 2, hspace=0.30, wspace=0.28)

        # Top-left: NN outputs + gait phase
        self.ax_nn = self.fig_imu.add_subplot(imu_gs[0, 0])
        # Top-right: Accelerometer
        self.ax_accel = self.fig_imu.add_subplot(imu_gs[0, 1])
        # Bottom-left: Angular speed
        self.ax_speed = self.fig_imu.add_subplot(imu_gs[1, 0])
        # Bottom-right: Angles
        self.ax_angle = self.fig_imu.add_subplot(imu_gs[1, 1])

        self.canvas_imu = FigureCanvasTkAgg(self.fig_imu, master=plot)
        self.canvas_imu.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Buffers for streaming (10 s window)
        self.window_sec, self.fs = 10, 50
        self.dt, self.max_samples = 1 / self.fs, int(self.window_sec * self.fs)
        self.tbuf = deque(maxlen=self.max_samples)
        self.accX = deque(maxlen=self.max_samples)
        self.accY = deque(maxlen=self.max_samples)
        self.accZ = deque(maxlen=self.max_samples)
        self.angX = deque(maxlen=self.max_samples)
        self.angZ = deque(maxlen=self.max_samples)
        self.theta = deque(maxlen=self.max_samples)       # θ from IMU (deg)
        self.alphaZ = deque(maxlen=self.max_samples)      # α from IMU (deg)
        self.alpha_enc = deque(maxlen=self.max_samples)   # α from encoder (deg)
        self.phase_buf = deque(maxlen=self.max_samples)
        self.torque_buf = deque(maxlen=self.max_samples)

        # NEW: buffers for NN outputs and gait phase
        self.gp_series  = deque(maxlen=self.max_samples)  # gait phase [0..1]
        self.cos_series = deque(maxlen=self.max_samples)  # filtered cos
        self.sin_series = deque(maxlen=self.max_samples)  # filtered sin

        self._t0 = time.perf_counter()
        self._next_t = 0.0

        self.lock_tau = tk.BooleanVar(master=self, value=False)
        ttk.Checkbutton(Torq_Ctrl_plt, text="Lock τ autoscale",
                        variable=self.lock_tau).place(relx=0.01, rely=0.02)

        self._mh = self.canvas.mpl_connect("motion_notify_event", self._on_move_tau)

        self.update_static_plots()

        # Start periodic plot update
        self.after(20, self._plot_update_loop)

        self.bind_all("<space>", lambda e: self.toggle_stream())
        self.bind_all("<Control-r>", lambda e: self.reset_params())
        self.bind_all("<Control-s>", lambda e: self.save_figs())

        self._log2("Torque GUI ready.")

        # Register this logger as the global Plots-tab logger
        global LOG2_FUNC
        LOG2_FUNC = self._log2

    # ---------------- Helpers: Log for Tab 2 ----------------
    def _log2(self, msg: str):
        if not hasattr(self, "log2"):
            return
        ts = time.strftime("%H:%M:%S")
        self.log2.configure(state="normal")
        self.log2.insert("end", f"[{ts}] {msg}\n")
        self.log2.see("end")
        self.log2.configure(state="disabled")

    # ---------------- Toolbar actions ----------------
    def toggle_stream(self):
        self.streaming = not self.streaming
        self.play_btn.config(text="▶ Resume IMU" if not self.streaming else "⏸ Pause IMU")
        self._log2("IMU streaming: " + ("PAUSED" if not self.streaming else "RESUMED"))

    def reset_params(self):
        for s, val in [
            (self.gp_percent, 50), (self.alpha_deg, 0), (self.bw_gain, 1),
            (self.mu, 0.5), (self.sigma, 0.17),
            (self.c, 0.6), (self.d, 0.02), (self.p, 0.8),
            (self.a0, -10), (self.a1, 10), (self.krep, 1),
        ]:
            s.set(val)
        self.prev_tau = None
        self._update_model()
        self._log2("Parameters reset to defaults.")

    def save_figs(self):
        try:
            path = filedialog.asksaveasfilename(
                title="Save main figure as...",
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")]
            )
            if not path:
                return
            self.fig.savefig(path, dpi=150, bbox_inches="tight")
            base, ext = path.rsplit(".", 1)
            imu_path = f"{base}_imu.{ext}"
            self.fig_imu.savefig(imu_path, dpi=150, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Figures saved:\n• {path}\n• {imu_path}")
            self._log2(f"Saved figures to {path} and {imu_path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))
            self._log2(f"Save error: {e}")

    def save_params(self):
        m = self._get_sliders()
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            self._log2("Save params canceled.")
            return
        try:
            with open(path, "w") as f:
                json.dump(m, f, indent=2)
            self._log2(f"Parameters saved: {path}")
        except Exception as e:
            messagebox.showerror("Save Params Error", str(e))
            self._log2(f"Save params error: {e}")

    def load_params(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            self._log2("Load params canceled.")
            return
        try:
            with open(path) as f:
                m = json.load(f)
            self._set_sliders(m)
            self._log2(f"Parameters loaded: {path}")
        except Exception as e:
            messagebox.showerror("Load Params Error", str(e))
            self._log2(f"Load params error: {e}")

    def freeze_curve(self):
        GP = np.linspace(0, 1, 600)
        alpha = np.deg2rad(float(self.alpha_deg.get()))
        self.prev_tau = (GP, self.model.tau(GP, alpha))
        self.update_static_plots()
        self._log2("Current τ curve frozen for comparison.")

    def apply_preset(self):
        name = self.preset_var.get()
        self._set_sliders(self.presets[name])
        self._log2(f"Preset applied: {name}")

    def _set_sliders(self, m):
        self.gp_percent.set(m["gp"])
        self.alpha_deg.set(m["alpha"])
        self.bw_gain.set(m["bw"])
        self.mu.set(m["mu"])
        self.sigma.set(m["sigma"])
        self.c.set(m["c"])
        self.d.set(m["d"])
        self.p.set(m["p"])
        self.a0.set(m["a0"])
        self.a1.set(m["a1"])
        self.krep.set(m["krep"])
        self._update_model()

    def _get_sliders(self):
        return dict(
            gp=self.gp_percent.get(), alpha=self.alpha_deg.get(), bw=self.bw_gain.get(),
            mu=self.mu.get(), sigma=self.sigma.get(), c=self.c.get(), d=self.d.get(), p=self.p.get(),
            a0=self.a0.get(), a1=self.a1.get(), krep=self.krep.get()
        )

    def _update_model(self):
        m = self.model
        m.mu = float(self.mu.get())
        m.sigma = float(self.sigma.get())
        m.c = float(self.c.get())
        m.d = float(self.d.get())
        m.p = float(self.p.get())
        m.BW_gain = float(self.bw_gain.get())
        m.alpha0 = np.deg2rad(float(self.a0.get()))
        m.alpha1 = np.deg2rad(float(self.a1.get()))
        m.K_rep = float(self.krep.get())
        self.update_static_plots()

    def _shade_gait(self):
        phases = [(0, 10, "HS"), (10, 50, "Stance"), (50, 60, "TO"), (60, 100, "Swing")]
        for a, b, name in phases:
            self.ax_tau.axvspan(a, b, alpha=0.07)
            self.ax_tau.text(
                (a + b) / 2,
                0.98,
                name,
                transform=self.ax_tau.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
            )

    def _on_move_tau(self, ev):
        if ev.inaxes != self.ax_tau:
            return
        self.update_static_plots()
        try:
            self.ax_tau.axvline(ev.xdata, ls=":", lw=1)
            self.ax_tau.axhline(ev.ydata, ls=":", lw=1)
            self.canvas.draw_idle()
        except Exception:
            pass

    def update_static_plots(self):
        GP = np.linspace(0, 1, 600)
        alpha = np.deg2rad(float(self.alpha_deg.get()))
        gp_now = float(self.gp_percent.get()) / 100.0
        tau = self.model.tau(GP, alpha)
        

        self.ax_tau.clear()
        self.ax_tau.plot(GP * 100.0, tau, lw=2, label="τ(GP, α)")
        self.ax_tau.axvline(gp_now * 100.0, ls="--", lw=1)
        tau_now = self.model.tau(gp_now, alpha)
        self.ax_tau.scatter(gp_now * 100.0, tau_now, s=55, zorder=5, label="Now")

        SAFE_MIN, SAFE_MAX = -0.6, 0.8
        self.ax_tau.axhspan(SAFE_MIN, SAFE_MAX, alpha=0.08)
        self.ax_tau.text(
            0.99,
            0.99,
            "Safe band",
            transform=self.ax_tau.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )

        self.ax_tau.grid(True, which="major", alpha=0.30)
        self.ax_tau.grid(True, which="minor", alpha=0.15)
        self.ax_tau.minorticks_on()
        self.ax_tau.set_xlim(0, 100)
        if not self.lock_tau.get():
            pad = (np.max(np.abs(tau)) * 0.15) + 1e-9
            self.ax_tau.set_ylim(np.min(tau) - pad, np.max(tau) + pad)
        self.ax_tau.set_title("Torque τ(GP, α)")
        self.ax_tau.set_xlabel("Gait phase GP [%]")
        self.ax_tau.set_ylabel("Torque (scaled)")

        self._shade_gait()

        self.ax_phase.clear()
        self.ax_phase.plot([0, 100], [0.5, 0.5], lw=2)
        self.ax_phase.scatter([gp_now * 100.0], [0.5], s=60, zorder=5)
        self.ax_phase.set_xlim(0, 100)
        self.ax_phase.get_yaxis().set_visible(False)
        self.ax_phase.set_xticks([0, 25, 50, 75, 100])
        self.ax_phase.tick_params(axis="x", labelsize=8)

        self.ax_f.clear()
        self.ax_f.plot(GP * 100.0, self.model.f(GP))
        self.ax_f.set_title("f(GP)")
        self.ax_f.grid(True, alpha=0.3)

        self.ax_s.clear()
        self.ax_s.plot(GP * 100.0, self.model.s(GP))
        self.ax_s.set_title("s(GP)")
        self.ax_s.grid(True, alpha=0.3)

        if self.prev_tau is not None:
            GPp, taup = self.prev_tau
            self.ax_tau.plot(GPp * 100.0, taup, lw=1, linestyle="--", label="Frozen τ")
            
        try:
            global app
            live_gp = getattr(app, "last_gp", None)
            live_alpha = getattr(app, "angle_alpha", None)
        except NameError:
            live_gp = None
            live_alpha = None
        
        if live_gp is not None and live_alpha is not None:
            try:
                alpha_live_rad = math.radians(float(live_alpha))
                tau_live = self.model.tau(float(live_gp), alpha_live_rad)
                # marker "x" so you can distinguish it from the slider point
                self.ax_tau.scatter(float(live_gp) * 100.0, tau_live, s=40, marker="x", c="red", zorder=6, label="Live tau(GP, alpha_encoder)")
            except Exception:
                pass
            

        self.ax_tau.legend(loc="upper left", fontsize=8)
        self.status.config(
            text=f"GP={self.gp_percent.get():.1f}% | α={self.alpha_deg.get():.1f}° | BW={self.bw_gain.get():.2f}"
        )
        self.canvas.draw_idle()

    def _plot_update_loop(self):
        """
        Periodically update the IMU plots using REAL data from the Application
        (accel_x_G, accel_y_G, accel_z_G, gyro_x, gyro_z, angle_theta_q/angle_theta,
         angle_alpha_q, and encoder angle_alpha, plus NN outputs).
        """
        if self.streaming:
            try:
                t = time.perf_counter() - self._t0
                self.tbuf.append(t)

                # Default values (in case app is not available yet)
                ax = ay = az = wx = wz = th = al_q = al_enc = 0.0
                gp_now = cos_now = sin_now = 0.0

                try:
                    # Read live data from the global Application instance
                    global app
                    ax = getattr(app, "accel_x_G", 0.0)
                    ay = getattr(app, "accel_y_G", 0.0)
                    az = getattr(app, "accel_z_G", 0.0)
                    wx = getattr(app, "gyro_x", 0.0)
                    wz = getattr(app, "gyro_z", 0.0)
                    th = getattr(app, "angle_theta_q", getattr(app, "angle_theta", 0.0))
                    al_q = getattr(app, "angle_alpha_q", 0.0)
                    al_enc = getattr(app, "angle_alpha", 0.0)

                    # NN outputs
                    gp_now  = getattr(app, "last_gp", 0.0)
                    cos_now = getattr(app, "last_cos_filt", 0.0)
                    sin_now = getattr(app, "last_sin_filt", 0.0)
                except NameError:
                    # app not defined yet; keep zeros
                    pass
                # --- estimated torque from GP + encoder angle ---
                try:
                    alpha_enc_rad = math.radians(float(al_enc))
                    tau_now = self.model.tau(float(gp_now), alpha_enc_rad)
                except Exception:
                    tau_now = 0.0

                self.torque_buf.append(tau_now)

                # Append to buffers
                self.accX.append(ax)
                self.accY.append(ay)
                self.accZ.append(az)
                self.angX.append(wx)
                self.angZ.append(wz)
                self.theta.append(th)
                self.alphaZ.append(al_q)
                self.alpha_enc.append(al_enc)

                # NN buffers
                self.gp_series.append(gp_now)
                self.cos_series.append(cos_now)
                self.sin_series.append(sin_now)

                # Time axis for rolling window
                if self.tbuf:
                    t0_view = self.tbuf[-1] - self.window_sec
                    x = np.array(self.tbuf) - max(t0_view, 0)
                else:
                    x = np.array([0.0])

                self._draw_imu(x)
            except Exception as e:
                self._log2(f"IMU plot update error: {e}")
        self.update_static_plots()

        self.after(40, self._plot_update_loop)

    def _draw_imu(self, x):
        # ===========================
        # TOP-LEFT: NN outputs + Gait Phase
        # ===========================
        self.ax_nn.clear()

        gp_vals   = list(self.gp_series)
        cos_vals  = list(self.cos_series)
        sin_vals  = list(self.sin_series)

        n = len(x)
        if len(gp_vals)  < n: gp_vals  = [0.0] * (n - len(gp_vals))  + gp_vals
        if len(cos_vals) < n: cos_vals = [0.0] * (n - len(cos_vals)) + cos_vals
        if len(sin_vals) < n: sin_vals = [0.0] * (n - len(sin_vals)) + sin_vals

        self.ax_nn.plot(x, cos_vals, label="cos_filtered", linewidth=1.2)
        self.ax_nn.plot(x, sin_vals, label="sin_filtered", linewidth=1.2)
        self.ax_nn.plot(x, gp_vals,  label="GP (0–1)", linewidth=1.5)

        self.ax_nn.set_title("NN Outputs & Estimated Gait Phase")
        self.ax_nn.set_ylabel("Value")
        self.ax_nn.set_xlim(0, self.window_sec)
        self.ax_nn.grid(True, alpha=0.25)
        self.ax_nn.legend(loc="upper right", fontsize=8)

        # ===========================
        # TOP-RIGHT: Accelerations in g
        # ===========================
        self.ax_accel.clear()
        self.ax_accel.plot(x, list(self.accX), label="accX [g]")
        self.ax_accel.plot(x, list(self.accY), label="accY [g]")
        self.ax_accel.plot(x, list(self.accZ), label="accZ [g]")
        self.ax_accel.legend(loc="upper right", fontsize=8)
        self.ax_accel.set_ylabel("g")
        self.ax_accel.set_title("Accelerometer [X, Y, Z]")
        self.ax_accel.set_xlim(0, self.window_sec)
        self.ax_accel.grid(True, alpha=0.25)

        # ===========================
        # BOTTOM-LEFT: Angular speeds
        # ===========================
        self.ax_speed.clear()
        self.ax_speed.plot(x, list(self.angX), label="ωx")
        self.ax_speed.plot(x, list(self.angZ), label="ωz")
        self.ax_speed.plot(x, list(self.torque_buf), label="τ (est.)", linewidth=1.5)
        
        self.ax_speed.legend(loc="upper right", fontsize=8)
        self.ax_speed.set_ylabel("rad/s")
        self.ax_speed.set_title("Angular Speed [X, Z]")
        self.ax_speed.set_xlim(0, self.window_sec)
        self.ax_speed.grid(True, alpha=0.25)

        # ===========================
        # BOTTOM-RIGHT: Angles
        # ===========================
        self.ax_angle.clear()
        self.ax_angle.plot(x, list(self.theta),      label="θ (IMU)")
        self.ax_angle.plot(x, list(self.alphaZ),     label="α (IMU)")
        self.ax_angle.plot(x, list(self.alpha_enc),  label="α (encoder)")
        self.ax_angle.legend(loc="upper right", fontsize=8)
        self.ax_angle.set_ylabel("deg")
        self.ax_angle.set_xlabel("Time [s]")
        self.ax_angle.set_title("Angle Position [θ, α_IMU, α_enc]")
        self.ax_angle.set_xlim(0, self.window_sec)
        self.ax_angle.grid(True, alpha=0.25)

        self.canvas_imu.draw_idle()


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ankle Exoskeleton Control GUI By Tinsae Tesfamichael")
        self.geometry("1000x600")
        self.minsize(1200, 800)

        self.no_row = 0
        self.no_col = 0

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
        self.enc_ok    = 1

        # Shared NN outputs for plots
        self.last_gp = 0.0
        self.last_cos_filt = 0.0
        self.last_sin_filt = 0.0

        self.pre = Preprocessor50Hz(n_features=7, max_rows=10, cutoff_hz=5.0, fs=50.0)

        self.is_it_safe = False
        self.desired_cmd_id = CMD_IDLE
        self.requested_torque_gui = 0.0
        self.requested_speed_gui  = 0.0
        self.direction_flag       = 1

        self.connected_var = tk.BooleanVar(value=False)
        self.current_port = tk.StringVar(value=DEFAULT_PORT)
        self.current_baud = tk.IntVar(value=DEFAULT_BAUD)

        self.encoder_cal_status = tk.StringVar(value="Not calibrated")
        self.sys_cal_status     = tk.StringVar(value="Not calibrated")
        self.gyro_cal_status    = tk.StringVar(value="Not calibrated")
        self.mag_cal_status     = tk.StringVar(value="Not calibrated")
        self.accel_cal_status   = tk.StringVar(value="Not calibrated")

        # ---- calibration logging state ----
        self._calibrating = False
        self._last_calib_tuple_logged = (-1, -1, -1, -1)

        self._init_styles()

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        container = tk.Frame(self)
        container.grid(row=0, column=0, sticky="nsew")

        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(container, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        vscroll = ttk.Scrollbar(container, orient="vertical",
                                command=self.canvas.yview)
        vscroll.grid(row=0, column=1, sticky="ns")

        self.canvas.configure(yscrollcommand=vscroll.set)

        self.main_frame = tk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window(
            (0, 0),
            window=self.main_frame,
            anchor="nw"
        )

        def _on_frame_configure(event):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        self.main_frame.bind("<Configure>", _on_frame_configure)

        def _on_canvas_configure(event):
            self.canvas.itemconfig(self.canvas_window, width=event.width)

        self.canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)

        for r in range(6):
            self.main_frame.grid_rowconfigure(r, weight=0)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.build_mainbar()
        self.build_logobar()
        self.build_toolbar()
        self.build_calibration_bar()
        self.Tabs()
        self.build_controls()
        self.build_inputs()
        self.status_bar()
        self.build_log()

        threading.Thread(target=self._reader_worker, daemon=True).start()
        threading.Thread(target=self._nn_worker, daemon=True).start()

        self._refresh_ports_combo()
        self.after(10, self.poll_serial_binary)
        self.after(250, self._update_status)

    def _init_styles(self):
        s = ttk.Style(self)
        try:
            s.theme_use("aquativo")
        except tk.TclError:
            pass
        s.configure("Ok.TLabel",   foreground="#059669")
        s.configure("Warn.TLabel", foreground="#dc2626")
        s.configure("Info.TLabel", foreground="#374151")

    def build_mainbar(self):
        self.main_bar = tk.Frame(self.main_frame, bd=0)
        self.main_bar.grid(row=self.no_row, column=self.no_col, sticky="ew")
        for c in range(1):
            self.main_bar.columnconfigure(c, weight=1)

    def build_logobar(self):
        logo_frame = tk.Frame(self.main_bar, bd=0, relief=tk.RIDGE)
        logo_frame.grid(row=0, column=0, rowspan=2, columnspan=4, sticky="ew")
        if Image and ImageTk:
            try:
                img = Image.open(r"C:\Users\Tinsae Tesfamichael\Desktop\Thesis\[_Final_code]\Final_Code\Final\src\icon.png")
                dimensions_w = img.width // 6
                dimensions_h = img.height // 6
                try:
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
        self.toolbar = tk.Frame(self.main_bar, bd=1)
        self.toolbar.grid(row=self.no_row+2, column=self.no_col, sticky="ew")

        for c in range(8):
            self.toolbar.columnconfigure(c, weight=0)
        self.toolbar.columnconfigure(8, weight=1)

        ttk.Label(self.toolbar, text="Port").grid(row=0, column=0, sticky="w")
        self.port_combo = ttk.Combobox(
            self.toolbar,
            textvariable=self.current_port,
            width=15,
            values=["COM1", "COM2", "COM3", "COM4", "/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyS0"],
        )
        self.port_combo.grid(row=0, column=1, padx=(4, 10), sticky="w")

        ttk.Label(self.toolbar, text="Baud").grid(row=0, column=2, sticky="w")
        self.baud_combo = ttk.Combobox(
            self.toolbar,
            textvariable=self.current_baud,
            width=10,
            values=[9600, 19200, 38400, 57600, 115200, 230400],
        )
        self.baud_combo.grid(row=0, column=3, padx=(4, 10), sticky="w")

        self.connect_btn = ttk.Button(self.toolbar, text="Connect", command=self._connect_clicked, state="normal")
        self.connect_btn.grid(row=0, column=4, padx=2)

        self.disconnect_btn = ttk.Button(self.toolbar, text="Disconnect", command=self._disconnect_clicked, state="disabled")
        self.disconnect_btn.grid(row=0, column=5, padx=2)

        self.safe_btn = tk.Button(self.toolbar, text="SAFE LATCH: OFF", command=self.on_isitsafe, width=18, bg="#ff0000")
        self.safe_btn.grid(row=0, column=6, padx=(14, 0))

        self.conn_label = ttk.Label(self.toolbar, text="Not connected")
        self.conn_label.grid(row=0, column=8, sticky="e")

    def build_calibration_bar(self):
        cal_bar = ttk.LabelFrame(self.main_bar, text="Calibration status", padding=(8, 6))
        cal_bar.grid(row=self.no_row+3, column=0, columnspan=2, sticky="ew", padx=4, pady=0)
        for c in range(10):
            cal_bar.columnconfigure(c, weight=1)

        ttk.Label(cal_bar, text="Encoder:").grid(row=0, column=0, sticky="e", padx=(0, 6))
        self.encoder_cal_lbl = ttk.Label(cal_bar, textvariable=self.encoder_cal_status, style="Warn.TLabel")
        self.encoder_cal_lbl.grid(row=0, column=1, sticky="w")

        ttk.Label(cal_bar, text="System:").grid(row=0, column=2, sticky="e", padx=(10, 6))
        self.sys_cal_lbl = ttk.Label(cal_bar, textvariable=self.sys_cal_status, style="Warn.TLabel")
        self.sys_cal_lbl.grid(row=0, column=3, sticky="w")

        ttk.Label(cal_bar, text="Gyroscope:").grid(row=0, column=4, sticky="e", padx=(10, 6))
        self.gyro_cal_lbl = ttk.Label(cal_bar, textvariable=self.gyro_cal_status, style="Warn.TLabel")
        self.gyro_cal_lbl.grid(row=0, column=5, sticky="w")

        ttk.Label(cal_bar, text="Accelerometer:").grid(row=0, column=6, sticky="e", padx=(10, 6))
        self.accel_cal_lbl = ttk.Label(cal_bar, textvariable=self.accel_cal_status, style="Warn.TLabel")
        self.accel_cal_lbl.grid(row=0, column=7, sticky="w")

        ttk.Label(cal_bar, text="Magnetometer:").grid(row=0, column=8, sticky="e", padx=(10, 6))
        self.mag_cal_lbl = ttk.Label(cal_bar, textvariable=self.mag_cal_status, style="Warn.TLabel")
        self.mag_cal_lbl.grid(row=0, column=9, sticky="w")

    def Tabs(self):
        tabs_bar_main = ttk.Notebook(self.main_frame)
        tabs_bar_main.grid(row=self.no_row+4, column=self.no_col, sticky="nsew")
        self.main_frame.grid_rowconfigure(self.no_row+4, weight=1)

        self.main_tab = ttk.Frame(tabs_bar_main)
        tabs_bar_main.add(self.main_tab, text="Main")
        self.main_tab.columnconfigure(0, weight=1)
        self.main_tab.columnconfigure(1, weight=1)

        self.plot_tab = ttk.Frame(tabs_bar_main)
        tabs_bar_main.add(self.plot_tab, text="Plots")

        scrollbar = ttk.Scrollbar(self.plot_tab, orient="vertical")
        scrollbar.pack(side="right", fill="y")

        canvas = tk.Canvas(self.plot_tab, yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar.config(command=canvas.yview)

        TorqueGUI(canvas)

    def build_controls(self):
        btns = ttk.LabelFrame(self.main_tab, text="Controls", padding=(8, 6))
        btns.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        for c in range(3):
            btns.columnconfigure(c, weight=1)

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
        inputs = ttk.LabelFrame(self.main_tab, text="Inputs", padding=(8, 6))
        inputs.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        for c in range(7):
            inputs.columnconfigure(c, weight=0)
        inputs.columnconfigure(6, weight=1)

        ttk.Label(inputs, text="Torque (raw)").grid(row=0, column=0, sticky="w")
        self.torque_in = tk.DoubleVar(value=self.requested_torque_gui)
        self.torque_spin = ttk.Spinbox(
            inputs,
            from_=-20000,
            to=20000,
            increment=1,
            width=10,
            textvariable=self.torque_in,
            state="disabled",
        )
        self.torque_spin.grid(row=0, column=1, padx=6, sticky="w")
        self.torque_set_btn = ttk.Button(inputs, text="Set", command=self.on_set_torque, state="disabled")
        self.torque_set_btn.grid(row=0, column=2, padx=(0, 8))

        ttk.Label(inputs, text="Speed cmd").grid(row=1, column=0, sticky="w")
        self.speed_in = tk.DoubleVar(value=self.requested_speed_gui)
        self.speed_spin = ttk.Spinbox(
            inputs,
            from_=0,
            to=20000,
            increment=0.5,
            width=10,
            textvariable=self.speed_in,
            state="disabled",
        )
        self.speed_spin.grid(row=1, column=1, padx=6, sticky="w")
        self.speed_set_btn = ttk.Button(inputs, text="Set", command=self.on_set_speed, state="disabled")
        self.speed_set_btn.grid(row=1, column=2, padx=(0, 8))

        self.dir_var = tk.IntVar(value=self.direction_flag)
        dir_frame = ttk.Frame(inputs)
        dir_frame.grid(row=2, column=0, columnspan=3, sticky="w")
        self.rb_fwd = ttk.Radiobutton(
            dir_frame, text="Forward", value=1, variable=self.dir_var,
            command=self.on_set_direction, state="disabled"
        )
        self.rb_rev = ttk.Radiobutton(
            dir_frame, text="Reverse", value=0, variable=self.dir_var,
            command=self.on_set_direction, state="disabled"
        )
        self.rb_fwd.pack(side="left")
        self.rb_rev.pack(side="left")

        self.inputs_status = ttk.Label(inputs, text="", style="Info.TLabel")
        self.inputs_status.grid(row=3, column=0, columnspan=7, sticky="w", pady=(6, 0))

        self.inputs_widgets = [
            self.torque_spin,
            self.speed_spin,
            self.rb_fwd,
            self.rb_rev,
            self.torque_set_btn,
            self.speed_set_btn,
        ]

    def status_bar(self):
        status = ttk.LabelFrame(self.main_tab, text="Status", padding=(8, 6))
        status.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        for c in range(2):
            status.columnconfigure(c, weight=1)

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
        log_frame = ttk.LabelFrame(self.main_tab, text="Log", padding=(8, 6))
        log_frame.grid(row=1, column=1, sticky="nsew", padx=0, pady=0)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log = tk.Text(log_frame, height=12, wrap="none", state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        vs = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        hs = ttk.Scrollbar(log_frame, orient="horizontal", command=self.log.xview)
        self.log.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")

        sb = ttk.Frame(self, padding=(8, 4))
        sb.grid(row=4, column=0, sticky="ew")
        sb.columnconfigure(0, weight=1)
        self.statusbar_msg = tk.StringVar(value="Ready")
        ttk.Label(sb, textvariable=self.statusbar_msg).grid(row=0, column=0, sticky="w")

    def _row(self, parent, r, label, var):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="e", padx=(0, 6))
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

    def quaternion_to_euler(self):
        qw = self.quaternion_w
        qx = self.quaternion_x
        qy = self.quaternion_y
        qz = self.quaternion_z
        t0 = +2.0 * (qw * qx + qy * qz)
        t1 = +1.0 - 2.0 * (qx * qx + qy * qy)
        roll_rad = math.atan2(t0, t1)
        t2 = +2.0 * (qw * qy - qz * qx)
        t2 = max(-1.0, min(1.0, t2))
        pitch_rad = math.asin(t2)
        t3 = +2.0 * (qw * qz + qx * qy)
        t4 = +1.0 - 2.0 * (qy * qy + qz * qz)
        yaw_rad = math.atan2(t3, t4)
        self.roll = roll_rad
        self.pitch = pitch_rad
        self.yaw = yaw_rad
        self.angle_alpha_q = math.degrees(self.yaw)
        self.angle_theta_q = math.degrees(roll_rad)
        self.angle_theta = self.angle_theta_q

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
            global RUN_NN

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

    def _update_safe_btn(self):
        if self.is_it_safe:
            self.safe_btn.configure(text="SAFE LATCH: ON", bg="#22cc55")
        else:
            self.safe_btn.configure(text="SAFE LATCH: OFF", bg="#ee4444")

    def on_isitsafe(self):
        self.is_it_safe = not self.is_it_safe
        self._update_safe_btn()
        if not self.is_it_safe:
            self.encoder_cal_status.set("Not calibrated")
            self.sys_cal_status.set("Not calibrated")
            self.gyro_cal_status.set("Not calibrated")
            self.mag_cal_status.set("Not calibrated")
            self.accel_cal_status.set("Not calibrated")
        self._log(f"Safety latch {'ON' if self.is_it_safe else 'OFF'}.")

    def on_set_torque(self):
        try:
            val = float(self.torque_in.get())
        except ValueError:
            messagebox.showerror("Invalid torque", "Enter a number for torque.")
            return
        self.requested_torque_gui = val
        self.inputs_status.config(text=f"Torque set to {val:.0f} (raw)")
        self._log(f"Requested torque = {val}")

    def on_set_speed(self):
        try:
            val = float(self.speed_in.get()) * 100.0
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
        self._tx_frame(
            frame,
            f"TX CONTROL τ={self.requested_torque_gui:.1f} spd={self.requested_speed_gui:.1f} dir={self.direction_flag}",
        )
        global RUN_NN, USE_FILTER
        USE_FILTER = True
        RUN_NN = True
        self.status_var.set("CONTROL mode")

    def on_stop(self):
        self.desired_cmd_id = CMD_STOP
        self._tx_frame(build_cmd_stop(), "TX STOP (EMERGENCY_STOP)")
        self.status_var.set("STOPPED")
        global RUN_NN, USE_FILTER
        RUN_NN = False
        USE_FILTER = False

    def on_zero(self):
        self.desired_cmd_id = CMD_OFFSETTING
        self._tx_frame(build_cmd_offsetting(), "TX OFFSETTING (encoder zero)")

    def on_calibrate(self):
        self.desired_cmd_id = CMD_CALIBRATE
        self._tx_frame(build_cmd_calibrate(), "TX CALIBRATE")
        # start continuous calibration logging
        self._calibrating = True
        self._last_calib_tuple_logged = (-1, -1, -1, -1)
        msg = "Calibration started... move the device as required until all levels reach 3."
        self._log(msg)
        # mirror to plots tab if available
        if LOG2_FUNC is not None:
            LOG2_FUNC(msg)
        self.status_var.set("Calibrating...")

    def on_request_status(self):
        self.desired_cmd_id = CMD_READING
        self._tx_frame(build_cmd_reading(), "TX READING / Status Request")
        self.status_var.set("Requesting Status...")
        at = getattr(self, "angle_theta_q", 0.0)
        self._log(
            f"RX {self.status_var.get()} θ={math.radians(at):.2f}rad α={math.radians(self.angle_alpha):.2f}rad τ={self.torque_meas:.1f} T={self.temp_meas:.2f}°C spd={self.speed_meas:.2f}deg/s"
        )

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
        self.destroy()

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
                self.angle_alpha = parsed["angleAlpha"]
                self.speed_meas  = parsed["speed"]
                self.torque_meas = parsed["torque"]
                self.temp_meas   = parsed["temp"]
                self.angle_alpha_rads = math.radians(self.angle_alpha)

                self.accel_x = parsed["accelX"]
                self.accel_y = parsed["accelY"]
                self.accel_z = parsed["accelZ"]
                self.gyro_x  = parsed["gyroX"]
                self.gyro_z  = parsed["gyroZ"]

                self.quaternion_w = parsed["quatW"]
                self.quaternion_x = parsed["quatX"]
                self.quaternion_y = parsed["quatY"]
                self.quaternion_z = parsed["quatZ"]

                self.imu_fully = int(parsed["imuFully"])
                self.enc_ok    = int(parsed["encOK"])

                self.quaternion_to_euler()
                self.status_var.set(state_code_to_name(parsed["state_code"]))

                # Convert to g for plotting
                self.accel_x_G = -self.accel_x / 9.8
                self.accel_y_G = -self.accel_y / 9.8
                self.accel_z_G = self.accel_z / 9.8

                if RUN_NN:
                    streaming_data = np.array(
                        [
                            self.accel_x_G,
                            self.accel_y_G,
                            self.accel_z_G,
                            self.gyro_x,
                            self.gyro_z,
                            math.radians(self.angle_alpha_q),
                            math.radians(self.angle_theta_q),
                        ],
                        dtype=float,
                    )

                    with data_lock:
                        global data_in
                        data_in = self.pre.step(streaming_data)

                    new_data_available.set()

        self.after(10, self.poll_serial_binary)

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

        global cSys_level, cG_level, cA_level, cM_level
        self.calib_tuple_var.set(f"(S,G,A,M) = {cSys_level},{cG_level},{cA_level},{cM_level}")

        if cSys_level == 3:
            self.sys_cal_status.set("Fully calibrated")
        elif cSys_level > 0:
            self.sys_cal_status.set("Calibrating...")
        else:
            self.sys_cal_status.set("Not calibrated")

        if cG_level == 3:
            self.gyro_cal_status.set("Fully calibrated")
        elif cG_level > 0:
            self.gyro_cal_status.set("Calibrating...")
        else:
            self.gyro_cal_status.set("Not calibrated")

        if cA_level == 3:
            self.accel_cal_status.set("Fully calibrated")
        elif cA_level > 0:
            self.accel_cal_status.set("Calibrating...")
        else:
            self.accel_cal_status.set("Not calibrated")

        if cM_level == 3:
            self.mag_cal_status.set("Fully calibrated")
        elif cM_level > 0:
            self.mag_cal_status.set("Calibrating...")
        else:
            self.mag_cal_status.set("Not calibrated")

        if self.enc_ok == 1:
            self.encoder_cal_status.set("Fully calibrated")

        # ---- continuous calibration logging ----
        calib_tuple = (cSys_level, cG_level, cA_level, cM_level)
        if self._calibrating and calib_tuple != self._last_calib_tuple_logged:
            self._last_calib_tuple_logged = calib_tuple
            msg = (
                f"Calibration levels -> System:{cSys_level} "
                f"Gyro:{cG_level} Accel:{cA_level} Mag:{cM_level}"
            )
            self._log(msg)
            if LOG2_FUNC is not None:
                LOG2_FUNC(msg)

        # stop calibration mode once fully calibrated
        if self._calibrating and (cSys_level == 3 and cG_level == 3 and cA_level == 3 and cM_level == 3):
            self._calibrating = False
            msg = "Calibration complete: all sensors fully calibrated (S=3,G=3,A=3,M=3)."
            self._log(msg)
            if LOG2_FUNC is not None:
                LOG2_FUNC(msg)
            self.status_var.set("Calibrated")

        self.after(250, self._update_status)

    def _nn_worker(self):
        """
        Background worker for the neural network:
        - Waits for new standardized data (new_data_available event)
        - If RUN_NN is True, runs the model and updates last_cos_filt / last_sin_filt / last_gp
        - Logs NN status to the main log
        """
        global response_bytes, current_reply, RUN_NN
        model = None

        while True:
            # Wait until streaming code gives us a new 10x7 window
            new_data_available.wait()
            new_data_available.clear()

            # If NN is not enabled, just ignore this batch
            if not RUN_NN:
                continue

            # Lazy-load model once (with patched LSTM to ignore time_major)
            if model is None and load_model is not None:
                try:
                    if PatchedLSTM is not None:
                        model = load_model(
                            MODEL_PATH,
                            custom_objects={"LSTM": PatchedLSTM},
                        )
                    else:
                        model = load_model(MODEL_PATH)

                    self._log(f"NN loaded: {MODEL_PATH}")
                except Exception as e:
                    self._log(f"NN load failed: {e}")
                    model = None
                    # Don't crash thread; just skip this batch
                    continue

            if model is None:
                # load_model is None or failed – nothing to do
                continue

            try:
                
                with data_lock:
                    window = data_in.copy()

                # Ensure shape is (10, 7)
                if window.shape[0] < 10:
                    pad_rows = 10 - window.shape[0]
                    window = np.vstack([np.zeros((pad_rows, 7), dtype=np.float32), window])

                # Run NN
                raw_out = model(window.reshape((1, 10, 7)), training=False).numpy()

                # Handle both possible output shapes:
                #  - (batch, 2)              -> direct
                #  - (batch, timesteps, 2)   -> take last timestep
                if raw_out.ndim == 3:
                    # (1, T, 2) -> (1, 2) by taking last time step
                    raw_out = raw_out[:, -1, :]
                elif raw_out.ndim != 2:
                    self._log(f"Unexpected NN output shape: {raw_out.shape}")
                    continue

                NN_out = raw_out[0]  # shape (2,)
                cos_val = float(NN_out[0])
                sin_val = float(NN_out[1])

                # Filter and normalize to unit circle
                cos_filt, sin_filt = filter_sin_cos(
                    cos_val, sin_val,
                    alpha=ALPHA,
                    enable_filter=USE_FILTER
                )

                # Pack for Arduino (if needed)
                cos_i = int(max(-1.0, min(1.0, cos_filt)) * CONVERTER)
                sin_i = int(max(-1.0, min(1.0, sin_filt)) * CONVERTER)
                packed = struct.pack('>ii', cos_i, sin_i)
                response_bytes = packed
                current_reply  = packed

                # Gait phase
                gait_phase = GP_estimation(sin_filt, cos_filt)

                # Expose to GUI plots
                self.last_gp = gait_phase
                self.last_cos_filt = cos_filt
                self.last_sin_filt = sin_filt
                
                

                self._log(
                    f"NN cos={cos_val:.3f} sin={sin_val:.3f} | "
                    f"filt cos={cos_filt:.3f} sin={sin_filt:.3f} | "
                    f"GP={gait_phase:.3f}"
                )
                
                
            except Exception as e:
                self._log(f"NN worker error: {e}")


if __name__ == "__main__":
    app = Application()
    app.mainloop()
