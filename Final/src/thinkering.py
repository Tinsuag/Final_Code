import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import math
import traceback
import numpy as np

# If pyserial isn't installed, GUI still opens
try:
    import serial
    from serial.tools import list_ports
except Exception:
    serial = None
    list_ports = None

# ======================
# USER SERIAL CONFIG (defaults; can be changed in GUI)
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
CONVERTER  = 2 ** 16
USE_FILTER = False
ALPHA      = 0.8

data_in = np.zeros((10, 7))
current_reply = None
data_lock = threading.Lock()
new_data_available = threading.Event()

# ======================
# PROTOCOL CONSTANTS
# ======================
START_BYTE   = 0xAA
PKT_CMD_ID   = 0x20
PKT_TELEM_ID = 0x30

# ======================
# SERIAL HANDLE (managed by GUI)
# ======================
SER = None


def safe_open_serial(port, baud):
    """Open serial port safely; return serial or None."""
    global SER
    if serial is None:
        return None
    try:
        ser = serial.Serial(port, baud, timeout=0.01)
        time.sleep(2.0)  # give MCU time to reset
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


class MiniGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Control Panel — Protocol Table Edition")
        self.geometry("1120x760")
        self.minsize(980, 640)

        # ---------- palette / style ----------
        self.style = ttk.Style(self)
        # Prefer 'clam' for consistent ttk rendering
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Colors for small LED indicators
        self.COLOR_OK = "#22cc55"
        self.COLOR_BAD = "#ee4444"
        self.COLOR_WARN = "#ffcc33"
        self.COLOR_DIM = "#888888"

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
        self.yaw   = 0.0
        self.pitch = 0.0
        self.roll  = 0.0

        self.imu_fully = 0
        self.enc_ok    = 0

        # ---------- command-side state ----------
        self.is_it_safe = False
        self.desired_cmd_id = CMD_IDLE
        self.requested_torque_gui = 200.0
        self.requested_speed_gui  = 0.0
        self.direction_flag       = 1  # 1 fwd, 0 rev

        # ---------- connection state ----------
        self.connected_var = tk.BooleanVar(value=False)
        self.current_port = tk.StringVar(value=DEFAULT_PORT)
        self.current_baud = tk.IntVar(value=DEFAULT_BAUD)

        # ---------- top-level layout ----------
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)

        self._build_menubar()
        self._build_toolbar_connection()   # connection controls
        self._build_main_panels()          # inputs, buttons, status, orientation, log
        self._build_statusbar()            # bottom status bar

        # ---------- background tasks ----------
        self._last_packet = None
        self.reader_thread = threading.Thread(target=self._reader_worker, daemon=True)
        self.reader_thread.start()

        self.after(150, self.poll_serial_binary)
        self.after(200, self._update_status)
        self.after(500, self._refresh_ports_combo)

        # Keyboard shortcuts
        self.bind_all("<space>", lambda e: self.on_isitsafe())
        self.bind_all("<s>",     lambda e: self.on_start())
        self.bind_all("<x>",     lambda e: self.on_stop())
        self.bind_all("<i>",     lambda e: self.on_init())
        self.bind_all("<c>",     lambda e: self.on_calibrate())
        self.bind_all("<z>",     lambda e: self.on_zero())
        self.bind_all("<r>",     lambda e: self.on_request_status())

        self.protocol("WM_DELETE_WINDOW", self.on_exit)

    # ================= UI BUILDERS =================
    def _build_menubar(self):
        m = tk.Menu(self)
        filem = tk.Menu(m, tearoff=0)
        filem.add_command(label="Connect", command=self._connect_clicked, accelerator="Ctrl+K")
        filem.add_command(label="Disconnect", command=self._disconnect_clicked, accelerator="Ctrl+Shift+K")
        filem.add_separator()
        filem.add_command(label="Exit", command=self.on_exit)
        m.add_cascade(label="File", menu=filem)

        self.bind_all("<Control-k>", lambda e: self._connect_clicked())
        self.bind_all("<Control-K>", lambda e: self._connect_clicked())
        self.bind_all("<Control-Shift-k>", lambda e: self._disconnect_clicked())
        self.bind_all("<Control-Shift-K>", lambda e: self._disconnect_clicked())
        self.config(menu=m)

    def _build_toolbar_connection(self):
        bar = ttk.Frame(self, padding=(8, 6))
        bar.grid(row=0, column=0, sticky="ew")
        for c in range(8):
            bar.columnconfigure(c, weight=0)
        bar.columnconfigure(7, weight=1)

        ttk.Label(bar, text="Port").grid(row=0, column=0, sticky="w")
        self.port_combo = ttk.Combobox(bar, textvariable=self.current_port, width=14, state="normal")
        self.port_combo.grid(row=0, column=1, padx=(4, 10), sticky="w")

        ttk.Label(bar, text="Baud").grid(row=0, column=2, sticky="w")
        self.baud_combo = ttk.Combobox(bar, textvariable=self.current_baud, width=10,
                                       values=[9600, 19200, 38400, 57600, 115200, 230400, 460800])
        self.baud_combo.grid(row=0, column=3, padx=(4, 10), sticky="w")

        self.connect_btn = ttk.Button(bar, text="Connect", command=self._connect_clicked)
        self.connect_btn.grid(row=0, column=4, padx=2)

        self.disconnect_btn = ttk.Button(bar, text="Disconnect", command=self._disconnect_clicked, state="disabled")
        self.disconnect_btn.grid(row=0, column=5, padx=2)

        # Safety latch big toggle – use tk.Button for reliable bg color
        self.safe_btn = tk.Button(bar, text="SAFE LATCH: OFF (press SPACE)",
                                  command=self.on_isitsafe, width=22, relief="raised")
        self.safe_btn.grid(row=0, column=6, padx=(14, 0))
        self._update_safe_btn_color()

        self.conn_label = ttk.Label(bar, text="Not connected", foreground=self.COLOR_DIM)
        self.conn_label.grid(row=0, column=7, sticky="e")

    def _build_main_panels(self):
        wrap = ttk.Frame(self, padding=(8, 4))
        wrap.grid(row=1, column=0, sticky="nsew")
        wrap.rowconfigure(1, weight=1)
        wrap.columnconfigure(0, weight=1)
        wrap.columnconfigure(1, weight=1)

        # ------ Inputs ------
        inputs = ttk.LabelFrame(wrap, text="Inputs", padding=8)
        inputs.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        inputs.columnconfigure(6, weight=1)

        ttk.Label(inputs, text="Torque (raw)").grid(row=0, column=0, sticky="w")
        self.torque_in = tk.DoubleVar(value=self.requested_torque_gui)
        ttk.Spinbox(inputs, from_=-10000, to=10000, increment=1, width=10,
                    textvariable=self.torque_in).grid(row=0, column=1, padx=6, sticky="w")
        ttk.Button(inputs, text="Set", command=self.on_set_torque).grid(row=0, column=2, padx=(0, 8))

        ttk.Label(inputs, text="Speed cmd").grid(row=0, column=3, sticky="w")
        self.speed_in = tk.DoubleVar(value=self.requested_speed_gui)
        ttk.Spinbox(inputs, from_=-10000, to=10000, increment=0.5, width=10,
                    textvariable=self.speed_in).grid(row=0, column=4, padx=6, sticky="w")
        ttk.Button(inputs, text="Set", command=self.on_set_speed).grid(row=0, column=5, padx=(0, 8))

        # Direction radio
        self.dir_var = tk.IntVar(value=1)
        dir_frame = ttk.Frame(inputs)
        dir_frame.grid(row=0, column=6, sticky="w")
        ttk.Radiobutton(dir_frame, text="Forward", value=1, variable=self.dir_var,
                        command=self.on_set_direction).pack(side="left")
        ttk.Radiobutton(dir_frame, text="Reverse", value=0, variable=self.dir_var,
                        command=self.on_set_direction).pack(side="left")

        # ------ Control buttons ------
        btns = ttk.LabelFrame(wrap, text="Controls", padding=8)
        btns.grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        for c in range(6):
            btns.columnconfigure(c, weight=1)

        self.start_btn = ttk.Button(btns, text="Start (S)", command=self.on_start, state="disabled")
        self.stop_btn  = ttk.Button(btns, text="Stop (X)", command=self.on_stop, state="disabled")
        self.init_btn  = ttk.Button(btns, text="Init (I)", command=self.on_init, state="disabled")
        self.cal_btn   = ttk.Button(btns, text="Calibrate (C)", command=self.on_calibrate, state="disabled")
        self.read_btn  = ttk.Button(btns, text="Request Status (R)", command=self.on_request_status, state="disabled")
        self.idle_btn  = ttk.Button(btns, text="Idle", command=self.on_idle, state="disabled")
        self.zero_btn  = ttk.Button(btns, text="Off-set Encoder (Z)", command=self.on_zero, state="disabled")
        self.clear_btn = ttk.Button(btns, text="Clear Log", command=self.on_clear_log)
        self.exit_btn  = ttk.Button(btns, text="Exit", command=self.on_exit)

        self.start_btn.grid(row=0, column=0, padx=2, pady=2)
        self.stop_btn.grid(row=0, column=1, padx=2, pady=2)
        self.idle_btn.grid(row=0, column=2, padx=2, pady=2)
        self.init_btn.grid(row=1, column=0, padx=2, pady=2)
        self.cal_btn.grid(row=1, column=1, padx=2, pady=2)
        self.read_btn.grid(row=1, column=2, padx=2, pady=2)
        self.zero_btn.grid(row=1, column=3, padx=2, pady=2)
        self.clear_btn.grid(row=2, column=0, padx=2, pady=2)
        self.exit_btn.grid(row=2, column=1, padx=2, pady=2)

        # ------ Status + Orientation + Log (two rows) ------
        lower_left = ttk.LabelFrame(wrap, text="Live Status", padding=8)
        lower_left.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        lower_left.columnconfigure(1, weight=1)

        self.status_var = tk.StringVar(value="--")
        self.temp_var   = tk.StringVar(value="--")
        self.torque_var = tk.StringVar(value="--")
        self.speed_var  = tk.StringVar(value="--")
        self.angl_theta_var = tk.StringVar(value="--")
        self.angl_alpha_var = tk.StringVar(value="--")
        self.calib_tuple_var = tk.StringVar(value="(S,G,A,M) = 0,0,0,0")
        self.encoder_ok_var  = tk.StringVar(value="0")
        self.conn_var = tk.StringVar(value="Disconnected")

        self._row(lower_left, 0, "State:", self.status_var)
        self._row(lower_left, 1, "Temperature (°C):", self.temp_var)
        self._row(lower_left, 2, "Torque (raw):", self.torque_var)
        self._row(lower_left, 3, "Speed (deg/s):", self.speed_var)
        self._row(lower_left, 4, "Angle θ (deg):", self.angl_theta_var)
        self._row(lower_left, 5, "Euler α (deg):", self.angl_alpha_var)
        self._row(lower_left, 6, "IMU Calib (S,G,A,M):", self.calib_tuple_var)
        self._row(lower_left, 7, "Encoder OK (1/0):", self.encoder_ok_var)
        self._row(lower_left, 8, "Connection:", self.conn_var)

        # Calibration / indicators
        calib_frame = ttk.LabelFrame(lower_left, text="Calibration", padding=8)
        calib_frame.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        for c in range(6):
            calib_frame.columnconfigure(c, weight=0)

        ttk.Label(calib_frame, text="IMU overall").grid(row=0, column=0, sticky="w")
        self.imu_light = tk.Canvas(calib_frame, width=14, height=14, highlightthickness=0)
        self.imu_light.grid(row=0, column=1, sticky="w", padx=(4, 12))
        self.imu_text = tk.StringVar(value="Uncalibrated")
        ttk.Label(calib_frame, textvariable=self.imu_text).grid(row=0, column=2, sticky="w")

        ttk.Label(calib_frame, text="Encoder").grid(row=0, column=3, sticky="w", padx=(16, 4))
        self.enc_light = tk.Canvas(calib_frame, width=14, height=14, highlightthickness=0)
        self.enc_light.grid(row=0, column=4, sticky="w", padx=(4, 12))
        self.enc_text = tk.StringVar(value="Uncalibrated")
        ttk.Label(calib_frame, textvariable=self.enc_text).grid(row=0, column=5, sticky="w")

        # IMU detail tuple
        detail = ttk.Frame(calib_frame)
        detail.grid(row=1, column=0, columnspan=6, sticky="w", pady=(6, 0))
        ttk.Label(detail, text="SYS").grid(row=0, column=0, sticky="w")
        self.cal_sys_light = tk.Canvas(detail, width=14, height=14, highlightthickness=0)
        self.cal_sys_light.grid(row=0, column=1, sticky="w", padx=(4, 12))
        ttk.Label(detail, text="GYR").grid(row=0, column=2, sticky="w")
        self.cal_gyr_light = tk.Canvas(detail, width=14, height=14, highlightthickness=0)
        self.cal_gyr_light.grid(row=0, column=3, sticky="w", padx=(4, 12))
        ttk.Label(detail, text="ACC").grid(row=0, column=4, sticky="w")
        self.cal_acc_light = tk.Canvas(detail, width=14, height=14, highlightthickness=0)
        self.cal_acc_light.grid(row=0, column=5, sticky="w", padx=(4, 12))
        ttk.Label(detail, text="MAG").grid(row=0, column=6, sticky="w")
        self.cal_mag_light = tk.Canvas(detail, width=14, height=14, highlightthickness=0)
        self.cal_mag_light.grid(row=0, column=7, sticky="w", padx=(4, 12))

        # Orientation group
        orient = ttk.LabelFrame(wrap, text="Orientation", padding=8)
        orient.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        orient.columnconfigure(0, weight=1)
        orient.columnconfigure(1, weight=1)

        quat_frame = ttk.LabelFrame(orient, text="Quaternion", padding=8)
        eul_frame  = ttk.LabelFrame(orient, text="Euler (deg)", padding=8)
        quat_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        eul_frame.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)

        self.quat_x_var = tk.StringVar(value="--")
        self.quat_y_var = tk.StringVar(value="--")
        self.quat_z_var = tk.StringVar(value="--")
        self.quat_w_var = tk.StringVar(value="--")

        ttk.Label(quat_frame, text="X").grid(row=0, column=0, sticky="w")
        ttk.Label(quat_frame, text="Y").grid(row=0, column=1, sticky="w")
        ttk.Label(quat_frame, text="Z").grid(row=0, column=2, sticky="w")
        ttk.Label(quat_frame, textvariable=self.quat_x_var, font=("TkDefaultFont", 10, "bold")).grid(row=1, column=0, sticky="e")
        ttk.Label(quat_frame, textvariable=self.quat_y_var, font=("TkDefaultFont", 10, "bold")).grid(row=1, column=1, sticky="e")
        ttk.Label(quat_frame, textvariable=self.quat_z_var, font=("TkDefaultFont", 10, "bold")).grid(row=1, column=2, sticky="e")

        ttk.Label(quat_frame, text="W").grid(row=2, column=0, sticky="w")
        ttk.Label(quat_frame, textvariable=self.quat_w_var, font=("TkDefaultFont", 10, "bold")).grid(row=2, column=1, sticky="e")

        self.yaw_var   = tk.StringVar(value="--")
        self.pitch_var = tk.StringVar(value="--")
        self.roll_var  = tk.StringVar(value="--")

        ttk.Label(eul_frame, text="Yaw").grid(row=0, column=0, sticky="w")
        ttk.Label(eul_frame, text="Pitch").grid(row=1, column=0, sticky="w")
        ttk.Label(eul_frame, text="Roll").grid(row=2, column=0, sticky="w")

        ttk.Label(eul_frame, textvariable=self.yaw_var, font=("TkDefaultFont", 10, "bold")).grid(row=0, column=1, sticky="e")
        ttk.Label(eul_frame, textvariable=self.pitch_var, font=("TkDefaultFont", 10, "bold")).grid(row=1, column=1, sticky="e")
        ttk.Label(eul_frame, textvariable=self.roll_var, font=("TkDefaultFont", 10, "bold")).grid(row=2, column=1, sticky="e")

        # Log
        log_frame = ttk.LabelFrame(wrap, text="Log", padding=8)
        log_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(log_frame, height=12, wrap="none", state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        vs = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        hs = ttk.Scrollbar(log_frame, orient="horizontal", command=self.log.xview)
        self.log.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")

        # Initialize LEDs to dim so GUI doesn't look dead
        for cv in (self.imu_light, self.enc_light, self.cal_sys_light, self.cal_gyr_light, self.cal_acc_light, self.cal_mag_light):
            self._set_light(cv, self.COLOR_DIM)

    def _build_statusbar(self):
        sb = ttk.Frame(self, padding=(8, 4))
        sb.grid(row=2, column=0, sticky="ew")
        sb.columnconfigure(0, weight=1)
        self.statusbar_msg = tk.StringVar(value="Ready")
        ttk.Label(sb, textvariable=self.statusbar_msg).grid(row=0, column=0, sticky="w")

    # ================= HELPERS =================
    def _row(self, parent, r, label, var):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=2, pady=2)
        ttk.Label(parent, textvariable=var, font=("TkDefaultFont", 10, "bold")).grid(row=r, column=1, sticky="e", padx=2, pady=2)

    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", f"{msg}\n")
        self.log.see("end")
        self.log.configure(state="disabled")
        # mirror to statusbar
        self.statusbar_msg.set(msg)

    def _set_light(self, canvas: tk.Canvas, color: str):
        canvas.delete("all")
        canvas.create_oval(2, 2, 12, 12, fill=color, outline=color)

    def _refresh_ports_combo(self):
        """Refresh available serial ports periodically."""
        if list_ports:
            try:
                ports = [p.device for p in list_ports.comports()]
            except Exception:
                ports = []
        else:
            ports = []
        if ports:
            # keep current selection if present
            cur = self.current_port.get()
            self.port_combo["values"] = ports
            if cur not in ports:
                self.current_port.set(ports[0])
        else:
            self.port_combo["values"] = []
        self.after(3000, self._refresh_ports_combo)

    def quaternion_to_euler(self):
        qw = self.quaternion_w
        qx = self.quaternion_x
        qy = self.quaternion_y
        qz = self.quaternion_z

        t0 = +2.0*(qw*qx + qy*qz)
        t1 = +1.0 - 2.0*(qx*qx + qy*qy)
        roll_rad = math.atan2(t0, t1)

        t2 = +2.0*(qw*qy - qz*qx)
        t2 = max(-1.0, min(1.0, t2))
        pitch_rad = math.sin(t2) if abs(t2) <= 1e-8 else math.asin(t2)

        t3 = +2.0*(qw*qz + qx*qy)
        t4 = +1.0 - 2.0*(qy*qy + qz*qz)
        yaw_rad = math.atan2(t3, t4)

        self.roll = math.degrees(roll_rad)
        self.pitch = math.degrees(pitch_rad)
        self.yaw = math.degrees(yaw_rad)

        # store pitch as "alpha"
        self.angle_alpha = self.pitch

    # ================= CONNECTION HANDLERS =================
    def _connect_clicked(self):
        port = self.current_port.get()
        baud = int(self.current_baud.get())
        safe_close_serial()
        ser = safe_open_serial(port, baud)
        if ser and ser.is_open:
            self.connected_var.set(True)
            self._log(f"Connected: {port} @ {baud}")
            self.conn_label.configure(text=f"Connected: {port}@{baud}", foreground=self.COLOR_OK)
            self.conn_var.set(f"Port:{port} @ {baud}")
            self._set_controls_enabled(True)
        else:
            self.connected_var.set(False)
            self._log(f"Failed to connect: {port} @ {baud}")
            self.conn_label.configure(text="Not connected", foreground=self.COLOR_BAD)
            self.conn_var.set("Port: (disconnected)")
            self._set_controls_enabled(False)
        self._update_connect_buttons()

    def _disconnect_clicked(self):
        safe_close_serial()
        self.connected_var.set(False)
        self._log("Disconnected")
        self.conn_label.configure(text="Not connected", foreground=self.COLOR_DIM)
        self.conn_var.set("Port: (disconnected)")
        self._set_controls_enabled(False)
        self._update_connect_buttons()

    def _update_connect_buttons(self):
        if self.connected_var.get():
            self.connect_btn.configure(state="disabled")
            self.disconnect_btn.configure(state="normal")
        else:
            self.connect_btn.configure(state="normal")
            self.disconnect_btn.configure(state="disabled")

    def _set_controls_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for b in (self.start_btn, self.stop_btn, self.init_btn, self.cal_btn,
                  self.read_btn, self.idle_btn, self.zero_btn):
            b.configure(state=state)

    # ================= UI CALLBACKS =================
    def _update_safe_btn_color(self):
        # visually emphasize the latch; explicit tk.Button for cross-platform bg
        if self.is_it_safe:
            self.safe_btn.configure(text="SAFE LATCH: ON (press SPACE)", bg="#22cc55", activebackground="#22cc55")
        else:
            self.safe_btn.configure(text="SAFE LATCH: OFF (press SPACE)", bg="#ee4444", activebackground="#ee4444")
        self.after(150, self._update_safe_btn_color)

    def on_isitsafe(self):
        self.is_it_safe = not self.is_it_safe
        if self.is_it_safe:
            self._log("Safety latch ON: control packets allowed.")
        else:
            self._log("Safety latch OFF: control packets blocked.")

    def on_set_torque(self):
        try:
            val = float(self.torque_in.get())
        except ValueError:
            messagebox.showerror("Invalid torque", "Enter a number for torque.")
            return
        self.requested_torque_gui = val
        self._log(f"Requested torque = {val}")

    def on_set_speed(self):
        try:
            val = float(self.speed_in.get())
        except ValueError:
            messagebox.showerror("Invalid speed", "Enter a number for speed.")
            return
        self.requested_speed_gui = abs(val)  # magnitude only; sign via direction
        self._log(f"Requested speed magnitude = {self.requested_speed_gui}")

    def on_set_direction(self):
        self.direction_flag = 1 if self.dir_var.get() == 1 else 0
        self._log(f"Direction = {'Forward' if self.direction_flag else 'Reverse'}")

    def _tx_frame(self, frame_bytes, log_label):
        global SER
        if SER is None or not getattr(SER, "is_open", False):
            self._log("TX blocked: serial disconnected.")
            return
        try:
            SER.write(frame_bytes)
            self._log(log_label)
        except Exception as e:
            self._log(f"TX error: {e}")

    def on_start(self):
        if not self.is_it_safe:
            self._log("CONTROL blocked: safety latch OFF.")
            return
        self.desired_cmd_id = CMD_CONTROL
        try:
            frame = build_cmd_control(self.requested_torque_gui,
                                      self.requested_speed_gui,
                                      self.direction_flag)
        except NameError:
            self._log("build_cmd_control() not defined.")
            return
        self._tx_frame(frame, f"TX CONTROL τ={self.requested_torque_gui:.1f} "
                              f"spd={self.requested_speed_gui:.1f} dir={self.direction_flag}")

    def on_stop(self):
        self.desired_cmd_id = CMD_STOP
        try:
            frame = build_cmd_stop()
        except NameError:
            self._log("build_cmd_stop() not defined.")
            return
        self._tx_frame(frame, "TX STOP")

    def on_init(self):
        self.desired_cmd_id = CMD_INITIALIZING
        try:
            frame = build_cmd_init()
        except NameError:
            self._log("build_cmd_init() not defined.")
            return
        self._tx_frame(frame, "TX INITIALIZING")

    def on_idle(self):
        self.desired_cmd_id = CMD_IDLE
        try:
            frame = build_cmd_idle()
        except NameError:
            self._log("build_cmd_idle() not defined.")
            return
        self._tx_frame(frame, "TX IDLE")

    def on_zero(self):
        self.desired_cmd_id = CMD_OFFSETTING
        try:
            frame = build_cmd_offsetting()
        except NameError:
            self._log("build_cmd_offsetting() not defined.")
            return
        self._tx_frame(frame, "TX OFFSETTING (encoder zero)")

    def on_calibrate(self):
        self.desired_cmd_id = CMD_CALIBRATE
        try:
            frame = build_cmd_calibrate()
        except NameError:
            self._log("build_cmd_calibrate() not defined.")
            return
        self._tx_frame(frame, "TX CALIBRATE")

    def on_request_status(self):
        self.desired_cmd_id = CMD_READING
        try:
            frame = build_cmd_reading()
        except NameError:
            self._log("build_cmd_reading() not defined.")
            return
        self._tx_frame(frame, "TX READING / Status Request")

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

    # ================= BACKGROUND RX =================
    def _reader_worker(self):
        while True:
            try:
                pkt = read_one_packet_blocking(timeout=0.05)
            except NameError:
                # protocol function not provided; keep thread alive
                time.sleep(0.1)
                pkt = None
            except Exception as e:
                # swallow serial errors; loop continues
                self._log(f"RX thread error: {e}")
                time.sleep(0.1)
                pkt = None

            if pkt is None:
                continue
            (pkt_id, payload) = pkt
            self._last_packet = (pkt_id, payload)

    def poll_serial_binary(self):
        data = getattr(self, "_last_packet", None)
        self._last_packet = None

        if data:
            (pkt_id, payload) = data
            try:
                parsed = parse_telemetry_payload(pkt_id, payload)
            except NameError:
                parsed = None
                self._log("parse_telemetry_payload() not defined.")
            except Exception as e:
                parsed = None
                self._log(f"Parse error: {e}")

            if parsed:
                # mirror parsed telemetry into GUI variables
                self.angle_theta = parsed.get("angleTheta", 0.0)
                self.speed_meas  = parsed.get("speed", 0.0)
                self.torque_meas = parsed.get("torque", 0.0)
                self.temp_meas   = parsed.get("temp", 0.0)

                self.accel_x = parsed.get("accelX", 0.0)
                self.accel_y = parsed.get("accelY", 0.0)
                self.accel_z = parsed.get("accelZ", 0.0)
                self.gyro_x  = parsed.get("gyroX", 0.0)
                self.gyro_z  = parsed.get("gyroZ", 0.0)

                self.quaternion_w = parsed.get("quatW", 0.0)
                self.quaternion_x = parsed.get("quatX", 0.0)
                self.quaternion_y = parsed.get("quatY", 0.0)
                self.quaternion_z = parsed.get("quatZ", 0.0)

                self.imu_fully = int(parsed.get("imuFully", 0))
                self.enc_ok    = int(parsed.get("encOK", 0))

                # update Euler + alpha
                self.quaternion_to_euler()

                # update label for state
                try:
                    st_name = state_code_to_name(parsed.get("state_code", 0))
                except NameError:
                    st_name = f"state={parsed.get('state_code', 0)}"
                self.status_var.set(st_name)

                # compact debug line
                self._log(f"RX {st_name} θ={self.angle_theta:.2f}° τ={self.torque_meas:.1f} spd={self.speed_meas:.2f}")

        self.after(120, self.poll_serial_binary)

    # ================= PERIODIC LABEL REFRESH =================
    def _update_status(self):
        # numbers
        self.temp_var.set(f"{self.temp_meas:.2f}")
        self.torque_var.set(f"{self.torque_meas:.1f}")
        self.speed_var.set(f"{self.speed_meas:.2f}")
        self.angl_theta_var.set(f"{self.angle_theta:.2f}")
        self.angl_alpha_var.set(f"{self.angle_alpha:.2f}")

        # connection text
        if SER and getattr(SER, "is_open", False):
            self.conn_var.set(f"Port:{SER.port} @ {SER.baudrate}")
        else:
            self.conn_var.set("Port: (disconnected)")

        # quaternion
        self.quat_w_var.set(f"{self.quaternion_w:.4f}")
        self.quat_x_var.set(f"{self.quaternion_x:.4f}")
        self.quat_y_var.set(f"{self.quaternion_y:.4f}")
        self.quat_z_var.set(f"{self.quaternion_z:.4f}")

        # euler
        self.yaw_var.set(f"{self.yaw:.2f}")
        self.pitch_var.set(f"{self.pitch:.2f}")
        self.roll_var.set(f"{self.roll:.2f}")

        # IMU + encoder indicators
        if self.imu_fully == 1:
            self._set_light(self.imu_light, self.COLOR_OK)
            self.imu_text.set("Fully calibrated")
        else:
            self._set_light(self.imu_light, self.COLOR_BAD)
            self.imu_text.set("Not calibrated")

        if self.enc_ok == 1:
            self._set_light(self.enc_light, self.COLOR_OK)
            self.enc_text.set("Calibrated")
        else:
            self._set_light(self.enc_light, self.COLOR_BAD)
            self.enc_text.set("Uncalibrated")

        # Detail LEDs are placeholders until you pass actual values
        for cv in (self.cal_sys_light, self.cal_gyr_light, self.cal_acc_light, self.cal_mag_light):
            self._set_light(cv, self.COLOR_WARN)

        self.encoder_ok_var.set(str(self.enc_ok))
        # If you later parse individual (S,G,A,M) levels, update here:
        # self.calib_tuple_var.set(f"(S,G,A,M) = {sys},{gyr},{acc},{mag}")

        self.after(250, self._update_status)


if __name__ == "__main__":
    MiniGUI().mainloop()
