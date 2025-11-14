import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import time
from collections import deque
import numpy as np
from math import radians
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Optional logo support (won't break if Pillow not installed) ---
try:
    from PIL import Image, ImageTk
    PIL_OK = True
except Exception:
    PIL_OK = False


# ======================
# Tab 2 model (pure math, no heavy deps)
# ======================
class AssistiveTorque:
    """
    τ(GP, α) = BW_gain · f(GP) · s(GP) · r(α)^(-1)
      f(GP) = exp( - (GP-μ)^2 / (2σ^2) )
      s(GP) = (p+1)/(exp((GP-c)/d) + 1) - p
      r(α)  = 1 + Krep*(exp(α0-α) + exp(α-α1))
    GP in [0,1], angles in radians internally.
    """
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


# ======================
# Tab 1: Home page (visual only)
# ======================
class HomePage(ttk.Frame):
    def __init__(self, master, logo_path=None):
        super().__init__(master)
        self.pack(fill=tk.BOTH, expand=True)

        # grid
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)
        self.grid_rowconfigure(3, weight=1)
        self.grid_rowconfigure(4, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # --- Logo bar ---
        logo_frame = tk.Frame(self, bd=0)
        logo_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        if PIL_OK and logo_path:
            try:
                img = Image.open(logo_path)
                w, h = max(1, img.width // 6), max(1, img.height // 6)
                try:
                    img = img.resize((w, h), resample=Image.Resampling.LANCZOS)
                except Exception:
                    img = img.resize((w, h))
                self._logo_img = ImageTk.PhotoImage(img)
                tk.Label(logo_frame, image=self._logo_img).pack(padx=6, pady=4)
            except Exception:
                tk.Label(logo_frame, text="Ankle Exoskeleton Control", font=("Segoe UI", 14, "bold")).pack(padx=10, pady=8)
        else:
            tk.Label(logo_frame, text="Ankle Exoskeleton Control", font=("Segoe UI", 14, "bold")).pack(padx=10, pady=8)

        # --- Toolbar ---
        bar = ttk.Frame(self, padding=(8, 6))
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        for c in range(9): bar.columnconfigure(c, weight=0)
        bar.columnconfigure(8, weight=1)

        ttk.Label(bar, text="Port").grid(row=0, column=0, sticky="w")
        ttk.Combobox(bar, width=15, values=[], state="disabled").grid(row=0, column=1, padx=(4, 10), sticky="w")

        ttk.Label(bar, text="Baud").grid(row=0, column=2, sticky="w")
        ttk.Combobox(bar, width=10, values=[9600,19200,38400,57600,115200,230400], state="disabled").grid(row=0, column=3, padx=(4, 10), sticky="w")

        ttk.Button(bar, text="Connect", state="disabled").grid(row=0, column=4, padx=2)
        ttk.Button(bar, text="Disconnect", state="disabled").grid(row=0, column=5, padx=2)

        self._safe = False
        self.safe_btn = tk.Button(bar, text="SAFE LATCH: OFF", width=18, bg="#ee4444", command=self._toggle_safe_visual)
        self.safe_btn.grid(row=0, column=6, padx=(14, 0))

        ttk.Label(bar, text="Not connected").grid(row=0, column=8, sticky="e")

        # --- Calibration bar ---
        cal_bar = ttk.LabelFrame(self, text="Calibration status", padding=(8, 6))
        cal_bar.grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6))
        for c in range(8): cal_bar.columnconfigure(c, weight=1)

        self.enc = tk.StringVar(value="Not calibrated")
        self.sys = tk.StringVar(value="Not calibrated")
        self.gyr = tk.StringVar(value="Not calibrated")
        self.mag = tk.StringVar(value="Not calibrated")

        ttk.Label(cal_bar, text="Encoder:").grid(row=0, column=0, sticky="e", padx=(0, 6))
        ttk.Label(cal_bar, textvariable=self.enc, foreground="#dc2626").grid(row=0, column=1, sticky="w")
        ttk.Label(cal_bar, text="System:").grid(row=0, column=2, sticky="e", padx=(10, 6))
        ttk.Label(cal_bar, textvariable=self.sys, foreground="#dc2626").grid(row=0, column=3, sticky="w")
        ttk.Label(cal_bar, text="Gyroscope:").grid(row=0, column=4, sticky="e", padx=(10, 6))
        ttk.Label(cal_bar, textvariable=self.gyr, foreground="#dc2626").grid(row=0, column=5, sticky="w")
        ttk.Label(cal_bar, text="Magnetometer:").grid(row=0, column=6, sticky="e", padx=(10, 6))
        ttk.Label(cal_bar, textvariable=self.mag, foreground="#dc2626").grid(row=0, column=7, sticky="w")

        # --- Controls ---
        btns = ttk.LabelFrame(self, text="Controls", padding=(8, 6))
        btns.grid(row=3, column=0, sticky="nsew", padx=(6, 3), pady=(0, 6))
        for c in range(3): btns.columnconfigure(c, weight=1)
        texts = [
            ("Init","Idle","Control"),
            ("Request Status","Calibrate","Off-set Encoder"),
            ("STOP","Clear Log","Exit"),
        ]
        for r,row in enumerate(texts):
            for c,txt in enumerate(row):
                ttk.Button(btns, text=txt, state="disabled").grid(row=r, column=c, padx=2, pady=2, sticky="ew")

        # --- Inputs ---
        inputs = ttk.LabelFrame(self, text="Inputs", padding=(8, 6))
        inputs.grid(row=3, column=1, sticky="nsew", padx=(3, 6), pady=(0, 6))
        for c in range(7): inputs.columnconfigure(c, weight=0)
        inputs.columnconfigure(6, weight=1)

        ttk.Label(inputs, text="Torque (raw)").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(inputs, from_=-20000, to=20000, increment=1, width=10, state="disabled").grid(row=0, column=1, padx=6)
        ttk.Button(inputs, text="Set", state="disabled").grid(row=0, column=2, padx=(0, 8))
        ttk.Label(inputs, text="Speed cmd").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(inputs, from_=0, to=20000, increment=0.5, width=10, state="disabled").grid(row=1, column=1, padx=6)
        ttk.Button(inputs, text="Set", state="disabled").grid(row=1, column=2, padx=(0, 8))
        df = ttk.Frame(inputs); df.grid(row=2, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(df, text="Forward", value=1, state="disabled").pack(side="left")
        ttk.Radiobutton(df, text="Reverse", value=0, state="disabled").pack(side="left")

        # --- Status ---
        status = ttk.LabelFrame(self, text="Status", padding=(8, 6))
        status.grid(row=4, column=0, sticky="nsew", padx=(6, 3), pady=(0, 6))
        for c in range(2): status.columnconfigure(c, weight=1)

        def row(parent, r, label, value="--"):
            ttk.Label(parent, text=label).grid(row=r, column=0, sticky="e", padx=(0, 6))
            ttk.Label(parent, text=value).grid(row=r, column=1, sticky="w")
        row(status, 0, "State:", "Ready")
        row(status, 1, "Temperature (°C):")
        row(status, 2, "Torque (raw):")
        row(status, 3, "Speed (deg/s):")
        row(status, 4, "Angle θ (deg):")
        row(status, 5, "Euler α (deg):")
        row(status, 6, "IMU Calib (S,G,A,M):", "(0,0,0,0)")
        row(status, 7, "Encoder OK (1/0):", "0")
        row(status, 8, "Connection:", "Port: (disconnected)")

        # --- Log (Tab 1) ---
        log_frame = ttk.LabelFrame(self, text="Log", padding=(8, 6))
        log_frame.grid(row=4, column=1, sticky="nsew", padx=(3, 6), pady=(0, 6))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log = tk.Text(log_frame, height=12, wrap="none", state="disabled")
        log.grid(row=0, column=0, sticky="nsew")
        vs = ttk.Scrollbar(log_frame, orient="vertical", command=log.yview)
        hs = ttk.Scrollbar(log_frame, orient="horizontal", command=log.xview)
        log.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")

    def _toggle_safe_visual(self):
        self._safe = not self._safe
        if self._safe:
            self.safe_btn.configure(text="SAFE LATCH: ON", bg="#22cc55")
            for var in [self.enc, self.sys, self.gyr, self.mag]:
                var.set("Calibrated")
        else:
            self.safe_btn.configure(text="SAFE LATCH: OFF", bg="#ee4444")
            for var in [self.enc, self.sys, self.gyr, self.mag]:
                var.set("Not calibrated")


# ======================
# Tab 2: Assistive Torque GUI (live fake IMU) + its own Log
# ======================
class TorqueGUI(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill=tk.BOTH, expand=True)
        self.model = AssistiveTorque()
        self.streaming = True
        self.prev_tau = None

        # --- TOP TOOLBAR ---
        toolbar = ttk.Frame(self, padding=(8, 6))
        toolbar.pack(fill=tk.X, side=tk.TOP)
        ttk.Label(toolbar, text="Assistive Torque Controller", font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT)

        # Presets
        self.presets = {
            "Default": dict(gp=50, alpha=0, bw=1, mu=0.5, sigma=0.17, c=0.6, d=0.02, p=0.8, a0=-10, a1=10, krep=1),
            "Heel-Strike Assist": dict(gp=5, alpha=-5, bw=1.2, mu=0.1, sigma=0.12, c=0.18, d=0.02, p=0.6, a0=-12, a1=8, krep=1.2),
            "Push-Off Boost": dict(gp=60, alpha=6, bw=1.5, mu=0.6, sigma=0.14, c=0.55, d=0.02, p=0.9, a0=-8, a1=12, krep=0.8),
        }
        preset_box = ttk.Frame(toolbar); preset_box.pack(side=tk.LEFT, padx=(12, 6))
        ttk.Label(preset_box, text="Preset:").pack(side=tk.LEFT)
        self.preset_var = tk.StringVar(value="Default")
        ttk.Combobox(preset_box, width=18, textvariable=self.preset_var,
                     values=list(self.presets.keys()), state="readonly").pack(side=tk.LEFT, padx=4)
        ttk.Button(preset_box, text="Apply", command=self.apply_preset).pack(side=tk.LEFT)

        # Buttons
        btns = ttk.Frame(toolbar); btns.pack(side=tk.RIGHT)
        self.play_btn = ttk.Button(btns, text="⏸ Pause IMU", command=self.toggle_stream)
        self.play_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="⟳ Reset", command=self.reset_params).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="⬇ Save Figures", command=self.save_figs).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="💾 Save Params", command=self.save_params).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="📂 Load Params", command=self.load_params).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="📌 Freeze τ", command=self.freeze_curve).pack(side=tk.LEFT, padx=4)

        # --- SPLIT: LEFT (controls + τ) | RIGHT (plots + log) ---
        splitter = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        splitter.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # LEFT pane
        left_pane = ttk.Frame(splitter)
        splitter.add(left_pane, weight=1)
        ctrl = ttk.Frame(left_pane); ctrl.pack(fill=tk.BOTH, expand=True)
        ctrl.grid_rowconfigure(0, weight=1)
        ctrl.grid_rowconfigure(1, weight=4)
        ctrl.grid_columnconfigure(0, weight=1)

        # Parameters (scrollable)
        params_card = ttk.LabelFrame(ctrl, text="Parameters")
        params_card.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6, 4))
        params_wrap = ttk.Frame(params_card); params_wrap.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(params_wrap, height=240, highlightthickness=0)
        vbar = ttk.Scrollbar(params_wrap, orient="vertical", command=canvas.yview)
        self.params_inner = ttk.Frame(canvas)
        self.params_inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        inner = canvas.create_window((0, 0), window=self.params_inner, anchor="nw")
        canvas.configure(yscrollcommand=vbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        vbar.pack(side="right", fill="y")
        params_wrap.bind("<Configure>", lambda e: canvas.itemconfig(inner, width=params_wrap.winfo_width()))

        # τ model plot
        Torq_Ctrl_plt = ttk.LabelFrame(ctrl, text="Assistive Torque Model")
        Torq_Ctrl_plt.grid(row=1, column=0, sticky="nsew", padx=6, pady=(4, 6))
        Torq_Ctrl_plt.grid_rowconfigure(0, weight=1); Torq_Ctrl_plt.grid_columnconfigure(0, weight=1)

        # RIGHT pane
        right_pane = ttk.Frame(splitter)
        splitter.add(right_pane, weight=2)
        right_pane.rowconfigure(0, weight=3)
        right_pane.rowconfigure(1, weight=2)
        right_pane.rowconfigure(2, weight=2)
        right_pane.columnconfigure(0, weight=1)

        # IMU plots block
        plot = ttk.LabelFrame(right_pane, text="Plots")
        plot.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6, 3))
        plot.grid_rowconfigure(0, weight=1); plot.grid_columnconfigure(0, weight=1)

        # --- Matplotlib: τ + f/s/r ---
        self.fig = Figure(figsize=(6.6, 3.8), dpi=100, layout="constrained")
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[2.6, 1.2], hspace=0.28, wspace=0.28)
        self.ax_tau = self.fig.add_subplot(gs[:, 0])
        self.ax_f = self.fig.add_subplot(gs[0, 1])
        self.ax_s = self.fig.add_subplot(gs[1, 1])
        self.ax_r = self.fig.add_subplot(gs[2, 1])
        self.ax_phase = self.ax_tau.inset_axes([0.08, 0.03, 0.84, 0.12])
        self.canvas = FigureCanvasTkAgg(self.fig, master=Torq_Ctrl_plt)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # --- IMU figure ---
        self.fig_imu = Figure(figsize=(6.6, 3.2), dpi=100, layout="constrained")
        imu_gs = self.fig_imu.add_gridspec(3, 1, hspace=0.24)
        self.ax_accel = self.fig_imu.add_subplot(imu_gs[0, 0])
        self.ax_speed = self.fig_imu.add_subplot(imu_gs[1, 0])
        self.ax_angle = self.fig_imu.add_subplot(imu_gs[2, 0])
        self.canvas_imu = FigureCanvasTkAgg(self.fig_imu, master=plot)
        self.canvas_imu.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # --- LOG (Tab 2) ---
        log_frame2 = ttk.LabelFrame(right_pane, text="Log")
        log_frame2.grid(row=1, column=0, sticky="nsew", padx=6, pady=(3, 6))
        log_frame2.rowconfigure(0, weight=1)
        log_frame2.columnconfigure(0, weight=1)
        self.log2 = tk.Text(log_frame2, height=10, wrap="none", state="disabled")
        self.log2.grid(row=0, column=0, sticky="nsew")
        vs2 = ttk.Scrollbar(log_frame2, orient="vertical", command=self.log2.yview)
        hs2 = ttk.Scrollbar(log_frame2, orient="horizontal", command=self.log2.xview)
        self.log2.configure(yscrollcommand=vs2.set, xscrollcommand=hs2.set)
        vs2.grid(row=0, column=1, sticky="ns")
        hs2.grid(row=1, column=0, sticky="ew")

        # --- STATUS LINE (bottom of tab 2) ---
        self.status = ttk.Label(right_pane, anchor="w")
        self.status.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 6))

        # --------------- SLIDER HELPER (fixed to not fire init callbacks) ---------------
        def add_slider(frame, label, frm, to, init, cmd, unit=""):
            row = ttk.Frame(frame, height=26)
            row.pack(fill=tk.X, pady=2, padx=6)
            row.pack_propagate(False)

            ttk.Label(row, text=label, width=14, anchor="w").pack(side=tk.LEFT)
            s = ttk.Scale(row, from_=frm, to=to, orient=tk.HORIZONTAL, length=170)
            val = ttk.Label(row, width=8, anchor="e")
            val.pack(side=tk.RIGHT)
            s.pack(side=tk.LEFT, padx=6)

            # Set initial value and label BEFORE binding command (prevents early callback)
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

        # Collapsible groups
        def collapsible(parent, title):
            f = ttk.Frame(parent); bar = ttk.Frame(f); bar.pack(fill=tk.X, pady=(3, 0))
            sv = tk.StringVar(value="▼ " + title); lbl = ttk.Label(bar, textvariable=sv); lbl.pack(side=tk.LEFT, padx=2)
            body = ttk.Frame(f); body.pack(fill=tk.X, padx=2, pady=(2,0))
            def toggle(_=None):
                if body.winfo_manager(): body.pack_forget(); sv.set("► " + title)
                else: body.pack(fill=tk.X, padx=2, pady=(2,0)); sv.set("▼ " + title)
            bar.bind("<Button-1>", toggle); lbl.bind("<Button-1>", toggle)
            return f, body

        # ---- Create all sliders (attributes exist before any callback runs) ----
        grp_core, core_body = collapsible(self.params_inner, "Core Controls"); grp_core.pack(fill=tk.X)
        self.gp_percent = add_slider(core_body, "GP (%)", 0, 100, 50, self.update_static_plots, "%")
        self.alpha_deg  = add_slider(core_body, "α (deg)", -30, 30, 0, self.update_static_plots, "°")
        self.bw_gain    = add_slider(core_body, "BW gain", 0, 2.5, 1, self._update_model)

        grp_f, body_f = collapsible(self.params_inner, "f(GP)"); grp_f.pack(fill=tk.X)
        self.mu    = add_slider(body_f, "μ", 0, 1, 0.5, self._update_model)
        self.sigma = add_slider(body_f, "σ", 0.05, 0.4, 0.17, self._update_model)

        grp_s, body_s = collapsible(self.params_inner, "s(GP)"); grp_s.pack(fill=tk.X)
        self.c   = add_slider(body_s, "c", 0, 1, 0.6, self._update_model)
        self.d   = add_slider(body_s, "d", 0.005, 0.1, 0.02, self._update_model)
        self.p   = add_slider(body_s, "p", 0, 1.5, 0.8, self._update_model)

        grp_r, body_r = collapsible(self.params_inner, "r(α)"); grp_r.pack(fill=tk.X)
        self.a0   = add_slider(body_r, "α0 (deg)", -40, 0, -10, self._update_model, "°")
        self.a1   = add_slider(body_r, "α1 (deg)", 0, 40, 10, self._update_model, "°")
        self.krep = add_slider(body_r, "Krep", 0, 5, 1, self._update_model)

        # IMU sim buffers
        self.window_sec, self.fs = 10, 50
        self.dt, self.max_samples = 1/self.fs, int(self.window_sec * self.fs)
        self.tbuf, self.accX, self.accY, self.accZ = [deque(maxlen=self.max_samples) for _ in range(4)]
        self.angX, self.angZ, self.theta, self.alphaZ = [deque(maxlen=self.max_samples) for _ in range(4)]
        self.t0 = time.perf_counter(); self._next_t = 0

        # Autoscale lock
        self.lock_tau = tk.BooleanVar(master=self, value=False)
        ttk.Checkbutton(Torq_Ctrl_plt, text="Lock τ autoscale", variable=self.lock_tau).place(relx=0.01, rely=0.02)

        # Mouse crosshair
        self._mh = self.canvas.mpl_connect("motion_notify_event", self._on_move_tau)

        # First draw AFTER all widgets exist
        self.update_static_plots()
        self._imu_loop()

        # Shortcuts
        self.bind_all("<space>", lambda e: self.toggle_stream())
        self.bind_all("<Control-r>", lambda e: self.reset_params())
        self.bind_all("<Control-s>", lambda e: self.save_figs())

        self._log2("Torque GUI ready.")

    # ---------------- Helpers: Log for Tab 2 ----------------
    def _log2(self, msg: str):
        if not hasattr(self, "log2"): return
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
            if not path: return
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
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
        if not path:
            self._log2("Save params canceled.")
            return
        try:
            with open(path, "w") as f: json.dump(m, f, indent=2)
            self._log2(f"Parameters saved: {path}")
        except Exception as e:
            messagebox.showerror("Save Params Error", str(e))
            self._log2(f"Save params error: {e}")

    def load_params(self):
        path = filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if not path:
            self._log2("Load params canceled.")
            return
        try:
            with open(path) as f: m = json.load(f)
            self._set_sliders(m)
            self._log2(f"Parameters loaded: {path}")
        except Exception as e:
            messagebox.showerror("Load Params Error", str(e))
            self._log2(f"Load params error: {e}")

    def freeze_curve(self):
        GP = np.linspace(0, 1, 600)
        alpha = radians(float(self.alpha_deg.get()))
        self.prev_tau = (GP, self.model.tau(GP, alpha))
        self.update_static_plots()
        self._log2("Current τ curve frozen for comparison.")

    def apply_preset(self):
        name = self.preset_var.get()
        self._set_sliders(self.presets[name])
        self._log2(f"Preset applied: {name}")

    def _set_sliders(self, m):
        self.gp_percent.set(m["gp"]); self.alpha_deg.set(m["alpha"]); self.bw_gain.set(m["bw"])
        self.mu.set(m["mu"]); self.sigma.set(m["sigma"])
        self.c.set(m["c"]); self.d.set(m["d"]); self.p.set(m["p"])
        self.a0.set(m["a0"]); self.a1.set(m["a1"]); self.krep.set(m["krep"])
        self._update_model()

    def _get_sliders(self):
        return dict(
            gp=self.gp_percent.get(), alpha=self.alpha_deg.get(), bw=self.bw_gain.get(),
            mu=self.mu.get(), sigma=self.sigma.get(), c=self.c.get(), d=self.d.get(), p=self.p.get(),
            a0=self.a0.get(), a1=self.a1.get(), krep=self.krep.get()
        )

    def _update_model(self):
        m = self.model
        m.mu, m.sigma, m.c, m.d, m.p, m.BW_gain = map(float,
            [self.mu.get(), self.sigma.get(), self.c.get(), self.d.get(), self.p.get(), self.bw_gain.get()])
        m.alpha0, m.alpha1, m.K_rep = radians(float(self.a0.get())), radians(float(self.a1.get())), float(self.krep.get())
        self.update_static_plots()

    def _shade_gait(self):
        phases = [(0,10,"HS"), (10,50,"Stance"), (50,60,"TO"), (60,100,"Swing")]
        for a,b,name in phases:
            self.ax_tau.axvspan(a, b, alpha=0.07)
            self.ax_tau.text((a+b)/2, 0.98, name,
                             transform=self.ax_tau.get_xaxis_transform(),
                             ha="center", va="top", fontsize=8)

    def _on_move_tau(self, ev):
        if ev.inaxes != self.ax_tau: return
        self.update_static_plots()
        try:
            self.ax_tau.axvline(ev.xdata, ls=":", lw=1)
            self.ax_tau.axhline(ev.ydata, ls=":", lw=1)
            self.canvas.draw_idle()
        except Exception:
            pass

    def update_static_plots(self):
        GP = np.linspace(0, 1, 600)
        alpha = radians(float(self.alpha_deg.get()))
        gp_now = float(self.gp_percent.get()) / 100
        tau = self.model.tau(GP, alpha)

        self.ax_tau.clear()
        self.ax_tau.plot(GP * 100, tau, lw=2, label="τ(GP, α)")
        self.ax_tau.axvline(gp_now * 100, ls="--", lw=1)
        self.ax_tau.scatter(gp_now * 100, self.model.tau(gp_now, alpha), s=55, zorder=5, label="Now")

        SAFE_MIN, SAFE_MAX = -0.6, 0.8
        self.ax_tau.axhspan(SAFE_MIN, SAFE_MAX, alpha=0.08)
        self.ax_tau.text(0.99, 0.99, "Safe band", transform=self.ax_tau.transAxes, ha="right", va="top", fontsize=8)

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
        self.ax_phase.scatter([gp_now * 100], [0.5], s=60, zorder=5)
        self.ax_phase.set_xlim(0, 100); self.ax_phase.get_yaxis().set_visible(False)
        self.ax_phase.set_xticks([0, 25, 50, 75, 100]); self.ax_phase.tick_params(axis='x', labelsize=8)

        self.ax_f.clear(); self.ax_f.plot(GP * 100, self.model.f(GP)); self.ax_f.set_title("f(GP)"); self.ax_f.grid(True, alpha=0.3)
        self.ax_s.clear(); self.ax_s.plot(GP * 100, self.model.s(GP)); self.ax_s.set_title("s(GP)"); self.ax_s.grid(True, alpha=0.3)
        a_scan = np.deg2rad(np.linspace(-40, 40, 400))
        self.ax_r.clear(); self.ax_r.plot(np.rad2deg(a_scan), self.model.r(a_scan)); self.ax_r.axvline(np.rad2deg(alpha), ls="--", lw=1)
        self.ax_r.set_title("r(α)"); self.ax_r.grid(True, alpha=0.3)

        if self.prev_tau is not None:
            GPp, taup = self.prev_tau
            self.ax_tau.plot(GPp * 100, taup, lw=1, linestyle="--", label="Frozen τ")

        self.ax_tau.legend(loc="upper left", fontsize=8)
        self.status.config(text=f"GP={self.gp_percent.get():.1f}% | α={self.alpha_deg.get():.1f}° | BW={self.bw_gain.get():.2f}")
        self.canvas.draw_idle()

    def _imu_loop(self):
        if self.streaming:
            if not hasattr(self, "t0"):
                self.t0 = time.perf_counter()
            if not hasattr(self, "_next_t"):
                self._next_t = 0
            if not hasattr(self, "tbuf"):
                self.window_sec, self.fs = 10, 50
                self.dt, self.max_samples = 1/self.fs, int(self.window_sec * self.fs)
                self.tbuf, self.accX, self.accY, self.accZ = [deque(maxlen=self.max_samples) for _ in range(4)]
                self.angX, self.angZ, self.theta, self.alphaZ = [deque(maxlen=self.max_samples) for _ in range(4)]

            while self._next_t <= (time.perf_counter() - self.t0):
                t = self._next_t
                ax = 0.20*np.sin(2*np.pi*t) + 0.01*np.random.randn()
                ay = 0.10*np.cos(4*np.pi*t) + 0.01*np.random.randn()
                az = 1.00 + 0.05*np.sin(2*np.pi*t) + 0.01*np.random.randn()
                wx = 2.0*np.sin(2*np.pi*t) + 0.05*np.random.randn()
                wz = 1.5*np.cos(2*np.pi*t) + 0.05*np.random.randn()
                th = 15.0*np.sin(2*np.pi*t) + 0.5*np.random.randn()
                al = 10.0*np.cos(2*np.pi*t) + 0.5*np.random.randn()

                self.tbuf.append(t)
                for buf, val in zip([self.accX,self.accY,self.accZ,self.angX,self.angZ,self.theta,self.alphaZ],
                                    [ax,ay,az,wx,wz,th,al]):
                    buf.append(val)
                self._next_t += 1/self.fs

            if self.tbuf:
                t0_view = self.tbuf[-1] - self.window_sec
                x = np.array(self.tbuf) - max(t0_view, 0)
            else:
                x = np.array([0.0])
            self._draw_imu(x)

        self.after(20, self._imu_loop)

    def _draw_imu(self, x):
        self.ax_accel.clear()
        self.ax_accel.plot(x, list(self.accX), label="accX")
        self.ax_accel.plot(x, list(self.accY), label="accY")
        self.ax_accel.plot(x, list(self.accZ), label="accZ")
        self.ax_accel.legend(loc="upper right", fontsize=8)
        self.ax_accel.set_ylabel("g"); self.ax_accel.set_title("Accelerometer [X, Y, Z]")
        self.ax_accel.set_xlim(0, self.window_sec); self.ax_accel.grid(True, alpha=0.25)

        self.ax_speed.clear()
        self.ax_speed.plot(x, list(self.angX), label="ωx")
        self.ax_speed.plot(x, list(self.angZ), label="ωz")
        self.ax_speed.legend(loc="upper right", fontsize=8)
        self.ax_speed.set_ylabel("rad/s"); self.ax_speed.set_title("Angular Speed [X, Z]")
        self.ax_speed.set_xlim(0, self.window_sec); self.ax_speed.grid(True, alpha=0.25)

        self.ax_angle.clear()
        self.ax_angle.plot(x, list(self.theta), label="θ (X)")
        self.ax_angle.plot(x, list(self.alphaZ), label="α (Z)")
        self.ax_angle.legend(loc="upper right", fontsize=8)
        self.ax_angle.set_ylabel("deg"); self.ax_angle.set_xlabel("Time [s]")
        self.ax_angle.set_title("Angle Position [θ, α]")
        self.ax_angle.set_xlim(0, self.window_sec); self.ax_angle.grid(True, alpha=0.25)

        self.canvas_imu.draw_idle()


# ======================
# Notebook shell
# ======================
class App(tk.Tk):
    def __init__(self, logo_path=None):
        super().__init__()
        self.title("Ankle Exoskeleton Suite • Notebook")
        self.geometry("1280x880")
        self.minsize(1100, 720)
        try:
            ttk.Style(self).theme_use("clam")
        except tk.TclError:
            pass

        nb = ttk.Notebook(self); nb.pack(fill=tk.BOTH, expand=True)

        tab1 = ttk.Frame(nb); nb.add(tab1, text="Home")
        HomePage(tab1, logo_path=logo_path)

        tab2 = ttk.Frame(nb); nb.add(tab2, text="Assistive Torque GUI")
        TorqueGUI(tab2)

        nb.select(tab1)


if __name__ == "__main__":
    # Set logo_path to a real image if you want it on the Home tab
    app = App(logo_path=r"C:\Users\Tinsae Tesfamichael\Desktop\Thesis\[_Final_code]\Final_Code\Final\src\icon.png")
    app.mainloop()
