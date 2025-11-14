import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import time
import numpy as np
from math import radians
from collections import deque
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =========================================================
# Assistive Torque Model
# =========================================================
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


# =========================================================
# Notebook App
# =========================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Assistive Torque • Notebook")
        self.geometry("1280x880")
        self.minsize(1100, 720)

        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self.style.configure("Toolbar.TFrame", padding=(8, 6))
        self.style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
        self.style.configure("Hint.TLabel", foreground="#666")

        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True)

        # Tab 1 — empty
        tab1 = ttk.Frame(nb)
        nb.add(tab1, text="Home")
        ttk.Label(tab1, text="(Empty tab — drop content here later)", style="Hint.TLabel").pack(
            padx=16, pady=16, anchor="w"
        )

        # Tab 2 — GUI
        tab2 = ttk.Frame(nb)
        nb.add(tab2, text="Assistive Torque GUI")
        TorqueGUI(tab2)  # build GUI inside this tab
        nb.select(tab1)  # open tab1 first


# =========================================================
# Full GUI in a Frame
# =========================================================
class TorqueGUI(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill=tk.BOTH, expand=True)
        self.model = AssistiveTorque()

        # ---------- State ----------
        self.dark = False
        self.streaming = True

        # ---------- Toolbar ----------
        toolbar = ttk.Frame(self, style="Toolbar.TFrame")
        toolbar.pack(fill=tk.X, side=tk.TOP)
        ttk.Label(toolbar, text="Assistive Torque Controller", style="Header.TLabel").pack(side=tk.LEFT)

        btns = ttk.Frame(toolbar)
        btns.pack(side=tk.RIGHT)
        self.play_btn = ttk.Button(btns, text="⏸ Pause IMU", command=self.toggle_stream)
        self.play_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="⟳ Reset", command=self.reset_params).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="☾ Dark/Light", command=self.toggle_dark).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="⬇ Save Figures", command=self.save_figs).pack(side=tk.LEFT, padx=4)

        # ---------- Split View (PanedWindow: left controls / right plots) ----------
        splitter = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        splitter.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        left_pane = ttk.Frame(splitter)
        right_pane = ttk.Frame(splitter)
        splitter.add(left_pane, weight=1)
        splitter.add(right_pane, weight=2)

        # ---------- LEFT: Controls ----------
        # Controls frame with two rows: (1) Parameters (scroll) (2) Torque plot container
        ctrl = ttk.Frame(left_pane)
        ctrl.pack(fill=tk.BOTH, expand=True)
        ctrl.grid_rowconfigure(0, weight=1)  # params (smaller)
        ctrl.grid_rowconfigure(1, weight=4)  # torque plot (bigger)
        ctrl.grid_columnconfigure(0, weight=1)

        # Parameters (scrollable, compact sliders)
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

        # Torque plot container
        Torq_Ctrl_plt = ttk.LabelFrame(ctrl, text="Assistive Torque Model")
        Torq_Ctrl_plt.grid(row=1, column=0, sticky="nsew", padx=6, pady=(4, 6))
        Torq_Ctrl_plt.grid_rowconfigure(0, weight=1)
        Torq_Ctrl_plt.grid_columnconfigure(0, weight=1)

        # ---------- RIGHT: Plots ----------
        plot = ttk.LabelFrame(right_pane, text="Plots")
        plot.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        plot.grid_rowconfigure(0, weight=3)
        plot.grid_rowconfigure(1, weight=2)
        plot.grid_columnconfigure(0, weight=1)

        # IMU signals
        IMU_signal_Plt = ttk.LabelFrame(plot, text="IMU Signals (10 s window)")
        IMU_signal_Plt.grid(row=1, column=0, sticky="nsew")
        IMU_signal_Plt.grid_rowconfigure(0, weight=1)
        IMU_signal_Plt.grid_columnconfigure(0, weight=1)

        # ---------- Matplotlib: τ + f/s/r stacked ----------
        self.fig = Figure(figsize=(6.6, 3.8), dpi=100, layout="constrained")
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[2.6, 1.2], hspace=0.28, wspace=0.28)
        self.ax_tau = self.fig.add_subplot(gs[:, 0])
        self.ax_f = self.fig.add_subplot(gs[0, 1])
        self.ax_s = self.fig.add_subplot(gs[1, 1])
        self.ax_r = self.fig.add_subplot(gs[2, 1])
        # Phase strip inset
        self.ax_phase = self.ax_tau.inset_axes([0.08, 0.03, 0.84, 0.12])

        self.canvas = FigureCanvasTkAgg(self.fig, master=Torq_Ctrl_plt)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # IMU figure
        self.fig_imu = Figure(figsize=(6.6, 3.2), dpi=100, layout="constrained")
        imu_gs = self.fig_imu.add_gridspec(3, 1, hspace=0.24)
        self.ax_accel = self.fig_imu.add_subplot(imu_gs[0, 0])
        self.ax_speed = self.fig_imu.add_subplot(imu_gs[1, 0])
        self.ax_angle = self.fig_imu.add_subplot(imu_gs[2, 0])
        self.canvas_imu = FigureCanvasTkAgg(self.fig_imu, master=IMU_signal_Plt)
        self.canvas_imu.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # ---------- Compact sliders ----------
        def add_slider(frame, label, frm, to, init, cmd, unit=""):
            row = ttk.Frame(frame, height=26)
            row.pack(fill=tk.X, pady=2, padx=6)
            row.pack_propagate(False)
            ttk.Label(row, text=label, width=14, anchor="w").pack(side=tk.LEFT)
            s = ttk.Scale(row, from_=frm, to=to, orient=tk.HORIZONTAL, length=170)
            val = ttk.Label(row, width=8, anchor="e")
            val.pack(side=tk.RIGHT)
            s.pack(side=tk.LEFT, padx=6)
            def _on_move(_=None):
                cmd()
                val.config(text=f"{s.get():.2f}{unit}")
            s.configure(command=_on_move)
            s.set(init)
            val.config(text=f"{init:.2f}{unit}")
            return s

        # — Core live controls —
        self.gp_percent = add_slider(self.params_inner, "GP (%)", 0, 100, 50, self.update_static_plots, "%")
        self.alpha_deg  = add_slider(self.params_inner, "α (deg)", -30, 30, 0, self.update_static_plots, "°")
        self.bw_gain    = add_slider(self.params_inner, "BW gain", 0, 2.5, 1, self._update_model)

        ttk.Separator(self.params_inner, orient="horizontal").pack(fill=tk.X, pady=3)

        # — f(GP) —
        self.mu      = add_slider(self.params_inner, "μ", 0, 1, 0.5, self._update_model)
        self.sigma   = add_slider(self.params_inner, "σ", 0.05, 0.4, 0.17, self._update_model)

        # — s(GP) —
        self.c       = add_slider(self.params_inner, "c", 0, 1, 0.6, self._update_model)
        self.d       = add_slider(self.params_inner, "d", 0.005, 0.1, 0.02, self._update_model)
        self.p       = add_slider(self.params_inner, "p", 0, 1.5, 0.8, self._update_model)

        ttk.Separator(self.params_inner, orient="horizontal").pack(fill=tk.X, pady=3)

        # — r(α) —
        self.a0      = add_slider(self.params_inner, "α0 (deg)", -40, 0, -10, self._update_model, "°")
        self.a1      = add_slider(self.params_inner, "α1 (deg)", 0, 40, 10, self._update_model, "°")
        self.krep    = add_slider(self.params_inner, "Krep", 0, 5, 1, self._update_model)

        # ---------- IMU sim setup ----------
        self.window_sec, self.fs = 10, 50
        self.dt, self.max_samples = 1/self.fs, int(self.window_sec * self.fs)
        self.tbuf, self.accX, self.accY, self.accZ = [deque(maxlen=self.max_samples) for _ in range(4)]
        self.angX, self.angZ, self.theta, self.alphaZ = [deque(maxlen=self.max_samples) for _ in range(4)]
        self.t0 = time.perf_counter()
        self._next_t = 0

        # First draw + loop
        self.update_static_plots()
        self._imu_loop()

    # ---------- Toolbar actions ----------
    def toggle_stream(self):
        self.streaming = not self.streaming
        self.play_btn.config(text="▶ Resume IMU" if not self.streaming else "⏸ Pause IMU")

    def reset_params(self):
        # Reset sliders to defaults
        for s, val in [
            (self.gp_percent, 50), (self.alpha_deg, 0), (self.bw_gain, 1),
            (self.mu, 0.5), (self.sigma, 0.17),
            (self.c, 0.6), (self.d, 0.02), (self.p, 0.8),
            (self.a0, -10), (self.a1, 10), (self.krep, 1),
        ]:
            s.set(val)
        self._update_model()

    def toggle_dark(self):
        self.dark = not self.dark
        # Redraw figures with dark/light backgrounds
        self.update_static_plots(force_style=True)
        self._redraw_imu_axes()

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
            # Save IMU with suffix
            base, ext = path.rsplit(".", 1)
            imu_path = f"{base}_imu.{ext}"
            self.fig_imu.savefig(imu_path, dpi=150, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Figures saved:\n• {path}\n• {imu_path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    # ---------- Model sync ----------
    def _update_model(self):
        m = self.model
        m.mu, m.sigma, m.c, m.d, m.p, m.BW_gain = map(float,
            [self.mu.get(), self.sigma.get(), self.c.get(), self.d.get(), self.p.get(), self.bw_gain.get()])
        m.alpha0, m.alpha1, m.K_rep = radians(float(self.a0.get())), radians(float(self.a1.get())), float(self.krep.get())
        self.update_static_plots()

    # ---------- Theme helpers ----------
    def _apply_axes_theme(self, axes):
        if not isinstance(axes, (list, tuple)):
            axes = [axes]
        if self.dark:
            face = "#181a1b"
            edge = "#888"
            txt  = "#ddd"
        else:
            face = "white"
            edge = "#333"
            txt  = "#111"
        for ax in axes:
            ax.set_facecolor(face)
            for spine in ax.spines.values():
                spine.set_color(edge)
            ax.tick_params(colors=txt, which="both")
            ax.title.set_color(txt)
            ax.xaxis.label.set_color(txt)
            ax.yaxis.label.set_color(txt)

    # ---------- Static plots (τ + inset phase strip; f,s,r vertical) ----------
    def update_static_plots(self, force_style=False):
        GP = np.linspace(0, 1, 600)
        alpha = radians(float(self.alpha_deg.get()))
        gp_now = float(self.gp_percent.get()) / 100
        tau = self.model.tau(GP, alpha)

        # Clear & theme
        for ax in [self.ax_tau, self.ax_f, self.ax_s, self.ax_r]:
            ax.clear()

        # τ(GP, α)
        self.ax_tau.plot(GP * 100, tau, lw=2)
        self.ax_tau.axvline(gp_now * 100, ls="--", lw=1)
        self.ax_tau.scatter(gp_now * 100, self.model.tau(gp_now, alpha), s=55, zorder=5)
        self.ax_tau.grid(True, which="major", alpha=0.30)
        self.ax_tau.grid(True, which="minor", alpha=0.15)
        self.ax_tau.minorticks_on()
        self.ax_tau.set_xlim(0, 100)
        pad = (np.max(np.abs(tau)) * 0.15) + 1e-9
        self.ax_tau.set_ylim(np.min(tau) - pad, np.max(tau) + pad)
        self.ax_tau.set_title("Torque τ(GP, α)")
        self.ax_tau.set_xlabel("Gait phase GP [%]")
        self.ax_tau.set_ylabel("Torque (scaled)")

        # Inset phase strip
        self.ax_phase.clear()
        self.ax_phase.plot([0, 100], [0.5, 0.5], lw=2)
        self.ax_phase.scatter([gp_now * 100], [0.5], s=60, zorder=5)
        self.ax_phase.set_xlim(0, 100)
        self.ax_phase.get_yaxis().set_visible(False)
        self.ax_phase.set_xticks([0, 25, 50, 75, 100])
        self.ax_phase.tick_params(axis='x', labelsize=8)

        # Right stack
        self.ax_f.plot(GP * 100, self.model.f(GP)); self.ax_f.set_title("f(GP)"); self.ax_f.grid(True, alpha=0.3)
        self.ax_s.plot(GP * 100, self.model.s(GP)); self.ax_s.set_title("s(GP)"); self.ax_s.grid(True, alpha=0.3)
        a_scan = np.deg2rad(np.linspace(-40, 40, 400))
        self.ax_r.plot(np.rad2deg(a_scan), self.model.r(a_scan)); self.ax_r.set_title("r(α)"); self.ax_r.grid(True, alpha=0.3)

        # Theme
        if self.dark or force_style:
            self._apply_axes_theme([self.ax_tau, self.ax_f, self.ax_s, self.ax_r, self.ax_phase])

        self.canvas.draw_idle()

    # ---------- IMU live update loop (10 s rolling window) ----------
    def _imu_loop(self):
        if self.streaming:
            now = time.perf_counter() - self.t0
            while self._next_t <= now:
                t = self._next_t
                # Accel (g)
                ax = 0.20*np.sin(2*np.pi*t) + 0.01*np.random.randn()
                ay = 0.10*np.cos(4*np.pi*t) + 0.01*np.random.randn()
                az = 1.00 + 0.05*np.sin(2*np.pi*t) + 0.01*np.random.randn()
                # Angular speed (rad/s)
                wx = 2.0*np.sin(2*np.pi*t) + 0.05*np.random.randn()
                wz = 1.5*np.cos(2*np.pi*t) + 0.05*np.random.randn()
                # Angles (deg)
                th = 15.0*np.sin(2*np.pi*t) + 0.5*np.random.randn()
                al = 10.0*np.cos(2*np.pi*t) + 0.5*np.random.randn()

                self.tbuf.append(t)
                for buf, val in zip([self.accX,self.accY,self.accZ,self.angX,self.angZ,self.theta,self.alphaZ],
                                    [ax,ay,az,wx,wz,th,al]):
                    buf.append(val)
                self._next_t += 1/self.fs

            # Prepare relative x (last 10 s)
            if self.tbuf:
                t0_view = self.tbuf[-1] - self.window_sec
                x = np.array(self.tbuf) - max(t0_view, 0)
            else:
                x = np.array([0.0])

            self._draw_imu(x)

        self.after(int(1000/self.fs), self._imu_loop)

    def _redraw_imu_axes(self):
        self._apply_axes_theme([self.ax_accel, self.ax_speed, self.ax_angle])
        self.canvas_imu.draw_idle()

    def _draw_imu(self, x):
        # accel
        self.ax_accel.clear()
        self.ax_accel.plot(x, list(self.accX), label="accX")
        self.ax_accel.plot(x, list(self.accY), label="accY")
        self.ax_accel.plot(x, list(self.accZ), label="accZ")
        self.ax_accel.legend(loc="upper right", fontsize=8)
        self.ax_accel.set_ylabel("g")
        self.ax_accel.set_title("Accelerometer [X, Y, Z]")
        self.ax_accel.set_xlim(0, self.window_sec)
        self.ax_accel.grid(True, alpha=0.25)

        # angular speed
        self.ax_speed.clear()
        self.ax_speed.plot(x, list(self.angX), label="ωx")
        self.ax_speed.plot(x, list(self.angZ), label="ωz")
        self.ax_speed.legend(loc="upper right", fontsize=8)
        self.ax_speed.set_ylabel("rad/s")
        self.ax_speed.set_title("Angular Speed [X, Z]")
        self.ax_speed.set_xlim(0, self.window_sec)
        self.ax_speed.grid(True, alpha=0.25)

        # angles
        self.ax_angle.clear()
        self.ax_angle.plot(x, list(self.theta), label="θ (X)")
        self.ax_angle.plot(x, list(self.alphaZ), label="α (Z)")
        self.ax_angle.legend(loc="upper right", fontsize=8)
        self.ax_angle.set_ylabel("deg")
        self.ax_angle.set_xlabel("Time [s]")
        self.ax_angle.set_title("Angle Position [θ, α]")
        self.ax_angle.set_xlim(0, self.window_sec)
        self.ax_angle.grid(True, alpha=0.25)

        # Theme
        if self.dark:
            self._apply_axes_theme([self.ax_accel, self.ax_speed, self.ax_angle])

        self.canvas_imu.draw_idle()


# =========================================================
if __name__ == "__main__":
    app = App()
    app.mainloop()
