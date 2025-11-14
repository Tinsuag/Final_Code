import tkinter as tk
from tkinter import ttk
import numpy as np
from math import radians
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==============================
# Assistive torque model
# ==============================
class AssistiveTorque:
    """
    τ(GP, α) = BW_gain · f(GP) · s(GP) · r(α)^(-1)
    Defaults from your pages:
      f(GP) = exp( - (GP-μ)^2 / (2σ^2) )             with μ=0.5, σ=0.17
      s(GP) = (p+1)/(exp((GP-c)/d) + 1) - p           with c=0.6, d=0.02, p=0.8
      r(α)  = 1 + Krep*(exp(α0-α) + exp(α-α1))        limits α0=-10°, α1=10°, Krep=1.0
    Angles are radians internally. GP is 0..1.
    """
    def __init__(self,
                 mu=0.5, sigma=0.17,
                 c=0.6, d=0.02, p=0.8,
                 alpha0_deg=-10.0, alpha1_deg=10.0,
                 K_rep=1.0,
                 BW_gain=1.0):
        self.mu = mu
        self.sigma = sigma
        self.c = c
        self.d = d
        self.p = p
        self.alpha0 = np.deg2rad(alpha0_deg)
        self.alpha1 = np.deg2rad(alpha1_deg)
        self.K_rep = K_rep
        self.BW_gain = BW_gain

    def f(self, GP):
        GP = np.asarray(GP)
        return np.exp(-((GP - self.mu) ** 2) / (2.0 * self.sigma ** 2))

    def s(self, GP):
        GP = np.asarray(GP)
        return (self.p + 1.0) / (np.exp((GP - self.c) / self.d) + 1.0) - self.p

    def r(self, alpha):
        a = np.asarray(alpha)
        return 1.0 + self.K_rep * (np.exp(self.alpha0 - a) + np.exp(a - self.alpha1))

    def tau(self, GP, alpha):
        return self.BW_gain * self.f(GP) * self.s(GP) / self.r(alpha)


# ==============================
# Tkinter UI + Matplotlib plots
# ==============================
class TorqueApp(tk.Tk):
    def __init__(self):
        super().__init__() #  Initialize parent tk.Tk class 
        self.title("Assistive Torque vs Gait Phase")
        self.geometry("1000x950")

        # Model with defaults
        self.model = AssistiveTorque()

        # Left: controls
        ctrl = ttk.Frame(self, padding=10)
        ctrl.grid(row=0, column=0, sticky="nsew")
        self.grid_columnconfigure(1, weight=1) # Make right column expandable
        self.grid_rowconfigure(0, weight=1)    # Make row expandable
        
        self.grid_columnconfigure(0, weight=0) # Left column fixed width
        self.grid_columnconfigure(1, weight=1) # Right column expandable

        # Right: plots
        plot = ttk.Frame(self)
        plot.grid(row=0, column=1, sticky="nsew")  

        # --- Matplotlib figure (1 big + 3 small axes) ---
        self.fig = Figure(figsize=(8.2, 6.4), dpi=100) # 820x640 pixels
        gs = self.fig.add_gridspec(2, 2, height_ratios=[2.5, 1.2], width_ratios=[2, 1], hspace=0.45, wspace=0.35) 
        self.ax_tau = self.fig.add_subplot(gs[0, :])    # wide top row: τ(GP,α)
        self.ax_f   = self.fig.add_subplot(gs[1, 0])    # bottom-left: f(GP)
        self.ax_s   = self.fig.add_subplot(gs[1, 1])    # bottom-right top: s(GP)
        # We'll show r(α) as a vertical marker on ax_tau (angle limit band) and a small inline text

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- Controls (sliders) ---
        # Helper to add a labeled slider
        def add_slider(parent, text, frm, to, init, resolution, cmd, unit=""):
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=4)
            lab = ttk.Label(row, text=f"{text}:")
            lab.pack(side=tk.LEFT)
            val = ttk.Label(row, width=10, anchor="e")
            val.pack(side=tk.RIGHT)
            s = ttk.Scale(row, from_=frm, to=to, value=init, orient=tk.HORIZONTAL, command=lambda e: cmd() or val.config(text=f"{s.get():.3f}{unit}"))
            s.pack(fill=tk.X, padx=6)
            val.config(text=f"{s.get():.3f}{unit}")
            s.configure(length=260)
            return s

        self.alpha_deg = add_slider(ctrl, "α (deg)", -30, 30, 0.0, 0.1, self.update_plots, "°")
        self.bw_gain   = add_slider(ctrl, "BW gain", 0.0, 2.0, 1.0, 0.01, self._update_model)
        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=6)

        self.mu      = add_slider(ctrl, "μ", 0.0, 1.0, 0.5, 0.001, self._update_model)
        self.sigma   = add_slider(ctrl, "σ", 0.05, 0.40, 0.17, 0.001, self._update_model)
        self.c       = add_slider(ctrl, "c", 0.0, 1.0, 0.6, 0.001, self._update_model)
        self.d       = add_slider(ctrl, "d", 0.005, 0.10, 0.02, 0.001, self._update_model)
        self.p       = add_slider(ctrl, "p", 0.0, 1.5, 0.8, 0.001, self._update_model)
        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=6)

        self.a0      = add_slider(ctrl, "α0 (deg)", -40, 0, -10.0, 0.1, self._update_model, "°")
        self.a1      = add_slider(ctrl, "α1 (deg)", 0, 40, 10.0, 0.1, self._update_model, "°")
        self.krep    = add_slider(ctrl, "Krep", 0.0, 5.0, 1.0, 0.01, self._update_model)

        # First draw
        self.update_plots()

    # Pull slider values into model, then redraw
    def _update_model(self):
        self.model.mu     = float(self.mu.get())
        self.model.sigma  = float(self.sigma.get())
        self.model.c      = float(self.c.get())
        self.model.d      = float(self.d.get())
        self.model.p      = float(self.p.get())
        self.model.BW_gain = float(self.bw_gain.get())
        self.model.alpha0 = np.deg2rad(float(self.a0.get()))
        self.model.alpha1 = np.deg2rad(float(self.a1.get()))
        self.model.K_rep   = float(self.krep.get())
        self.update_plots()

    def update_plots(self):
        GP = np.linspace(0.0, 1.0, 400)
        alpha = radians(float(self.alpha_deg.get()))
        f = self.model.f(GP)
        s = self.model.s(GP)
        r = self.model.r(alpha)
        tau = self.model.tau(GP, alpha)

        # --- τ(GP, α) ---
        self.ax_tau.clear()
        self.ax_tau.plot(GP * 100.0, tau)  # GP in percent for the x-axis
        self.ax_tau.set_title("Torque τ(GP, α)")
        self.ax_tau.set_xlabel("GP [%]")
        self.ax_tau.set_ylabel("Torque (scaled)")
        self.ax_tau.grid(True, alpha=0.3)

        # Show α band (limits) as a legend text
        a0_deg = np.rad2deg(self.model.alpha0)
        a1_deg = np.rad2deg(self.model.alpha1)
        self.ax_tau.text(0.02, 0.95,
                         f"α = {np.rad2deg(alpha):.1f}° | α₀={a0_deg:.1f}°, α₁={a1_deg:.1f}°  | r(α)={float(r):.3f}",
                         transform=self.ax_tau.transAxes,
                         va="top", ha="left", fontsize=9,
                         bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

        # --- f(GP) ---
        self.ax_f.clear()
        self.ax_f.plot(GP * 100.0, f)
        self.ax_f.set_title("f(GP) — Gaussian peak")
        self.ax_f.set_xlabel("GP [%]")
        self.ax_f.set_ylabel("f")
        self.ax_f.grid(True, alpha=0.3)

        # --- s(GP) ---
        self.ax_s.clear()
        self.ax_s.plot(GP * 100.0, s)
        self.ax_s.set_title("s(GP) — smooth sign / range")
        self.ax_s.set_xlabel("GP [%]")
        self.ax_s.set_ylabel("s")
        self.ax_s.grid(True, alpha=0.3)

        self.canvas.draw_idle()


if __name__ == "__main__":
    app = TorqueApp()
    app.mainloop()
