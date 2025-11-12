import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg") # 
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class PlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Plotter")

        self.frame = ttk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.plot_button = ttk.Button(self.frame, text="Load Data and Plot", command=self.load_and_plot)
        self.plot_button.pack(pady=10)

        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(fill=tk.BOTH, expand=True)

    def load_and_plot(self):
        file_path = filedialog.askopenfilename(title="Select Data File", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if not file_path:
            return

        try:
            data = np.loadtxt(file_path)
            x = data[:, 0]
            y = data[:, 1]

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(x, y, marker='o', linestyle='-')
            ax.set_title("Data Plot")
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
            ax.grid(True)

            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load or plot data: {e}")
            


if __name__ == "__main__":
    root = tk.Tk()
    app = PlotApp(root)
    root.mainloop()