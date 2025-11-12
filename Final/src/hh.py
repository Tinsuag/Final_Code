import tkinter as tk
from PIL import Image, ImageTk

img_path = r"C:\Users\Tinsae Tesfamichael\Desktop\Thesis\[_Final_code]\Final\src\icon.png"

# Open image with PIL
pil_image = Image.open(img_path)

# Calculate new dimensions (halving both width and height)
new_width = pil_image.width // 10
new_height = pil_image.height // 10

# Resize image proportionally
resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

root = tk.Tk()
# Convert PIL image to PhotoImage
image = ImageTk.PhotoImage(resized_image)
dimensions = f"image size: {new_width}x{new_height}"
label = tk.Label(root, compound="top", image=image, text=dimensions)
label.pack()
root.mainloop()
