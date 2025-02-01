import torch
import torchvision.transforms as transforms
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import torch.nn as nn
import colorsys

# Check if CUDA is available and use GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Colorization Model
class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=4, dilation=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 3, kernel_size=5, stride=1, padding=4, dilation=2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = nn.functional.relu(self.bn4(self.conv4(x)))
        x = torch.sigmoid(self.conv5(x))
        return x

# HSV Presets for different eras
HSV_PRESETS = {
    "1920s": (0.02, 0.8, 0.9),
    "1950s": (0.05, 1.2, 1.1),
    "1980s": (0.1, 1.5, 1.3),
    "2000s": (0.0, 1.0, 1.0)
}

def load_model():
    model = torch.load(r"Models\task2_model.pt", map_location=device)
    model.to(device)
    model.eval()
    return model

def colorize_image(image, model):
    grayscale_image = image.convert("L")
    tensor = transforms.ToTensor()(grayscale_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        colorized = model(tensor).squeeze(0).cpu().numpy().transpose(1, 2, 0)

    colorized = (colorized * 255).astype(np.uint8)
    return Image.fromarray(colorized)

def adjust_hsv(image, hue_shift, saturation_scale, value_scale):
    np_img = np.array(image.convert("RGB"))
    hsv_img = np.apply_along_axis(lambda x: colorsys.rgb_to_hsv(x[0]/255.0, x[1]/255.0, x[2]/255.0), 2, np_img)
    hsv_img[..., 0] = ((hsv_img[..., 0] + hue_shift) % 1.0)
    hsv_img[..., 1] = np.clip(hsv_img[..., 1] * saturation_scale, 0, 1)
    hsv_img[..., 2] = np.clip(hsv_img[..., 2] * value_scale, 0, 1)
    rgb_img = np.apply_along_axis(lambda x: np.array(colorsys.hsv_to_rgb(x[0], x[1], x[2])) * 255, 2, hsv_img)
    return Image.fromarray(rgb_img.astype(np.uint8))

def apply_preset():
    preset = preset_var.get()
    if preset in HSV_PRESETS:
        hue_slider.set(HSV_PRESETS[preset][0] * 360)
        saturation_slider.set(HSV_PRESETS[preset][1] * 100)
        value_slider.set(HSV_PRESETS[preset][2] * 100)

def adjust_and_display_hsv():
    global colorized_img
    if colorized_img:
        hue_shift = hue_slider.get() / 360.0
        saturation_scale = saturation_slider.get() / 100.0
        value_scale = value_slider.get() / 100.0
        adjusted_img = adjust_hsv(colorized_img, hue_shift, saturation_scale, value_scale)
        adjusted_display = ImageTk.PhotoImage(adjusted_img)
        panel.config(image=adjusted_display)
        panel.image = adjusted_display

def load_image():
    global img, panel
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img_display = ImageTk.PhotoImage(img)
        panel.config(image=img_display)
        panel.image = img_display

def process_image():
    global img, colorized_img
    if img:
        model = load_model()
        colorized_img = colorize_image(img, model)
        display_img = ImageTk.PhotoImage(colorized_img)
        panel.config(image=display_img)
        panel.image = display_img

def save_image():
    if colorized_img:
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if file_path:
            colorized_img.save(file_path)

# Initialize the Tkinter GUI
root = tk.Tk()
root.title("Image Colourization")
root.geometry("800x600")

preset_var = tk.StringVar(value="2000s")
tk.Label(root, text="Select Era Preset:").grid(row=0, column=0, pady=5)
preset_dropdown = ttk.Combobox(root, textvariable=preset_var, values=list(HSV_PRESETS.keys()))
preset_dropdown.grid(row=0, column=1, pady=5)
preset_dropdown.bind("<<ComboboxSelected>>", lambda e: apply_preset())

tk.Button(root, text="Load Image", command=load_image).grid(row=1, column=0, pady=5)
tk.Button(root, text="Apply Colorization", command=process_image).grid(row=2, column=0, pady=5)
tk.Button(root, text="Save Image", command=save_image).grid(row=3, column=0, pady=5)

panel = tk.Label(root)
panel.grid(row=4, column=0, pady=5)

hue_slider = tk.Scale(root, from_=0, to=360, orient="horizontal", label="Hue")
hue_slider.grid(row=5, column=0)
saturation_slider = tk.Scale(root, from_=0, to=200, orient="horizontal", label="Saturation")
saturation_slider.grid(row=6, column=0)
value_slider = tk.Scale(root, from_=0, to=200, orient="horizontal", label="Value")
value_slider.grid(row=7, column=0)

tk.Button(root, text="Adjust HSV", command=adjust_and_display_hsv).grid(row=8, column=0, pady=5)
root.mainloop()
