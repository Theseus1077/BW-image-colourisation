import torch
import torchvision.transforms as transforms
import tkinter as tk
from tkinter import filedialog, ttk, Canvas, Scrollbar
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import torch.nn as nn

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

# Load the model
def load_model():
    model = torch.load(r"Models\Task3_model.pt", map_location=device)
    model.to(device)
    model.eval()
    return model

# Colorize image function
def colorize_image(image, model):
    grayscale_image = image.convert("L")
    tensor = transforms.ToTensor()(grayscale_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        colorized = model(tensor).squeeze(0).cpu().numpy().transpose(1, 2, 0)

    colorized = (colorized * 255).astype(np.uint8)
    return Image.fromarray(colorized)

# Mouse selection variables
start_x, start_y, end_x, end_y = None, None, None, None
draw_rectangle = None

# Function to handle mouse press
def on_mouse_press(event):
    global start_x, start_y, draw_rectangle
    start_x, start_y = event.x, event.y
    if draw_rectangle:
        canvas.delete(draw_rectangle)

# Function to handle mouse drag (draws rectangle)
def on_mouse_drag(event):
    global draw_rectangle, start_x, start_y
    if draw_rectangle:
        canvas.delete(draw_rectangle)
    draw_rectangle = canvas.create_rectangle(start_x, start_y, event.x, event.y, outline="red", width=2, dash=(4, 2))

# Function to handle mouse release
def on_mouse_release(event):
    global start_x, start_y, end_x, end_y, img, colorized_img
    end_x, end_y = event.x, event.y
    
    if img:
        model = load_model()
        grayscale_img = img.convert("L")
        
        start_x_clip = max(0, min(start_x, img.width - 1))
        start_y_clip = max(0, min(start_y, img.height - 1))
        end_x_clip = max(0, min(end_x, img.width - 1))
        end_y_clip = max(0, min(end_y, img.height - 1))
        
        selected_region = grayscale_img.crop((start_x_clip, start_y_clip, end_x_clip, end_y_clip))
        colorized_region = colorize_image(selected_region, model)
        
        colorized_img = grayscale_img.convert("RGB")
        colorized_img.paste(colorized_region, (start_x_clip, start_y_clip))
        
        display_img = ImageTk.PhotoImage(colorized_img)
        canvas.itemconfig(image_on_canvas, image=display_img)
        canvas.image = display_img

# Load image function
def load_image():
    global img, canvas, image_on_canvas
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).convert("RGB")
        img_display = ImageTk.PhotoImage(img)
        
        canvas.config(scrollregion=canvas.bbox("all"))
        canvas.itemconfig(image_on_canvas, image=img_display)
        canvas.image = img_display

# Save image function
def save_image():
    if colorized_img:
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if file_path:
            colorized_img.save(file_path)

# Initialize the Tkinter GUI
root = tk.Tk()
root.title("Interactive Image Colourization")
root.geometry("1000x700")

# Main container frame
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Canvas with scrollbars
canvas_frame = tk.Frame(main_frame)
canvas_frame.pack(fill=tk.BOTH, expand=True)

canvas = Canvas(canvas_frame, bg="white")
scroll_x = Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=canvas.xview)
scroll_y = Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
canvas.configure(xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)
scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
canvas.pack(fill=tk.BOTH, expand=True)

# Placeholder image
dummy_img = Image.new("RGB", (800, 600), "gray")
dummy_display = ImageTk.PhotoImage(dummy_img)
image_on_canvas = canvas.create_image(0, 0, anchor=tk.NW, image=dummy_display)
canvas.image = dummy_display

# Buttons Frame
buttons_frame = tk.Frame(root)
buttons_frame.pack(fill=tk.X, pady=10)

tk.Button(buttons_frame, text="Load Image", command=load_image).pack(side=tk.LEFT, padx=10)
tk.Button(buttons_frame, text="Save Image", command=save_image).pack(side=tk.LEFT, padx=10)

# Bind mouse events to canvas
canvas.bind("<ButtonPress-1>", on_mouse_press)
canvas.bind("<B1-Motion>", on_mouse_drag)
canvas.bind("<ButtonRelease-1>", on_mouse_release)

root.mainloop()
