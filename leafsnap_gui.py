import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all logs, 1=info, 2=warnings, 3=errors
import tensorflow as tf

# leafsnap_gui_aesthetic.py

TF_ENABLE_ONEDNN_OPTS=0
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageTk
import numpy as np

# Load model
model = MobileNetV2(weights='imagenet')

# Function to predict image
def predict_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        preds = model.predict(x)
        decoded = decode_predictions(preds, top=3)[0]
        
        # Display predictions
        result_text = "Top 3 Predictions:\n"
        for i, (imagenet_id, label, score) in enumerate(decoded):
            result_text += f"{i+1}. {label} ({score*100:.2f}%)\n"
        
        result_label.config(text=result_text)
    
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to open file dialog
def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
    )
    if file_path:
        # Display image
        img = Image.open(file_path)
        img.thumbnail((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        
        # Predict
        predict_image(file_path)

# Tkinter GUI
root = tk.Tk()
root.title("🍃 LeafSnap Mini 🌿")
root.geometry("550x650")
root.configure(bg="black")
root.resizable(False, False)

# Heading
heading = tk.Label(root, text="LeafSnap Mini", font=("Comic Sans MS", 24, "bold"), bg="black", fg="#39FF14")
heading.pack(pady=15)

# Subheading / Description
subheading = tk.Label(
    root, text="Upload a plant or animal image to see predictions!",
    font=("Helvetica", 12, "italic"), bg="black", fg="#FFFFFF"
)
subheading.pack(pady=5)

# Upload Button
upload_btn = tk.Button(
    root, text="📂 Upload Image", command=upload_image,
    bg="#FF6F61", fg="white", font=("Helvetica", 14, "bold"),
    activebackground="#FF3B30", relief="raised", bd=3
)
upload_btn.pack(pady=15)

# Image Display
image_frame = tk.Frame(root, bg="black", bd=2, relief="sunken")
image_frame.pack(pady=10)
image_label = tk.Label(image_frame, bg="black")
image_label.pack()

# Prediction Result
result_label = tk.Label(root, text="Select an image to predict.", font=("Helvetica", 14), bg="black", fg="#39FF14", justify="left")
result_label.pack(pady=15)

# Optional: Add a small aesthetic doodle / GIF (animated) at the bottom
try:
    doodle_img = Image.open("C:/Users/fatim/Downloads/doodle.gif")  # Place doodle.gif in the project folder and add file path
    doodle_tk = ImageTk.PhotoImage(doodle_img)
    doodle_label = tk.Label(root, image=doodle_tk, bg="black")
    doodle_label.pack(pady=10)
except:
    pass  # If no GIF, skip

root.mainloop()
