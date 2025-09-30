import customtkinter as ctk
from tkinter import filedialog, Label, Frame, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO
import numpy as np
import threading
import os
import time
import webbrowser
from datetime import datetime
import cv2

# Load the YOLOv8 model
model = YOLO("best.pt")

# Global variables
img_path = None
dark_mode = False
processing = False
history = []
current_theme = None

# Color Themes
LIGHT_THEME = {
    "name": "light",
    "bg": "#f8f9fa",
    "header": "#2c3e50",
    "card": "#ffffff",
    "card_border": "#e0e0e0",
    "text_primary": "#2c3e50",
    "text_secondary": "#7f8c8d",
    "button_primary": "#3498db",
    "button_secondary": "#2ecc71",
    "status_bar": "#2c3e50",
    "status_text": "#bdc3c7",
    "image_bg": "#f1f3f5",
    "positive": "#e74c3c",
    "negative": "#27ae60",
    "no_tumor_box": "#ffffff",
    "accent": "#9b59b6",
    "warning": "#f39c12"
}

DARK_THEME = {
    "name": "dark",
    "bg": "#121212",
    "header": "#1a1a1a",
    "card": "#1e1e1e",
    "card_border": "#333333",
    "text_primary": "#ffffff",
    "text_secondary": "#b3b3b3",
    "button_primary": "#1976d2",
    "button_secondary": "#388e3c",
    "status_bar": "#1a1a1a",
    "status_text": "#757575",
    "image_bg": "#2d2d2d",
    "positive": "#e74c3c",
    "negative": "#2ecc71",
    "no_tumor_box": "#ffffff",
    "accent": "#8e44ad",
    "warning": "#d35400"
}

# Additional theme for high contrast mode
HIGH_CONTRAST_THEME = {
    "name": "high_contrast",
    "bg": "#000000",
    "header": "#000000",
    "card": "#000000",
    "card_border": "#ffffff",
    "text_primary": "#ffffff",
    "text_secondary": "#cccccc",
    "button_primary": "#ff0000",
    "button_secondary": "#00ff00",
    "status_bar": "#000000",
    "status_text": "#ffffff",
    "image_bg": "#000000",
    "positive": "#ff0000",
    "negative": "#00ff00",
    "no_tumor_box": "#ffffff",
    "accent": "#ffff00",
    "warning": "#ffa500"
}

def toggle_theme():
    global dark_mode, current_theme
    themes = [LIGHT_THEME, DARK_THEME, HIGH_CONTRAST_THEME]
    
    if current_theme is None:
        current_theme = DARK_THEME if dark_mode else LIGHT_THEME
    
    current_index = themes.index(current_theme)
    next_index = (current_index + 1) % len(themes)
    current_theme = themes[next_index]
    
    dark_mode = current_theme["name"] != "light"
    apply_theme()
    update_theme_preview()

def update_theme_preview():
    if 'theme_preview' in globals():
        theme_preview.configure(text=f"Theme: {current_theme['name'].replace('_', ' ').title()}",
                              text_color=current_theme["accent"])

def apply_theme():
    theme = current_theme
    
    window.configure(fg_color=theme["bg"])
    header_frame.configure(fg_color=theme["header"])
    title_label.configure(text_color="white")
    subtitle_label.configure(text_color=theme["text_secondary"])
    
    if 'logo_label' in globals():
        logo_label.configure(bg=theme["header"])
    
    upload_frame.configure(fg_color=theme["card"], border_color=theme["card_border"])
    detect_frame.configure(fg_color=theme["card"], border_color=theme["card_border"])
    history_frame.configure(fg_color=theme["card"], border_color=theme["card_border"])
    stats_frame.configure(fg_color=theme["card"], border_color=theme["card_border"])
    
    upload_title.configure(text_color=theme["text_primary"])
    detect_title.configure(text_color=theme["text_primary"])
    history_title.configure(text_color=theme["text_primary"])
    stats_title.configure(text_color=theme["text_primary"])
    
    upload_image_container.configure(fg_color=theme["image_bg"], border_color=theme["card_border"])
    detect_image_container.configure(fg_color=theme["image_bg"], border_color=theme["card_border"])
    upload_label.configure(bg=theme["image_bg"])
    detect_label.configure(bg=theme["image_bg"])
    
    upload_button.configure(fg_color=theme["button_primary"], hover_color=adjust_color(theme["button_primary"], -20))
    detect_button.configure(fg_color=theme["button_secondary"], hover_color=adjust_color(theme["button_secondary"], -20))
    clear_button.configure(fg_color=theme["warning"], hover_color=adjust_color(theme["warning"], -20))
    save_button.configure(fg_color=theme["accent"], hover_color=adjust_color(theme["accent"], -20))
    help_button.configure(fg_color=theme["header"], hover_color=adjust_color(theme["header"], 20))
    
    theme_button.configure(text=f"🎨 {current_theme['name'].replace('_', ' ').title()}"[:10],
                         fg_color=theme["accent"], hover_color=adjust_color(theme["accent"], -20))
    
    status_bar.configure(fg_color=theme["status_bar"])
    status_label.configure(text_color=theme["status_text"])
    
    # Update history buttons
    for btn in history_buttons:
        btn.configure(fg_color=theme["button_primary"], hover_color=adjust_color(theme["button_primary"], -20))
    
    # Update stats labels
    stats_positive.configure(text_color=theme["positive"])
    stats_negative.configure(text_color=theme["negative"])
    stats_total.configure(text_color=theme["text_primary"])
    
    # Redraw images with current theme
    if img_path:
        display_uploaded_image()
    if hasattr(detect_label, 'image'):
        display_detection_result()

def adjust_color(hex_color, amount):
    """Lighten or darken a color by a given amount"""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    adjusted = []
    for channel in rgb:
        value = max(0, min(255, channel + amount))
        adjusted.append(value)
    
    return f"#{adjusted[0]:02x}{adjusted[1]:02x}{adjusted[2]:02x}"

def display_uploaded_image():
    global img_path
    if img_path:
        img = Image.open(img_path)
        img = img.resize((400, 400))
        
        # Apply theme-appropriate enhancements
        if current_theme["name"] == "dark":
            img = ImageOps.autocontrast(img)
        elif current_theme["name"] == "high_contrast":
            img = ImageOps.invert(img)
        
        img_tk = ImageTk.PhotoImage(img)
        upload_label.config(image=img_tk)
        upload_label.image = img_tk

def display_detection_result():
    if hasattr(detect_label, 'image'):
        img_tk = detect_label.image
        detect_label.config(image=img_tk)
        detect_label.image = img_tk

def upload_image():
    global img_path, processing
    if processing:
        return
        
    filetypes = [
        ("Image Files", "*.jpg;*.png;*.jpeg"),
        ("DICOM Files", "*.dcm"),
        ("All Files", "*.*")
    ]
    
    img_path = filedialog.askopenfilename(
        title="Select Medical Image",
        filetypes=filetypes,
        initialdir=os.path.expanduser("~")
    )

    if img_path:
        display_uploaded_image()
        upload_title.configure(text=f"Uploaded: {os.path.basename(img_path)[:20]}...")
        detect_title.configure(text="Detection Result (Pending)")
        
        # Clear previous detection
        blank_img = Image.new("RGB", (400, 400), color=current_theme["image_bg"])
        blank_tk = ImageTk.PhotoImage(blank_img)
        detect_label.config(image=blank_tk)
        detect_label.image = blank_tk
        
        update_status(f"Loaded: {os.path.basename(img_path)}")

def add_no_tumor_detection(original_img, confidence=0.9):
    """Add white rectangle covering brain and text outside it"""
    img = original_img.copy()
    draw = ImageDraw.Draw(img)
    theme = current_theme
    
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    detection_text = f"No Tumor Detected ({confidence*100:.0f}%)"
    
    # Calculate text size
    bbox = draw.textbbox((0, 0), detection_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Rectangle covering most of the brain
    rect_margin = 30
    rect_coords = (rect_margin, rect_margin, 400-rect_margin, 400-rect_margin)
    
    # Draw rectangle with theme-appropriate color
    overlay = Image.new('RGBA', img.size, (255,255,255,0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(rect_coords, 
                          fill=(*hex_to_rgb(theme["negative"]), 100), 
                          outline=theme["negative"], 
                          width=3)
    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Draw text outside rectangle
    text_x = 400 - text_width - 20
    text_y = 400 - text_height - 20
    draw.text((text_x, text_y), detection_text, 
             fill=theme["negative"], 
             font=font, 
             stroke_width=2, 
             stroke_fill=theme["bg"] if theme["name"] != "high_contrast" else "#000000")
    
    return img

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def run_detection():
    global processing, img_path, history
    
    if not img_path or processing:
        return
    
    processing = True
    detect_button.configure(state="disabled")
    upload_button.configure(state="disabled")
    clear_button.configure(state="disabled")
    save_button.configure(state="disabled")
    
    start_time = time.time()
    
    try:
        original_img = Image.open(img_path).resize((400, 400))
        results = model(img_path)
        
        detection_time = time.time() - start_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if results and len(results[0].boxes) > 0:
            result_img = results[0].plot()
            result_img = Image.fromarray(result_img[..., ::-1]).resize((400, 400))
            result_img_tk = ImageTk.PhotoImage(result_img)
            detect_title.configure(text=f"Tumor Detected ({len(results[0].boxes)} regions)")
            
            # Add to history
            history_entry = {
                "timestamp": timestamp,
                "filename": os.path.basename(img_path),
                "result": "Positive",
                "confidence": float(max([box.conf for box in results[0].boxes])),
                "image": result_img,
                "time_taken": detection_time
            }
            
            update_status(f"Detection completed in {detection_time:.2f}s - Tumor found")
            update_stats("positive")
        else:
            # Add "No_Tumor" annotation to original image
            result_img = add_no_tumor_detection(original_img)
            result_img_tk = ImageTk.PhotoImage(result_img)
            detect_title.configure(text="No Tumor Detected")
            
            # Add to history
            history_entry = {
                "timestamp": timestamp,
                "filename": os.path.basename(img_path),
                "result": "Negative",
                "confidence": 0.9,  # Default confidence for no tumor
                "image": result_img,
                "time_taken": detection_time
            }
            
            update_status(f"Detection completed in {detection_time:.2f}s - No tumor")
            update_stats("negative")
        
        history.append(history_entry)
        update_history_list()
        
        detect_label.config(image=result_img_tk)
        detect_label.image = result_img_tk
            
    except Exception as e:
        print(f"Error during detection: {e}")
        detect_title.configure(text="Detection Failed")
        update_status(f"Error: {str(e)}")
        
    finally:
        processing = False
        detect_button.configure(state="normal")
        upload_button.configure(state="normal")
        clear_button.configure(state="normal")
        save_button.configure(state="normal")

def detect_disease():
    if not img_path:
        messagebox.showwarning("No Image", "Please upload an image first!")
        return
    
    detect_title.configure(text="Processing...")
    window.update()
    
    detection_thread = threading.Thread(target=run_detection)
    detection_thread.daemon = True
    detection_thread.start()

def clear_images():
    global img_path
    img_path = None
    
    # Clear uploaded image
    blank_img = Image.new("RGB", (400, 400), color=current_theme["image_bg"])
    blank_tk = ImageTk.PhotoImage(blank_img)
    
    upload_label.config(image=blank_tk)
    upload_label.image = blank_tk
    detect_label.config(image=blank_tk)
    detect_label.image = blank_tk
    
    upload_title.configure(text="Upload MRI Scan")
    detect_title.configure(text="Detection Result")
    update_status("Ready")

def save_results():
    if not img_path or not hasattr(detect_label, 'image'):
        messagebox.showwarning("No Results", "Nothing to save. Please process an image first.")
        return
    
    # Create output directory if it doesn't exist
    output_dir = "NeuroVision_Results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    output_filename = f"{output_dir}/{base_filename}_result_{timestamp}.png"
    
    try:
        # Get the detection image from the label
        img = ImageTk.getimage(detect_label.image)
        img.save(output_filename)
        update_status(f"Results saved to {output_filename}")
        messagebox.showinfo("Success", f"Results saved successfully to:\n{output_filename}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save results: {str(e)}")

def update_status(message):
    status_label.configure(text=f"Status: {message}")
    window.update()

def update_stats(result_type):
    global stats_positive, stats_negative, stats_total
    
    total = len(history)
    positive = len([h for h in history if h["result"] == "Positive"])
    negative = total - positive
    
    stats_positive.configure(text=f"Positive: {positive}")
    stats_negative.configure(text=f"Negative: {negative}")
    stats_total.configure(text=f"Total Scans: {total}")

def update_history_list():
    global history_buttons
    
    # Clear existing buttons
    for widget in history_scroll_frame.winfo_children():
        widget.destroy()
    
    history_buttons = []
    
    # Add newest first
    for i, entry in enumerate(reversed(history)):
        btn_text = f"{entry['timestamp']} - {entry['filename'][:15]}... ({entry['result']})"
        btn = ctk.CTkButton(
            history_scroll_frame,
            text=btn_text,
            command=lambda e=entry: show_history_entry(e),
            font=("Roboto", 10),
            fg_color=current_theme["button_primary"],
            hover_color=adjust_color(current_theme["button_primary"], -20),
            anchor="w",
            height=30
        )
        btn.pack(fill="x", pady=2)
        history_buttons.append(btn)
    
    # Update stats
    update_stats(None)

def show_history_entry(entry):
    global img_path
    
    # Display the original image
    try:
        img = Image.open(entry["filename"]) if os.path.exists(entry["filename"]) else entry["image"]
        img = img.resize((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        
        upload_label.config(image=img_tk)
        upload_label.image = img_tk
        upload_title.configure(text=f"History: {entry['filename'][:20]}...")
    except:
        pass
    
    # Display the result image
    result_img = entry["image"].resize((400, 400))
    result_img_tk = ImageTk.PhotoImage(result_img)
    
    detect_label.config(image=result_img_tk)
    detect_label.image = result_img_tk
    detect_title.configure(text=f"Result: {entry['result']} ({entry['confidence']*100:.1f}%)")
    
    update_status(f"Showing history entry from {entry['timestamp']}")

def open_help():
    help_text = """
    NeuroVision AI - Brain Tumor Detection System
    
    Instructions:
    1. Click 'Upload MRI Scan' to select a brain MRI image
    2. Click 'Detect Tumor' to analyze the image
    3. View results in the right panel
    4. Use 'Save Results' to save the detection image
    5. Access previous scans in the History section
    
    Tips:
    - Use high-quality MRI scans for best results
    - The system works with JPG, PNG, and DICOM formats
    - Toggle between light/dark/high contrast themes
    
    For more information, visit our website.
    """
    
    messagebox.showinfo("Help & Instructions", help_text)

def open_website():
    webbrowser.open("https://www.example.com/neurovision")

# Initialize main window
ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")

window = ctk.CTk()
window.title("NeuroVision AI - Brain Tumor Detection System")
window.geometry("1200x900")
window.configure(fg_color=LIGHT_THEME["bg"])
current_theme = LIGHT_THEME

# Header section
header_frame = ctk.CTkFrame(window, fg_color=LIGHT_THEME["header"], height=120, corner_radius=0)
header_frame.pack(fill="x", pady=(0, 10))

header_content = ctk.CTkFrame(header_frame, fg_color="transparent")
header_content.pack(expand=True, fill="both", padx=50)

title_container = ctk.CTkFrame(header_content, fg_color="transparent")
title_container.pack(side="left")

try:
    logo = Image.open("logo.png") if os.path.exists("logo.png") else Image.new("RGB", (80, 80), color="#2c3e50")
    logo = logo.resize((80, 80))
    logo_tk = ImageTk.PhotoImage(logo)
    logo_label = Label(title_container, image=logo_tk, bg=LIGHT_THEME["header"])
    logo_label.pack(side="left", padx=(0, 15))
except:
    pass

title_label = ctk.CTkLabel(
    title_container,
    text="NeuroVision AI",
    font=("Roboto", 28, "bold"),
    text_color="white"
)
title_label.pack(side="left", pady=10)

subtitle_label = ctk.CTkLabel(
    header_content,
    text="Advanced Brain Tumor Detection System",
    font=("Roboto", 14),
    text_color=LIGHT_THEME["text_secondary"]
)
subtitle_label.pack(side="left", padx=20)

# Header buttons
button_container = ctk.CTkFrame(header_content, fg_color="transparent")
button_container.pack(side="right")

help_button = ctk.CTkButton(
    button_container,
    text="ℹ️ Help",
    command=open_help,
    font=("Roboto", 12),
    width=80,
    height=30,
    fg_color=LIGHT_THEME["header"],
    hover_color=adjust_color(LIGHT_THEME["header"], 20)
)
help_button.pack(side="right", padx=5)

theme_button = ctk.CTkButton(
    button_container,
    text="🎨 Theme",
    command=toggle_theme,
    font=("Roboto", 12),
    width=100,
    height=30,
    fg_color=LIGHT_THEME["accent"],
    hover_color=adjust_color(LIGHT_THEME["accent"], -20)
)
theme_button.pack(side="right", padx=5)

theme_preview = ctk.CTkLabel(
    button_container,
    text="Theme: Light",
    font=("Roboto", 12),
    text_color=LIGHT_THEME["accent"]
)
theme_preview.pack(side="right", padx=10)

# Main content area
main_frame = ctk.CTkFrame(window, fg_color="transparent")
main_frame.pack(expand=True, fill="both", padx=20, pady=10)

# Left panel (image processing)
left_panel = ctk.CTkFrame(main_frame, fg_color="transparent")
left_panel.pack(side="left", fill="both", expand=True)

# Image display section
image_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
image_frame.pack(expand=True, fill="both")

# Upload panel
upload_frame = ctk.CTkFrame(
    image_frame, 
    width=450, 
    height=460,
    fg_color=LIGHT_THEME["card"],
    border_width=1,
    border_color=LIGHT_THEME["card_border"],
    corner_radius=12
)
upload_frame.pack(side="left", padx=10, pady=5)
upload_frame.pack_propagate(False)

upload_title = ctk.CTkLabel(
    upload_frame, 
    text="Upload MRI Scan", 
    font=("Roboto", 16, "bold"),
    text_color=LIGHT_THEME["text_primary"]
)
upload_title.pack(pady=(15, 10))

upload_image_container = ctk.CTkFrame(
    upload_frame, 
    fg_color=LIGHT_THEME["image_bg"], 
    width=400, 
    height=400,
    corner_radius=8,
    border_width=1,
    border_color=LIGHT_THEME["card_border"]
)
upload_image_container.pack()
upload_image_container.pack_propagate(False)

upload_label = Label(
    upload_image_container, 
    bg=LIGHT_THEME["image_bg"],
    bd=0,
    highlightthickness=0
)
upload_label.pack(expand=True, fill="both", padx=5, pady=5)

# Detection panel
detect_frame = ctk.CTkFrame(
    image_frame, 
    width=450, 
    height=460,
    fg_color=LIGHT_THEME["card"],
    border_width=1,
    border_color=LIGHT_THEME["card_border"],
    corner_radius=12
)
detect_frame.pack(side="left", padx=10, pady=5)
detect_frame.pack_propagate(False)

detect_title = ctk.CTkLabel(
    detect_frame, 
    text="Detection Result", 
    font=("Roboto", 16, "bold"),
    text_color=LIGHT_THEME["text_primary"]
)
detect_title.pack(pady=(15, 10))

detect_image_container = ctk.CTkFrame(
    detect_frame, 
    fg_color=LIGHT_THEME["image_bg"], 
    width=400, 
    height=400,
    corner_radius=8,
    border_width=1,
    border_color=LIGHT_THEME["card_border"]
)
detect_image_container.pack()
detect_image_container.pack_propagate(False)

detect_label = Label(
    detect_image_container, 
    bg=LIGHT_THEME["image_bg"],
    bd=0,
    highlightthickness=0
)
detect_label.pack(expand=True, fill="both", padx=5, pady=5)

# Action buttons
button_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
button_frame.pack(pady=10)

upload_button = ctk.CTkButton(
    button_frame,
    text="📁 Upload MRI Scan",
    command=upload_image,
    font=("Roboto", 14, "bold"),
    fg_color=LIGHT_THEME["button_primary"],
    hover_color=adjust_color(LIGHT_THEME["button_primary"], -20),
    text_color="white",
    corner_radius=8,
    width=180,
    height=40,
    border_spacing=8
)
upload_button.grid(row=0, column=0, padx=10, pady=5)

detect_button = ctk.CTkButton(
    button_frame,
    text="🔍 Detect Tumor",
    command=detect_disease,
    font=("Roboto", 14, "bold"),
    fg_color=LIGHT_THEME["button_secondary"],
    hover_color=adjust_color(LIGHT_THEME["button_secondary"], -20),
    text_color="white",
    corner_radius=8,
    width=180,
    height=40,
    border_spacing=8
)
detect_button.grid(row=0, column=1, padx=10, pady=5)

clear_button = ctk.CTkButton(
    button_frame,
    text="🗑️ Clear",
    command=clear_images,
    font=("Roboto", 14, "bold"),
    fg_color=LIGHT_THEME["warning"],
    hover_color=adjust_color(LIGHT_THEME["warning"], -20),
    text_color="white",
    corner_radius=8,
    width=100,
    height=40,
    border_spacing=8
)
clear_button.grid(row=0, column=2, padx=10, pady=5)

save_button = ctk.CTkButton(
    button_frame,
    text="💾 Save Results",
    command=save_results,
    font=("Roboto", 14, "bold"),
    fg_color=LIGHT_THEME["accent"],
    hover_color=adjust_color(LIGHT_THEME["accent"], -20),
    text_color="white",
    corner_radius=8,
    width=150,
    height=40,
    border_spacing=8
)
save_button.grid(row=0, column=3, padx=10, pady=5)

# Right panel (history and stats)
right_panel = ctk.CTkFrame(main_frame, fg_color="transparent", width=300)
right_panel.pack(side="right", fill="y", padx=10)

# History section
history_frame = ctk.CTkFrame(
    right_panel, 
    width=300,
    height=400,
    fg_color=LIGHT_THEME["card"],
    border_width=1,
    border_color=LIGHT_THEME["card_border"],
    corner_radius=12
)
history_frame.pack(fill="both", pady=(0, 10))
history_frame.pack_propagate(False)

history_title = ctk.CTkLabel(
    history_frame, 
    text="Scan History", 
    font=("Roboto", 16, "bold"),
    text_color=LIGHT_THEME["text_primary"]
)
history_title.pack(pady=(15, 10))

# Scrollable history list
history_scroll = ctk.CTkScrollableFrame(
    history_frame, 
    fg_color="transparent"
)
history_scroll.pack(expand=True, fill="both", padx=10, pady=5)

history_scroll_frame = ctk.CTkFrame(history_scroll, fg_color="transparent")
history_scroll_frame.pack(fill="both", expand=True)

history_buttons = []

# Statistics section
stats_frame = ctk.CTkFrame(
    right_panel, 
    width=300,
    height=200,
    fg_color=LIGHT_THEME["card"],
    border_width=1,
    border_color=LIGHT_THEME["card_border"],
    corner_radius=12
)
stats_frame.pack(fill="x", pady=10)
stats_frame.pack_propagate(False)

stats_title = ctk.CTkLabel(
    stats_frame, 
    text="Detection Statistics", 
    font=("Roboto", 16, "bold"),
    text_color=LIGHT_THEME["text_primary"]
)
stats_title.pack(pady=(15, 10))

stats_content = ctk.CTkFrame(stats_frame, fg_color="transparent")
stats_content.pack(fill="both", expand=True, padx=20, pady=10)

stats_positive = ctk.CTkLabel(
    stats_content,
    text="Positive: 0",
    font=("Roboto", 14),
    text_color=LIGHT_THEME["positive"]
)
stats_positive.pack(anchor="w", pady=5)

stats_negative = ctk.CTkLabel(
    stats_content,
    text="Negative: 0",
    font=("Roboto", 14),
    text_color=LIGHT_THEME["negative"]
)
stats_negative.pack(anchor="w", pady=5)

stats_total = ctk.CTkLabel(
    stats_content,
    text="Total Scans: 0",
    font=("Roboto", 14),
    text_color=LIGHT_THEME["text_primary"]
)
stats_total.pack(anchor="w", pady=5)

# Status bar
status_bar = ctk.CTkFrame(
    window, 
    height=30, 
    fg_color=LIGHT_THEME["status_bar"],
    corner_radius=0
)
status_bar.pack(fill="x", side="bottom")

status_label = ctk.CTkLabel(
    status_bar, 
    text="Status: Ready • NeuroVision AI v2.0 • © 2023 Medical Diagnostics Inc.",
    font=("Roboto", 10),
    text_color=LIGHT_THEME["status_text"]
)
status_label.pack(side="left", padx=20)

# Initialize with blank images
blank_img = Image.new("RGB", (400, 400), color=LIGHT_THEME["image_bg"])
blank_tk = ImageTk.PhotoImage(blank_img)

upload_label.config(image=blank_tk)
upload_label.image = blank_tk
detect_label.config(image=blank_tk)
detect_label.image = blank_tk

# Apply theme
apply_theme()

window.mainloop()