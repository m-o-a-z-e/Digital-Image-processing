# Digital Image Processing (DIP) Application

![DIP Toolbox Screenshot](https://github.com/m-o-a-z-e/Digital-Image-processing/blob/main/Screenshot%20(93).png)

## 📌 Overview

This project is a comprehensive **Digital Image Processing (DIP)** application with a graphical user interface (GUI) built using **Python**, **OpenCV**, **NumPy**, and **customtkinter**.  
It allows users to perform various image processing tasks such as:
- Filtering  
- Histogram operations  
- Thresholding  
- Color manipulation  
- Point operations  
- Edge detection  
- Noise handling  
- Morphological operations

---

## ✨ Features

### 🎨 1. Color Manipulation
- Adjust intensities of **Red**, **Green**, and **Blue** channels  
- Swap color channels (e.g., Red ↔ Green, Red ↔ Blue)  
- Remove specific color channels  

### 🧹 2. Spatial Filters
- **Linear Filters:** Average, Laplacian  
- **Non-Linear Filters:** Median, Mode, Maximum, Minimum  

### 📊 3. Histogram Processing
- Histogram stretching  
- Histogram equalization (manual & OpenCV-based)  

### ⚫ 4. Thresholding
- Global thresholding  
- Otsu’s automatic thresholding  
- Adaptive thresholding  
- Vertical adaptive Otsu thresholding  

### ⚙️ 5. Point Operations
- Add, subtract, or divide pixel values  
- Compute the negative (complement) of an image  

### 🧠 6. Edge Detection
- Sobel edge detection (gradients in X and Y directions)  

### 🎛️ 7. Noise Addition & Restoration
- Add Salt-and-Pepper or Gaussian noise  
- Restore using average, median, or outlier filters  
- Image averaging for noise reduction  

### 🧱 8. Morphological Operations
- Dilation, erosion, and opening  
- Boundary extraction: internal, external, morphological gradient  

---

## 🧩 Project Modules

- `color_manipulation.py`: Color adjustment and swapping  
- `filtersim.py`: Spatial filters (linear & non-linear)  
- `histim.py`: Histogram operations  
- `thresholding.py`: Thresholding techniques  
- `sobel.py`: Sobel edge detection  
- `Image_Restoration_ed.py`: Noise addition & restoration  
- `Point_operation.py`: Point-wise operations  
- `Morphological_gradient.py`: Morphological operations  
- `main.py`: Main GUI application

---

## File Structure

```
dip-toolbox/
├── main.py                 # Main application GUI
├── thresholding.py         # Thresholding operations
├── color_manipulation.py   # Color channel operations
├── filtersim.py            # Filter implementations
├── histim.py               # Histogram processing
├── Image_Restoration_ed.py # Noise generation and removal
├── Morphological_gradient.py # Morphological operations
├── Point_operation.py      # Basic point operations
├── sobel.py                # Edge detection
└── README.md               # This file
```

## 🛠️ Installation

### ✅ Prerequisites
- Python 3.8+
- Required libraries:

```bash
pip install opencv-python numpy customtkinter Pillow
```

## 🚀 Run The Project

```bash
python main.py
```

## 🧪 Usage

### 📤 Upload an Image  
Click “Upload Image” to select an image file.

### 🛠️ Apply an Operation  
Choose a function (e.g., Color Manipulation), adjust parameters, then click “Apply”.

### 🖼️ View Results  
The processed image is displayed beside the original.

---

## 👨‍💻 Contributors

- **Moaz Hany** – Filters, Histogram Processing, Main Application Review & Edits  
- **Abdelrahman Salah** – Color Manipulation, Thresholding  
- **Nour Mohamed** – Point Operations, Noise Restoration  
- **Hager Mostafa** – Sobel Edge Detection, Morphological Operations  
- **Ammar Amgad** – Main Application Development  

---

## 📜 License

This project is licensed under the **MIT License**.  
See the LICENSE file for more information.

---

## Acknowledgments

- OpenCV for powerful image processing capabilities
- CustomTkinter for the modern GUI interface
- NumPy for efficient array operations

## Contact

For questions or suggestions, please contact: Moaz hany at [moaz.h.sabry@gmail.com]
