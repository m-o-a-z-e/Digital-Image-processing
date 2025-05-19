"""
main.py

This is the main application file for the Digital Image Processing project.
It provides a graphical user interface (GUI) using customtkinter for applying various image processing operations.
The application allows users to upload images and perform operations such as filtering, histogram processing,
thresholding, color manipulation, point operations, edge detection, noise addition/restoration, and morphological operations.
All processing functions are imported from custom modules and results are displayed interactively.
made by [Ammar Amgad] and reviewed by [Moaz Hany]
"""

# Import required libraries
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import cv2
import numpy as np

# Import custom modules for image processing operations
import filtersim
import histim
import thresholding
import color_manipulation
import Point_operation
import sobel
import Image_Restoration_ed
import Morphological_gradient

# Set up the appearance of the customtkinter application
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class DIPApp(ctk.CTk):
    """
    Main application class for Digital Image Processing project.
    Handles GUI setup and image processing operations.
    """
    def __init__(self):
        """
        Initialize the application window and set up basic properties.
        """
        super().__init__()
        self.title("Digital Image Processing Project")
        self.geometry("1400x900")
        self.image_path = None
        self.original_image = None
        self.images = {}
        self.current_thresh_mode = "Basic Global"
        
        self.main_scrollable_frame = ctk.CTkScrollableFrame(self)
        self.main_scrollable_frame.pack(fill="both", expand=True)
        
        self.setup_layout()

    def setup_layout(self):
        """
        Set up the main GUI layout including buttons, frames and image display.
        """
        # Upload button for loading images
        container = self.main_scrollable_frame
        
        # Upload button for loading images
        self.upload_button = ctk.CTkButton(container, text="\U0001F4C2 Upload Image", command=self.load_image)
        self.upload_button.pack(pady=10)

        # Scrollable frame for thumbnail display
        self.thumbnail_frame_container = ctk.CTkScrollableFrame(container, height=220, orientation="horizontal")
        self.thumbnail_frame_container.pack(pady=10, fill="x")
        self.thumbnail_frame = self.thumbnail_frame_container

        # Frame for operation options
        self.options_frame = ctk.CTkFrame(container, height=160)
        self.options_frame.pack(pady=10, fill="x")

        # Label for displaying main image
        self.main_image_label = ctk.CTkLabel(container, text="")
        self.main_image_label.pack(pady=20)

    def load_image(self):
        """
        Open a file dialog to select an image and load it into the application.
        Generates thumbnails for various processing operations.
        """
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            self.image_path = path
            image = cv2.imread(path)
            self.original_image = image
            self.generate_thumbnails(image)

    def generate_thumbnails(self, image):
        """
        Generate thumbnails for various image processing operations.
        
        Args:
            image: The input image to process
        """
        self.images.clear()
        # Clear existing thumbnails
        for widget in self.thumbnail_frame.winfo_children():
            widget.destroy()

        # Apply various image processing operations
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        stretched = histim.hist_stretching(image)
        red_changed = color_manipulation.change_red(self.image_path, 40)
        sobel_img = sobel.sobel_edge_detection(self.image_path)[1]
        if len(sobel_img.shape) == 2:
            sobel_img = cv2.cvtColor(sobel_img, cv2.COLOR_GRAY2BGR)
        median_filtered = filtersim.apply_filter(image, "median")

        # Noise generation and morphological operations
        gaussian_noised = Image_Restoration_ed.add_gaussian_noise(image)
        if len(gaussian_noised.shape) == 2:
            gaussian_noised = cv2.cvtColor(gaussian_noised, cv2.COLOR_GRAY2BGR)

        salt_noised = Image_Restoration_ed.add_salt_pepper_noise(image)
        if len(salt_noised.shape) == 2:
            salt_noised = cv2.cvtColor(salt_noised, cv2.COLOR_GRAY2BGR)

        # Morphological operations
        dilation = Morphological_gradient.apply_dilation(image)
        erosion = Morphological_gradient.apply_erosion(image)
        opening = Morphological_gradient.apply_opening(image)
        int_b = Morphological_gradient.internal_boundary(image)
        ext_b = Morphological_gradient.external_boundary(image)
        morph_grad = Morphological_gradient.morphological_gradient(image)

        def ensure_bgr(img):
            """Ensure image is in BGR format (3 channels)"""
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img

        # Dictionary of all available processing operations
        thumbnails = {
            "Original": image,
            "Grayscale": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            "Threshold": cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
            "Histogram Stretch": stretched,
            "Histogram Equalization": histim.histogram_equalization(image),
            "Color Manipulation": red_changed,
            "Edge (Sobel)": sobel_img,
            "Point Operations": image,
            "Linear & Non Linear Filters": median_filtered,
            "Gaussian Noise": gaussian_noised,
            "Salt & Pepper": salt_noised,
            "Dilation": ensure_bgr(dilation),
            "Erosion": ensure_bgr(erosion),
            "Opening": ensure_bgr(opening),
            "Boundary Extraction": ensure_bgr(int_b),
        }

        # Create buttons for each processing operation
        for i, (name, img) in enumerate(thumbnails.items()):
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((150, 150))
            img_ctk = ctk.CTkImage(light_image=img_pil, size=(150, 150))
            btn = ctk.CTkButton(self.thumbnail_frame, image=img_ctk, text=name, compound="top", width=160, height=180,
                                command=lambda n=name: self.show_options(n))
            btn.grid(row=0, column=i, padx=10)
            self.images[name] = img

    def show_options(self, filter_name):
        """
        Show options for the selected filter/processing operation.
        
        Args:
            filter_name: Name of the selected filter/operation
        """
        # Clear previous options
        for widget in self.options_frame.winfo_children():
            widget.destroy()

        # Add title for the options section
        label = ctk.CTkLabel(self.options_frame, text=f"\u2699 Options for: {filter_name}", font=ctk.CTkFont(size=16, weight="bold"))
        label.pack(pady=5)

        # Threshold-specific options
        if filter_name == "Threshold":
            self.threshold_dropdown = ctk.CTkOptionMenu(self.options_frame, 
                                                    values=["Basic Global", "Automatic", "Adaptive", "Vertical Otsu"], 
                                                    command=self.update_threshold_mode)
            self.threshold_dropdown.set("Basic Global")
            self.threshold_dropdown.pack(pady=5)

            self.threshold_slider = ctk.CTkSlider(self.options_frame, 
                                                from_=0, 
                                                to=255, 
                                                number_of_steps=255)
            self.threshold_slider.set(127)
            
            self.threshold_value_label = ctk.CTkLabel(self.options_frame, text="Threshold: 127")
            
            self.threshold_apply_button = ctk.CTkButton(self.options_frame, 
                                                    text="Apply", 
                                                    command=self.apply_threshold)
            
            self.current_thresh_mode = "Basic Global"
            self.update_threshold_mode("Basic Global")

        # Color manipulation options
        elif filter_name == "Color Manipulation":
            for widget in self.options_frame.winfo_children():
                widget.destroy()

            color_ops = ["Change", "Swap", "Eliminate"]
            color_choices = ["Red", "Green", "Blue"]
            swap_choices = ["R↔G", "R↔B", "G↔B"]

            op_dropdown = ctk.CTkOptionMenu(self.options_frame, values=color_ops,
                                            command=lambda op: handle_color_operation(op_dropdown.get()))
            op_dropdown.set("Change")
            op_dropdown.pack(pady=5)

            color_dropdown = ctk.CTkOptionMenu(self.options_frame, values=color_choices)
            color_dropdown.set("Red")
            color_dropdown.pack(pady=5)

            swap_dropdown = ctk.CTkOptionMenu(self.options_frame, values=swap_choices)
            swap_dropdown.set("R↔G")
            swap_dropdown.pack(pady=5)
            swap_dropdown.pack_forget()

            self.color_slider = ctk.CTkSlider(self.options_frame, from_=-100, to=100, number_of_steps=200,
                                            command=lambda val: self.apply_color_operation("Change", color_dropdown.get(), int(float(val))))
            self.color_slider.set(0)
            self.color_slider.pack(pady=10)

            def handle_color_operation(op_type):
                """Handle different color operation types"""
                if op_type == "Change":
                    self.color_slider.configure(command=lambda val: self.apply_color_operation("Change", color_dropdown.get(), int(float(val))))
                    self.color_slider.set(0)
                    self.color_slider.pack(pady=10)
                    color_dropdown.pack(pady=5)
                    swap_dropdown.pack_forget()
                elif op_type == "Eliminate":
                    color_dropdown.pack(pady=5)
                    swap_dropdown.pack_forget()
                    self.color_slider.pack_forget()
                    color_dropdown.configure(command=lambda val: self.apply_color_operation("Eliminate", val))
                    self.apply_color_operation("Eliminate", color_dropdown.get())
                elif op_type == "Swap":
                    swap_dropdown.pack(pady=5)
                    color_dropdown.pack_forget()
                    self.color_slider.pack_forget()
                    self.apply_color_operation("Swap", swap_dropdown.get())

            swap_dropdown.configure(command=lambda val: self.apply_color_operation("Swap", val))

        # Filter options (linear and non-linear)
        elif filter_name == "Linear & Non Linear Filters":
            self.add_filter_section("Linear Filter", ["average", "laplacian"], self.apply_linear_filter)
            self.add_filter_section("Non-Linear Filter", ["maximum", "minimum", "median", "mode"], self.apply_non_linear_filter)

        # Salt & pepper noise restoration options
        elif filter_name == "Salt & Pepper":
            ctk.CTkButton(self.options_frame, text="Average Filter", command=self.restore_sp_average).pack(pady=3)
            ctk.CTkButton(self.options_frame, text="Median Filter", command=self.restore_sp_median).pack(pady=3)
            ctk.CTkButton(self.options_frame, text="Outlier Method", command=self.restore_outlier).pack(pady=3)

        # Gaussian noise restoration options
        elif filter_name == "Gaussian Noise":
            ctk.CTkButton(self.options_frame, 
                        text="Image Averaging", 
                        command=self.prepare_image_averaging).pack(pady=3)
            ctk.CTkButton(self.options_frame, 
                        text="Average Filter", 
                        command=self.restore_gaussian).pack(pady=3)

        # Boundary extraction options
        elif filter_name == "Boundary Extraction":
            dropdown = ctk.CTkOptionMenu(self.options_frame, values=["Internal", "External", "Morphological"], command=self.apply_boundary)
            dropdown.pack(pady=5)
            
        # Histogram equalization (no options, just apply)
        elif filter_name == "Histogram Equalization":
            self.apply_hist_equalization()

        # Point operation options
        elif filter_name == "Point Operations":
            for widget in self.options_frame.winfo_children():
                widget.destroy()

            slider = ctk.CTkSlider(self.options_frame, from_=0, to=255, command=lambda value: None)
            slider.set(0)
            slider.pack_forget()

            def handle_point_op(op_type):
                """Handle different point operations"""
                if op_type == "add":
                    slider.configure(from_=0, to=220)
                    slider.set(0)
                    slider.configure(command=lambda value: self.apply_point_op("add", value))
                    slider.pack(pady=5)

                elif op_type == "subtract":
                    slider.configure(from_=0, to=220)
                    slider.set(0)
                    slider.configure(command=lambda value: self.apply_point_op("subtract", value))
                    slider.pack(pady=5)

                elif op_type == "divide":
                    slider.configure(from_=1, to=10)
                    slider.set(1)
                    slider.configure(command=lambda value: self.apply_point_op("divide", value))
                    slider.pack(pady=5)

                elif op_type == "complement":
                    slider.pack_forget()
                    self.apply_point_op("complement")

            # Create buttons for each point operation
            for name, op in [
                ("Add", "add"),
                ("Subtract", "subtract"),
                ("Divide", "divide"),
                ("Complement", "complement")]:
                ctk.CTkButton(self.options_frame, text=name, command=lambda op=op: handle_point_op(op)).pack(pady=2)
        
        # Display the main image with the selected operation
        self.display_main_image(filter_name)

    def update_threshold_preview(self, value):
        """Update the threshold value preview label"""
        threshold_value = int(float(value))
        self.threshold_value_label.configure(text=f"Threshold: {threshold_value}")

    def update_threshold_mode(self, mode):
        """
        Update the threshold mode and adjust UI accordingly.
        
        Args:
            mode: Selected threshold mode
        """
        self.current_thresh_mode = mode
        
        if mode in ["Basic Global", "Automatic"]:
            self.threshold_slider.pack(pady=5)
            self.threshold_value_label.pack()
            self.threshold_apply_button.pack(pady=5)
            
            self.threshold_slider.configure(command=self.update_threshold_preview)
        else:
            self.threshold_slider.pack_forget()
            self.threshold_value_label.pack_forget()
            self.threshold_apply_button.pack(pady=5)

    def apply_threshold(self):
        """Apply the selected threshold operation to the image"""
        if self.current_thresh_mode == "Adaptive":
            result = thresholding.Adaptive_thresholding(self.original_image)
        elif self.current_thresh_mode == "Vertical Otsu":
            result = thresholding.vertical_adaptive_otsu(self.original_image)
        elif self.current_thresh_mode == "Automatic":
            value = int(self.threshold_slider.get())
            result = thresholding.Automatic_thresholding(self.original_image, value)
        else:
            value = int(self.threshold_slider.get())
            result = thresholding.Basic_Global_Thresholding(self.original_image, value)

        self.display_main_image("Threshold", override_img=cv2.cvtColor(result, cv2.COLOR_GRAY2BGR))

    def apply_color_operation(self, op_type, color, value=None):
        """
        Apply color manipulation operations to the image.
        
        Args:
            op_type: Type of operation (Change/Swap/Eliminate)
            color: Color channel to operate on
            value: Value for change operations (optional)
        """
        if op_type == "Change":
            if color == "Red":
                result = color_manipulation.change_red(self.image_path, value)
            elif color == "Green":
                result = color_manipulation.change_green(self.image_path, value)
            elif color == "Blue":
                result = color_manipulation.change_blue(self.image_path, value)
        elif op_type == "Swap":
            if color == "R↔G":
                result = color_manipulation.swap_red_green(self.image_path)
            elif color == "R↔B":
                result = color_manipulation.swap_red_blue(self.image_path)
            elif color == "G↔B":
                result = color_manipulation.swap_green_blue(self.image_path)
        elif op_type == "Eliminate":
            if color == "Red":
                result = color_manipulation.eliminate_red(self.image_path)
            elif color == "Green":
                result = color_manipulation.eliminate_green(self.image_path)
            elif color == "Blue":
                result = color_manipulation.eliminate_blue(self.image_path)

        self.display_main_image("Color Manipulation", override_img=result)

    def add_filter_section(self, title, options, command):
        """
        Add a filter section to the options frame.
        
        Args:
            title: Section title
            options: List of filter options
            command: Function to call when filter is applied
        """
        ctk.CTkLabel(self.options_frame, text=title, font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 2))
        dropdown = ctk.CTkOptionMenu(self.options_frame, values=options)
        dropdown.pack(pady=2)
        ctk.CTkButton(self.options_frame, text="Apply", command=lambda: command(dropdown.get())).pack(pady=2)

    def apply_linear_filter(self, ftype):
        """Apply linear filter to the image"""
        result = filtersim.apply_filter(self.original_image, ftype)
        self.display_main_image("Linear & Non Linear Filters", override_img=result)

    def apply_non_linear_filter(self, ftype):
        """Apply non-linear filter to the image"""
        result = filtersim.apply_filter(self.original_image, ftype)
        self.display_main_image("Linear & Non Linear Filters", override_img=result)

    def apply_point_op(self, op, value=None):
        """
        Apply point operations to the image.
        
        Args:
            op: Operation to perform (add/subtract/divide/complement)
            value: Value for the operation (optional)
        """
        if value is None:
            value = 50

        if op == "add":
            result = Point_operation.add_value_to_image(self.original_image, value)
        elif op == "subtract":
            result = Point_operation.subtract_value_from_image(self.original_image, value)
        elif op == "divide":
            result = Point_operation.divide_image_by_value(self.original_image, value)
        elif op == "complement":
            result = Point_operation.complement_image(self.original_image)

        self.display_main_image("Point Operations", override_img=result)
    def prepare_image_averaging(self):
        """Open file dialog to select second image for averaging"""
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            self.second_image = cv2.imread(path)
            self.apply_image_averaging()

    def apply_image_averaging(self):
        """Apply image averaging between original and selected second image"""
        if hasattr(self, 'second_image') and self.second_image is not None:
            if self.second_image.shape != self.original_image.shape:
                self.second_image = cv2.resize(self.second_image, 
                                            (self.original_image.shape[1], 
                                            self.original_image.shape[0]))
                
            restored = Image_Restoration_ed.image_averaging(self.original_image, self.second_image)
            
            if len(restored.shape) == 2:
                restored = cv2.cvtColor(restored, cv2.COLOR_GRAY2BGR)
                
            self.display_main_image("Gaussian Noise", override_img=restored)
        else:
            print("Please select a second image first")

    def apply_boundary(self, btype):
        """
        Apply boundary extraction operations.
        
        Args:
            btype: Boundary type (Internal/External/Morphological)
        """
        if btype == "Internal":
            result = Morphological_gradient.internal_boundary(self.original_image)
        elif btype == "External":
            result = Morphological_gradient.external_boundary(self.original_image)
        elif btype == "Morphological":
            result = Morphological_gradient.morphological_gradient(self.original_image)
        self.display_main_image("Boundary Extraction", override_img=cv2.cvtColor(result, cv2.COLOR_GRAY2BGR))

    def restore_outlier(self):
        """Restore salt & pepper noise using outlier method"""
        noisy = Image_Restoration_ed.add_salt_pepper_noise(self.original_image)
        gray_noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY) if len(noisy.shape) == 3 else noisy
        restored = Image_Restoration_ed.outlier_filter(gray_noisy)
        self.display_main_image("Salt & Pepper", override_img=cv2.cvtColor(restored, cv2.COLOR_GRAY2BGR))

    def restore_sp_average(self):
        """Restore salt & pepper noise using average filter"""
        noisy = Image_Restoration_ed.add_salt_pepper_noise(self.original_image)
        restored = Image_Restoration_ed.average_filter(cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY))
        self.display_main_image("Salt & Pepper", override_img=cv2.cvtColor(restored, cv2.COLOR_GRAY2BGR))

    def restore_sp_median(self):
        """Restore salt & pepper noise using median filter"""
        noisy = Image_Restoration_ed.add_salt_pepper_noise(self.original_image)
        restored = Image_Restoration_ed.median_filter(cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY))
        self.display_main_image("Salt & Pepper", override_img=cv2.cvtColor(restored, cv2.COLOR_GRAY2BGR))

    def restore_gaussian(self):
        """Restore gaussian noise using average filter"""
        noisy = Image_Restoration_ed.add_gaussian_noise(self.original_image)
        restored = Image_Restoration_ed.average_filter(cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY))
        self.display_main_image("Gaussian Noise", override_img=cv2.cvtColor(restored, cv2.COLOR_GRAY2BGR))

    def restore_gaussian_avg(self):
        """Legacy function - now handled by prepare_image_averaging"""
        self.prepare_image_averaging()

    def apply_hist_equalization(self):
        """Apply histogram equalization to the image"""
        result = histim.histogram_equalization(self.original_image)
        self.display_main_image("Histogram Equalization", override_img=result)

    def update_red(self, value):
        """Update red channel value (legacy function)"""
        result = color_manipulation.change_red(self.image_path, value)
        self.display_main_image("Color Manipulation", override_img=result)

    def display_main_image(self, name, override_img=None):
        """
        Display the main image with processing results.
        
        Args:
            name: Name of the processing operation
            override_img: Optional image to display instead of stored one
        """
        processed_img = override_img if override_img is not None else self.images.get(name)
        if processed_img is None:
            return

        # Resize images for display
        original_resized = cv2.resize(self.original_image, (400, 400))
        processed_resized = cv2.resize(processed_img, (400, 400))

        # Ensure images are in BGR format
        if len(processed_resized.shape) == 2:
            processed_resized = cv2.cvtColor(processed_resized, cv2.COLOR_GRAY2BGR)
        if len(original_resized.shape) == 2:
            original_resized = cv2.cvtColor(original_resized, cv2.COLOR_GRAY2BGR)
            
        # Create separator between original and processed images
        separator_color = (34, 34, 34)
        separator = np.full((400, 50, 3), separator_color, dtype=np.uint8)

        # Combine images side by side
        combined = cv2.hconcat([original_resized, separator, processed_resized])

        # Convert to RGB for display in Tkinter
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(combined_rgb)

        # Resize and display
        img_pil = img_pil.resize((880, 400))
        self.main_ctk_image = ctk.CTkImage(light_image=img_pil, size=(880, 400))
        self.main_image_label.configure(image=self.main_ctk_image, text="")


if __name__ == "__main__":
    # Create and run the application
    app = DIPApp()
    app.mainloop()