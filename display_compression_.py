import os
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from sklearn.decomposition import TruncatedSVD
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ImageCompressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SVD Image Compression Analyzer")
        
        # Data collection
        self.metrics_data = []
        self.current_experiment = None
        
        # Image variables
        self.image_files = []
        self.current_index = 0
        self.original_image = None
        self.processed_image = None
        self.max_k = 100
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the user interface"""
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls
        control_frame = tk.Frame(main_frame, width=300, padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Experiment controls
        tk.Label(control_frame, text="Experiment Setup", font=('Arial', 12, 'bold')).pack(pady=5)
        tk.Button(control_frame, text="New Experiment", command=self.new_experiment).pack(fill=tk.X, pady=2)
        tk.Button(control_frame, text="Load Images", command=self.load_image_folder).pack(fill=tk.X, pady=2)
        
        # Image navigation
        nav_frame = tk.Frame(control_frame)
        nav_frame.pack(pady=5)
        tk.Button(nav_frame, text="◄ Previous", command=self.prev_image).pack(side=tk.LEFT)
        tk.Button(nav_frame, text="Next ►", command=self.next_image).pack(side=tk.LEFT)
        
        # SVD controls
        self.k_slider = tk.Scale(control_frame, from_=1, to=self.max_k, orient="horizontal", 
                                label="Components (k)", command=self.update_display)
        self.k_slider.set(20)
        self.k_slider.pack(fill=tk.X, pady=5)
        
        # Analysis controls
        tk.Button(control_frame, text="Run Full Analysis", command=self.run_full_analysis).pack(fill=tk.X, pady=2)
        tk.Button(control_frame, text="Save CSV Report", command=self.save_csv_report).pack(fill=tk.X, pady=2)
        tk.Button(control_frame, text="Generate Plots", command=self.generate_plots).pack(fill=tk.X, pady=2)
        
        # Right panel - Display
        display_frame = tk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Image display
        img_display_frame = tk.Frame(display_frame)
        img_display_frame.pack()
        
        self.original_label = tk.Label(img_display_frame, text="Original")
        self.original_label.pack(side=tk.LEFT, padx=10)
        
        self.processed_label = tk.Label(img_display_frame, text="Compressed")
        self.processed_label.pack(side=tk.LEFT, padx=10)
        
        # Metrics display
        self.metrics_label = tk.Label(display_frame, text="", justify=tk.LEFT)
        self.metrics_label.pack(pady=10)
        
        # Plot display
        self.plot_frame = tk.Frame(display_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)
        self.current_plot = None
    
    def new_experiment(self):
        """Initialize a new experiment"""
        self.metrics_data = []
        self.current_experiment = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        tk.messagebox.showinfo("New Experiment", f"Started experiment {self.current_experiment}")
    
    def update_slider_range(self):
        """Update slider max based on current image dimensions"""
        if self.original_image is not None:
            h, w = self.original_image.shape
            self.max_k = min(h, w) - 1
            self.k_slider.config(to=self.max_k)
            self.k_slider.set(min(50, self.max_k))
    
    def load_image_folder(self):
        """Load all images from a selected folder"""
        if not self.current_experiment:
            self.new_experiment()
            
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return
            
        self.image_files = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        
        for file in os.listdir(folder_path):
            if file.lower().endswith(valid_extensions):
                self.image_files.append(os.path.join(folder_path, file))
        
        if self.image_files:
            self.current_index = 0
            self.load_current_image()
    
    def load_current_image(self):
        """Load and process the current image"""
        if not self.image_files:
            return
            
        try:
            img = Image.open(self.image_files[self.current_index]).convert('L')
            self.original_image = np.array(img, dtype=np.uint8)
            self.update_slider_range()
            self.update_display()
            
            self.root.title(f"SVD Compression - {os.path.basename(self.image_files[self.current_index])} "
                          f"(k: 1-{self.max_k}) | Exp: {self.current_experiment}")
        except Exception as e:
            print(f"Error loading image: {e}")
    
    def process_image(self, k):
        if self.original_image is None:
            return None
            
        img_float = self.original_image.astype(np.float64) / 255.0
        svd = TruncatedSVD(n_components=k)
        reduced = svd.fit_transform(img_float)
        restored = svd.inverse_transform(reduced)
        return restored
    
    def calculate_metrics(self, original, compressed, k):
        h, w = original.shape
        
        # Storage calculations
        original_bytes = (h * w * 8) / 1024
        compressed_bytes = (k * (h + w + 1) * 8) / 1024
        
        # Quality metrics
        original_float = original.astype(np.float64) / 255.0
        compressed_clipped = np.clip(compressed, 0, 1)
        
        mse = mean_squared_error(original_float, compressed_clipped)
        psnr = peak_signal_noise_ratio(original_float, compressed_clipped, data_range=1.0)
        ratio = compressed_bytes / original_bytes
        
        return {
            'filename': os.path.basename(self.image_files[self.current_index]),
            'dimensions': f"{w}x{h}",
            'k': k,
            'max_k': self.max_k,
            'mse': mse,
            'psnr': psnr,
            'compression_ratio': ratio,
            'original_kb': original_bytes,
            'compressed_kb': compressed_bytes
        }
    
    def update_display(self, event=None):
        """Update the GUI with current image and metrics"""
        if self.original_image is None:
            return
            
        k = self.k_slider.get()
        compressed = self.process_image(k)
        
        if compressed is None:
            return
            
        # Convert to display format
        compressed_uint8 = np.clip(compressed * 255, 0, 255).astype(np.uint8)
        
        # Calculate metrics
        metrics = self.calculate_metrics(self.original_image, compressed, k)
        self.metrics_data.append(metrics.copy())  # Store for analysis
        
        # Update images
        original_disp = Image.fromarray(self.original_image)
        compressed_disp = Image.fromarray(compressed_uint8)
        
        original_tk = ImageTk.PhotoImage(original_disp)
        compressed_tk = ImageTk.PhotoImage(compressed_disp)
        
        self.original_label.config(image=original_tk)
        self.original_label.image = original_tk
        self.processed_label.config(image=compressed_tk)
        self.processed_label.image = compressed_tk
        
        # Update metrics
        warning = " ⚠️ Compressed larger!" if metrics['compressed_kb'] > metrics['original_kb'] else ""
        self.metrics_label.config(
            text=f"Image {self.current_index+1}/{len(self.image_files)}\n"
                 f"File: {metrics['filename']}\n"
                 f"Dimensions: {metrics['dimensions']}\n"
                 f"k: {k}/{self.max_k} | MSE: {metrics['mse']:.4f} | PSNR: {metrics['psnr']:.2f} dB\n"
                 f"Original: {metrics['original_kb']:.1f} KB | Compressed: {metrics['compressed_kb']:.1f} KB\n"
                 f"Ratio: {metrics['compression_ratio']:.2f}:1{warning}"
        )
    
    def run_full_analysis(self):
        """Run analysis across all k values for current image"""
        if self.original_image is None:
            return
            
        k_values = np.linspace(5, self.max_k, num=10, dtype=int)
        for k in k_values:
            compressed = self.process_image(k)
            metrics = self.calculate_metrics(self.original_image, compressed, k)
            self.metrics_data.append(metrics)
        
        tk.messagebox.showinfo("Analysis Complete", f"Completed analysis for {len(k_values)} k values")
        self.generate_plots()
    
    def save_csv_report(self):
        """Save collected metrics to CSV file"""
        if not self.metrics_data:
            tk.messagebox.showerror("Error", "No metrics data to save")
            return
            
        df = pd.DataFrame(self.metrics_data)
        filename = f"svd_metrics_{self.current_experiment}.csv"
        df.to_csv(filename, index=False)
        tk.messagebox.showinfo("Report Saved", f"Metrics saved to {filename}")
    
    def generate_plots(self):
        """Generate and display analysis plots"""
        if not self.metrics_data:
            tk.messagebox.showerror("Error", "No metrics data to plot")
            return
            
        df = pd.DataFrame(self.metrics_data)
        
        # Clear previous plot
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # PSNR vs k
        ax1.plot(df['k'], df['psnr'], 'b-o')
        ax1.set_title('PSNR vs Number of Components')
        ax1.set_xlabel('k (components)')
        ax1.set_ylabel('PSNR (dB)')
        ax1.grid(True)
        
        # Compression ratio vs k
        ax2.plot(df['k'], df['compression_ratio'], 'r-o')
        ax2.set_title('Compression Ratio vs Number of Components')
        ax2.set_xlabel('k (components)')
        ax2.set_ylabel('Compression Ratio')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Embed plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Save plot
        plot_filename = f"svd_plots_{self.current_experiment}.png"
        fig.savefig(plot_filename, dpi=300)
        self.current_plot = plot_filename
    
    def prev_image(self):
        """Load previous image in folder"""
        if len(self.image_files) > 1:
            self.current_index = (self.current_index - 1) % len(self.image_files)
            self.load_current_image()
    
    def next_image(self):
        """Load next image in folder"""
        if len(self.image_files) > 1:
            self.current_index = (self.current_index + 1) % len(self.image_files)
            self.load_current_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCompressionApp(root)
    root.mainloop()