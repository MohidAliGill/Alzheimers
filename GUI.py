import tkinter as tk
from tkinter import filedialog, messagebox
import os
import uuid
import nibabel as nib
import numpy as np
from PIL import Image, ImageTk
import subprocess
import SimpleITK as sitk
import ants  # For registration
from nilearn import image  # For image processing and resampling
from skimage import measure

# Environment setup for FSL
os.environ['FSLDIR'] = '/Users/mohidaligill/Documents/Alzheimers/fsl'  # Update this path to your FSL installation directory
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['FSLDIR'], 'bin')
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['FSLDIR'], 'share/fsl/bin')

# Verify the BET path
bet_path = os.path.join(os.environ['FSLDIR'], 'bin', 'bet')

# Path to the MNI152 template
template_path = '/Users/mohidaligill/Documents/Alzheimers/fsl/pkgs/fsl-data_standard-2208.0-0/data/standard/MNI152_T1_1mm_brain.nii.gz'

# Path to the Harvard-Oxford Subcortical Atlas
atlas_path = "/Users/mohidaligill/Documents/Alzheimers/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz"

# Labels for different brain parts
labels = {
    "left_hippocampus": 9,
    "right_hippocampus": 19,
    "left_amygdala": 10,
    "right_amygdala": 20,
    "left_cortex": 2,
    "right_cortex": 13,
    "left_lateral_ventricle": 3,
    "right_lateral_ventricle": 14
}


class MRIViewerApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("MRI Loader, Segmentation, and Viewer")
        self.geometry("1200x600")

        # Create a frame for the buttons to align them in a grid
        button_frame = tk.Frame(self)
        button_frame.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        # Button to load the MRI
        load_button = tk.Button(button_frame, text="Load MRI", command=self.load_mri)
        load_button.grid(row=0, column=0, padx=5)

        # Button for skull stripping
        skull_strip_button = tk.Button(button_frame, text="Skull Strip", command=self.skull_strip)
        skull_strip_button.grid(row=0, column=1, padx=5)

        # Button for normalization and correction
        normalise_button = tk.Button(button_frame, text="Normalize & Correct", command=self.normalise_and_correct)
        normalise_button.grid(row=0, column=2, padx=5)

        # Button for registration to MNI152 template
        register_button = tk.Button(button_frame, text="Register to MNI152", command=self.register_to_template)
        register_button.grid(row=0, column=3, padx=5)

        # Button for segmentation
        segment_button = tk.Button(button_frame, text="Segment Brain Parts", command=self.segment_brain_parts)
        segment_button.grid(row=0, column=4, padx=5)
        
        # Button for feature extraction
        extract_button = tk.Button(button_frame, text="Extract Features", command=self.extract_features)
        extract_button.grid(row=0, column=5, padx=5)
        
        # Button for classification
        extract_button = tk.Button(button_frame, text="Classify", command=self.classify)
        extract_button.grid(row=0, column=6, padx=5)

        # Canvas to display the original MRI slice
        self.canvas_original = tk.Canvas(self, width=240, height=256, bg='white')
        self.canvas_original.grid(row=1, column=0, padx=10, pady=10)

        # Canvas to display the skull-stripped MRI slice
        self.canvas_stripped = tk.Canvas(self, width=240, height=256, bg='white')
        self.canvas_stripped.grid(row=1, column=1, padx=10, pady=10)

        # Canvas to display the normalized and bias-corrected MRI slice
        self.canvas_normalised = tk.Canvas(self, width=240, height=256, bg='white')
        self.canvas_normalised.grid(row=1, column=2, padx=10, pady=10)

        # Canvas to display the registered image
        self.canvas_registered = tk.Canvas(self, width=240, height=256, bg='white')
        self.canvas_registered.grid(row=1, column=3, padx=10, pady=10)

        # Label and Entry for Average Hippocampus Sphericity
        self.label_sphericity = tk.Label(self, text="Average Hippocampus Sphericity:")
        self.label_sphericity.grid(row=2, column=0, padx=10, pady=10, sticky="e")
        self.entry_sphericity = tk.Entry(self, width=20, font=("Arial", 14), state="disabled")
        self.entry_sphericity.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        # Label and Entry for Average Cortex Volume
        self.label_cortex_volume = tk.Label(self, text="Average Cortex Volume:")
        self.label_cortex_volume.grid(row=2, column=2, padx=10, pady=10, sticky="e")
        self.entry_cortex_volume = tk.Entry(self, width=20, font=("Arial", 14), state="disabled")
        self.entry_cortex_volume.grid(row=2, column=3, padx=10, pady=10, sticky="w")
        
         # Label and Entry for Average Cortex Spherecity
        self.label_cortex_sphericity = tk.Label(self, text="Average Cortex Sphericity:")
        self.label_cortex_sphericity.grid(row=3, column=0, padx=10, pady=10, sticky="e")
        self.entry_cortex_sphericity = tk.Entry(self, width=20, font=("Arial", 14), state="disabled")
        self.entry_cortex_sphericity.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        
        # Label and Entry for Lateral Ventricle Compactness Ratio
        self.label_ventricle_compactness_ratio = tk.Label(self, text="Lateral Ventricle Compactness Ratio:")
        self.label_ventricle_compactness_ratio.grid(row=3, column=2, padx=10, pady=10, sticky="e")
        self.entry_ventricle_compactness_ratio = tk.Entry(self, width=20, font=("Arial", 14), state="disabled")
        self.entry_ventricle_compactness_ratio.grid(row=3, column=3, padx=10, pady=10, sticky="w")
        
        # Label and Entry for Lateral Ventricle Sphericity Ratio
        self.label_ventricle_sphericity_ratio = tk.Label(self, text="Lateral Ventricle Sphericity Ratio:")
        self.label_ventricle_sphericity_ratio.grid(row=4, column=0, padx=10, pady=10, sticky="e")
        self.entry_ventricle_sphericity_ratio = tk.Entry(self, width=20, font=("Arial", 14), state="disabled")
        self.entry_ventricle_sphericity_ratio.grid(row=4, column=1, padx=10, pady=10, sticky="w")
        
        # Label and Entry for Amygdala Volume
        self.label_amygdala_volume = tk.Label(self, text="Amygdala Volume:")
        self.label_amygdala_volume.grid(row=4, column=2, padx=10, pady=10, sticky="e")
        self.entry_amygdala_volume = tk.Entry(self, width=20, font=("Arial", 14), state="disabled")
        self.entry_amygdala_volume.grid(row=4, column=3, padx=10, pady=10, sticky="w")

        # Placeholder for the loaded MRI image, unique ID, and file paths
        self.loaded_image = None
        self.unique_id = None
        self.save_folder = None
        self.saved_mri_path = None
        self.corrected_image_path = None
        self.stripped_mri_path = None  # Path to the skull-stripped image
        self.registered_mri_path = None  # Path to the registered image
        self.average_hippocampus_sphericity = None
        self.average_cortex_volume = None
        self.average_cortex_sphericity = None
        self.ventricle_compactness_ratio = None
        self.ventricle_sphericity_ratio = None
        self.amygdala_volume = None
        
    def create_unique_folder(self):
        """ Create a unique folder to save the images """
        self.unique_id = str(uuid.uuid4())  # Generate a unique ID
        self.save_folder = os.path.join(os.getcwd(), self.unique_id)
        os.makedirs(self.save_folder, exist_ok=True)  # Create the folder
        return self.save_folder

    def save_image(self, image_data, filename):
        """ Save the NIfTI image in the specified folder """
        img_path = os.path.join(self.save_folder, filename)
        nib.save(nib.Nifti1Image(image_data, np.eye(4)), img_path)
        return img_path

    def load_mri(self):
        """ Load an MRI image, save it in a unique folder, and display a slice """
        file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii *.nii.gz")])
        if file_path:
            try:
                # Create a unique folder and save the original MRI image
                self.create_unique_folder()
                self.loaded_image = nib.load(file_path).get_fdata()
                self.saved_mri_path = self.save_image(self.loaded_image, f"image.nii")

                # Display the original image in the first canvas
                self.display_slice(self.loaded_image, self.canvas_original)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def display_slice(self, image_data, canvas, slice_idx=None):
        """ Display a single MRI slice on the given canvas """
        if image_data is not None:
            # If no specific slice index is provided, show the middle slice
            if slice_idx is None:
                slice_idx = image_data.shape[2] // 2

            # Extract the middle slice from the 3D image volume
            slice_img = image_data[:, :, slice_idx]

            # Normalize the image data for display
            slice_img = 255 * (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img))
            slice_img = slice_img.astype(np.uint8)

            # Convert the NumPy array to a PIL Image and then to ImageTk
            img = ImageTk.PhotoImage(image=Image.fromarray(slice_img))

            # Display the image on the given canvas
            canvas.create_image(0, 0, anchor="nw", image=img)
            canvas.image = img  # Keep a reference to avoid garbage collection

    def skull_strip(self):
        """ Perform skull stripping and display the result in a separate canvas """
        if self.saved_mri_path is not None:
            try:
                self.stripped_mri_path = os.path.join(self.save_folder, 'stripped.nii.gz')
                result = subprocess.run([bet_path, self.saved_mri_path, self.stripped_mri_path, '-R'], capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Load the skull-stripped image
                    stripped_image = nib.load(self.stripped_mri_path).get_fdata()

                    # Display the skull-stripped image in the second canvas
                    self.display_slice(stripped_image, self.canvas_stripped)
                else:
                    # Handle skull stripping error
                    messagebox.showerror("Error", f"Skull stripping failed: {result.stderr}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to perform skull stripping: {str(e)}")

    def normalize_intensity(self, data):
        """ Normalize intensity between 0 and 1 """
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def bias_field_correction(self, file_path, output_path):
        """ Perform bias field correction using SimpleITK """
        image = sitk.ReadImage(file_path)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected_image = corrector.Execute(image)
        sitk.WriteImage(corrected_image, output_path)
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Bias field corrected file not found: {output_path}")

    def normalise_and_correct(self):
        """ Load the skull-stripped image, perform normalization and bias correction """
        if self.stripped_mri_path is not None:
            try:
                # Normalize intensity
                stripped_image = nib.load(self.stripped_mri_path)
                stripped_image_data = stripped_image.get_fdata()
                normalized_data = self.normalize_intensity(stripped_image_data)
                normalized_img = nib.Nifti1Image(normalized_data, stripped_image.affine)

                # Save the normalized image (before bias correction)
                normalized_image_path = self.save_image(normalized_data, 'normalized.nii.gz')

                # Perform bias field correction
                self.corrected_image_path = os.path.join(self.save_folder, 'normalised_and_corrected.nii.gz')
                self.bias_field_correction(normalized_image_path, self.corrected_image_path)

                # Load and display the corrected image in the third canvas
                corrected_image = nib.load(self.corrected_image_path).get_fdata()
                self.display_slice(corrected_image, self.canvas_normalised)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to normalize and correct image: {str(e)}")

    def register_to_template(self):
        """ Register the skull-stripped image to the MNI152 template and display the result """
        if self.corrected_image_path is not None:
            try:
                # Register the image
                self.registered_mri_path = os.path.join(self.save_folder, 'transformed.nii.gz')
                self.perform_registration(self.corrected_image_path, template_path, self.registered_mri_path)

                # Load and display the registered image in the fourth canvas
                registered_image = nib.load(self.registered_mri_path).get_fdata()
                self.display_slice(registered_image, self.canvas_registered)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to register image to template: {str(e)}")

    def perform_registration(self, moving_image_path, fixed_image_path, output_prefix):
        """ Perform image registration to MNI152 template using ANTs """
        fixed = ants.image_read(fixed_image_path)
        moving = ants.image_read(moving_image_path)
        registration = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyN')
        warped_image = registration['warpedmovout']
        warped_image_path = output_prefix
        ants.image_write(warped_image, warped_image_path)
        return warped_image_path

    def segment_brain_parts(self):
        """ Segment the brain parts using the atlas and save the results """
        if self.registered_mri_path is not None:
            try:
                registered_mri = nib.load(self.registered_mri_path)
                registered_mri_data = registered_mri.get_fdata()

                atlas_img = nib.load(atlas_path)
                atlas_data = atlas_img.get_fdata()

                for label in labels:
                    label_out_path = os.path.join(self.save_folder, f"{label}.nii.gz")
                    label_mask = np.isin(atlas_data, labels[label]).astype(np.float32)
                    label_mask_img = nib.Nifti1Image(label_mask, atlas_img.affine)

                    resampled_mask_img = image.resample_to_img(label_mask_img, registered_mri, interpolation='nearest')
                    label_data = image.math_img("img1 * img2", img1=registered_mri, img2=resampled_mask_img)
                    label_img = nib.Nifti1Image(label_data.get_fdata(), registered_mri.affine)

                    nib.save(label_img, label_out_path)

                # Notify the user of successful segmentation
                messagebox.showinfo("Success", "Brain parts segmented and saved successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to segment brain parts: {str(e)}")
    
    def extract_sphericity(self, volume, surface_area):
        """Calculate sphericity from volume and surface area."""
        if surface_area == 0:
            return 0
        return (np.pi**(1/3)) * (6 * volume)**(2/3) / surface_area  
    
    def extract_volume(self, segmented_image_path):
        """Calculate the volume of the segmented brain part from the image."""
        segmented_img = nib.load(segmented_image_path)
        segmented_data = segmented_img.get_fdata()
        volume_voxels = np.sum(segmented_data)  # Volume in voxels
        voxel_volume = np.prod(segmented_img.header.get_zooms())  # Voxel size in mm³
        return volume_voxels * voxel_volume  # Volume in mm³
    
    def extract_compactness(self, volume, surface_area):
        if volume == 0:
            return 0
        return surface_area**2 / volume  # Compactness

    def extract_surface_area(self, segmented_image_path):
        """Calculate the surface area of the segmented brain part using marching cubes."""
        segmented_img = nib.load(segmented_image_path)
        segmented_data = segmented_img.get_fdata()
        
        # Ensure segmented data is binary
        segmented_data = (segmented_data > 0).astype(np.uint8)
        
        # Surface area using marching cubes
        verts, faces, _, _ = measure.marching_cubes(segmented_data, level=0)
        surface_area = measure.mesh_surface_area(verts, faces)
        
        return surface_area  # Surface area in mm²
    
    def extract_features(self):
        """Extract features such as the sphericity for hippocampus and cortex volume."""
        try:
            # --- Hippocampus Calculations ---
            left_hippocampus_path = os.path.join(self.save_folder, "left_hippocampus.nii.gz")
            right_hippocampus_path = os.path.join(self.save_folder, "right_hippocampus.nii.gz")

            left_volume = self.extract_volume(left_hippocampus_path)
            left_surface_area = self.extract_surface_area(left_hippocampus_path)
            self.left_sphericity = self.extract_sphericity(left_volume, left_surface_area)

            right_volume = self.extract_volume(right_hippocampus_path)
            right_surface_area = self.extract_surface_area(right_hippocampus_path)
            self.right_sphericity = self.extract_sphericity(right_volume, right_surface_area)

            self.average_hippocampus_sphericity = (self.left_sphericity + self.right_sphericity) / 2

            # --- Cortex Calculations ---
            left_cortex_path = os.path.join(self.save_folder, "left_cortex.nii.gz")
            right_cortex_path = os.path.join(self.save_folder, "right_cortex.nii.gz")

            left_cortex_volume = self.extract_volume(left_cortex_path)
            right_cortex_volume = self.extract_volume(right_cortex_path)
            self.average_cortex_volume = (left_cortex_volume + right_cortex_volume) / 2
            
            left_cortex_surface_area = self.extract_surface_area(left_cortex_path)
            right_cortex_surface_area = self.extract_surface_area(right_cortex_path)
            
            self.left_cortex_sphericity = self.extract_sphericity(left_cortex_volume, left_cortex_surface_area)
            self.right_cortex_sphericity = self.extract_sphericity(right_cortex_volume, right_cortex_surface_area)
            self.average_cortex_sphericity = (self.left_cortex_sphericity + self.right_cortex_sphericity) / 2
            
            # ----- Ventricle Compactness ---
            left_ventricle_path = os.path.join(self.save_folder, "left_lateral_ventricle.nii.gz")
            right_ventricle_path = os.path.join(self.save_folder, "right_lateral_ventricle.nii.gz")
            
            left_lateral_ventricle_volume = self.extract_volume(left_ventricle_path)
            right_lateral_ventricle_volume = self.extract_volume(right_ventricle_path)
            
            left_lateral_ventricle_surface_area = self.extract_surface_area(left_ventricle_path)
            right_lateral_ventricle_surface_area = self.extract_surface_area(right_ventricle_path)
            
            left_ventricle_compactness = self.extract_compactness(left_lateral_ventricle_volume, left_lateral_ventricle_surface_area)
            right_ventricle_compactness = self.extract_compactness(right_lateral_ventricle_volume, right_lateral_ventricle_surface_area)
            
            self.ventricle_compactness_ratio = left_ventricle_compactness / right_ventricle_compactness
            
            # ----- Ventricle Spherecity ---
            
            left_ventricle_spherecity = self.extract_sphericity(left_lateral_ventricle_volume, left_lateral_ventricle_surface_area)
            right_ventricle_spherecity = self.extract_sphericity(right_lateral_ventricle_volume, right_lateral_ventricle_surface_area)
            
            self.ventricle_sphericity_ratio = left_ventricle_spherecity / right_ventricle_spherecity
            
            # ---- Amygdala Volume ---
            left_amygdala_path = os.path.join(self.save_folder, "left_amygdala.nii.gz")
            right_amygdala_path = os.path.join(self.save_folder, "right_amygdala.nii.gz")
            
            left_amygdala_volume = self.extract_volume(left_amygdala_path)
            right_amygdala_volume = self.extract_volume(right_amygdala_path)
            
            self.amygdala_volume = (left_amygdala_volume + right_amygdala_volume) / 2
    
            # Update GUI
            self.entry_sphericity.config(state="normal")
            self.entry_sphericity.delete(0, tk.END)
            self.entry_sphericity.insert(0, f"{self.average_hippocampus_sphericity:.3f}")
            self.entry_sphericity.config(state="disabled")

            self.entry_cortex_volume.config(state="normal")
            self.entry_cortex_volume.delete(0, tk.END)
            self.entry_cortex_volume.insert(0, f"{self.average_cortex_volume:.3f}")
            self.entry_cortex_volume.config(state="disabled")
            
            self.entry_cortex_sphericity.config(state="normal")
            self.entry_cortex_sphericity.delete(0, tk.END)
            self.entry_cortex_sphericity.insert(0, f"{self.average_cortex_sphericity:.3f}")
            self.entry_cortex_sphericity.config(state="disabled")
            
            self.entry_ventricle_compactness_ratio.config(state="normal")
            self.entry_ventricle_compactness_ratio.delete(0, tk.END)
            self.entry_ventricle_compactness_ratio.insert(0, f"{self.ventricle_compactness_ratio:.3f}")
            self.entry_ventricle_compactness_ratio.config(state="disabled")
            
            self.entry_ventricle_sphericity_ratio.config(state="normal")
            self.entry_ventricle_sphericity_ratio.delete(0, tk.END)
            self.entry_ventricle_sphericity_ratio.insert(0, f"{self.ventricle_sphericity_ratio:.3f}")
            self.entry_ventricle_sphericity_ratio.config(state="disabled")
            
            self.entry_amygdala_volume.config(state="normal")
            self.entry_amygdala_volume.delete(0, tk.END)
            self.entry_amygdala_volume.insert(0, f"{self.amygdala_volume:.3f}")
            self.entry_amygdala_volume.config(state="disabled")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract features: {str(e)}")
            
    def classify(self):
        pass


if __name__ == "__main__":
    app = MRIViewerApp()
    app.mainloop()