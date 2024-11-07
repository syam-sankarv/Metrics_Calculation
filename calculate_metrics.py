# Required Libraries
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
import numpy as np
from PIL import Image
import glob
import os
import torch
from torchvision import transforms
import re
import pandas as pd
import openpyxl

target_size = (512, 512)

# Image Preprocessing and Utility Functions
def preprocess_image(image, target_size, threshold=20, target_values=(0, 1)):
    """
    Resizes the image to the target size and binarizes it by thresholding.
    All values below the threshold are set to 0 and values above it are set to the target max value (e.g., 1).
    """
    # Resize image
    image = image.resize(target_size) 
    
    # Convert image to tensor
    tensor = transforms.ToTensor()(image)
    
    # Binarize the tensor: Apply thresholding to convert to binary [0, 1]
    tensor = (tensor > (threshold / 255.0)).float()  # Values above threshold set to 1.0
    
    # Convert binary values to match the target (either 0 and 1)
    tensor = tensor * target_values[1]
    
    return tensor

def compute_iou(y_pred, y_true):
    """Compute IoU metric."""
    # Flatten tensors
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    
    # Ensure binary values [0, 1]
    y_pred = (y_pred > 0).int()
    y_true = (y_true > 0).int()
    
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    
    IoU = intersection / union.astype(np.float32)
    
    return np.nanmean(IoU)

def compute_recall(y_pred, y_true):
    """Compute recall metric."""
    y_pred = y_pred.flatten().cpu().numpy()
    y_true = y_true.flatten().cpu().numpy()
    
    y_pred = (y_pred > 0).astype(int)
    y_true = (y_true > 0).astype(int)
    
    recall = recall_score(y_true, y_pred)
    return recall

def compute_precision(y_pred, y_true):
    """Compute precision metric."""
    y_pred = y_pred.flatten().cpu().numpy()
    y_true = y_true.flatten().cpu().numpy()
    
    y_pred = (y_pred > 0).astype(int)
    y_true = (y_true > 0).astype(int)
    
    precision = precision_score(y_true, y_pred)
    return precision

def compute_f1(precision, recall):
    """Compute F1 score."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# Main Processing Function
def process_masks(predicted_mask_dir, ground_truth_dir, output_excel_path, target_size=target_size):
    masks = glob.glob(os.path.join(predicted_mask_dir, '*.png'))
    ground_truth_files = os.listdir(ground_truth_dir)

    d = []
    for file in masks:
        # Load the predicted mask image
        mask_file = Image.open(os.path.join(predicted_mask_dir, file))
        
        # Debugging: Print mask file size and tensor unique values
        print(f"Processing mask file: {file}")
        print(f"Mask file size: {mask_file.size}")
        
        # Extract the plot number and height from filename
        number = re.search('plot_(.*)_flight', file).group(1)
        height = re.search('_X(.*).png', file).group(1)
        
        # Load the corresponding ground truth file
        ground_truth_file = f'plot_{number}_flight_X10.tiff'
        ground_truth_image = Image.open(os.path.join(ground_truth_dir, ground_truth_file))
        
        # Preprocess the images (resize, threshold, binarize)
        ground_truth_tensor = preprocess_image(ground_truth_image, target_size)
        
        mask_tensor = preprocess_image(mask_file, target_size)
        
        
        # Debugging: Print image shapes and unique values of tensors
        print(f"Ground truth tensor shape: {ground_truth_tensor.shape}, unique values: {torch.unique(ground_truth_tensor)}")
        print(f"Mask tensor shape: {mask_tensor.shape}, unique values: {torch.unique(mask_tensor)}")
        
        # Calculate metrics
        iou = compute_iou(mask_tensor, ground_truth_tensor)
        recall = compute_recall(mask_tensor, ground_truth_tensor)
        precision = compute_precision(mask_tensor, ground_truth_tensor)
        f1 = compute_f1(precision, recall)
        
        # Store results
        d.append(
            {
                'IoU': iou,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Plot': file,
                'Species': 'Centaurea cyanus',
                'Height': height
            }
        )

        print(f"Finished processing: {file}\n")

    # Convert results to DataFrame
    Centaurea = pd.DataFrame(d)
    
    # Save the DataFrame as an Excel file
    Centaurea.to_excel(output_excel_path, index=False)
    print(f"Results saved to: {output_excel_path}")
    print(Centaurea)

# Call the processing function
predicted_mask_dir = r"D:\STUDY\Thesis\comparison-model\comparison-model\attention-unet✔\aug\predict_height"
ground_truth_dir = r"D:\STUDY\Thesis\ground_truth"
output_excel_path = r"D:\STUDY\Thesis\comparison-model\comparison-model\attention-unet✔\Centaurea_results-512-aug.xlsx"
process_masks(predicted_mask_dir, ground_truth_dir, output_excel_path)
