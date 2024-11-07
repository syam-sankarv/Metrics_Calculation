
---

# Image Mask Processing and Evaluation

This project is a part of my thesis and evaluates predicted image masks by comparing them with ground truth images, calculating metrics like Intersection over Union (IoU), Precision, Recall, and F1 Score. The results are saved in an Excel file.

## Requirements

To run this code, install the following Python libraries:

- `numpy`
- `PIL` (Python Imaging Library, for image processing)
- `torch` (PyTorch, for tensor operations)
- `torchvision` (for image transformations)
- `sklearn` (for metrics)
- `pandas` (for data handling)
- `openpyxl` (for Excel file handling)

Install these packages with:
```bash
pip install numpy pillow torch torchvision scikit-learn pandas openpyxl
```

## Code Overview

### 1. **Image Preprocessing**

The function `preprocess_image` resizes each image to a target size and converts it into a binary format using a threshold. This helps standardize images for accurate comparison.

### 2. **Metric Calculations**

Several metrics are computed to assess how well the predicted masks match the ground truth images:

- **IoU (Intersection over Union):** Measures overlap between predicted and actual mask.
- **Precision:** Indicates how many pixels identified as "object" by the model are correct.
- **Recall:** Shows how many actual object pixels are correctly identified by the model.
- **F1 Score:** Combines Precision and Recall to provide a balanced metric.

### 3. **Processing Masks**

The `process_masks` function performs the main tasks:

- Loads predicted mask and ground truth images.
- Processes each image using `preprocess_image`.
- Calculates the above metrics using custom functions (`compute_iou`, `compute_recall`, `compute_precision`, and `compute_f1`).
- Saves results in an Excel file, including each mask’s IoU, Precision, Recall, F1 Score, file name, and additional metadata.

### 4. **Run the Code**

To execute the code, specify the paths:

1. `predicted_mask_dir`: Directory containing the predicted mask images.
2. `ground_truth_dir`: Directory with the ground truth images.
3. `output_excel_path`: Path to save the results as an Excel file.

Then, run the `process_masks` function with these paths:
```python
predicted_mask_dir = r"path\to\predicted\masks"
ground_truth_dir = r"path\to\ground\truth\images"
output_excel_path = r"path\to\output\Centaurea_results.xlsx"
process_masks(predicted_mask_dir, ground_truth_dir, output_excel_path)
```

## Example Usage

If your predicted masks are in `D:\STUDY\Thesis\predict_height` and ground truth images are in `D:\STUDY\Thesis\ground_truth`, save results to an Excel file as follows:

```python
predicted_mask_dir = r"D:\STUDY\Thesis\predict_height"
ground_truth_dir = r"D:\STUDY\Thesis\ground_truth"
output_excel_path = r"D:\STUDY\Thesis\Centaurea_results.xlsx"
process_masks(predicted_mask_dir, ground_truth_dir, output_excel_path)
```

## Output

An Excel file will be saved at the specified `output_excel_path`, containing each image’s metrics (`IoU`, `Precision`, `Recall`, and `F1 Score`), as well as image identifiers (`Plot`, `Species`, and `Height`).

## Additional Notes

- Ensure all predicted mask files are in `.png` format.
- Ensure each ground truth image has a corresponding predicted mask for proper comparison.
- Adjust `target_size` and `threshold` in the `preprocess_image` function as needed for specific use cases.

---
