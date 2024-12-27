import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torchmetrics.classification import JaccardIndex, Dice

#___________________________________________________________________________________________________
# General Functions
# __________________________________________________________________________________________________

def get_image_path(folder_name, file_name):
    return os.path.join("Img", folder_name, file_name)

def display_image(image, title):
     # Display the filtered image
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def filter_image_top(image):
   # Get the shape of the image
    height, width = image.shape    
    # Create a filter matrix with ones
    filter_matrix = np.ones((height, width))
    # Set rows y = 0 to y = 175 to zeros
    filter_matrix[0:176, :] = 0  # Note: Includes row 175
    # Multiply the filter matrix by the image
    filtered_image = image * filter_matrix
    return filtered_image

def is_8bit_image(image):
    """
    Check if the given image is an 8-bit grayscale or RGB image.

    Parameters:
        image (numpy.ndarray): The image to check.

    Returns:
        bool: True if the image is 8-bit, False otherwise.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    
    # Check the data type and value range
    return image.dtype == np.uint8 and image.min() >= 0 and image.max() <= 255

def create_8_bit_image(image):
    return (image * 255).astype(np.uint8)

def is_grayscale(image):
    """
    Determines if an image is grayscale.

    Args:
        image (numpy.ndarray): The input image (loaded using cv2 or similar).

    Returns:
        bool: True if the image is grayscale, False otherwise.
    """
    # Check if the image has only one channel (already grayscale)
    if len(image.shape) == 2:
        return True
    # If it has three channels (color), check if all channels are identical
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Compare all channels
        return np.all(image[:, :, 0] == image[:, :, 1]) and np.all(image[:, :, 1] == image[:, :, 2])

    # If neither condition is met, the image is not grayscale
    return False

# //////////////////////////////////////////////////////////////////////////////////////////////////


#___________________________________________________________________________________________________
# Evaluation Functions
# __________________________________________________________________________________________________

def calculate_accuracy(ground_truth, prediction):
    """
    Calculate the accuracy of the predicted mask compared to the ground truth mask.

    Parameters:
    - ground_truth: 2D numpy array (8-bit) representing the ground truth mask.
    - prediction: 2D numpy array (8-bit) representing the predicted mask.

    Returns:
    - accuracy: A float value between 0 and 1 representing the accuracy.
    """
    # Ensure the inputs are numpy arrays
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)

    # Ensure the masks are binary (0 and 1 only)
    # ground_truth = (ground_truth > 0).astype(int)
    # prediction = (prediction > 0).astype(int)
    
    # Check that both images have the same shape
    if ground_truth.shape != prediction.shape:
        raise ValueError("Ground truth and prediction must have the same shape")
    
    # Calculate the number of correct pixels
    correct_pixels = np.sum(ground_truth == prediction)
    
    # Calculate the total number of pixels
    total_pixels = ground_truth.size
    
    # Calculate the accuracy
    accuracy = correct_pixels / total_pixels
    return accuracy


def calculate_weighted_accuracy(gtm_image, binary_mask):
    """
    Calculates the weighted accuracy for binary segmentation.

    Parameters:
        gtm_image (np.ndarray): Ground truth binary mask (2D array).
        binary_mask (np.ndarray): Predicted binary mask (2D array).

    Returns:
        float: The weighted accuracy.
    """
    # Ensure the masks are binary (0 and 1 only)
    gtm_image = (gtm_image > 0).astype(int)
    binary_mask = (binary_mask > 0).astype(int)

    # Total number of pixels
    total_pixels = gtm_image.size

    # Count the number of white (1) and black (0) pixels in the ground truth
    num_white_pixels = gtm_image.sum()
    num_black_pixels = total_pixels - num_white_pixels

    # Calculate the percentage of black and white pixels
    P_b = num_black_pixels / total_pixels
    P_w = 1 - P_b  # P_w is complementary to P_b

    # Count true positives for white and black pixels
    T_w = ((gtm_image == 1) & (binary_mask == 1)).sum()  # True white pixels
    T_b = ((gtm_image == 0) & (binary_mask == 0)).sum()  # True black pixels

    # Weighted accuracy calculation
    weighted_acc = ((P_b * T_w) + (P_w * T_b)) / total_pixels

    return weighted_acc


# def calculate_dice_score(gtm_image, binary_mask):
#     """
#     Calculates the Dice Score for binary segmentation.

#     Parameters:
#         gtm_image (np.ndarray): Ground truth binary mask (2D array).
#         binary_mask (np.ndarray): Predicted binary mask (2D array).

#     Returns:
#         float: The Dice Score.
#     """
#     # Calculate the intersection and the sum of the ground truth and predicted masks
#     intersection = np.sum((gtm_image == binary_mask))
#     # union = np.sum(gtm_image) + np.sum(binary_mask)
#     union = gtm_image.size + binary_mask.size

#     # Calculate the Dice Score
#     if union == 0:
#         return 1.0  # If both are empty, consider it as a perfect match
#     else:
#         return 2 * intersection / union

def calculate_dice_score(gtm_image, binary_mask):
    """
    Calculates the Dice Score for binary segmentation using torchmetrics.

    Parameters:
        gtm_image (np.ndarray): Ground truth binary mask (2D array).
        binary_mask (np.ndarray): Predicted binary mask (2D array).

    Returns:
        float: The Dice Score, which is a measure of similarity between the ground truth and predicted masks.
    """

    # Ensure that the values are binary (0 or 1), thresholding if necessary
    gtm_image = (gtm_image > 0).astype(np.int64)  # Convert to binary: 0 or 1
    binary_mask = (binary_mask > 0).astype(np.int64)  # Convert to binary: 0 or 1
    # Convert the numpy arrays to torch tensors
    gtm_image_tensor = torch.tensor(gtm_image, dtype=torch.int64)
    binary_mask_tensor = torch.tensor(binary_mask, dtype=torch.int64)

    # Initialize the Dice metric for binary class (2 classes: 0 and 1)
    try:
        dice = Dice(average="micro", num_classes=2)
    except ValueError as e:
        print(e)
    # Compute and return the Dice score for all classes
    return dice(binary_mask_tensor, gtm_image_tensor).cpu().numpy()

def calculate_dice_score_for_1_class(gtm_image, binary_mask):
    """
    Calculates the Dice Score for the 1 class (white pixels) in binary segmentation.

    Parameters:
        gtm_image (np.ndarray): Ground truth binary mask (2D array).
        binary_mask (np.ndarray): Predicted binary mask (2D array).

    Returns:
        float: The Dice Score for the 1 class.
    """
    gtm_image = (gtm_image > 0).astype(int)
    binary_mask = (binary_mask > 0).astype(int)
    # Calculate the intersection where both masks have 1
    intersection = np.sum((gtm_image == 1) & (binary_mask == 1))

    # Calculate the total number of pixels where either mask has 1 (union)
    total_positive_pixels = np.sum((gtm_image == 1)) +  np.sum((binary_mask == 1))
    
    # Calculate the Dice Score for the 1 class (positive class)
    if total_positive_pixels == 0:
        return 1.0  # If there are no 1s in either mask, consider it as a perfect match

    else:
        return 2 * intersection / total_positive_pixels
    

def calculate_jaccard_index(gtm_image, binary_mask):
    """
    Calculates the Jaccard Index (IoU) for multiclass segmentation using torchmetrics.

    Parameters:
        gtm_image (np.ndarray): Ground truth multi-class mask (2D array for multiclass).
        binary_mask (np.ndarray): Predicted multi-class mask (2D array for multiclass).

    Returns:
        float: The Jaccard Index for the specified number of classes.
    """

    # Ensure that the values are binary (0 or 1), thresholding if necessary
    gtm_image = (gtm_image > 0).astype(np.int64)  # Convert to binary: 0 or 1
    binary_mask = (binary_mask > 0).astype(np.int64)  # Convert to binary: 0 or 1
    # Convert the numpy arrays to torch tensors
    gtm_image_tensor = torch.tensor(gtm_image, dtype=torch.int64)
    binary_mask_tensor = torch.tensor(binary_mask, dtype=torch.int64)

    # Initialize the JaccardIndex metric for multiclass with the specified number of classes
    jaccard = JaccardIndex(task="binary", average="weighted")

    # Compute and return the Jaccard Index for all classes
    return jaccard(binary_mask_tensor, gtm_image_tensor).cpu().numpy()

# //////////////////////////////////////////////////////////////////////////////////////////////////
