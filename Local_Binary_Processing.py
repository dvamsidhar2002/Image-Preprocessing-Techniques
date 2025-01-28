import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage import exposure

def preprocess_image(image_path):
    """
    Preprocess the input image for LBP feature extraction.

    Parameters:
    - image_path: Path to the input image

    Returns:
    - Grayscale preprocessed image
    """
    # Load the image
    image = cv2.imread(image_path)

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to a fixed size (optional)
    gray_image = cv2.resize(gray_image, (256, 256))  # Resize to 256x256

    # Normalize the image to range [0, 1]
    gray_image = gray_image / 255.0

    return gray_image

def extract_lbp_features(image, radius=1, n_points=8):
    """
    Extract LBP features from the input image.

    Parameters:
    - image: Grayscale image
    - radius: Radius of the circle for LBP (typically 1 or 2)
    - n_points: Number of points to sample (typically 8 or 16)

    Returns:
    - LBP image
    - LBP histogram (feature vector)
    """
    # Compute LBP
    lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')

    # Compute histogram of LBP values (flattened)
    lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

    # Normalize the histogram to make it invariant to lighting changes
    lbp_hist = lbp_hist.astype('float')
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # Avoid division by zero

    return lbp_image, lbp_hist

def visualize_lbp(image, lbp_image):
    """
    Visualize the original image and its LBP representation.

    Parameters:
    - image: Original grayscale image
    - lbp_image: LBP transformed image
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Show original image
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Show LBP image
    ax2.imshow(lbp_image, cmap=plt.cm.gray)
    ax2.set_title('LBP Image')
    ax2.axis('off')

    plt.show()

def extract_and_visualize_lbp(image_path):
    """
    Extract and visualize LBP features from the given image.

    Parameters:
    - image_path: Path to the input image
    """
    # Step 1: Preprocess the image
    image = preprocess_image(image_path)

    # Step 2: Extract LBP features
    lbp_image, lbp_hist = extract_lbp_features(image)

    # Step 3: Visualize the LBP features
    visualize_lbp(image, lbp_image)

    # Optionally, you can return the LBP histogram for further processing
    return lbp_hist

if __name__ == "__main__":
    # Example usage
    image_path = 'F:/Brain Tumor Detection/Dataset/yes/y0.jpg'  # Replace with your image path
    lbp_features = extract_and_visualize_lbp(image_path)
    print("Extracted LBP histogram:", lbp_features)
    print("LBP histogram length:", len(lbp_features))
