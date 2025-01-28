import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

# Load the brain MRI image (convert to grayscale if not already)
image_path = 'F:/Brain Tumor Detection/Dataset/Training/yes/y0.jpg'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define Gabor filter parameters
theta = 0  # Orientation of the filter, can vary from 0 to np.pi
frequency = 0.1  # Frequency of the sinusoidal wave in the filter

# Create a Gabor kernel
kernel = cv2.getGaborKernel((31, 31), 4.0, theta, 1.0/frequency, 0.5, 0, ktype=cv2.CV_32F)

# Apply the Gabor filter on the image
filtered_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)

# Display original and filtered images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title("Gabor Filtered Image")
plt.show()