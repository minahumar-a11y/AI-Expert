import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(image, title, colormap=None):
    """Utility function to display an image."""
    plt.figure()
    if len(image.shape) == 2: # Grayscale image
        plt.imshow(image, cmap='gray')
    else: # Color image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def interactive_edge_detection(image_path):
    """Interactive activity for edge detection and filtering."""

    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found!")
        return

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    display_image(gray_image, 'Grayscale Image', colormap='gray')

    print("Select an option:")
    print("1. Sobel Edge Detection")
    print("2. Canny Edge Detection")
    print("3. Laplacian Edge Detection")
    print("4. Gaussian Smoothing")
    print("5. Median Filtering")
    print("6. Exit")

    while True:
        choice = input("Enter your choice (1-6): ")

        if choice == "1":
            # Sobel Edge Detection
            ksize = 3
            sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=ksize)
            sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=ksize)
            combined_sobel = cv2.bitwise_or(sobelx.astype(np.uint8), sobely.astype(np.uint8))
            display_image(combined_sobel, 'Combined Sobel')

        elif choice == "2":
            # Canny Edge Detection
            # Default thresholds for Canny (default: 100 and 200)
            lower_thresh = int(input("Enter lower threshold: "))
            upper_thresh = int(input("Enter upper threshold: "))
            edges = cv2.Canny(gray_image, lower_thresh, upper_thresh)
            display_image(edges, 'Canny Edge Detection')

        elif choice == "3":
            # Laplacian Edge Detection
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            display_image(laplacian.astype(np.uint8), 'Laplacian Edge Detection') # .astype(np.uint8) added for correct display

        elif choice == "4":
            # Gaussian Smoothing
            # Kernel size for Gaussian Blur (must be odd, default: 5)
            kernel_size = int(input("Enter kernel size (odd number): "))
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            display_image(blurred, 'Gaussian Filtered')

        elif choice == "5":
            # Median Filtering
            # Kernel size for Median filtering (must be odd, default: 5)
            kernel_size = int(input("Enter kernel size (odd number): "))
            median_filtered = cv2.medianBlur(image, kernel_size)
            display_image(median_filtered, 'Median Filtered')

        elif choice == "6":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please select a number between 1 and 6.")

# Provide the path to an image for the activity
interactive_edge_detection('example.jpg')