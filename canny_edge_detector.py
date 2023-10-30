import numpy as np
import cv2

def canny_edge_detector(original_image, low_threshold, high_threshold, kernel_size=5, sigma=1.4, name = ""):
    # Step 1: Convert the image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow(f"Gray Image - {name}", gray_image)

    # Step 2: Apply Gaussian blur to reduce noise and smooth the image
    blurred_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), sigma)
    cv2.imshow(f"Blurred Image - {name}", blurred_image)

    # Step 3: Compute the gradient magnitude and direction using Sobel filters
    gradient_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    # Step 4: Non-maximum suppression to thin out the edges
    gradient_direction = np.degrees(gradient_direction)
    gradient_direction[gradient_direction < 0] += 180  # Convert angles to positive range

    suppressed_image = np.zeros_like(gradient_magnitude)
    rows, cols = gradient_magnitude.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle = gradient_direction[i, j]

            q = 255
            r = 255

            # Vertical edge
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = gradient_magnitude[i, j + 1]
                r = gradient_magnitude[i, j - 1]
            # Diagonal 1
            elif 22.5 <= angle < 67.5:
                q = gradient_magnitude[i + 1, j - 1]
                r = gradient_magnitude[i - 1, j + 1]
            # Horizontal edge
            elif 67.5 <= angle < 112.5:
                q = gradient_magnitude[i + 1, j]
                r = gradient_magnitude[i - 1, j]
            # Diagonal 2
            elif 112.5 <= angle < 157.5:
                q = gradient_magnitude[i - 1, j - 1]
                r = gradient_magnitude[i + 1, j + 1]

            if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                suppressed_image[i, j] = gradient_magnitude[i, j]

    # Step 5: Double thresholding to identify strong and weak edges
    max_val = np.max(suppressed_image)
    low_threshold_val = max_val * low_threshold
    high_threshold_val = max_val * high_threshold

    strong_edges = suppressed_image > high_threshold_val
    weak_edges = (suppressed_image >= low_threshold_val) & (suppressed_image <= high_threshold_val)

    # Step 6: Edge tracking by hysteresis
    edges_final = np.zeros_like(suppressed_image)
    edges_final[strong_edges] = 255

    # Use recursive search for weak edges connected to strong edges
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if weak_edges[i, j]:
                if (
                    np.any(strong_edges[i - 1 : i + 2, j - 1 : j + 2])
                    or np.any(strong_edges[i, j - 1 : j + 2])
                ):
                    edges_final[i, j] = 255

    # Define a kernel for dilation
    dilation_kernel = np.ones((6, 6), np.uint8)

    # Dilate the edges
    dilated_edges = cv2.dilate(edges_final, dilation_kernel, iterations=1)

    # Define a kernel for closing
    closing_kernel = np.ones((16, 16), np.uint8)

    # Perform closing operation
    closed_edges = np.array(cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, closing_kernel)).astype(dtype='uint8')

    # Find contours in the closed_edges
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segment_colors = np.random.randint(0, 255, (len(contours), 3), dtype=np.uint8)

    colour_image = np.zeros_like(original_image)

    for i, contour in enumerate(contours):
        colour = segment_colors[i].tolist()
        cv2.drawContours(colour_image, [contour], -1, colour, thickness=cv2.FILLED)

    cv2.imshow(f"Colour Image - {name}", colour_image)
    cv2.imshow(f"Original Image - {name}", original_image)
    cv2.imshow(f"Canny Edges - {name}", edges_final)

    return edges_final

# Adjust the thresholds, kernel_size, and sigma based on your preference
low_threshold = 0.05
high_threshold = 0.20
kernel_size = 7
sigma = 1.7

# Load an image and apply Canny edge detection
image_paths = ("on_white_background.png", "on_black_background.png", "color.png")

for image_path in image_paths:
    original_image = cv2.imread(image_path)
    canny_edge_detector(original_image, low_threshold, high_threshold, kernel_size, sigma, image_path)


# Display the results
cv2.waitKey(0)
cv2.destroyAllWindows()

