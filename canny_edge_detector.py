import numpy as np
import numpy.typing as npt
import cv2
from copy import copy
from dataclasses import dataclass

Image = npt.NDArray[np.uint8]
Contour = npt.NDArray[np.int_]
@dataclass
class RegionOfInterest:
    x: int
    y: int
    width: int
    height: int

def image_segmentation(edges: Image, original_image: Image) -> tuple[Image, list[Contour]]:
    """Given an image of contours, colour in the largest objects

    Args:
        edges (Image): Image of segmented edges
        original_image (Image): Original image
    
    Returns:
        Image: New image with the largest contours colored in
        list[Contours]: List of the coloured in contours
    """
    # Step 1: Dilating the edges
    dilation_kernel = np.ones((6, 6), np.uint8)
    dilated_edges = cv2.dilate(edges, dilation_kernel, iterations=1)

    # Step 2: Closing the edges
    closing_kernel = np.ones((16, 16), np.uint8)
    closed_edges = np.array(
        cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, closing_kernel)).astype(dtype='uint8'
    )

    # Step 3: Find contours in the closed_edges
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4: Color in the contours
    segmented_image = np.zeros_like(original_image)
    colours = np.random.randint(0, 255, (len(contours), 3), dtype=np.uint8)
    for i, contour in enumerate(contours):
        colour = colours[i].tolist()
        cv2.drawContours(segmented_image, [contour], -1, colour, thickness=cv2.FILLED)
    
    return segmented_image, contours

def find_rois(
    segmented_image: Image, 
    contours: list[Contour], 
    height_fraction: float, 
    width_fraction: float
) -> tuple[Image, list[RegionOfInterest]]:
    """Find the ROIs of objects that fall within a certain size range
    
    Args:
        segmented_image (Image): Image with the largest contours colored in
        contours (list[Contour]): List of contours of the segmentated image
        height_fraction (float): Size threshold for the minimum height of the object to be delcared as significant
        width_fraction (float): Size theshold for the minimum width of the object to be declared as significant
    
    Returns:
        Image: Image with ROIs annotated
        list[RegionOfInterest]: List of ROIs of significance
    """
    # Step 1: Setup variables and bou ds
    roi_image = copy(segmented_image)
    rois_of_interest: list[RegionOfInterest] = []
    image_height, image_width, _ = segmented_image.shape


    for contour in contours:
        # Step 2: Calculate a bounding box around the image
        x, y, width, height = cv2.boundingRect(contour)

        # Step 3: Ensure the object is within desired size
        if width/image_width < width_fraction or height/image_height < height_fraction:
            continue

        # Step 4: Annotate image
        cv2.rectangle(roi_image,(x,y),(x+width,y+height),(255,255,255),2)
        cv2.putText(
            roi_image, f"x: {x}, y: {y}", (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2
        )
        cv2.putText(
            roi_image, f"w: {width}, h: {height}", (x+5, y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2
        )
        rois_of_interest.append(RegionOfInterest(x, y, width, height))
    
    return roi_image, rois_of_interest

def canny_edge_detector(
    original_image: Image, 
    low_threshold: float, 
    high_threshold: float, 
    kernel_size: int, 
    sigma: float, 
) -> Image:
    """Given an image and thresholds, perform Canny edge detection
    
    Args:
        original_image (Image): Image to perform detection on
        low_threshold (float): Low threshold (fraction) for discarding weak edges
        high_threshold (float): High threshold (fraction) for keeping strong edges
        kernel_size (int): Kernel size for performing Gaussian Blur
        sigma (float): Sigma for performing Gaussian Blue
    
    Returns:
        Image: Image with contours
    """
    # Step 1: Converting the image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Step 2: Applying Gaussian blur to reduce noise and smooth the image
    blurred_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), sigma)

    # Step 3: Computing the gradient magnitude and direction using Sobel filters
    gradient_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    # Step 4: Applying non-maximum suppression to thin out the edges
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

    # Step 5: Using double thresholding to identify strong and weak edges
    max_val = np.max(suppressed_image)
    low_threshold_val = max_val * low_threshold
    high_threshold_val = max_val * high_threshold

    strong_edges = suppressed_image > high_threshold_val
    weak_edges = (suppressed_image >= low_threshold_val) & (suppressed_image <= high_threshold_val)

    final_edges_image = np.zeros_like(suppressed_image)
    final_edges_image[strong_edges] = 255

    # Step 6: Edge tracking by hysteresis
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if weak_edges[i, j]:
                if (
                    np.any(strong_edges[i - 1 : i + 2, j - 1 : j + 2])
                    or np.any(strong_edges[i, j - 1 : j + 2])
                ):
                    final_edges_image[i, j] = 255
    
    return final_edges_image

def main():
    # Adjust the thresholds, kernel_size, and sigma for the edge detection
    low_threshold = 0.05
    high_threshold = 0.175
    kernel_size = 7
    sigma = 1.7

    # Object size bounds for determining ROIs
    height_fraction = 0.1
    width_fraction = 0.1

    # List of images
    image_paths = ("on_white_background.png", "on_black_background.png", "color.png")

    for image_path in image_paths:
        # Read the original image
        original_image = cv2.imread(image_path)

        # Extract edges using canny edge
        edges_image = canny_edge_detector(
            original_image, low_threshold, high_threshold, kernel_size, sigma
        )

        # Perform simple image segmentation
        segmented_image, segmented_contours = image_segmentation(edges_image, original_image)

        # Calculate the ROIs of significant objects
        bound_image, rois = find_rois(
            segmented_image, segmented_contours, height_fraction, width_fraction
        )

        # Display the results
        cv2.imshow(f"Original Image - {image_path}", original_image)
        cv2.imshow(f"Canny Edges - {image_path}", edges_image)
        cv2.imshow(f"Segmented Image - {image_path}", segmented_image)
        cv2.imshow(f"Bounded Objects - {image_path}", bound_image)

    # Display the results
    cv2.waitKey(0)
    input("Press ENTER to exit: ")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

