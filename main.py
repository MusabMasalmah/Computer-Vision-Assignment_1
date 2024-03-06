import math

import cv2
import numpy as np
# Read the image

#******************************************Q1P1P2*************************************************************
image = cv2.imread('C:/Users/OMEN/OneDrive/Desktop/1st_Sem_University/Comp. Vision/Assighment/Assign1/Q1/Input.jpg',
                   cv2.IMREAD_GRAYSCALE)
"""
# Check if the image is loaded successfully
if image is not None:
    # Normalize pixel values to the range [0, 1]
    normalized_image = image / 255.0

    # Apply power law transformation with gamma=0.4
    gamma = 0.4
    transformed_image = np.power(normalized_image, gamma)

    # Scale the values back to the range [0, 255]
    transformed_image = np.uint8(transformed_image * 255)

    # Display the original and transformed images
    cv2.imshow('Transformed Image (Gamma=0.4)', transformed_image)

    cv2.imshow('Original Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('P2_transformed.jpg', transformed_image)
else:
    print("Error: Unable to load the image.")


#*******************************************************P3*************************************************
# Check if the image is loaded successfully
if image is not None:
    # Generate Gaussian noise with zero mean and variance = 40
    mean = 0
    variance = 40
    sigma = np.sqrt(variance)
    height, width = image.shape
    gaussian_noise = np.random.normal(mean, sigma, (height, width))

    # Add the noise to the image
    noisy_image = image + gaussian_noise

    # Clip pixel values to be within the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    # Display the original and noisy images
    cv2.imshow('Original Image', image)
    cv2.imshow('Noisy Image (Variance = 40)', noisy_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('P3_Noisy.jpg', noisy_image)
else:
    print("Error: Unable to load the image.")


#********************************************P4**********************************************************************
noisy_image = cv2.imread('C:/Users/OMEN/OneDrive/Desktop/1st_Sem_University/Comp. Vision/Assighment/Assign1/Q1/P3_Noisy.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if noisy_image is not None:
    # Define the 5x5 mean filter kernel
    mean_filter = np.ones((5, 5), np.float32) / 25

    # Apply the mean filter to the noisy image
    filtered_image = cv2.filter2D(noisy_image, -1, mean_filter)

    # Display the original and noisy images
    cv2.imshow('Noisy Image (Variance = 40)', noisy_image)
    cv2.imshow('Filtered Image', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('P4_filtered_image.jpg', filtered_image)
else:
    print("Error: Unable to load the noisy image.")


#**************************************************************P5 P6 ***************************************
# Check if the image is loaded successfully
if image is not None:
    # Add salt and pepper noise to the original image
    noise_density = 0.1
    noisy_image = np.copy(image)

    # Salt noise
    salt_coords = [np.random.randint(0, high, int(noise_density * high)) for high in image.shape[:2]]
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    # Pepper noise
    pepper_coords = [np.random.randint(0, high, int(noise_density * high)) for high in image.shape[:2]]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    # Apply a 7x7 median filter to the noisy image
    median_filtered_image = cv2.medianBlur(noisy_image, 7)

    # Apply a 7x7 mean filter to the noisy image
    mean_filtered_image = cv2.blur(noisy_image, (7, 7))

    # Display the original, noisy, and median-filtered images
    cv2.imshow('Original Image', image)
    cv2.imshow('Noisy Image (Salt and Pepper Noise)', noisy_image)
    cv2.imshow('Median Filtered Image (7x7)', median_filtered_image)
    cv2.imshow('Mean Filtered Image (7x7)', mean_filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('P5_Salt_Paper.jpg', noisy_image)
    cv2.imwrite('P5_median_fil.jpg', median_filtered_image)
    cv2.imwrite('P6_mean_fil.jpg', mean_filtered_image)
else:
    print("Error: Unable to load the image.")

#*****************************************************P7*****************************************************

if image is not None:
    # Sobel kernels for horizontal and vertical edges

    # Initialize arrays to store Sobel x and Sobel y responses
    sobel_x = np.zeros_like(image, dtype=np.float64)
    sobel_y = np.zeros_like(image, dtype=np.float64)

    # Convolve the image with Sobel kernels
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            # Perform convolution for Sobel x
            sobel_x[i, j] = int(image[i - 1, j - 1] * -1) + int(image[i - 1, j] * -2) + int(
                image[i - 1, j + 1] * -1) + int(image[i + 1, j - 1] * 1) + int(image[i + 1, j] * 2) + int(
                image[i + 1, j + 1] * 1)

            # Perform convolution for Sobel y
            sobel_y[i, j] = int(image[i - 1, j - 1] * 1) + int(image[i, j - 1] * 2) + int(
                image[i + 1, j - 1] * 1) + int(image[i - 1, j + 1] * -1) + int(image[i, j + 1] * -2) + int(
                image[i + 1, j + 1] * -1)

    sobel_response = np.zeros_like(image, dtype=np.float64)
    print(sobel_response)
    # Combine the horizontal and vertical responses
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            sobel_response[i][j] = math.sqrt(sobel_x[i][j]**2 + sobel_y[i][j]**2)

    # Display the original image, Sobel x, Sobel y, and Sobel response
    cv2.imshow('Original Image', image)
    cv2.imshow('Sobel Response', sobel_response.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('P7_Salt_Paper.jpg', sobel_response.astype(np.uint8))


else:
    print("Error: Unable to load the image.")


#**********************************************************Q2**********************************************************

image1 = cv2.imread('C:/Users/OMEN/OneDrive/Desktop/1st_Sem_University/Comp. Vision/Assighment/Assign1/Q2/House1.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('C:/Users/OMEN/OneDrive/Desktop/1st_Sem_University/Comp. Vision/Assighment/Assign1/Q2/House2.jpg', cv2.IMREAD_GRAYSCALE)

image_rows = 0
image_cols = 0
def myImageFilter(input_image, filter_type, sigma=None):
    if filter_type == 1:
        # Develop Averaging filter(3, 3) mask
        filter = np.ones([3, 3], dtype=int)
        filter = filter / 9

    elif filter_type == 2:
        # Develop Averaging filter(3, 3) mask
        filter = np.ones([5, 5], dtype=int)
        filter = filter / 25

    elif filter_type == 3:
        filter = generateGaussianKernel(sigma)  # Gaussian filter
    elif filter_type == 4:
        filter = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])  # Sobel X filter
    elif filter_type == 5:
        filter = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])  # Sobel Y filter
    elif filter_type == 6:
        filter = np.array([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]])  # Prewitt X filter
    elif filter_type == 7:
        filter = np.array([[-1, -1, -1],
                           [0, 0, 0],
                           [1, 1, 1]])  # Prewitt Y filter
    else:
        raise ValueError("Invalid filter type. Choose from 1, 2, 3, 4, 5, 6, 7.")

    # Get the dimensions of the input image and the filter
    image_rows, image_cols = input_image.shape
    filter_rows, filter_cols = filter.shape

    # Compute the padding required for the convolution
    pad_rows = filter_rows // 2
    pad_cols = filter_cols // 2

    # Pad the input image
    padded_image = np.pad(input_image, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode='constant')

    # Initialize the output image
    output_image = np.zeros((image_rows, image_cols))

    # Perform convolution
    for i in range(image_rows):
        for j in range(image_cols):
            # Extract the region of interest from the padded image
            roi = padded_image[i:i + filter_rows, j:j + filter_cols]

            # Perform element-wise multiplication and sum
            output_image[i, j] = np.sum(roi * filter)

    return output_image


def generateGaussianKernel(sigma):
    size = int(2 * sigma + 1)
    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - size // 2) ** 2 + (y - size // 2) ** 2)/ (2 * sigma ** 2)),(size, size))
    return kernel / np.sum(kernel)


# Apply different filters using filter types 1, 2, 3, 4, 5, 6, 7
result_filter1 = myImageFilter(image1, 1)
result_filter2 = myImageFilter(image1, 2)
result_filter3_sigma1 = myImageFilter(image1, 3, sigma=1)
result_filter3_sigma2 = myImageFilter(image1, 3, sigma=2)
result_filter3_sigma3 = myImageFilter(image1, 3, sigma=3)

result_filter4_sobel_x = myImageFilter(image1, 4)
result_filter5_sobel_y = myImageFilter(image1, 5)
sobel_response = np.zeros((476, 640), dtype=np.float64)
for i in range(1, 475):  # Loop from 0 to 639
    for j in range(1, 640):  # Loop from 0 to 475
        sobel_response[i][j] = math.sqrt(result_filter4_sobel_x[i][j] ** 2 + result_filter5_sobel_y[i][j] ** 2)

sobel_response = cv2.normalize(sobel_response, None, 0, 1023, cv2.NORM_MINMAX)

result_filter6_prewitt_x = myImageFilter(image1, 6)
result_filter7_prewitt_y = myImageFilter(image1, 7)
prewitt_response = np.zeros((476, 640), dtype=np.float64)

for i in range(1, 475):  # Loop from 0 to 639
    for j in range(1, 640):  # Loop from 0 to 475
        prewitt_response[i][j] = math.sqrt(result_filter6_prewitt_x[i][j] ** 2 + result_filter7_prewitt_y[i][j] ** 2)

prewitt_response = cv2.normalize(prewitt_response, None, 0, 1023, cv2.NORM_MINMAX)

result_filter1_2 = myImageFilter(image2, 1)
result_filter2_2 = myImageFilter(image2, 2)
result_filter3_sigma1_2 = myImageFilter(image2, 3, sigma=1)
result_filter3_sigma2_2 = myImageFilter(image2, 3, sigma=2)
result_filter3_sigma3_2 = myImageFilter(image2, 3, sigma=3)

result_filter4_sobel_x_2 = myImageFilter(image2, 4)
result_filter5_sobel_y_2 = myImageFilter(image2, 5)
sobel_response_2 = np.zeros((768, 1024), dtype=np.float64)

for i in range(1, 768):  # Loop from 0 to 639
    for j in range(1, 1024):  # Loop from 0 to 475
        sobel_response_2[i][j] = math.sqrt(result_filter4_sobel_x_2[i][j] ** 2 + result_filter5_sobel_y_2[i][j] ** 2)
sobel_response_2 = cv2.normalize(sobel_response_2, None, 0, 1023, cv2.NORM_MINMAX)

result_filter6_prewitt_x_2 = myImageFilter(image2, 6)
result_filter7_prewitt_y_2 = myImageFilter(image2, 7)
prewitt_response_2 = np.zeros((768, 1024), dtype=np.float64)

for i in range(1, 768):  # Loop from 0 to 639
    for j in range(1, 1024):  # Loop from 0 to 475
        prewitt_response_2[i][j] = math.sqrt(result_filter6_prewitt_x_2[i][j] ** 2 + result_filter7_prewitt_y_2[i][j] ** 2)

prewitt_response_2 = cv2.normalize(prewitt_response_2, None, 0, 1023, cv2.NORM_MINMAX)




# Display results using cv2.imshow
cv2.imshow('Original Image 1', image1)
cv2.imshow('Averaging Filter (3x3)', result_filter1.astype(np.uint8))
cv2.imshow('Averaging Filter (5x5)', result_filter2.astype(np.uint8))
cv2.imshow('Gaussian Filter (σ=1)', result_filter3_sigma1.astype(np.uint8))
cv2.imshow('Gaussian Filter (σ=2)', result_filter3_sigma2.astype(np.uint8))
cv2.imshow('Gaussian Filter (σ=3)', result_filter3_sigma3.astype(np.uint8))
cv2.imshow('Sobel Filter', sobel_response.astype(np.uint8))
cv2.imshow('prewitt response Filter', prewitt_response.astype(np.uint8))



cv2.imshow('Original Image 1_2', image2)
cv2.imshow('Averaging Filter (3x3)_2', result_filter1_2.astype(np.uint8))
cv2.imshow('Averaging Filter (5x5)_2', result_filter2_2.astype(np.uint8))
cv2.imshow('Gaussian Filter (σ=1)_2', result_filter3_sigma1_2.astype(np.uint8))
cv2.imshow('Gaussian Filter (σ=2)_2', result_filter3_sigma2_2.astype(np.uint8))
cv2.imshow('Gaussian Filter (σ=3)_2', result_filter3_sigma3_2.astype(np.uint8))
cv2.imshow('Sobel Filter_2', sobel_response_2.astype(np.uint8))
cv2.imshow('prewitt response Filter_2', prewitt_response_2.astype(np.uint8))


# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('Q2_H1_AF(3x3).jpg', result_filter1.astype(np.uint8))
cv2.imwrite('Q2_H1_AF(5x5).jpg', result_filter2.astype(np.uint8))
cv2.imwrite('Q2_H2_AF(3x3).jpg', result_filter1_2.astype(np.uint8))
cv2.imwrite('Q2_H2_AF(5x5).jpg', result_filter2_2.astype(np.uint8))
cv2.imwrite('Q2_H1_GF_S1.jpg', result_filter3_sigma1.astype(np.uint8))
cv2.imwrite('Q2_H1_GF_S2.jpg', result_filter3_sigma2.astype(np.uint8))
cv2.imwrite('Q2_H1_GF_S3.jpg', result_filter3_sigma3.astype(np.uint8))
cv2.imwrite('Q2_H2_GF_S1.jpg', result_filter3_sigma1_2.astype(np.uint8))
cv2.imwrite('Q2_H2_GF_S2.jpg', result_filter3_sigma2_2.astype(np.uint8))
cv2.imwrite('Q2_H2_GF_S3.jpg', result_filter3_sigma3_2.astype(np.uint8))
cv2.imwrite('Q3_H1_SF.jpg', sobel_response.astype(np.uint8))
cv2.imwrite('Q3_H2_SF.jpg', sobel_response_2.astype(np.uint8))
cv2.imwrite('Q4_H1_PF.jpg', prewitt_response.astype(np.uint8))
cv2.imwrite('Q4_H2_PF.jpg', prewitt_response_2.astype(np.uint8))



# Read the noisy images
P3_image1 = cv2.imread('C:/Users/OMEN/OneDrive/Desktop/1st_Sem_University/Comp. Vision/Assighment/Assign1/Q3/Noisyimage1.jpg', cv2.IMREAD_GRAYSCALE)
P3_image2 = cv2.imread('C:/Users/OMEN/OneDrive/Desktop/1st_Sem_University/Comp. Vision/Assighment/Assign1/Q3/Noisyimage2.jpg', cv2.IMREAD_GRAYSCALE)

# Apply 5x5 Averaging filter
averaging_filter = np.ones((5, 5), dtype=np.float32) / 25
P3_result_averaging1 = cv2.filter2D(P3_image1, -1, averaging_filter)
P3_result_averaging2 = cv2.filter2D(P3_image2, -1, averaging_filter)

# Apply 5x5 Median filter
P3_result_median1 = cv2.medianBlur(P3_image1, 5)
P3_result_median2 = cv2.medianBlur(P3_image2, 5)

# Display the results
cv2.imshow('Noisy Image 1', P3_image1)
cv2.imshow('Averaging Filter Result 1', P3_result_averaging1)
cv2.imshow('Median Filter Result 1', P3_result_median1)

cv2.imshow('Noisy Image 2', P3_image2)
cv2.imshow('Averaging Filter Result 2', P3_result_averaging2)
cv2.imshow('Median Filter Result 2', P3_result_median2)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('Q3_I1_AF.jpg', P3_result_averaging1.astype(np.uint8))
cv2.imwrite('Q3_I1_MF.jpg', P3_result_median1.astype(np.uint8))
cv2.imwrite('Q3_I2_AF.jpg', P3_result_averaging2.astype(np.uint8))
cv2.imwrite('Q3_I2_MF.jpg', P3_result_median2.astype(np.uint8))


# Read the image
P4_image = cv2.imread('C:/Users/OMEN/OneDrive/Desktop/1st_Sem_University/Comp. Vision/Assighment/Assign1/Q4/Q_4.jpg', cv2.IMREAD_GRAYSCALE)

# Compute gradient magnitude using Sobel gradients
gradient_x = cv2.Sobel(P4_image, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(P4_image, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

# Stretch the resulting magnitude for better visualization
gradient_magnitude_stretched = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Compute histogram of gradient magnitude
hist, bins = np.histogram(gradient_magnitude, bins=256, range=[0, 256])

# Compute gradient orientation (angle of gradient vector)
gradient_orientation = np.arctan2(gradient_y, gradient_x)

# Compute histogram of gradient orientation (angle between 0 and 2*pi)
hist_orientation, bins_orientation = np.histogram(gradient_orientation, bins=360, range=[0, 2*np.pi])

# Display the results using cv2.imshow

# Stretched Gradient Magnitude
cv2.imshow('Stretched Gradient Magnitude', gradient_magnitude_stretched)

# Histogram of Gradient Magnitude
hist_image = np.zeros((100, 256), dtype=np.uint8)
hist_normalized = cv2.normalize(hist, None, 0, hist_image.shape[0], cv2.NORM_MINMAX)

# Draw histogram for gradient magnitude
for i in range(256):
    cv2.line(hist_image, (i, int(hist_image.shape[0])), (i, int(hist_image.shape[0] - hist_normalized[i][0])), 255)

cv2.imshow('Histogram of Gradient Magnitude', hist_image)

# Gradient Orientation
cv2.imshow('Gradient Orientation', gradient_orientation)

# Histogram of Gradient Orientation
hist_orientation_image = np.zeros((100, 360), dtype=np.uint8)
hist_orientation_normalized = cv2.normalize(hist_orientation, None, 0, hist_orientation_image.shape[0], cv2.NORM_MINMAX)

# Draw histogram for gradient orientation
for i in range(360):
    cv2.line(hist_orientation_image, (i, int(hist_orientation_image.shape[0])), (i, int(hist_orientation_image.shape[0] - hist_orientation_normalized[i][0])), 255)

cv2.imshow('Histogram of Gradient Orientation', hist_orientation_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('Q4_P1.jpg', gradient_magnitude_stretched.astype(np.uint8))
cv2.imwrite('Q4_P2.jpg', hist_image.astype(np.uint8))
cv2.imwrite('Q4_P3.jpg', gradient_orientation.astype(np.uint8))
cv2.imwrite('Q4_P4.jpg', hist_orientation_image.astype(np.uint8))


# Load images in grayscale
Q5_image1 = cv2.imread('C:/Users/OMEN/OneDrive/Desktop/1st_Sem_University/Comp. Vision/Assighment/Assign1/Q5/walk_1.jpg', cv2.IMREAD_GRAYSCALE)
Q5_image2 = cv2.imread('C:/Users/OMEN/OneDrive/Desktop/1st_Sem_University/Comp. Vision/Assighment/Assign1/Q5/walk_2.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the images have been loaded successfully
if Q5_image1 is None or Q5_image2 is None:
    print("Error loading images.")
else:
    result_image = cv2.subtract(Q5_image1, Q5_image2)


    # Display the grayscale images
    cv2.imshow('Walk 1 - Grayscale', Q5_image1)
    cv2.imshow('Walk 2 - Grayscale', Q5_image2)
    # Display the result
    cv2.imshow('Result Image', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cv2.imwrite('Q5_Gray1.jpg', Q5_image1.astype(np.uint8))
cv2.imwrite('Q5_Gray2.jpg', Q5_image2.astype(np.uint8))
cv2.imwrite('Q5_result.jpg', result_image.astype(np.uint8))
"""


# Load the image
Q6_image = cv2.imread('C:/Users/OMEN/OneDrive/Desktop/1st_Sem_University/Comp. Vision/Assighment/Assign1/Q6/Q_4.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image has been loaded successfully
if Q6_image is None:
    print("Error loading the image.")
else:
    # Apply Canny edge detector with different threshold values
    thresholds = [50, 100, 150]  # You can adjust these threshold values
    canny_results = []

    for threshold in thresholds:
        edges = cv2.Canny(Q6_image, threshold, threshold * 2)
        canny_results.append(edges)

        # Display the result for each threshold value
        cv2.imshow(f'Threshold={threshold}', edges)
        cv2.imwrite(f'Threshold={threshold}.jpg', edges.astype(np.uint8))

    # Display the original image
    cv2.imshow('Original Image', Q6_image)

    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
