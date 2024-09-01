import cv2 as cv
import numpy as np
from PIL import Image
from skimage import measure

def otsuThresholding(img):
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])

    total = img.size
    sum_val = np.sum(np.arange(256) * hist)

    sumB = 0
    wB = 0
    wF = 0
    varMax = 0
    threshold = 0

    for t in range(256):
        wB += hist[t]               # Weight Background
        if wB == 0:
            continue

        wF = total - wB             # Weight Foreground
        if wF == 0:
            break

        sumB += t * hist[t]

        mB = sumB / wB              # Mean Background
        mF = (sum_val - sumB) / wF  # Mean Foreground

        # Calculate Between class variance
        varBetween = wB * wF * (mB - mF) * (mB - mF)

        # Check if new max is found
        if varBetween > varMax:
            varMax = varBetween
            threshold = t

    return threshold

def binaryMorph(img, threshold):

    binary_img = (img > threshold).astype(np.uint8) * 255

    # Custom Erosion
    img_eroded = customErosion(binary_img)

    # Custom Dilation
    img_dilated = customDilation(binary_img)

    return img_eroded, img_dilated

def customDilation(binary_img):
    rows, cols = binary_img.shape # Getting number of rows and cols in img
    dilated_img = np.zeros_like(binary_img) # array initialised with zeros

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            neighborhood = binary_img[i - 1:i + 2, j - 1:j + 2] #checking the neighbours of each pixel
            if np.all(neighborhood == 255):                      
                dilated_img[i, j] = 255

    return dilated_img

def customErosion(binary_img):
    rows, cols = binary_img.shape # Getting number of rows and cols in img
    eroded_img = np.zeros_like(binary_img) # array initialised with zeros

    for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if binary_img[i + 1, j - 1] == 255: #checking the neighbours of each pixel
                    eroded_img[i - 1:i + 2, j - 1:j + 2] = 255 #setting neighbor pixel to white if pixel if white
    

    return eroded_img

def connectedComp(img):
    labeled_img = np.copy(img)
    labeled_components, num_components = measure.label(img, connectivity=2, return_num=True) # this line assigns a unique label, and you total number of connected comps

    for label in range(1, num_components + 1):
        component_mask = (labeled_components == label).astype(np.uint8) # creating mask for current comp
        contours = measure.find_contours(component_mask, level=0.5, fully_connected='low') # Finds contours at that level

        if contours:
            contour = contours[0]
            #calculating area and perim of each component
            area = np.sum(component_mask)
            perimeter = np.sum(np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1)))

            if perimeter == 0:
                continue  

            circularity = 4 * np.pi * area / (perimeter ** 2) # Circularity calculation 

            print("Circularity:", circularity)

            if 0.896 < circularity < 1.11: # Read on stackoverflow that a perfect circle threshold is 1.11 and in our example most were 0.896 so set range
                print("PASS")
            else:
                print("FAIL")

    return num_components, labeled_img

total_processing_time = 0

# Read and process each image
for i in range(1, 16):
    img_path = 'C:\\Users\\vadim\\OneDrive - Technological University Dublin\\Year 4\\Year 4 Sem 2\\Computer Vision\\Assignment\\CaOring - Vadims Prociks - B00132224\\img\\Oring' + str(i) + '.jpg'

    # Load the image using OpenCV
    img = cv.imread(img_path, 0)

    start_time = cv.getTickCount()

    otsu_thresh = otsuThresholding(img)

    _, thresholded_img = cv.threshold(img, otsu_thresh, 255, cv.THRESH_BINARY)

    # Running dilation and erosion 4 times and passing the images through to each other
    for _ in range(4):
        img_dilated = customDilation(thresholded_img)

    for _ in range(4):
        img_eroded = customErosion(img_dilated)
    num_components, labeled_img = connectedComp(img_eroded)

    end_time = cv.getTickCount()
    elapsed_time = (end_time - start_time) / cv.getTickFrequency()
    total_processing_time += elapsed_time

    # Displaying the results from processed images
    # cv.imshow('Original Image', img)
    cv.imshow('Otsu Thresholded Image', thresholded_img)
    cv.imshow('Dilated Image', img_dilated)
    cv.imshow('Eroded Image', img_eroded)

    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_GRAY2BGR)
    cv.putText(labeled_img, str(total_processing_time), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)

    cv.imshow('Labeled Image', labeled_img)

    cv.waitKey(0)
    cv.destroyAllWindows()

# Displaying total processing time
#print("Total Processing Time: {:.2f} seconds".format(total_processing_time))                          