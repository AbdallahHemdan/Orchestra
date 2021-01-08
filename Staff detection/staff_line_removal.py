import cv2
import numpy as np
import skimage.io as io
from skimage.filters import threshold_otsu
from skimage import color
from skimage import feature
import matplotlib.pyplot as plt
from skimage import measure


def staff_removal(in_img, thresh):
    filterOne = np.array([[0, 0, 0],[1, 1, 1],[0, 0, 0]], np.uint8)
    filterTwo = np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]], np.uint8)
    erosion = cv2.erode(in_img, filterOne, iterations=1)
    outputImg = cv2.dilate(erosion, filterTwo, iterations=1)

    return outputImg;

def thresh_callback(thresh):
    edges = cv2.Canny(gray,thresh,thresh*2)
    drawing = np.zeros(img.shape,np.uint8)     # Image to draw the contours
    contours,hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    

    one_contour = np.zeros(img.shape,np.uint8)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
    
        thickness = 2
        color = (255, 0, 0) 
        one_contour = cv2.rectangle(one_contour, (x, y), (x + w, y + h), color, thickness) 
    io.imshow(gray)
    io.show()
    io.imshow(one_contour)
    io.show()
    
    
def segmentation(gray):
    original_image = gray.copy()
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilate = cv2.dilate(thresh, kernel , iterations=4)

    cv2.imshow("thresh", thresh)
    cv2.imshow("dilate", dilate)

    # Find contours in the image
    cnts = cv2.findContours(dilate.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    contours = []

    threshold_min_area = 0
    threshold_max_area = 3000

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area > threshold_min_area and area < threshold_max_area:
            original_image = cv2.rectangle(original_image, (x,y), (x+w, y+h), (0,255,0),1)
            contours.append(c)

    print('contours detected: {}'.format(len(contours)))

    cv2.imshow("original", original_image)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()