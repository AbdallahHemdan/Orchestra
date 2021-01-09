import cv2
import numpy as np
import skimage.io as io
from skimage import color
from copy import deepcopy
from skimage import feature
from skimage import measure
import matplotlib.pyplot as plt
from collections import Counter
from skimage.filters import threshold_otsu

def get_staff_lines(width, height, in_img, threshold):
    # initial_lines: list of all initial lines that maybe extended #
    initial_lines = []
    
    # row_histogram: histogram of all row, contains number of black cell for each row #
    row_histogram = [0] * width
    
    # staff_lines: list of staff lines in our image #
    staff_lines = []
    staff_lines_thicknesses = []
    
    # Calculate histogram for all row #
    for r in range(width):
        for c in range(height):
            if in_img[r][c] == 0:
                row_histogram[r] += 1
    
    # Get only rows which have black pixels larger that threshold #
    for row in range(len(row_histogram)):
        if row_histogram[row] >= (width * threshold):
            initial_lines.append(row)

            
    # it: iterator over all doubtful lines #
    it = 0
    
    # cur_thinkneed: current thickness of line which may extended #  
    cur_thickness = 1
    
    while it < len(initial_lines) - 1:
        # Save starting row of staff line #
        if cur_thickness == 1:
            staff_lines.append(initial_lines[it])

        # Try to extend line thicknees #
        # If Failed: 1.save current thickness, 2.rest thickness #
        if initial_lines[it] + 1 == initial_lines[it + 1]:
            cur_thickness += 1
        else:
            staff_lines_thicknesses.append(cur_thickness)
            cur_thickness = 1

        it += 1
        
    staff_lines_thicknesses.append(cur_thickness + 1)
    
    # If all staff lines thickness are equal, then return one of them #
    if all(thickness == staff_lines_thicknesses[0] for thickness in staff_lines_thicknesses):
        return staff_lines_thicknesses[0], staff_lines
    return staff_lines_thicknesses, staff_lines

def remove_single_line(line_thickness, line_start, in_img, width):
    # line_end: end pixel of the current staff line #
    line_end = line_start + line_thickness - 1
    
    
    for col in range(width):
        if in_img.item(line_start, col) == 0 or in_img.item(line_end, col) == 0:
            # If current staff is clear (up-down), then remove it directly #
            if in_img.item(line_start - 1, col) == 255 and in_img.item(line_end + 1, col) == 255:
                for j in range(line_thickness):
                    in_img.itemset((line_start + j, col), 255)
                    
            # If current staff can be extended up, then extend #
            elif in_img.item(line_start - 1, col) == 255 and in_img.item(line_end + 1, col) == 0:
                thick = line_thickness + 1
                if thick < 1:
                    thick = 1
                for j in range(int(thick)):
                    in_img.itemset((line_start + j, col), 255)

            # If current staff can be extended down, then extend #                            
            elif in_img.item(line_start - 1, col) == 0 and in_img.item(line_end + 1, col) == 255:
                thick = line_thickness + 1
                if thick < 1:
                    thick = 1
                for j in range(int(thick)):
                    in_img.itemset((line_end - j, col), 255)
    return in_img

def remove_staff_lines(in_img, width, staff_lines, staff_lines_thicknesses):
    it = 0
    
    # Iterate over all staff lines and remove them line by line#
    while it < len(staff_lines):
        line_start = staff_lines[it]
        line_thickness = staff_lines_thicknesses[it]
        in_img = remove_single_line(line_thickness, line_start, in_img, width)
        
        it += 1
    return in_img

def segmentation(gray):
    original_image = gray.copy()
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilate = cv2.dilate(thresh, kernel , iterations=4)

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

def fix_rotation(img):
    skew_img = cv2.bitwise_not(img)  # Invert image

    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(skew_img > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    # rotate the image to deskew it
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return angle, rotated
