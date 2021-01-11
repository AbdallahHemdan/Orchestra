import cv2
import numpy as np
import skimage.io as io
from skimage import color
from copy import deepcopy
from skimage import feature
from skimage import measure
import matplotlib.pyplot as plt
from collections import Counter

# Methods of Staff line removal and segmentation #
def get_staff_lines(width, height, in_img, threshold):
    # initial_lines: list of all initial lines that maybe extended #
    initial_lines = []

    # row_histogram: histogram of all row, contains number of black cell for each row #
    row_histogram = [0] * height

    # staff_lines: list of staff lines in our image #
    staff_lines = []
    staff_lines_thicknesses = []

    # Calculate histogram for rows #
    for r in range(height):
        for c in range(width):
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

    while it < len(initial_lines):
        # Save starting row of staff line #
        if cur_thickness == 1:
            staff_lines.append(initial_lines[it])

        if it == int(len(initial_lines) - 1):
            staff_lines_thicknesses.append(cur_thickness)

        # Try to extend line thickness #
        # If Failed: 1.save current thickness, 2.rest thickness #
        elif initial_lines[it] + 1 == initial_lines[it + 1]:
            cur_thickness += 1
        else:
            staff_lines_thicknesses.append(cur_thickness)
            cur_thickness = 1

        it += 1

    # Return the staff lines thicknesses and staff lines
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
                if (col > 0 and in_img.item(line_end + 1, col - 1) != 0) and (col < width - 1 and in_img.item(line_end + 1, col + 1) != 0):
                    thick = line_thickness + 1
                    if thick < 1:
                        thick = 1
                    for j in range(int(thick)):
                        in_img.itemset((line_start + j, col), 255)

            # If current staff can be extended down, then extend #
            elif in_img.item(line_start - 1, col) == 0 and in_img.item(line_end + 1, col) == 255:
                if (col > 0 and in_img.item(line_start - 1, col - 1) != 0) and (col < width - 1 and in_img.item(line_start - 1, col + 1) != 0):
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

def cut_image_into_buckets(in_img, staff_lines):
    # List of cutted buckets images and positions of cutting #
    cutted_images = []
    cutting_position = []

    it = 0
    lst_slice = 0
    no_of_buckets = len(staff_lines) // 5
    while it < no_of_buckets - 1:
        _str = staff_lines[it * 5 + 4]
        _end = staff_lines[it * 5 + 5]

        mid_row = (_end + _str) // 2
        cutting_position.append(lst_slice)
        cutted_images.append(in_img[lst_slice: mid_row, :])

        it += 1
        lst_slice = mid_row

    cutting_position.append(lst_slice)
    cutted_images.append(in_img[lst_slice: in_img.shape[0], :])
    return cutting_position, cutted_images

def get_ref_lines(cut_positions, staff_lines):
    ref_lines = []
    no_of_buckets = len(staff_lines) // 5
    lines_spacing = []

    for it in range(no_of_buckets):
        line_spacing = (staff_lines[it * 5 + 4] - staff_lines[it * 5]) / 4
        lines_spacing.append(line_spacing)

        ref_line = staff_lines[it * 5 + 4] + line_spacing * 1.5
        ref_lines.append(ref_line)

    return ref_lines, lines_spacing

def segmentation(height_before, in_img):
    n, m = in_img.shape

    blurred = cv2.GaussianBlur(in_img, (3, 3), 0)
    thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY_INV)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours in the image
    cnts = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]


    threshold_min_area = 0
    threshold_max_area = n * m

    symbols = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area > threshold_min_area and area < threshold_max_area:
            symbols.append([x, y + height_before, x + w, y + h + height_before])

    return symbols
