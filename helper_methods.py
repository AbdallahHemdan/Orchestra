import numpy as np
import cv2


def clean_and_cut(img):
    img[img > 200] = 255
    img[img <= 200] = 0

    img = 255 - img

    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones((img.shape[0], img.shape[1]), dtype="uint8") * 255
    contours = sorted(contours, key=cv2.contourArea)

    for i in range(len(contours) - 1):
        cv2.drawContours(mask, [contours[i]], -1, 0, -1)

    img = cv2.bitwise_and(img, img, mask=mask)

    white = np.argwhere(img == 255)

    x, y, w, h = cv2.boundingRect(white)
    img = img[x:x+w, y:y+h]

    img = 255 - img
    return img


def clean_rabbish(img):

    return img


def cut_shape_only(img):

    return img
