import cv2
import random
import numpy as np
from sklearn import svm

# List of maps and needed variables #

########## Variables ##########
random_seed = 42
random.seed(random_seed)
target_img_size = (32, 32)
np.random.seed(random_seed)
classifiers = {
    'SVM': svm.LinearSVC(random_state=random_seed)
}

direct_labels = ['x', 'b', 'clef', 'dot', 'hash', 'd', 't_2', 't_4', 'symbol_bb', 'barline']
direct_texts = {'x':'##', 'b':'&', 'hash':'#', 'd':'', 'symbol_bb':'&&', 'dot':'.', 'clef':'', 't_2':'2', 't_4':'4', 'barline':''}

chords_order = 'cdefgabcdefgab'
direct_a = {'a_1':'/1', 'a_2':'/2', 'a_4':'/4', 'a_8':'/8', 'a_16':'/16','a_32':'/32', 'a_2_flipped':'/2', 'a_4_flipped':'/4', 'a_8_flipped':'/8', 'a_16_flipped':'/16', 'a_32_flipped':'/32'}

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

def get_a_character(distance, line_spacing, flipped = 0):
    if flipped and distance < 4.25 * line_spacing:
        return 'b1'
    if distance < flipped * (4.5 * line_spacing) + .25 * line_spacing:
        return 'c' + (1 - flipped) * '1' + (flipped) * '2'
    if distance < flipped * (4.5 * line_spacing) + .75 * line_spacing:
        return 'd' + (1 - flipped) * '1' + (flipped) * '2'
    if distance < flipped * (4.5 * line_spacing) + 1.25 * line_spacing:
        return 'e' + (1 - flipped) * '1' + (flipped) * '2'
    if distance < flipped * (4.5 * line_spacing) + 1.75 * line_spacing:
        return 'f' + (1 - flipped) * '1' + (flipped) * '2'
    if distance < flipped * (4.5 * line_spacing) + 2.25 * line_spacing:
        return 'g' + (1 - flipped) * '1' + (flipped) * '2'
    if distance < flipped * (4.5 * line_spacing) + 2.75 * line_spacing:
        return 'a' + (1 - flipped) * '1' + (flipped) * '2'
    if distance < flipped * (4.5 * line_spacing) + 3.25 * line_spacing:
        return 'b2'
    return 'b1'

def get_nxt(chord_type, char):
    pos = chords_order.find(char)
    if chord_type == 'chord_2':
        return '{' + f'{char}1/4,{chords_order[pos + 2]}1/4' + '}'

    if chord_type == 'chord_3':
        return '{' + f'{char}1/4,{chords_order[pos + 2]}1/4,{chords_order[pos + 4]}1/4' + '}'
    
    if chord_type == 'chord_special':
        return '{' + f'{char}1/4,{chords_order[pos + 1]}1/4,{chords_order[pos + 3]}1/4' + '}'
    
    if chord_type == 'chord_3_2':
        return '{' + f'{char}1/4,{chords_order[pos + 3]}1/4,{chords_order[pos + 5]}1/4' + '}'
    
def get_a_chord(chord_type, distance, line_spacing, flipped = 0):
    if distance < .25 * line_spacing:
        return get_nxt(chord_type, 'c')
    if distance < .75 * line_spacing:
        return get_nxt(chord_type, 'd')
    if distance < 1.25 * line_spacing:
        return get_nxt(chord_type, 'e')
    if distance < 1.75 * line_spacing:
        return get_nxt(chord_type, 'f')
    if distance < 2.25 * line_spacing:
        return get_nxt(chord_type, 'g')
    if distance < 2.75 * line_spacing:
        return get_nxt(chord_type, 'a')
    if distance < 3.25 * line_spacing:
        return get_nxt(chord_type, 'b')
    return get_nxt(chord_type, 'b')

def text_operation(label, ref_line, line_spacing, y1, y2): 
    if label in direct_labels:
        return direct_texts[label]
    
    if label.startswith('chord_'):
        distance = ref_line - y2
        return get_a_chord(label, distance, line_spacing)
            
    if not(label.endswith('flipped')): 
        distance = 0
        if label.startswith('a_'):
            distance = ref_line - y2
            character = get_a_character(distance, line_spacing)

            return f'{character}{direct_a[label]}'
    else: # flipped
        distance = 0
        if label.startswith('a_'):
            distance = ref_line - y1
            character = get_a_character(distance, line_spacing, 1)

            return character + direct_a[label]

def cut_boundaries(cur_symbol, no_of_cuts, y2):
    last_x = 0
    cutted_boundaries = []
    step = cur_symbol.shape[1] // no_of_cuts

    for i in range(no_of_cuts):
        cur = cur_symbol[:, last_x:last_x + step]
        last_x += step

        white = np.argwhere(cur == 255)
        white = sorted(white, key=lambda x: x[0])

        y_min, y_max = white[0][0], white[-1][0]
        ret_boundary = [0, y_min, cur.shape[1] - 1, y_max]
        
        diff_y1, diff_y2 = cur.shape[0] - y_min, cur.shape[0] - y_max
        ret_boundary[1] = y2 - diff_y1
        ret_boundary[3] = y2 - diff_y2
        
        cutted_boundaries.append(ret_boundary)

    return cutted_boundaries

def preprocess_img(img_path):
    # 1. Read desired image #
    img = cv2.imread(img_path, 0)
    
    # 2. Remove noise (odd pixels) from the image and save it #
    img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

    # 3. Binarize image using combination of (global + otsu) thresholding and save it #
    threshold, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # 4. Return image shape (width, height) and processed image # 
    n, m = img.shape
    return n, m, img

def extract_hog_features(img):
    img = cv2.resize(img, target_img_size)
    win_size = (32, 32)
    cell_size = (4, 4)
    block_size_in_cells = (2, 2)

    block_size = (block_size_in_cells[1] * cell_size[1],
                  block_size_in_cells[0] * cell_size[0])
    block_stride = (cell_size[1], cell_size[0])
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size,
                            block_stride, cell_size, nbins)
    h = hog.compute(img)
    h = h.flatten()
    return h.flatten()
