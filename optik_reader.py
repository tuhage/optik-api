from jdeskew.estimator import get_angle
from jdeskew.utility import rotate
import numpy as np
import cv2

def indexToLetter(index):
    return chr(index+64)

def get_result_from_question(cropped,row,column):
    h, w = cropped.shape[:2]

    height_step = h // row
    width_step = w // column

    result = []

    for i in range(column):
        max_black = 0
        max_black_index = 0
        max_black_ratio = 0
        for j in range(2,row):
            x = i*width_step
            y = j*height_step
            roi = cropped[y:y+height_step, x:x+width_step]
            
            black_pixels = np.sum((roi == 0))
            black_ratio = black_pixels / (height_step * width_step)
                        
            if black_pixels > max_black:
                max_black = black_pixels
                max_black_index = j
                max_black_ratio = black_ratio
                
        if max_black_ratio > 0.25:
            result.append(max_black_index-2)
        else:
            result.append(' ')
            
    return ''.join(map(str, result))

def find_kitapcik(cropped,row,column):
    h, w = cropped.shape[:2]
    height_step = h // row
    width_step = w // column

    result = []
    for i in range(column):
        max_black = 0
        for j in range(2,3):
            x = i*width_step
            y = j*height_step
            roi = cropped[y:y+height_step, x:x+width_step]
            black_pixels = np.sum((roi == 0))
                   
            if black_pixels > max_black:
                max_black = black_pixels
                
            result.append(max_black)
    return indexToLetter(result.index(max(result))+1)

def rects_to_lines(rects,hg):
    rects.sort(key=lambda x: x[1])
    rects_lines = []
    for i in range(len(rects)):
        if i == 0:
            rects_lines.append([rects[i]])
        else:
            if rects[i][1] - rects[i-1][1] < hg*0.01:
                rects_lines[-1].append(rects[i])
            else:
                rects_lines.append([rects[i]])
                
    for i in range(len(rects_lines)):
        rects_lines[i].sort(key=lambda x: x[0])
        
    i = 0
    while i < len(rects_lines) - 1:
        mean_y = sum([y for x, y, w, h in rects_lines[i]]) // len(rects_lines[i])
        max_height = max([h for x, y, w, h in rects_lines[i]])
        mean_y_next = sum([y for x, y, w, h in rects_lines[i + 1]]) // len(rects_lines[i + 1])
        max_height_next = max([h for x, y, w, h in rects_lines[i + 1]])
        
        if mean_y_next - mean_y < hg * 0.1:
            if max_height_next > max_height:
                rects_lines.pop(i)
            else:
                rects_lines.pop(i + 1)
            i = 0
        else:
            i += 1
    return rects_lines

def get_result_from_answers(cropped, row, column):
    h, w = cropped.shape[:2]

    height_step = h // row
    width_step = w // column

    result = []
    for i in range(row):
        max_black = 0
        max_black_index = 0
        max_black_ratio = 0
        for j in range(1,column):
            x = j * width_step
            y = i * height_step
            roi = cropped[y:y+height_step, x:x+width_step]
            
            black_pixels = np.sum((roi == 0))
            black_ratio = black_pixels / (height_step * width_step)

            if black_pixels > max_black:
                max_black = black_pixels
                max_black_index = j
                max_black_ratio = black_ratio
        if max_black_ratio > 0.25:
            result.append(indexToLetter(max_black_index))
        else:
            result.append(' ')
    return ''.join(map(str, result))


def get_question_points(rects_lines,thresh_image):
    question_points = []
    for i in range(len(rects_lines[0])):
        t = rects_lines[0][i]
        cropped = thresh_image[t[1]:t[1]+t[3], t[0]:t[0]+t[2]]
        cropped_dilated = cv2.morphologyEx(cropped, cv2.MORPH_DILATE, np.ones((5,5), np.uint8))
        c=get_result_from_question(cropped_dilated,12,3 if i == 0 else 2)
        question_points.append(c)
    return question_points

def get_student_number(rects_lines,thresh_image):
    t=rects_lines[1][0]
    cropped = thresh_image[t[1]:t[1]+t[3], t[0]:t[0]+t[2]]
    cropped_dilated = cv2.morphologyEx(cropped, cv2.MORPH_DILATE, np.ones((5,5), np.uint8))
    return get_result_from_question(cropped_dilated,18,9)

def get_book_type(rects_lines,thresh_image):
    t=rects_lines[1][1]
    cropped = thresh_image[t[1]:t[1]+t[3], t[0]:t[0]+t[2]]
    cropped_dilated = cv2.morphologyEx(cropped, cv2.MORPH_DILATE, np.ones((5,5), np.uint8))
    return find_kitapcik(cropped_dilated,6,1)

def get_student_answers(rects_lines,thresh_image):
    t=rects_lines[2][0]
    cropped = thresh_image[t[1]:t[1]+t[3], t[0]:t[0]+t[2]]
    cropped_dilated = cv2.morphologyEx(cropped, cv2.MORPH_DILATE, np.ones((5,5), np.uint8))
    angle = get_angle(cropped_dilated)
    cropped_dilated = rotate(cropped_dilated, angle)
    h, w = cropped_dilated.shape[:2]
    crop_height = int(h * 0.05)
    cropped_dilated = cropped_dilated[crop_height:h, 0:w]

    h, w = cropped_dilated.shape[:2]
    crop_width = w // 2
    left = cropped_dilated[:, 0:crop_width]
    right = cropped_dilated[:, crop_width:w]
    return get_result_from_answers(left, 30, 6) + get_result_from_answers(right, 30, 6)

def image_to_rects_lines():
    img = cv2.imread("images/temp.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    angle = get_angle(gray)
    gray = rotate(gray, angle)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    th3 = cv2.morphologyEx(th3, cv2.MORPH_ERODE, np.ones((3,3), np.uint8))
    th3 = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    
    contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_copy = img.copy()
    
    wg, hg = img.shape[:2]
    rects = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        roi = th3[y:y+h, x:x+w]        
        if 1.5*w>h:
            continue
        if roi.size < wg*hg*0.0025:
            continue
        if h>hg*0.7:
            continue
        
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0,65,255), 2)
        rects.append((x, y, w, h))
        
    rects_lines = rects_to_lines(rects,hg)
    return rects_lines,th3


def read_optik_form():
    rects_lines,thresh_image = image_to_rects_lines()
    question_points = get_question_points(rects_lines,thresh_image)
    student_number = get_student_number(rects_lines,thresh_image)
    book_type = get_book_type(rects_lines,thresh_image)
    student_answers = get_student_answers(rects_lines,thresh_image)
    return student_number + book_type + ''.join(question_points) + student_answers 

