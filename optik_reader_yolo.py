from ultralytics import YOLO
import fitz 
from datetime import datetime
import io
import os
from jdeskew.estimator import get_angle
from jdeskew.utility import rotate
import matplotlib.pyplot as plt
import numpy as np
import cv2


model_first_stage = YOLO("best3.pt")  # load an official model
model_second_stage = YOLO("bestc.pt")  # load an official model



def rects_to_lines_new(rects,hg):
    rects.sort(key=lambda x: x[0])
    q, s, a,k = [], [], [], []
    for i in range(len(rects)):
        if rects[i][4] == 0:
            q.append(rects[i])
        elif rects[i][4] == 1:
            s.append(rects[i])
        elif rects[i][4] == 2:
            a.append(rects[i])
        else:
            k.append(rects[i])
            
    
    return [q[0],s[0],a[0],k]

def return_result_stage_2(roi):
    cv2.imwrite(f"temp.png", roi)
    result_roi=model_second_stage.predict("temp.png",save=True,show_labels=False)
    return result_roi

def get_all_stage_2_results(img,rects_lines):
    roi1 = img[rects_lines[0][1]:rects_lines[0][1]+rects_lines[0][3], rects_lines[0][0]:rects_lines[0][0]+rects_lines[0][2]]
    roi2 = img[rects_lines[1][1]:rects_lines[1][1]+rects_lines[1][3], rects_lines[1][0]:rects_lines[1][0]+rects_lines[1][2]]
    roi3 = img[rects_lines[2][1]:rects_lines[2][1]+rects_lines[2][3], rects_lines[2][0]:rects_lines[2][0]+rects_lines[2][2]]
    roi4s = []
    for i in range(len(rects_lines[3])):
        roi4s.append(img[rects_lines[3][i][1]:rects_lines[3][i][1]+rects_lines[3][i][3], rects_lines[3][i][0]:rects_lines[3][i][0]+rects_lines[3][i][2]])
        
    result_roi1 = return_result_stage_2(roi1)
    result_roi2 = return_result_stage_2(roi2)
    result_roi3 = return_result_stage_2(roi3)
    result_roi4s = []
    for roi4 in roi4s:
        result_roi4s.append(return_result_stage_2(roi4))    
    return result_roi1,result_roi2,result_roi3,result_roi4s

def organize_matrix(data, threshold=10,mode=0):
    sorted_data = sorted(data, key=lambda x: x[1])
    new_sorted_data = []    
    
    tc = []
    t = [sorted_data[0]]
    for i in range(1, len(sorted_data)):
        if sorted_data[i][1] - sorted_data[i-1][1] <= threshold:
            t.append(sorted_data[i])
        else:
            if len(t) > 1:
                new_sorted_data.extend(t)
            t=[sorted_data[i]]
    
    if mode == 0:
        new_sorted_data=sorted(new_sorted_data, key=lambda x: x[0])
    else:
        new_sorted_data=sorted(data, key=lambda x: x[0])
    
    columns = []
    current_column = [new_sorted_data[0]]
    
    for i in range(1, len(new_sorted_data)):
        if new_sorted_data[i][0] - new_sorted_data[i-1][0] <= threshold:
            current_column.append(new_sorted_data[i])
        else:
            current_sorted= sorted(current_column, key=lambda x: x[1])
            filtered_sorted = filter_close_elements(current_sorted)
            columns.append(sorted(filtered_sorted, key=lambda x: x[1]))
            current_column = [new_sorted_data[i]]
    
    columns.append(sorted(current_column, key=lambda x: x[1]))
    
    
    max_rows = max(len(col) for col in columns)
    matrix = [[(0,0,0,0,0) for _ in range(len(columns))] for _ in range(max_rows)]
    
    for j, col in enumerate(columns):
        for i, value in enumerate(col):
            matrix[i][j] = value
            
    return matrix


def get_matrix_from_results(results,mode=0):
    for result in results:
        rects = []
        for bbox in result.boxes:
            x1,y1,x2,y2 = map(int, bbox.xyxy[0])
            rects.append((x1,y1,x2-x1,y2-y1,int(bbox.cls[0]),float(bbox.conf[0])))
    return organize_matrix(rects,mode=mode)

def find_first_one_indices(matrix):
    column_count = len(matrix[0])
    first_one_indices = []
    
    for col in range(column_count):
        index_with_one = ' '
        for row in range(len(matrix)):
            if matrix[row][col][4] == 1:
                index_with_one = row
                break
        first_one_indices.append(index_with_one)
    
    return first_one_indices

def find_first_one_in_rows(matrix):
    indices_or_space = []
    
    for row in matrix:
        found_one = False
        for col in range(len(row)):
            if row[col][4] == 1:
                indices_or_space.append(col)  
                found_one = True
                break
        if not found_one:
            indices_or_space.append(' ') 
    
    return indices_or_space


def split_matrix(matrix):
    row_count = len(matrix)
    column_count = len(matrix[0])
    split_index = column_count // 2
    left_matrix = [row[:split_index] for row in matrix]
    right_matrix = [row[split_index:] for row in matrix]
    
    return left_matrix, right_matrix

def filter_close_elements(elements):
    filtered_elements = []
    
    prev = None 
    for current in elements:
        if prev is not None:
            x_diff = abs(current[0] - prev[0])
            y_diff = abs(current[1] - prev[1])
            
            if x_diff < 5 and y_diff < 5:
                if current[-1] > prev[-1]:
                    prev = current
            else:
                filtered_elements.append(prev)
                prev = current
        else:
            prev = current
    if prev is not None:
        filtered_elements.append(prev)  
    return filtered_elements


def read_optik_form():
    
    try:
        results = model_first_stage.predict("images/temp.png",conf=0.4,iou=0.2)  # predict on an image
        if results[0]!= None:
            result = results[0]
            img = cv2.imread(result.path)
            rects = []
            for bbox in result.boxes:
                x1,y1,x2,y2 = map(int, bbox.xyxy[0])
                rects.append((x1,y1,x2-x1,y2-y1,int(bbox.cls[0])))

            rects_lines = rects_to_lines_new(rects, img.shape[0])


        answers,student_number,book_type,question_points = get_all_stage_2_results(img,rects_lines)
        
        student_number_matrix = get_matrix_from_results(student_number)
        student_number_str = ''.join([str(x) for x in find_first_one_indices(student_number_matrix)])
        
        answers_matrix = get_matrix_from_results(answers)
        row_count = len(answers_matrix)
        column_count = len(answers_matrix[0])
        left_matrix, right_matrix = split_matrix(answers_matrix)
        left,right=find_first_one_in_rows(left_matrix),find_first_one_in_rows(right_matrix)
        all_answers = [x if x == ' ' else chr(x+65) for x in left+right]
        answers_str = ''.join(all_answers[:60])
        
        book_type_matrix = get_matrix_from_results(book_type)
        book_type_array=find_first_one_indices(book_type_matrix)

        if book_type_array[0]==' ':
            book_type_str=' '
        else:
            book_type_str=chr(book_type_array[0]+65)
            
        question_matrixs = []
        question_points_result=[]
        for question_point in question_points:
            question_matrixs.append(get_matrix_from_results(question_point))
            question_points_result.append(''.join([str(x) for x in find_first_one_indices(question_matrixs[-1])]))
        questions_str = ''.join(question_points_result)
        
        
        return student_number_str + book_type_str + questions_str + answers_str 
    except Exception as e:
        print("Error : ",e)
        return "Okuma hatası lütfen tekrar deneyiniz."

