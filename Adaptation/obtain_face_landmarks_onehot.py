import cv2
import numpy as np
import os
import sys
#photo_list = open('crop_faces_align/photo.list')
photo_list = open('crop_faces_align/caricature.list')
dir_path = 'landmarks_onehot/caricature/'
#dir_path = 'small_landmarks_onehot/photo/'
for photo_name in photo_list.readlines():
    photo_name = photo_name.strip()
    #print(photo_name)
    landmark_path = 'face_landmarks_points/caricature/' + photo_name[:-3] + 'txt'
    #landmark_path = 'face_landmarks_points/photo/' + photo_name[:-3] + 'txt'
    save_path = dir_path + photo_name[:-3] + 'npy'
    landmark_points = open(landmark_path)
    #black_image = np.zeros((256, 256, 3), np.uint8)
    black_image = np.zeros((64, 64, 3), np.uint8)
    points = []
    for idx, point in enumerate(landmark_points.readlines()):
        point = point.strip()
        point = point.split(' ')
        tmp_point = (int(float(point[0])/4), int(float(point[1])/4))
        points.append(tmp_point)
    thickness = 1
    cv2.line(black_image,(points[4][0], points[0][1]), (points[7][0], points[0][1]), (1, 0, 0), 2*thickness)
    cv2.line(black_image, (points[1][0], min(points[5][1], points[6][1])), (points[1][0], points[12][1]), (2,0,0), 2*thickness)
    cv2.line(black_image, (points[14][0], points[2][1]), (points[16][0], points[2][1]), (3,0,0), 2*thickness)
    cv2.line(black_image, (points[3][0], min(points[5][1], points[6][1])), (points[3][0], points[12][1]), (4,0,0), 2*thickness)
    mouth_x_left = min(points[13][0], points[15][0], points[14][0], points[16][0])
    mouth_x_right = max(points[13][0], points[15][0], points[14][0], points[16][0])
    mouth_y_top = min(points[13][1], points[14][1], points[16][1], points[15][1])
    mouth_y_bottom = max(points[13][1], points[15][1], points[16][1], points[14][1])
    up_mouth = np.array( [[[mouth_x_left,mouth_y_top],[mouth_x_left,mouth_y_bottom],[mouth_x_right,mouth_y_bottom], [mouth_x_right, mouth_y_top]]], dtype=np.int32)
    cv2.fillPoly(black_image, up_mouth, (5,0,0))
    eye_x_left = min(points[4][0], points[5][0])
    eye_x_right = max(points[4][0], points[5][0])
    eye_y_top = min(points[4][1], points[5][1])
    eye_y_bottom = max(points[4][1], points[5][1])
    eye_y_center = int((eye_y_bottom + eye_y_top)/2)
    left_eye = np.array([[[eye_x_left, eye_y_top], [eye_x_left, eye_y_bottom], [eye_x_right, eye_y_bottom], [eye_x_right, eye_y_top]]], dtype=np.int32)
    cv2.line(black_image, (eye_x_left, eye_y_center), (eye_x_right, eye_y_center), (6,0,0), 2 * thickness)
    eye_x_left = min(points[6][0], points[7][0])
    eye_x_right = max(points[6][0], points[7][0])
    eye_y_top = min(points[6][1], points[7][1])
    eye_y_bottom = max(points[6][1], points[7][1])
    eye_y_center = int((eye_y_bottom + eye_y_top)/2)
    right_eye = np.array([[[eye_x_left, eye_y_top], [eye_x_left, eye_y_bottom], [eye_x_right, eye_y_bottom], [eye_x_right, eye_y_top]]], dtype=np.int32)
    cv2.line(black_image, (eye_x_left, eye_y_center), (eye_x_right, eye_y_center), (7,0,0), 2 * thickness)
    eye_x_left = min(points[8][0], points[9][0])
    eye_x_right = max(points[8][0], points[9][0])
    eye_y_top = min(points[8][1], points[9][1])
    eye_y_bottom = max(points[8][1], points[9][1])
    eye_y_center = int((eye_y_bottom + eye_y_top)/2)
    left_eye = np.array([[[eye_x_left, eye_y_top], [eye_x_left, eye_y_bottom], [eye_x_right, eye_y_bottom], [eye_x_right, eye_y_top]]], dtype=np.int32)
    cv2.line(black_image, (eye_x_left, eye_y_center), (eye_x_right, eye_y_center), (8,0,0), 2 * thickness)
    eye_x_left = min(points[10][0], points[11][0])
    eye_x_right = max(points[10][0], points[11][0])
    eye_y_top = min(points[10][1], points[11][1])
    eye_y_bottom = max(points[10][1], points[11][1])
    eye_y_center = int((eye_y_bottom + eye_y_top)/2)
    right_eye = np.array([[[eye_x_left, eye_y_top], [eye_x_left, eye_y_bottom], [eye_x_right, eye_y_bottom], [eye_x_right, eye_y_top]]], dtype=np.int32)
    cv2.line(black_image, (eye_x_left, eye_y_center), (eye_x_right, eye_y_center), (9,0,0), 2 * thickness)
    tmp_height = points[12][1] - int((points[9][1] + points[10][1])/2)
    tmp_width =  int((points[16][0] - points[14][0])/4)
    nose_x_left = points[12][0] - tmp_width
    nose_x_right = points[12][0] + tmp_width
    nose_y_top = points[12][1] - tmp_height
    nose_y_bottom = points[12][1]
    nose = np.array([[[nose_x_left, nose_y_top], [nose_x_left, nose_y_bottom], [nose_x_right, nose_y_bottom], [nose_x_right, nose_y_top]]], dtype=np.int32)
    cv2.fillPoly(black_image, nose, (10,0,0))
    landmark_onehot = np.zeros((11, 64, 64), np.uint8)
    for i in range(64):
        for j in range(64):
            landmark_onehot[black_image[i][j][0]][i][j] = 1
    np.save(save_path, landmark_onehot)

