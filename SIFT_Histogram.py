import cv2
import numpy as np
import os

def extract_sift_features(image_path):
    """Trích xuất đặc trưng SIFT từ ảnh"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(des1, des2):
    """So khớp đặc trưng SIFT giữa 2 ảnh"""
    if des1 is None or des2 is None:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return len(matches)

def find_similar_images(query_image_path, dataset_folder, top_k=3):
    """Tìm 3 ảnh giống nhất trong dataset"""
    # Trích xuất đặc trưng từ ảnh đầu vào
    _, query_des = extract_sift_features(query_image_path)
    
    # Lấy danh sách ảnh trong dataset
    image_files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) 
                  if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Tính toán số điểm khớp với từng ảnh trong dataset
    similarity_scores = []
    for img_path in image_files:
        _, des = extract_sift_features(img_path)
        matches = match_features(query_des, des)
        similarity_scores.append((img_path, matches))
    
    # Sắp xếp theo số điểm khớp giảm dần
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Trả về top_k ảnh giống nhất
    return [img_path for img_path, _ in similarity_scores[:top_k]]


