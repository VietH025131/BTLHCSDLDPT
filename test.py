import cv2
import numpy as np
import os
import pickle
from pathlib import Path

def extract_sift_features(image_path, features_cache=None):
    """Trích xuất và lưu đặc trưng SIFT từ ảnh"""
    # Tạo tên file cache dựa trên đường dẫn ảnh
    cache_path = None
    if features_cache:
        Path(features_cache).mkdir(parents=True, exist_ok=True)
        filename = Path(image_path).stem
        cache_path = os.path.join(features_cache, f"{filename}.pkl")
        
        # Kiểm tra nếu đã có cache
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
    
    # Nếu không có cache thì trích xuất mới
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None
    
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    # Lưu vào cache nếu được yêu cầu
    if cache_path and descriptors is not None:
        with open(cache_path, 'wb') as f:
            pickle.dump((keypoints, descriptors), f)
    
    return keypoints, descriptors

def match_features(des1, des2):
    """So khớp đặc trưng SIFT giữa 2 ảnh"""
    if des1 is None or des2 is None:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return len(matches)

def find_similar_images(query_image_path, dataset_folder, top_k=3, features_cache=None):
    """Tìm ảnh giống nhất trong dataset với khả năng lưu cache đặc trưng"""
    # Trích xuất đặc trưng từ ảnh đầu vào
    _, query_des = extract_sift_features(query_image_path, features_cache)
    
    # Lấy danh sách ảnh trong dataset
    image_files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) 
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    # Tính toán số điểm khớp với từng ảnh trong dataset
    similarity_scores = []
    for img_path in image_files:
        _, des = extract_sift_features(img_path, features_cache)
        matches = match_features(query_des, des)
        similarity_scores.append((img_path, matches))
    
    # Sắp xếp theo số điểm khớp giảm dần
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Trả về top_k ảnh giống nhất và số điểm khớp (để debug)
    return [(img_path, score) for img_path, score in similarity_scores[:top_k]]

# Ví dụ sử dụng
if __name__ == "__main__":
    query_img = "data:image/jpeg;base64,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"
    dataset_folder = "dataset_images"
    cache_dir = "features_cache"  # Thư mục lưu cache đặc trưng
    
    # Tìm ảnh giống nhau và lưu cache đặc trưng
    similar_images = find_similar_images(query_img, dataset_folder, top_k=3, features_cache=cache_dir)
    
    print("Top 3 ảnh giống nhất:")
    for i, (img_path, score) in enumerate(similar_images, 1):
        print(f"{i}. {img_path} (Số điểm khớp: {score})")