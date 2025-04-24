import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import SIFT_Histogram

dataset_folder = "source/dataset_images"

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        img = Image.open(file_path)
        img = img.resize((300, 300))  # Resize ảnh cho phù hợp
        img_tk = ImageTk.PhotoImage(img)
        lbl_image.config(image=img_tk)
        lbl_image.image = img_tk
        lbl_path.config(text=file_path)
        
        top_3_similar_images = SIFT_Histogram.find_similar_images(file_path, dataset_folder, top_k=3)

        # Hiển thị 3 ảnh kết quả
        for i, img_path in enumerate(top_3_similar_images):
            img = Image.open(img_path)
            img = img.resize((200, 200))  # Resize nhỏ hơn ảnh chính
            img_tk = ImageTk.PhotoImage(img)

            lbl_results[i].config(image=img_tk)
            lbl_results[i].image = img_tk
            lbl_filenames[i].config(text=img_path.split('/')[-1])  # Hiển thị tên file

    

# Tạo cửa sổ chính
root = tk.Tk()
root.title("Chọn ảnh đầu vào")

# Nút chọn ảnh
btn_open = tk.Button(root, text="Chọn ảnh", command=open_image)
btn_open.pack()

# Nút tìm

# Hiển thị đường dẫn ảnh
lbl_path = tk.Label(root, text="Chưa chọn ảnh")
lbl_path.pack()

# Hiển thị ảnh
lbl_image = tk.Label(root)
lbl_image.pack(pady=10)

# Tạo 3 nhãn ảnh kết quả và tên file
lbl_results = [tk.Label(root) for _ in range(3)]
lbl_filenames = [tk.Label(root, font=("Arial", 10)) for _ in range(3)]

# Sắp xếp ảnh kết quả theo hàng ngang
for i in range(3):
    lbl_results[i].pack(side=tk.LEFT, padx=10)
    lbl_filenames[i].pack(side=tk.LEFT, padx=10)

# Chạy ứng dụng
root.mainloop()



