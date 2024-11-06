import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Tải mô hình đã lưu
model = load_model('license_plate_detection_model.h5')
# Hàm tiền xử lý ảnh
def preprocess_image(image_path, img_size=(224, 224)):
    # Đọc ảnh
    img = cv2.imread(image_path)
    # Resize ảnh về kích thước mà mô hình yêu cầu
    img_resized = cv2.resize(img, img_size)
    # Chuẩn hóa ảnh
    img_resized = img_resized / 255.0
    # Thêm chiều batch cho ảnh
    img_resized = np.expand_dims(img_resized, axis=0)
    return img_resized
# Hàm phát hiện biển số xe
def detect_license_plate(image_path):
    # Tiền xử lý ảnh
    img = preprocess_image(image_path)
    
    # Dự đoán bounding box từ mô hình
    prediction = model.predict(img)[0]
    
    # Dự đoán bao gồm (x_center, y_center, width, height)
    x_center, y_center, width, height = prediction
    
    # Chuyển đổi các giá trị về pixel
    img_original = cv2.imread(image_path)
    img_height, img_width, _ = img_original.shape
    
    # Quy đổi các giá trị về pixel
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    
    # Vẽ bounding box lên ảnh
    cv2.rectangle(img_original, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Hiển thị ảnh với bounding box
    cv2.imshow("Detected License Plate", img_original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Thử phát hiện biển số từ một ảnh cụ thể
image_path = 'Car_Dataset/images/img1.jpg'  # Thay bằng đường dẫn tới ảnh của bạn
detect_license_plate(image_path)
