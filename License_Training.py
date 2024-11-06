import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Hàm để đọc label từ file .txt, bỏ qua giá trị class
def load_labels(label_path):
    with open(label_path, 'r') as file:
        line = file.readline().strip().split()
        # Bỏ qua giá trị class (vị trí 0), chỉ lấy 4 giá trị sau
        return [float(coord) for coord in line[1:]]  # lấy 4 giá trị sau `0` (x_center, y_center, width, height)


# Hàm để đọc ảnh và label
def load_data(image_dir, label_dir, img_size=(224, 224)):
    images = []
    labels = []
    
    # Lặp qua tất cả các file ảnh
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt'))
        
        if os.path.exists(label_path):
            # Đọc và tiền xử lý ảnh
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img = img / 255.0  # chuẩn hóa ảnh
            images.append(img)
            
            # Đọc label và chuyển đổi
            label = load_labels(label_path)
            labels.append(label)
    
    return np.array(images), np.array(labels)

# Đọc dữ liệu
image_dir = 'Car_Dataset/images'
label_dir = 'Car_Dataset/labels1'
X, y = load_data(image_dir, label_dir)

# Chia tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Xây dựng mô hình CNN
def build_model(input_shape=(224, 224, 3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(4, activation='sigmoid'))  # 4 đầu ra cho (x_center, y_center, width, height)
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])
    return model

# Tạo mô hình
model = build_model()
# Huấn luyện mô hình
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=8)
# Lưu mô hình
model.save('license_plate_detection_model.h5')
