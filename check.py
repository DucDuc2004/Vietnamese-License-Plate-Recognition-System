import cv2
import matplotlib.pyplot as plt

# Đọc ảnh
image = cv2.imread('Car_dataset/img1.jpg')

# Chuyển sang grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Lọc nhiễu bằng Gaussian Blur
gaussian_blur = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Lọc nhiễu bằng Median Blur
median_blur = cv2.medianBlur(gray_image, 5)

# Lọc nhiễu bằng Bilateral Filter
bilateral_filter = cv2.bilateralFilter(gray_image, 9, 75, 75)

# Hiển thị so sánh các ảnh
images = [gray_image, gaussian_blur, median_blur, bilateral_filter]
titles = ['Grayscale', 'Gaussian Blur', 'Median Blur', 'Bilateral Filter']

for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
