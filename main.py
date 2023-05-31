import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QHBoxLayout, QWidget, \
    QPushButton, QSizePolicy, QAction, QMessageBox, QInputDialog, QGridLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QRect, QTimer
import cv2
import numpy as np

class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cv_img = None


        self.init_ui()




    def init_ui(self):
        self.setWindowTitle('Görüntü İşleme Uygulaması')
        self.setGeometry(self.center_on_screen(1000, 1000))

        save_file_action = QAction('Kaydet', self)
        save_file_action.triggered.connect(self.save_image)

        self.statusBar()

        menubar = self.menuBar()
        file_menu = menubar.addMenu('Dosya')
        file_menu.addAction(save_file_action)

        self.image_label = QLabel()
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setScaledContents(False)

        grid = QGridLayout()

        # Define buttons and their grid positions
        button_data = [
            ('Resim Aç', self.open_image, (0, 0)),
            ('Yeniden Boyutlandır', self.resize_image, (0, 1)),
            ('Döndür', self.rotate_image, (0, 2)),
            ('Blur', self.blur_image, (0, 3)),
            ('Gri Tonlama', self.custom_gray_image, (0, 4)),
            ('Kontür Bul', self.find_contours, (1, 0)),
            ('Köşe Bul', self.detect_corners, (1, 1)),
            ('Kenar Bul', self.detect_edges, (1, 2)),
            ('Erosion', self.erosion_image, (1, 3)),
            ('Dilation', self.dilation_image, (1, 4)),
            ('Opening', self.opening_image, (2, 0)),
            ('Closing', self.closing_image, (2, 1)),
            ('Region Filling', self.region_filling_image, (2, 2)),
            ('Thinning', self.thinning_image, (2, 3)),
            ('Thickening', self.thickening_image, (2, 4)),
            ('RGB Uzayına Dönüştür', self.convert_to_rgb, (3, 0)),
            ('HSV Uzayına Dönüştür', self.convert_to_hsv, (3, 1)),
            ('Histogram Eşitleme', self.equalize_hist_image, (3, 2)),
            ('Color Thresholding', self.threshold_image, (3, 3)),
            ('Color Filtering', self.filter_image, (3, 4)),
            ('Color Reduction', self.color_reduction_image, (4, 0)),
            ('Color Enchancement', self.enhance_colors_image, (4, 1)),
            ('Color Inpainting', self.inpaint_image, (4, 2)),
            ('Color Correction', self.color_correction, (4, 3)),
            ('Color Segmentation', self.color_segmentation, (4, 4)),
            ('Binary Goruntu', self.custom_binary_image, (5, 0)),

            # ... add other buttons here in the same format ...
        ]

        for button_text, slot, position in button_data:
            button = QPushButton(button_text, self)
            button.clicked.connect(slot)
            grid.addWidget(button, *position)

        vbox = QVBoxLayout()
        vbox.addLayout(grid)
        vbox.addWidget(self.image_label)

        widget = QWidget()
        widget.setLayout(vbox)

        self.setCentralWidget(widget)

    def center_on_screen(self, width, height):
        screen = QApplication.desktop().screenGeometry()
        x = (screen.width() - width) // 2
        y = (screen.height() - height) // 2
        return QRect(x, y, width, height)

    def open_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, 'Görüntü Aç', '', 'Images (*.png *.xpm *.jpg *.bmp);;All Files (*)', options=options)

        if file_name:
            self.cv_img = cv2.imread(file_name)
            self.display_image()

    def save_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, 'Görüntüyü Kaydet', '', 'Images (*.png *.xpm *.jpg *.bmp);;All Files (*)', options=options)

        if file_name:
            cv2.imwrite(file_name, self.cv_img)

    def display_image(self):
        height, width, channel = self.cv_img.shape
        bytesPerLine = 3 * width
        qt_img = QImage(self.cv_img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qt_img)
        pixmap = pixmap.scaled(800, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.image_label.setPixmap(pixmap)

    def resize_image(self):
        if not self.check_image_loaded():
            return

        # Ask the user for the new width and height
        new_width, ok = QInputDialog.getInt(self, 'Yeni Genişlik', 'Yeni genişliği girin:', min=1, max=10000)
        if not ok:
            return
        new_height, ok = QInputDialog.getInt(self, 'Yeni Yükseklik', 'Yeni yüksekliği girin:', min=1, max=10000)
        if not ok:
            return

        # Resize the image
        self.cv_img = cv2.resize(self.cv_img, (new_width, new_height))

        self.display_image()

    def rotate_image(self):
        if not self.check_image_loaded():
            return

        rotation = cv2.ROTATE_90_CLOCKWISE

        self.cv_img = cv2.rotate(self.cv_img, rotation)

        self.display_image()

    def check_image_loaded(self):
        if self.cv_img is None:
            QMessageBox.warning(self, 'Hata', 'Lütfen önce bir resim seçin.')
            return False
        return True

    def blur_image(self):

        if not self.check_image_loaded():
            return

        self.cv_img = cv2.GaussianBlur(self.cv_img, (9, 9), 0)
        self.display_image()



    def gray_image(self):

        if not self.check_image_loaded():
            return
        self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2GRAY)
        self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_GRAY2BGR)
        return self.cv_img


    def custom_gray_image(self):
        if not self.check_image_loaded():
            return

        # Ağırlıklar
        r_weight = 0.2989
        g_weight = 0.5870
        b_weight = 0.1140

        # Görüntü boyutları
        height, width, _ = self.cv_img.shape

        # Gri görüntü için yeni bir matris oluştur
        gray_img = np.zeros((height, width), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                # Her pikselin BGR değerlerini al
                b, g, r = self.cv_img[y, x]

                # Gri değeri hesapla
                gray_value = int(b * b_weight + g * g_weight + r * r_weight)

                # Yeni görüntüde gri değeri atama
                gray_img[y, x] = gray_value

        # Gri görüntüyü BGR formatına dönüştür
        self.cv_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        # Görüntüyü güncelle
        self.display_image()


    """def binary_image(self):
        gray_img = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)
        self.cv_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        self.display_image()"""

    def detect_corners(self):
        if not self.check_image_loaded():
            return

        gray = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        self.cv_img[dst > 0.01 * dst.max()] = [0, 0, 255]

        # Update the display
        self.display_image()

    def detect_edges(self):
        if not self.check_image_loaded():
            return

        gray = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        self.cv_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Update the display
        self.display_image()


    ############ BINARY GORUNTU

    def binary_image(self, img):

        if not self.check_image_loaded():
            return
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)
        return binary_img

    def custom_binary_image(self):
        if not self.check_image_loaded():
            return

        # Resmi griye dönüştür
        gray_img = self.gray_image()

        # Eşik değerini belirle
        threshold = 128

        # Görüntüyü ikili formata çevir
        binary_img = np.zeros_like(gray_img)
        binary_img[gray_img > threshold] = 255

        self.cv_img = binary_img

        self.display_image()

    def erosion_image(self):
        if not self.check_image_loaded():
            return

        binary_img = self.binary_image(self.cv_img)

        kernel = np.ones((5, 5), np.uint8)
        self.cv_img = cv2.erode(binary_img, kernel)
        self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_GRAY2BGR)
        self.display_image()

    def dilation_image(self):
        if not self.check_image_loaded():
            return

        binary_img = self.binary_image(self.cv_img)

        kernel = np.ones((5, 5), np.uint8)
        self.cv_img = cv2.dilate(binary_img, kernel, iterations=1)
        self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_GRAY2BGR)
        self.display_image()


    def opening_image(self):

        if not self.check_image_loaded():
            return

        binary_img = self.binary_image(self.cv_img)

        kernel = np.ones((5, 5), np.uint8)
        self.cv_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
        self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_GRAY2BGR)
        self.display_image()

    def closing_image(self):

        if not self.check_image_loaded():
            return

        binary_img = self.binary_image(self.cv_img)

        kernel = np.ones((5, 5), np.uint8)
        self.cv_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
        self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_GRAY2BGR)
        self.display_image()

    def region_filling_image(self):

        if not self.check_image_loaded():
            return
        binary_img = self.binary_image(self.cv_img)

        seed_point = (0, 0)  # Köşeden başlayarak doldurma işlemi yapacağız.
        fill_value = 255
        lo_diff = 1
        hi_diff = 1

        mask = np.zeros((binary_img.shape[0] + 2, binary_img.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(binary_img, mask, seed_point, fill_value, loDiff=lo_diff, upDiff=hi_diff, flags=4)

        self.cv_img = cv2.bitwise_not(binary_img)
        self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_GRAY2BGR)
        self.display_image()



    def thinning_image(self):
        if not self.check_image_loaded():
            return

        binary_img = self.binary_image(self.cv_img)

        # Step 1: Create an empty skeleton
        size = np.size(binary_img)
        skel = np.zeros(binary_img.shape, np.uint8)

        # Get a Cross Shaped Kernel
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        # Repeat steps 2-4
        while True:
            # Step 2: Open the image
            open = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, element)
            # Step 3: Subtract open from the original image
            temp = cv2.subtract(binary_img, open)
            # Step 4: Erode the original image and refine the skeleton
            eroded = cv2.erode(binary_img, element)
            skel = cv2.bitwise_or(skel, temp)
            binary_img = eroded.copy()
            # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
            if cv2.countNonZero(binary_img) == 0:
                break

        self.cv_img = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
        self.display_image()

    def thickening_image(self):
        if not self.check_image_loaded():
            return

        binary_img = self.binary_image(self.cv_img)

        # Step 1: Create an empty thickened image
        thick = np.zeros(binary_img.shape, np.uint8)

        # Get a Cross Shaped Kernel
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        # Repeat steps 2-4
        while True:
            # Step 2: Close the image
            closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, element)
            # Step 3: Subtract the original image from the closed image
            temp = cv2.subtract(closed, binary_img)
            # Step 4: Dilate the original image and refine the thickened image
            dilated = cv2.dilate(binary_img, element)
            thick = cv2.bitwise_or(thick, temp)
            binary_img = dilated.copy()
            # Step 5: If there are no black pixels left ie.. the image has been completely dilated, quit the loop
            if cv2.countNonZero(binary_img) == np.size(binary_img):
                break

        self.cv_img = cv2.cvtColor(thick, cv2.COLOR_GRAY2BGR)
        self.display_image()

        ############ BINARY GORUNTU

    def convert_to_rgb(self):
        if not self.check_image_loaded():
            return

        self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2RGB)
        self.display_image()

    def convert_to_hsv(self):
        if not self.check_image_loaded():
            return

        self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2HSV)
        self.display_image()

    def threshold_image(self):
        if not self.check_image_loaded():
            return

        gray_image = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2GRAY)

        _, thresh_img = cv2.threshold(gray_image, thresh=60, maxval=255, type=cv2.THRESH_BINARY)
        self.cv_img = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
        self.display_image()

    def filter_image(self):

        # Fotoğrafı HSV'ye dönüştür
        self.cv_img =cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2HSV)

        # Kırmızı rengin alt ve üst sınırları
        """lower_red_1 = np.array([0, 70, 50])
        upper_red_1 = np.array([10, 255, 255])

        lower_red_2 = np.array([170, 70, 50])
        upper_red_2 = np.array([180, 255, 255])

        mask_red_1 = cv2.inRange(self.cv_img, lower_red_1, upper_red_1)
        mask_red_2 = cv2.inRange(self.cv_img, lower_red_2, upper_red_2)
        mask = cv2.add(mask_red_1, mask_red_2)"""

        """lower_red = np.array([30, 150, 50])
        upper_red = np.array([255, 255, 180])"""

        lower_blue = np.array([80, 50, 50])
        upper_blue = np.array([120, 255, 255])

        # Mavi renk aralığındaki pikselleri maskeleme
        mask = cv2.inRange(self.cv_img, lower_blue, upper_blue)

        #mask = cv2.inRange(self.cv_img, lower_red, upper_red)

        # Maskeyi oluştur
        #mask = cv2.inRange(self.cv_img, lower_red, upper_red)
        res = cv2.bitwise_and(self.cv_img, self.cv_img, mask=mask)
        self.cv_img=res

        self.display_image()

    def color_reduction_image(self):
        if not self.check_image_loaded():
            return

        k = 64  # Number of colors to reduce to
        data = self.cv_img.reshape((-1, 3))  # Reshape image data to a 2D array
        data = np.float32(data)  # Convert to float32 for k-means algorithm

        # Define criteria and apply k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert the centers back to uint8 and replace pixel values with the closest center
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        self.cv_img = res.reshape(self.cv_img.shape)
        self.display_image()

    def enhance_colors_image(self):
        if not self.check_image_loaded():
            return

        hsv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2HSV)

        # Increase saturation
        hsv_img[..., 1] = hsv_img[..., 1] * 1.5

        # Decrease brightness
        hsv_img[..., 2] = hsv_img[..., 2] * 0.8

        self.cv_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        self.display_image()


    def color_correction(self):
        if not self.check_image_loaded():
            return

        # Converting the image to YCrCb
        img_y_cr_cb = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2YCrCb)

        # Splitting the YCrCb image to Y, Cr and Cb channels
        y, cr, cb = cv2.split(img_y_cr_cb)

        # Applying histogram equalization on Y channel
        y_eq = cv2.equalizeHist(y)

        # Merging the equalized Y, original Cr and original Cb channels
        img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))

        # Converting the histogram equalized image to RGB format
        self.cv_img = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCrCb2BGR)

        self.display_image()

    def color_segmentation(self):
        if not self.check_image_loaded():
            return

        # Convert the image from BGR to HSV color space
        hsv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2HSV)

        # Define the range for blue color in HSV
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

        # Bitwise-AND mask and original image
        self.cv_img = cv2.bitwise_and(self.cv_img, self.cv_img, mask=mask)

        self.display_image()

    def find_contours(self):
        if not self.check_image_loaded():
            return

        # Convert the image to grayscale
        gray_img = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2GRAY)

        # Apply threshold
        ret, thresh = cv2.threshold(gray_img, 127, 255, 0)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Draw all contours
        cv2.drawContours(self.cv_img, contours, -1, (0, 255, 0), 3)

        # Update the display
        self.display_image()

    def equalize_hist_image(self):
        if not self.check_image_loaded():
            return

        gray_img = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2GRAY)
        eq_gray_img = cv2.equalizeHist(gray_img)
        self.cv_img = cv2.cvtColor(eq_gray_img, cv2.COLOR_GRAY2BGR)
        self.display_image()

    def inpaint_image(self):
        img = cv2.imread('OpenCV_Logo_B.png')  # input
        mask = cv2.imread('OpenCV_Logo_C.png', 0)  # mask

        self.cv_img = img.copy()
        self.mask = mask  # Store the mask for later use

        self.display_image()

        QTimer.singleShot(5000, self.perform_inpainting)  # 5000 milliseconds = 5 seconds

    def perform_inpainting(self):
        # Then inpaint and show the inpainted image
        self.cv_img = cv2.inpaint(self.cv_img, self.mask, 3, cv2.INPAINT_TELEA)  # Update self.cv_img
        self.display_image()  # Display the updated image






if __name__ == '__main__':
    app = QApplication(sys.argv)
    image_processor = ImageProcessor()
    image_processor.show()
    sys.exit(app.exec_())
