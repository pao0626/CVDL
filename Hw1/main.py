import os
import sys
import numpy as np
import cv2
import re
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow
from PyQt5.QtGui import QPixmap
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torchsummary
from matplotlib import pyplot as plt
from hw1_ui import Ui_hw1

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_hw1()
        self.ui.setupUi(self)
        
        # member initialization
        self.load_all_file = ""
        self.files = []  # list of file name of loadall

        self.l = ""
        self.r = ""
        
        self.load_img1 = ""
        self.load_img2 = ""

        self.load_Q5image = ""
        self.class_name = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


        # from tensorflow.keras.datasets import cifar10
        # (self.x_train, self.y_train), (self.x_test, self.y_test)=cifar10.load_data()

        # ref https://www.jianshu.com/p/d9c4fb366fe2
        self.w = 11
        self.h = 8
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) #Conditions for terminating the optimization algorithm
        self.cp_int = np.zeros((self.w*self.h, 3), dtype=np.float32) 
        self.cp_int[:, :2] = np.mgrid[0:self.w, 0:self.h].T.reshape(-1, 2)

        # ret, intrinsic(內參矩陣), distort(畸變係數), r_vecs(旋轉向量), t_vecs(平移向量) by cv2.calibrateCamera
        self.matrix = None

        # click connect
        self.ui.pushButton0_1.clicked.connect(self.loadall_click)
        self.ui.pushButton0_2.clicked.connect(self.loadl_click)
        self.ui.pushButton0_3.clicked.connect(self.loadr_click)
        self.ui.pushButton1_1.clicked.connect(self.corners_click)
        self.ui.pushButton1_2.clicked.connect(self.intrinsic_click)
        self.ui.pushButton1_3.clicked.connect(self._extrinsic_click)
        self.ui.pushButton1_4.clicked.connect(self.distortion_click)
        self.ui.pushButton1_5.clicked.connect(self.show_result_click)
        self.ui.pushButton2_1.clicked.connect(self.horizontally_click)
        self.ui.pushButton2_2.clicked.connect(self.vertically_click)
        self.ui.pushButton3_1.clicked.connect(self.disparity_click)
        self.ui.pushButton4_1.clicked.connect(self.load1_click)
        self.ui.pushButton4_2.clicked.connect(self.load2_click)
        self.ui.pushButton4_3.clicked.connect(self.keypoints_click)
        self.ui.pushButton4_4.clicked.connect(self.match_click)
        self.ui.pushButton5_1.clicked.connect(self.load_click)
        self.ui.pushButton5_2.clicked.connect(self.augmented_click)
        self.ui.pushButton5_3.clicked.connect(self.structure_click)
        self.ui.pushButton5_4.clicked.connect(self.acc_loss)
        self.ui.pushButton5_5.clicked.connect(self.infer_click)

        #group 0
    def loadall_click(self):
        try:
            self.load_all_file = str(QFileDialog.getExistingDirectory(self, "Select a Folder"))
            allbmp = [i for i in os.listdir(self.load_all_file) if re.search(".bmp", i)] #find all .bmp file
            self.files = sorted(allbmp, key=lambda x: int(x.split('.')[0]))
            print(self.files)
        except Exception as e:
            print(f"Error: {str(e)}")
    
    def loadl_click(self):
        try:
            self.l = str(QFileDialog.getOpenFileName(self, "Choose a file")[0])
            print(self.l)
        except Exception as e:
            print(f"Error: {str(e)}")
    
    def loadr_click(self):
        try:
            self.r = str(QFileDialog.getOpenFileName(self, "Choose a file")[0])
            print(self.r)
        except Exception as e:
            print(f"Error: {str(e)}")

    def calibaration(self):
        try:
            print("start calibaration")
            obj_points = [] # the points in world space
            img_points = [] # the points in image space (relevant to obj_points)
            for file in self.files:
                img = cv2.imread(os.path.join(self.load_all_file, file))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                ret, corners = cv2.findChessboardCorners(gray, (self.w, self.h), None)
                if(ret == True):
                    new_corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), self.criteria)
                    obj_points.append(self.cp_int)
                    img_points.append(new_corners)
            self.matrix = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
        except Exception as e:
            print(f"Error: {str(e)}")

    #group 1
    def corners_click(self):
        try:
            for file in self.files:
                img = cv2.imread(os.path.join(self.load_all_file, file))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #change image color to gray
                
                ret, corners = cv2.findChessboardCorners(gray, (self.w, self.h), None)
                if(ret == True):
                    new_corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), self.criteria)
                    img_out = cv2.drawChessboardCorners(img, (self.w, self.h), new_corners, ret)
                    cv2.namedWindow('Corners', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Corners', 480, 480)
                    cv2.imshow('Corners', img_out)
                    cv2.waitKey(1000)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error: {str(e)}")

    #f_x and f_y are the focal lengths in the horizontal and vertical directions and represent the camera's magnification.
    #s is the oblique mirror abnormality (usually 0, indicating no oblique mirror abnormality).
    #c_x and c_y are the positions of the principal point (primary optical center), representing the pixel coordinates of the image center
    #Intrinsic Matrix = [[f_x s c_x][0 f_y c_y][0 0 1]]
    def intrinsic_click(self):         
        try:
            if not self.matrix :
                self.calibaration()
            print("Intrinsic:", self.matrix[1], sep="\n")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    # Extrinsic Matrix = | R | t |
    def _extrinsic_click(self):
        try:
            if not self.matrix :
                self.calibaration()
            index = int(self.ui.spinBox.value()) - 1
            rotation_mat = cv2.Rodrigues(self.matrix[3][index])[0] # return matrix and vector len
            # print("rotation_mat", rotation_mat, sep="\n")
            translation_mat = self.matrix[4][index]
            # print("translation_mat", self.matrix[4][index], sep="\n")
            extrinsic_mat = np.hstack([rotation_mat, translation_mat])
            print("Extrinsic", extrinsic_mat, sep="\n")
        except Exception as e:
            print(f"Error: {str(e)}")       

    def distortion_click(self):
        try:
            if not self.matrix :
                self.calibaration()
            print("Intrinsic:", self.matrix[2], sep="\n")
        except Exception as e:
            print(f"Error: {str(e)}")

    #ref https://opencv-python-tutorials.readthedocs.io/zh/latest/7.%20%E7%9B%B8%E6%9C%BA%E6%A0%A1%E5%87%86%E5%92%8C3D%E9%87%8D%E5%BB%BA/7.1.%20%E7%9B%B8%E6%9C%BA%E6%A0%A1%E5%87%86/
    def show_result_click(self):
        try:
            if not self.matrix :
                self.calibaration()
            for file in self.files:
                img = cv2.imread(os.path.join(self.load_all_file, file))
                h, w = img.shape[:2]
                newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(self.matrix[1], self.matrix[2], (w, h), 1, (w, h))
                dst = cv2.undistort(img, self.matrix[1], self.matrix[2], None, newcameramatrix)

                x, y, w, h = roi
                dst = dst[y:y+h, x:x+w]
                img = cv2.resize(img, (480,480))
                dst = cv2.resize(dst, (480,480))
                imgs = np.hstack([dst, img])
                cv2.imshow("undistorted result", imgs)
                cv2.waitKey(1000)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error: {str(e)}")
    
    def draw(self, img, img_point, len):
        try:  
            img_point = np.int32(img_point).reshape(-1,2)
            for i in range(len):
                img = cv2.line(img, tuple(img_point[2*i]), tuple(img_point[2*i+1]),(0, 0, 255), 15) #img, line_start, line_end, color, width_pixel
            return img 
        except Exception as e:
            print(f"Error: {str(e)}")

    #group 2
    def horizontally_click(self):
        try:            
            self.calibaration()

            word = []
            text = self.ui.lineEdit.text().upper()
            lib = os.path.join(self.load_all_file,'Q2_lib/alphabet_lib_onboard.txt')
            fs = cv2.FileStorage(lib, cv2.FILE_STORAGE_READ)

            length = 0 #record alphabet len
            for i in range(len(text)):
                if text[i].encode('UTF-8').isalpha() and not text[i].isdigit():
                    word.append(fs.getNode(text[i]).mat())
                    length = length + 1

            pos_adjust=[[7,5,0],[4,5,0],[1,5,0],[7,2,0],[4,2,0],[1,2,0]] #for postiopn and order

            for i in range(length):
                for j in range(len(word[i])):
                    new_axis1 = [a + b for a, b in zip(word[i][j][0], pos_adjust[i])]
                    new_axis2 = [a + b for a, b in zip(word[i][j][1], pos_adjust[i])]
                    word[i][j][0]=new_axis1
                    word[i][j][1]=new_axis2

            for i in range(len(self.files)):
                img = cv2.imread(os.path.join(self.load_all_file, self.files[i]))
                rotation_vector= cv2.Rodrigues(self.matrix[3][i])[0]
                transform_vector = self.matrix[4][i]

                axis = []
                for j in range(length):
                    axis.append(np.array(word[j], dtype=np.float32).reshape(-1,3))
                    img_points, _ = cv2.projectPoints(axis[j], rotation_vector, transform_vector, self.matrix[1], self.matrix[2])
                    img_out =  self.draw(img, img_points, len(word[j]))              
                cv2.namedWindow('Augmented Reality', cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Augmented Reality", 480, 480)
                cv2.imshow('Augmented Reality',img_out)
                cv2.waitKey(1000)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error: {str(e)}")

    def vertically_click(self):
        try:
            self.calibaration()

            word = []
            text = self.ui.lineEdit.text().upper()
            lib = os.path.join(self.load_all_file,'Q2_lib/alphabet_lib_vertical.txt')
            fs = cv2.FileStorage(lib, cv2.FILE_STORAGE_READ)

            length = 0
            for i in range(len(text)):
                if text[i].encode('UTF-8').isalpha() and not text[i].isdigit():
                    word.append(fs.getNode(text[i]).mat())
                    length = length + 1

            pos_adjust=[[7,5,0],[4,5,0],[1,5,0],[7,2,0],[4,2,0],[1,2,0]]

            for i in range(length):
                for j in range(len(word[i])):
                    new_axis1 = [a + b for a, b in zip(word[i][j][0], pos_adjust[i])]
                    new_axis2 = [a + b for a, b in zip(word[i][j][1], pos_adjust[i])]
                    word[i][j][0]=new_axis1
                    word[i][j][1]=new_axis2

            for i in range(len(self.files)):
                img = cv2.imread(os.path.join(self.load_all_file, self.files[i]))
                rotation_vector= cv2.Rodrigues(self.matrix[3][i])[0]
                transform_vector = self.matrix[4][i]

                axis = []
                for j in range(length):
                    axis.append(np.array(word[j], dtype=np.float32).reshape(-1,3))
                    img_points, _ = cv2.projectPoints(axis[j], rotation_vector, transform_vector, self.matrix[1], self.matrix[2])
                    img_out =  self.draw(img, img_points, len(word[j]))            
                cv2.namedWindow('Augmented Reality_V', cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Augmented Reality_V", 480,480)
                cv2.imshow('Augmented Reality_V',img_out)
                cv2.waitKey(1000)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error: {str(e)}")

    #group 3
    def disparity_click(self):
        try:
            if self.l and self.r:
                imgL = cv2.imread(self.l)
                imgR = cv2.imread(self.r)
                imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
                imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

                stereo = cv2.StereoBM_create(numDisparities = 256, blockSize = 25)
                disparity = stereo.compute(imgL_gray, imgR_gray)
                disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.namedWindow('disparity',cv2.WINDOW_NORMAL)
                cv2.resizeWindow("disparity", 720, 540)
                cv2.imshow("disparity", disparity)

                def mouse(event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        imgR_cp = imgR.copy()
                        imgL_cp = imgL.copy()
                        if disparity[y, x] != 0:
                            cv2.circle(imgR_cp, (x-disparity[y, x], y), 10, (0,255,0), thickness = -1)
                            print('(x: {}, y: {}),dis: {}'.format(x, y, disparity[y, x]))
                            cv2.imshow("imgR", imgR_cp)
                            cv2.imshow("imgL", imgL_cp)
                        else:
                            print("Failure case")

                cv2.namedWindow('imgL',cv2.WINDOW_NORMAL)
                cv2.resizeWindow("imgL", 720, 540)
                cv2.imshow("imgL", imgL)
                cv2.namedWindow('imgR',cv2.WINDOW_NORMAL)
                cv2.resizeWindow("imgR", 720, 540)     
                cv2.imshow("imgR", imgR)
                cv2.setMouseCallback("imgL", mouse)
                cv2.waitKey(0)

            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error: {str(e)}")

    #group 4        
    def load1_click(self):
        try:
            self.load_img1 = str(QFileDialog.getOpenFileName(self, "Choose a file")[0])
            print(self.load_img1)
        except Exception as e:
            print(f"Error: {str(e)}")
            
    def load2_click(self):
        try:
            self.load_img2 = str(QFileDialog.getOpenFileName(self, "Choose a file")[0])
            print(self.load_img2)
        except Exception as e:
            print(f"Error: {str(e)}")

    def keypoints_click(self):
        try:
            img1 = cv2.imread(self.load_img1)
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

            sift = cv2.SIFT_create()
            key1, _ = sift.detectAndCompute(img1_gray, None)

            kp_image1 = cv2.drawKeypoints(img1_gray, key1, None, color=(0,255,0))
            cv2.namedWindow('Keypoints',cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Keypoints", 480, 480)
            cv2.imshow("Keypoints", kp_image1)
            cv2.waitKey()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error: {str(e)}")

    #ref https://www.cnblogs.com/silence-cho/p/15170216.html
    def match_click(self):
        try:
            img1 = cv2.imread(self.load_img1)
            img2 = cv2.imread(self.load_img2)
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            sift = cv2.SIFT_create()
            key1, des1 = sift.detectAndCompute(img1_gray, None)
            key2, des2 = sift.detectAndCompute(img2_gray, None)

            macher = cv2.BFMatcher()
            matches = macher.knnMatch(des1, des2, k=2)
            matches = sorted(matches, key=lambda x: x[0].distance / x[1].distance)
        
            goodMatches = []
            for m,n in matches:
                if m.distance < 0.75 * n.distance:
                    goodMatches.append([m])

            img3 = cv2.drawMatchesKnn(img1_gray, key1, img2_gray, key2, goodMatches, outImg = None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.namedWindow('Matched',cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Matched", 960, 480)
            cv2.imshow('Matched',img3)
            cv2.waitKey()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error: {str(e)}")

    #group 5     
    def load_click(self):
        try:
            self.load_Q5image = str(QFileDialog.getOpenFileName(self, "Choose a file")[0])
            print(self.load_Q5image)
            img = QPixmap(self.load_Q5image)
            self.ui.label_img.setPixmap(img) # ref https://sammypython.blogspot.com/2019/01/pyqt5-qpixmap.html
        except Exception as e:
            print(f"Error: {str(e)}")
    
    def augmented_click(self):
        try:
            image_folder = "C:/Users/p7611/Desktop/CVDLhw1/Dataset_CvDl_Hw1/Q5_image/Q5_1/"

            data_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(), 
                transforms.RandomVerticalFlip(),   
                transforms.RandomRotation(30)       
            ])

            image_names = os.listdir(image_folder)[:9]
            images = [Image.open(os.path.join(image_folder, name)) for name in image_names]

            plt.figure(figsize=(8, 8))

            for i in range(9):
                plt.subplot(3, 3, i + 1)
                augmented_image = data_transform(images[i])
                plt.imshow(augmented_image)
                image_name = image_names[i].replace(".png", "")
                plt.title(image_name)
                plt.axis("on")  
                plt.xticks(range(0, augmented_image.width, 10))  
                plt.yticks(range(0, augmented_image.height, 5)) 

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error: {str(e)}")

    def structure_click(self):
        try:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = models.vgg19_bn(num_classes=10)
            model = model.to(device)
            torchsummary.summary(model, (3, 32, 32))
        except Exception as e:
            print(f"Error: {str(e)}")

    def acc_loss(self):
        try:
            img_path = 'C:/Users/p7611/Desktop/CVDLhw1/loss_acc.png'
            img = plt.imread(img_path)
            plt.figure(figsize=(8, 12))
            plt.imshow(img)
            plt.axis('off')  # 可以關閉座標軸
            plt.show()
        except Exception as e:
            print(f"Error: {str(e)}")

    def infer_click(self):
        try:
            image = Image.open(self.load_Q5image)
            
            #建立 VGG-19 model 並載入預訓練參數
            model = models.vgg19_bn(num_classes=10)
            model.load_state_dict(torch.load('best_model.pth'))
            
            #將模型設定為評估（evaluation）模式
            model.eval()

            # Prepare the image for inference
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = transform(image).unsqueeze(0)

            # Perform inference
            with torch.no_grad():
                output = model(image)
            
            class_idx = torch.argmax(output)
            self.ui.label.setText(f"Predict = {self.class_name[class_idx]}")
            self.ui.label.adjustSize()

            # Plot probability distribution
            probabilities = torch.softmax(output, dim=1).numpy()[0]
            plt.figure()
            plt.bar(self.class_name, probabilities)
            plt.xlabel('Class')
            plt.ylabel('Probability')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.title('Probability of each class')
            plt.show()
    
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == '__main__':  
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())     