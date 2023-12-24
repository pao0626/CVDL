import os
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torchsummary import summary
from torch.utils.data import DataLoader
from PIL import Image
import random
from PyQt5.QtWidgets import QMainWindow, QApplication,  QFileDialog
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont
from hw2_ui import Ui_hw2


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_hw2()
        self.ui.setupUi(self)

        # 讓q4畫板呈現黑色
        self.SetGraph()

        self.ui.Image.clicked.connect(self.Image_click)
        self.ui.Video.clicked.connect(self.Video_click)     
        self.ui.Subtraction.clicked.connect(self.Subtraction_click)
        self.ui.Preprocessing.clicked.connect(self.Preprocessing_click)
        self.ui.Tracking.clicked.connect(self.Tracking_click)
        self.ui.Reduction.clicked.connect(self.Reduction_click)
        self.ui.Struction.clicked.connect(self.StructionVGG_click)
        self.ui.Acc_loss.clicked.connect(self.Acc_loss_click)
        self.ui.Predict.clicked.connect(self.Predict_click)
        self.ui.Reset.clicked.connect(self.Reset_click)
        self.ui.Load.clicked.connect(self.Load_click)
        self.ui.Show_Image.clicked.connect(self.ShowImage_click)
        self.ui.Struction_ResNet.clicked.connect(self.StructionResNet_click)
        self.ui.Comprasion.clicked.connect(self.Comprasion_click)
        self.ui.Inference.clicked.connect(self.Inference_click)

        self.video = ''
        self.image = ''

    def Image_click(self):
        try:
            self.image = str(QFileDialog.getOpenFileName(self, "Choose a file")[0])
            print(self.image)
        except Exception as e:
            print(f"Error: {str(e)}")

    def Video_click(self):
        try:
            self.video = str(QFileDialog.getOpenFileName(self, "Choose a file")[0])
            print(self.video)
        except Exception as e:
            print(f"Error: {str(e)}")

    #group 1
    def Subtraction_click(self):
        try:
            cap = cv2.VideoCapture(self.video)
            
            subtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=True)
                        
            while True:
                # 讀取一幀
                ret, frame = cap.read()
                if not ret:
                    break

                # 模糊處理
                blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

                # 使用背景移除器獲取背景遮罩
                mask = subtractor.apply(blurred_frame)

                # 通過位元與操作提取僅包含運動物體的幀
                result_frame = cv2.bitwise_and(blurred_frame, blurred_frame, mask=mask)
        
                cv2.imshow("Original Video", frame)
                cv2.imshow("Background Substraction Mask", mask)
                cv2.imshow("Background Substraction", result_frame)
                
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error: {str(e)}")
    
    #group 2   
    def Preprocessing_click(self):
        try:
            cap = cv2.VideoCapture(self.video)
            ret, frame = cap.read()
            if not ret:
                return
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            maxCorners = 1 # 檢測的最大角點數量
            qualityLevel = 0.3 # 角點的品質因子，高於閾值保留
            minDistance = 7 # 兩個角點的距離小於此值，則只保留一個
            blockSize = 7 # 角點檢測中使用的窗口大小
            
            corners = cv2.goodFeaturesToTrack(
                gray, maxCorners, qualityLevel, minDistance, blockSize
            )
            if corners is not None:
                x, y = corners[0][0]

                cv2.line(frame, (int(x) - 10, int(y)), (int(x) + 10, int(y)), (0, 0, 255), 4)
                cv2.line(frame, (int(x), int(y) - 10), (int(x), int(y) + 10), (0, 0, 255), 4)

            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("frame", 960, 540)
            cv2.imshow("frame", frame)
            cv2.waitKey(0) #delay 的值為 0，表示將無窮等待

            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error: {str(e)}")

    def Tracking_click(self):
        try:
            cap = cv2.VideoCapture(self.video)
            ret, prev_frame = cap.read()
            if not ret:
                return

            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

            maxCorners = 1
            qualityLevel = 0.3
            minDistance = 7
            blockSize = 7

            prev_corners = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners, qualityLevel, minDistance, blockSize
            )

            # Create an empty mask image
            mask = np.zeros_like(prev_frame)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 追蹤特徵點
                next_corners, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, prev_corners, None
                )
                
                # 畫線
                a, b = next_corners.ravel() 
                c, d = prev_corners.ravel() # same as prev_corners[0][0]
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 100, 255), 2)
                frame = cv2.line(
                    frame, (int(a) - 10, int(b)), (int(a) + 10, int(b)), (0, 0, 255), 4
                )
                frame = cv2.line(
                    frame, (int(a), int(b) - 10), (int(a), int(b) + 10), (0, 0, 255), 4
                )

                # 結合黃線和特徵圖
                output = cv2.add(frame, mask)

                # Show the frame with the trajectory lines
                cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("frame", 960, 540)
                cv2.imshow("frame", output)

                # 更新前一幀的圖片和特徵點
                prev_gray = gray.copy()
                prev_corners = next_corners.reshape(-1, 1, 2)
                
                if cv2.waitKey(15) & 0xFF == ord("q"):
                    break

            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
                print(f"Error: {str(e)}")

    #group 3            
    def Reduction_click(self):
        try:
            img = cv2.imread(self.image)
            # 轉成灰圖
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 正規化
            normalized_img = gray_img / 255.0

            # DO PCA
            w, h = gray_img.shape
            min_dim = min(w, h)
            mse_threshold = 3.0
            n = 1

            while True:
                pca = PCA(n_components=n)
                # 轉成一維進行降維
                transformed_image = pca.fit_transform(normalized_img.reshape(-1, min_dim))
                # 投射回原始維度
                reconstructed_img = pca.inverse_transform(transformed_image)

                # 用MSE(Mean Square Error)計算誤差
                mse = np.mean(((normalized_img - reconstructed_img.reshape(w, h)) * 255.0) ** 2)

                print("n:", n, "MSE: ",mse)
                if mse <= mse_threshold or n >= min_dim:
                    break

                n += 1
            
            # 印出結果
            print("Minimum n value:", n)

            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(normalized_img, cmap="gray")
            axs[0].set_title("Gray Scale image")
            axs[0].axis("off")

            axs[1].imshow(reconstructed_img.reshape(w, h), cmap="gray")
            axs[1].set_title("Reconstruction image (n={})".format(n))
            axs[1].axis("off")

            plt.show()
        except Exception as e:
                print(f"Error: {str(e)}")

    #group 4        
    def SetGraph(self):
        pixmap = QPixmap(501, 311)
        pixmap.fill(QColor(0, 0, 0))
        self.ui.paint.setText("")
        self.ui.paint.setPixmap(pixmap)
        self.ui.paint.mousePressEvent = self.MousePress
        self.ui.paint.mouseMoveEvent = self.MouseMove

    def drawLine(self, event, width):
        painter = QPainter(self.ui.paint.pixmap())
        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(width)
        painter.setPen(pen)
        painter.drawLine(
            event.pos().x(), event.pos().y(), event.pos().x(), event.pos().y()
        )
        painter.end()
        self.ui.paint.setPixmap(self.ui.paint.pixmap())

    def MousePress(self, event):
        # print("Press")
        self.drawLine(event, 5)
        
    def MouseMove(self, event):
        # print("Move")
        self.drawLine(event, 20)

    def StructionVGG_click(self):
        try:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(device)
            model = models.vgg19_bn(num_classes=10)
            model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
            model.to(device)
            summary(model, (1, 32, 32))
        except Exception as e:
                print(f"Error: {str(e)}")
    
    def Acc_loss_click(self):
        try:
            img_path = "C:/Users/p7611/Desktop/CVDLhw2/loss_acc.png"          
            img = QPixmap(img_path)
            img = img.scaled(501, 311)
            self.ui.paint.setPixmap(img)

            # 直接呈現圖片
            img = plt.imread(img_path)
            plt.figure(figsize=(8, 12))
            plt.imshow(img)
            plt.axis('off')  # 可以關閉座標軸
            plt.show()
        except Exception as e:
                print(f"Error: {str(e)}")
                
    def Predict_click(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=1),  # Ensure single channel for MNIST
            transforms.ToTensor()
        ])
        piximg = QPixmap(self.ui.paint.pixmap())
        piximg.save("Test.png")
        img = Image.open("Test.png")
        img = transform(img)
        img = img.unsqueeze(0) # 將(channels, height, width)擴展符合模型输入形状 (batch_size, channels, height, width)
        img = img.to(device)
        print("Trans Complete")

        model = models.vgg19_bn(num_classes=10)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        model = model.to(device)
        model.load_state_dict(torch.load('./model/best_model.pth'))
        model.eval()

        with torch.no_grad():
            output = model(img)
        print(output)

        # 畫出機率圖方法1
        output = torch.clamp(output, min=0) # 將輸出中的負值修正為零       
        labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]        
        plt.bar(labels, output[0].tolist())
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.tight_layout()
        plt.title('Probability of each class')
        plt.show()

        # 印出結果
        class_idx = torch.argmax(output) # 找出最大值
        font = QFont()
        font.setPointSize(16) # 字體大小
        font.setBold(True)    # 粗體
        self.ui.result.setFont(font)
        self.ui.result.setText(labels[class_idx])

    def Reset_click(self):
        try:      
            self.ui.paint.clear()  # 清空 QLabel 上的内容
            pixmap = QPixmap(501, 311)
            pixmap.fill(QColor(0, 0, 0))
            self.ui.paint.setPixmap(pixmap)
        except Exception as e:
            print(f"Error: {str(e)}")
    
    # group 5
    def Load_click(self):
        try:
            self.image = str(QFileDialog.getOpenFileName(self, "Choose a file")[0])
            print(self.image)
            img = QPixmap(self.image)
            self.ui.label_img.setPixmap(img)
        except Exception as e:
            print(f"Error: {str(e)}")

    def ShowImage_click(self):
        cat_dir = "./Dataset/inference_dataset/Cat"
        dog_dir = "./Dataset/inference_dataset/Dog"
        
        cat_list = os.listdir(cat_dir)
        dog_list = os.listdir(dog_dir)

        random_cat_filename = random.choice(cat_list)
        random_dog_filename = random.choice(dog_list)
               
        cat_img = Image.open(os.path.join(cat_dir, random_cat_filename))
        dog_img = Image.open(os.path.join(dog_dir, random_dog_filename))
        
        Resize = transforms.Resize((224, 224))
        
        cat_img = Resize(cat_img)
        dog_img = Resize(dog_img)
        
        plt.subplot(1, 2, 1)
        plt.title("Cat")
        plt.imshow(cat_img)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title("Dog")
        plt.imshow(dog_img)
        plt.axis('off')
        
        plt.show()
    
    def StructionResNet_click(self):
        try:
            model = models.resnet50()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.fc = nn.Sequential(
                nn.Linear(2048, 1),
                nn.Sigmoid()
            )
            model = model.to(device)
            summary(model, input_size=(3, 224, 224))
        except Exception as e:
            print(f"Error: {str(e)}")
    
    def Comprasion_click(self):
        try:
            img_path = 'C:/Users/p7611/Desktop/CVDLhw2/ResNet_Comparison.png'
            if not os.path.exists(img_path):
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                
                model = models.resnet50()
                model.fc = nn.Sequential(
                    nn.Linear(2048, 1),
                    nn.Sigmoid()
                )
                model = model.to(device)
                model.load_state_dict(torch.load('./model/ResNet_model.pth'))
                model.eval()

                modelNE = models.resnet50()
                modelNE.fc = nn.Sequential(
                    nn.Linear(2048, 1),
                    nn.Sigmoid()
                )
                modelNE = modelNE.to(device)
                modelNE.load_state_dict(torch.load('./model/ResNet_model_withoutErase.pth'))
                modelNE.eval()
                
                transform = transforms.Compose([
                    transforms.Resize((244, 244)),
                    transforms.ToTensor()
                ])
                test_dataset = datasets.ImageFolder(root='./Dataset/validation_dataset', transform=transform)
                testloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

                correct = 0
                correctNE = 0
                total = 0               
                threshold = 0.5

                with torch.no_grad():
                    for data in testloader:
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        outputsNE = modelNE(inputs)
                        labels = labels.float().view(-1, 1)
                        predicted = (outputs > threshold).float()
                        predictedNE = (outputsNE > threshold).float()
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        correctNE += (predictedNE == labels).sum().item()
                
                print("correct:",correct)
                print("correctNE:",correctNE)
                print("total:",total)
                accuracy = 100 * correct / total
                accuracyNE = 100 * correctNE / total
                print("accuracy_erase = ",accuracy)
                print("accuracy_noErase = ",accuracyNE)

                models_name = ['With Random erasing', 'Without Random erasing']
                accuracies = [accuracy, accuracyNE]
                bar_width = 0.5
                plt.bar(models_name, accuracies, color='blue', width=bar_width)
                plt.ylabel('Accuracy (%)')
                plt.title('Accuracy Comparison')
                for i, v in enumerate(accuracies):
                   plt.text(i, v + 0.01, str(round(v, 2)) + '%', ha='center', va='bottom')
                plt.savefig('ResNet_Comparison.png')
                plt.show()

            else:
                img = plt.imread(img_path)
                plt.imshow(img)
                plt.axis('off')
                plt.show()
                
        except Exception as e:
            print(f"Error: {str(e)}")

    def Inference_click(self):
        try:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = models.resnet50()
            model.fc = nn.Sequential(
                nn.Linear(2048, 1),
                nn.Sigmoid()
            )
            model = model.to(device)
            model.load_state_dict(torch.load('./model/ResNet_model.pth'))
            
            model.eval()

            transform = transforms.Compose([
                transforms.Resize((244, 244)),
                transforms.ToTensor()
            ])
            img = Image.open(self.image)
            img = transform(img).unsqueeze(0)
            img = img.to(device)

            with torch.no_grad():
                output = model(img)

            rounded_output = round(output.item(), 3)
            print(rounded_output)
            
            if(output.item() < 0.5):
                self.ui.label_q5.setText("Prediction = Cat")
                self.ui.label_q5.adjustSize()
            else:
                self.ui.label_q5.setText("Prediction = Dog")
                self.ui.label_q5.adjustSize()
    
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == '__main__':  
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())
