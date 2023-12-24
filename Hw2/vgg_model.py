import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm #progress bar
from torch.optim.lr_scheduler import StepLR

def train():
    try:
        if not os.path.exists("./model/best_model.pth") or not os.path.exists("loss_acc.png"):        
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(device)

            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()
            ])

            train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
            test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
            print("Train Dataset Size:", len(train_dataset))
            print("Test Dataset Size:", len(test_dataset))
            batch_size = 64
            trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            
            sample_inputs, sample_labels = next(iter(trainloader))
            print("Sample Input Shape:", sample_inputs.shape)
            print("Sample Label Shape:", sample_labels.shape)
            model = models.vgg19_bn(num_classes=10)
            model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
            model = model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.00005)
            # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

            num_epochs = 30
            train_loss_history = []
            train_acc_history = []
            val_loss_history = []
            val_acc_history = []

            # early stop
            best_val_acc = 0.0
            # epochs_without_improvement = 0
            # early_stopping_patience = 5 
            
            scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

            print("Start Training Model")
            for epoch in range(num_epochs):
                trainloader = tqdm(trainloader)

                # for train
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                for data in trainloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                train_loss_history.append(running_loss / len(trainloader))
                train_acc_history.append(100 * correct / total)

                scheduler.step()
                
                # for val
                model.eval()
                val_running_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for data in testloader:
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_running_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                current_val_acc = 100 * val_correct / val_total
                val_loss_history.append(val_running_loss / len(testloader))
                val_acc_history.append(100 * val_correct / val_total)

                if epoch == 0 or current_val_acc > best_val_acc:
                    torch.save(model.state_dict(), './model/best_model.pth')
                    best_val_acc = current_val_acc
                #     epochs_without_improvement = 0
                # else:
                #     epochs_without_improvement += 1

                print(f'Epoch [{epoch + 1}/{num_epochs}]')
                print(f'Training Loss: {train_loss_history[-1]:.4f}, Training Accuracy: {train_acc_history[-1]:.2f}%')
                print(f'Validation Loss: {val_loss_history[-1]:.4f}, Validation Accuracy: {val_acc_history[-1]:.2f}%')

                # If verification accuracy not improve for multiple consecutive epochs, stop training.
                # if epochs_without_improvement >= early_stopping_patience:
                #     print("Early stopping, no improvement in validation accuracy.")
                #     break

            _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

            # 第一个子图
            ax1.plot(train_loss_history, label='train_loss')
            ax1.plot(val_loss_history, label='val_loss')
            ax1.set_ylabel('loss')
            ax1.legend()
            ax1.set_title('Loss')

            # 第二个子图
            ax2.plot(train_acc_history, label='train_acc')
            ax2.plot(val_acc_history, label='val_acc')
            ax2.set_xlabel('epoch')
            ax2.set_ylabel('accuracy(%)')
            ax2.legend()
            ax2.set_title('Accuracy')

            # 调整子图之间的间距，使得标题和 x 轴标签不重叠
            plt.tight_layout()

            plt.savefig('loss_acc.png')
            plt.show()

            print("Training completed and model saved.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    train()