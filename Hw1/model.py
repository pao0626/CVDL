import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
from tqdm import tqdm #progress bar

def train():
    try:
        if not os.path.exists("best_model.pth") or not os.path.exists("loss_acc.png"):
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(device)
            model = models.vgg19_bn(num_classes=10)
            model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

            num_epochs = 50
            train_loss_history = []
            train_acc_history = []
            val_loss_history = []
            val_acc_history = []

            #early stop
            best_val_acc = 0.0
            epochs_without_improvement = 0
            early_stopping_patience = 5 
            
            # Learning rate scheduler
            # step_size = 10  
            # gamma = 0.1     
            # scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            
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

                # scheduler.step()
                
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
                    torch.save(model.state_dict(), 'best_model.pth')
                    best_val_acc = current_val_acc
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                print(f'Epoch [{epoch + 1}/{num_epochs}]')
                print(f'Training Loss: {train_loss_history[-1]:.4f}, Training Accuracy: {train_acc_history[-1]:.2f}%')
                print(f'Validation Loss: {val_loss_history[-1]:.4f}, Validation Accuracy: {val_acc_history[-1]:.2f}%')

                # If verification accuracy not improve for multiple consecutive epochs, stop training.
                if epochs_without_improvement >= early_stopping_patience:
                    print("Early stopping, no improvement in validation accuracy.")
                    break

            plt.figure(figsize=(8, 12))
            plt.subplot(2, 1, 1)
            plt.plot(train_loss_history, label='train_loss')
            plt.plot(val_loss_history, label='val_loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            plt.title('Loss')

            plt.subplot(2, 1, 2)
            plt.plot(train_acc_history, label='train_acc')
            plt.plot(val_acc_history, label='val_acc')
            plt.xlabel('epoch')
            plt.ylabel('accuracy(%)')
            plt.legend()
            plt.title('Accuracy')

            plt.savefig('loss_acc.png')
            plt.show()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    train()