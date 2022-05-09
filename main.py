# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import math
import random
import os
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pointsampler import PointSampler, Normalize, RandRotation_z, RandomNoise, ToTensor
from util import pointnetloss, plot_confusion_matrix, pcshow, visualize_rotate, read_off
from check_data import read_off as read
#from check_data import read_off
from PointCloudData import PointCloudData
from network import PointNet
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def main():
    data_path = 'dataset/ModelNet10/'
    folders = sorted(os.listdir(data_path))
    classes = {folder: i for i, folder in enumerate(folders)};

    with open(os.path.join(data_path,"bed/train/bed_0006.off")) as f:
      verts, faces = read_off(f)

    # file = os.path.join(data_path,"sofa/train/sofa_0336.off")
    # verts, faces = read(file)
    # face = np.array(faces)
    # face = face[:,1:]
    i, j, k = np.array(faces).T
    x, y, z = np.array(verts).T
    visualize_rotate([go.Mesh3d(x=x, y=y, z=z, color='blue', opacity=0.50, i=i, j=j, k=k)]).show()
    pcshow(x,y,z)

    pointcloud = PointSampler(3000)((verts, faces))
    pcshow(*pointcloud.T)

    norm_pointcloud = Normalize()(pointcloud)
    pcshow(*norm_pointcloud.T)

    rot_pointcloud = RandRotation_z()(norm_pointcloud)
    noisy_rot_pointcloud = RandomNoise()(rot_pointcloud)

    pcshow(*noisy_rot_pointcloud.T)

    ToTensor()(noisy_rot_pointcloud)
    data_transforms = transforms.Compose([
        PointSampler(1024),
        Normalize(),
        RandRotation_z(),
        RandomNoise(),
        ToTensor()
    ])

    train_ds = PointCloudData(data_path, transform=data_transforms)
    valid_ds = PointCloudData(data_path, valid=True, folder='test', transform=data_transforms)
    inv_classes = {i: cat for cat, i in train_ds.classes.items()}
    batch_size = 128

    print('Train dataset size: ', len(train_ds))
    print('Valid dataset size: ', len(valid_ds))
    print('Number of classes: ', len(train_ds.classes))
    print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())
    print('Class: ', inv_classes[train_ds[0]['category']])
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=batch_size, shuffle=False)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    pointnet = PointNet()
    pointnet.to(device)

    lr = 0.001

    trrloss,valloss,valacc = train(pointnet = pointnet,lr = lr,epochs=40,save=True,train_loader = train_loader,valid_loader = valid_loader)
    plot(epoch_loss=trrloss,val_loss=valloss,val_acc=valacc)
    test(pointnet=pointnet, test_loader=valid_loader,classes=classes)

    print("Finished Program")

def train(pointnet, lr = 0.001, epochs=40,save=True, train_loader = None,valid_loader=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(pointnet.parameters(), lr=lr)
    epoch_loss = []
    val_loss = []
    val_acc = []
    correct = total = 0
    # batch_val_loss = 0
    best_acc = 0
    for epoch in range(epochs):
        print("Epoch : ", epoch)
        pointnet.train()
        running_loss = 0.0
        batch_loss = batch_val_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = pointnet(inputs.transpose(1, 2))
            ##loss calculation
            loss = pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                      (epoch + 1, i + 1, len(train_loader), running_loss / 10))
                running_loss = 0.0

        epoch_loss.append(batch_loss / len(train_loader))
        pointnet.eval()

        # validation
        if valid_loader:
            with torch.no_grad():
                batch_val_loss = 0.0
                for data in valid_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    optimizer.zero_grad()
                    outputs, m3x3, m64x64 = pointnet(inputs.transpose(1, 2))
                    _, predicted = torch.max(outputs.data, 1)
                    ##loss calculation
                    loss = pointnetloss(outputs, labels, m3x3, m64x64)
                    batch_val_loss += loss.item()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                val_loss.append(batch_val_loss / len(valid_loader))

            acc = 100. * correct / total
            print('Valid accuracy at epoch %d is : %d' % (epoch, acc) + '%')
            if acc > best_acc:
                print("changing best model at epoch:", epoch)
                best_acc = acc
                torch.save(pointnet.state_dict(), "best_model" + ".pth")
            val_acc.append(acc)

        # save the model
        if save:
            torch.save(pointnet.state_dict(), "save_" + str(epoch) + ".pth")
    np.savetxt("epoch_loss_40.csv", epoch_loss, delimiter=",")
    np.savetxt("val_loss_40.csv", val_loss, delimiter=",")
    np.savetxt("val_acc_40.csv", val_acc, delimiter=",")
    return epoch_loss,val_loss,val_acc

def plot(epoch_loss, val_loss, val_acc):


    plt.figure(1)
    #Plotting Values
    plt.plot(epoch_loss, label = "Training Loss")
    plt.plot(val_loss,label = "Validation Loss")
    plt.legend()
    plt.title("Loss Function")
    plt.savefig("Loss Function_40.png")
    #plt.show()
    plt.figure(2)
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend()
    plt.title("Validation Accuracy")
    plt.savefig("Validation Accuracy_40.png")

def test(pointnet, test_loader,classes):
    data_path = 'dataset/ModelNet10/'
    folders = sorted(os.listdir(data_path))
    classes = {folder: i for i, folder in enumerate(folders)};
    data_transforms = transforms.Compose([
        PointSampler(1024),
        Normalize(),
        RandRotation_z(),
        RandomNoise(),
        ToTensor()
    ])
    batch_size = 128
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    pointnet = PointNet()
    pointnet.to(device)
    pointnet.load_state_dict(torch.load('best_model.pth'))
    pointnet.eval();
    all_preds = []
    all_labels = []
    valid_ds = PointCloudData(data_path, valid=True, folder='test', transform=data_transforms)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=batch_size)
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            print('Batch [%4d / %4d]' % (i + 1, len(valid_loader)))
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            outputs, __, __ = pointnet(inputs.transpose(1, 2))
            _, preds = torch.max(outputs.data, 1)
            all_preds += list(preds.cpu().detach().numpy())
            all_labels += list(labels.cpu().detach().numpy())
    cm = confusion_matrix(all_labels, all_preds);
    print(cm)
    plot_confusion_matrix(cm, list(classes.keys()), normalize=True)

if __name__ == '__main__':
   main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
