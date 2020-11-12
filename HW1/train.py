import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import time
import os
import PIL.Image as Image
from IPython.display import display

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

train_dir = 'data/training/'
test_dir = 'data/testing/'

train_tfms = transforms.Compose([transforms.Resize((400, 400)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(15),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = torchvision.datasets.ImageFolder(root=train_dir, transform = train_tfms)
trainloader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle=True)
cla_dic=dataset.class_to_idx

test_tfms = transforms.Compose([transforms.Resize((400, 400)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


dataset2 = torchvision.datasets.ImageFolder(root=test_dir, transform = test_tfms)
testloader = torch.utils.data.DataLoader(dataset2, batch_size = 32, shuffle=False)

def eval_model(model):
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data
            #images = images.to(device).half() # uncomment for half precision model
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100.0 * correct / total
    print('Accuracy of the network on the test images: %d %%' % (
        test_acc))
    return test_acc

def train_model(model, criterion, optimizer, n_epochs = 5):

    losses = []
    accuracies = []
    test_accuracies = []
    # set the model to train mode initially
    model.train()
    for epoch in range(n_epochs):
        since = time.time()
        running_loss = 0.0
        running_correct = 0.0
        best_acc = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs and assign them to cuda
            inputs, labels = data
            #inputs = inputs.to(device).half() # uncomment for half precision model
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # calculate the loss/acc later
            running_loss += loss.item()
            running_correct += (labels==predicted).sum().item()

        epoch_duration = time.time()-since
        epoch_loss = running_loss/len(trainloader)
        epoch_acc = 100/32*running_correct/len(trainloader)
        print("Epoch %s, duration: %d s, loss: %.4f, acc: %.4f" % (epoch+1, epoch_duration, epoch_loss, epoch_acc))

        losses.append(epoch_loss)
        accuracies.append(epoch_acc)


        print('Saving best model')
        state = {
            'net': model.state_dict(),
            'acc': epoch_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        src='./checkpoint/res34_'+str(epoch)+'.pth'
        torch.save(state, src)


        # re-set the model to train mode after validating
        model.train()
        since = time.time()
    print('Finished Training')
    return model, losses, accuracies


model_ft = models.resnet34(pretrained=True)
num_ftrs = model_ft.fc.in_features

# replace the last fc layer with an untrained one (requires grad by default)
model_ft.fc = nn.Linear(num_ftrs, 196)
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

model_ft, training_losses, training_accs = train_model(model_ft, criterion, optimizer, n_epochs=12)


# def find_classes(dir):
#     classes = os.listdir(dir)
#     classes.sort()
#     class_to_idx = {classes[i]: i for i in range(len(classes))}
#     return classes, class_to_idx
# classes, c_to_idx = find_classes('data/training/')

model_ft.eval()

testPIL_tfms = transforms.Compose([transforms.Resize((400, 400)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_tfms = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((400, 400)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

id =[]
pre = []
with open('gg.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')

    writer.writerow(['id','label'])
    for root, dirs, files in os.walk('data/test_img', topdown=False):
        for name in files:
            t=[]
            img=os.path.join(root, name)
            image = Image.open(img)
            if len(np.shape(image)) != 3:
                image = cv2.imread(img)
                image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
                image = test_tfms(image).float()
            else:
                image = testPIL_tfms(image).float()
            image = torch.autograd.Variable(image, requires_grad=True)
            image = image.unsqueeze(0)
            image = image.to(device)
            output = model_ft(image)
            conf, predicted = torch.max(output.data, 1)
            t.append(name[:-4])
            for cla, id in cla_dic.items():
                if id==predicted:
                    print(cla)
                    t.append(cla)
                    break;
            writer.writerow(t)
