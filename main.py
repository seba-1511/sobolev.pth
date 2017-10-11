#!/usr/bin/env python3

import torch as th
import torchvision as tv
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable as V
from torchvision import transforms

from lenet import LeNet
from sobolev import SobolevLoss

USE_SOBOLEV = False

student = LeNet()
teacher = LeNet()
teacher.load_state_dict(th.load('teacher.pth'))
student = student.cuda()
teacher = teacher.cuda()

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = tv.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = th.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = tv.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = th.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

label_loss = nn.CrossEntropyLoss()
distillation_loss = nn.MSELoss()
sobolev = SobolevLoss(weight=0.1)
s_opt = optim.SGD(
    student.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
t_opt = optim.SGD(
    teacher.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# Train
for epoch in range(100):
    epoch_distillation = 0.0
    epoch_teacher = 0.0
    epoch_student = 0.0
    epoch_sobolev = 0.0
    for X, y in trainloader:
        X, y = V(X.cuda()), V(y.cuda())
        s_preds = student(X)
        t_preds = teacher(X)
        s_loss = distillation_loss(s_preds, t_preds.detach())
        t_loss = label_loss(t_preds, y)
        s_opt.zero_grad()
        t_opt.zero_grad()
        s_loss.backward()
        t_loss.backward()
        if USE_SOBOLEV:
            sobolev_loss = sobolev(student.parameters(), teacher.parameters())
        s_opt.step()
        epoch_student += label_loss(s_preds, y).data[0]
        epoch_distillation += s_loss.data[0]
        epoch_teacher += t_loss.data[0]
        if USE_SOBOLEV:
            epoch_sobolev += sobolev_loss.data[0]
    print('*' * 20, 'Epoch ', epoch, '*' * 20)
    print('distillation_loss:', epoch_distillation / len(trainloader))
    print('student_loss: ', epoch_student / len(trainloader))
    print('teacher_loss:', epoch_teacher / len(trainloader))
    print('sobolev_loss: ', epoch_sobolev / len(trainloader))
    print('\n')

# Test
test_distillation = 0.0
test_teacher = 0.0
test_student = 0.0
test_sobolev = 0.0
for X, y in testloader:
    X, y = V(X.cuda()), V(y.cuda())
    s_preds = student(X)
    t_preds = teacher(X)
    s_loss = distillation_loss(s_preds, t_preds.detach())
    t_loss = label_loss(t_preds, y)
    s_opt.zero_grad()
    t_opt.zero_grad()
    s_loss.backward()
    t_loss.backward()
    if USE_SOBOLEV:
        sobolev_loss = sobolev(student.parameters(), teacher.parameters())
    test_student += label_loss(s_preds, y).data[0]
    test_distillation += s_loss.data[0]
    test_teacher += t_loss.data[0]
    if USE_SOBOLEV:
        test_sobolev += sobolev_loss.data[0]
print('*' * 20, 'Test Stats', '*' * 20)
print('distillation_loss:', test_distillation / len(testloader))
print('student_loss: ', test_student / len(testloader))
print('teacher_loss:', test_teacher / len(testloader))
print('sobolev_loss: ', test_sobolev / len(testloader))
print('\n')

th.save(student.state_dict(), './student.pth')
