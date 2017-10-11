#!/usr/bin/env python3

import torch as th
import torchvision as tv
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable as V
from torchvision import transforms

from lenet import LeNet


teacher = LeNet()
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

trainset = tv.datasets.CIFAR10(root='./data',
                               train=True,
                               download=True,
                               transform=transform_train)
trainloader = th.utils.data.DataLoader(trainset,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=2)

testset = tv.datasets.CIFAR10(root='./data',
                              train=False,
                              download=True,
                              transform=transform_test)
testloader = th.utils.data.DataLoader(testset,
                                      batch_size=100,
                                      shuffle=False,
                                      num_workers=2)

label_loss = nn.CrossEntropyLoss()
t_opt = optim.SGD(teacher.parameters(),
                  lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(t_opt, milestones=[100, 150])

# Train
for epoch in range(100):
    epoch_teacher = 0.0
    scheduler.step()
    for X, y in trainloader:
        X, y = V(X.cuda()), V(y.cuda())
        t_preds = teacher(X)
        t_loss = label_loss(t_preds, y)
        t_opt.zero_grad()
        t_loss.backward()
        t_opt.step()
        epoch_teacher += t_loss.data[0]
    print('*' * 20, 'Epoch ', epoch, '*' * 20)
    print('teacher_loss:', epoch_teacher / len(trainloader))
    print('\n')

# Test
test_teacher = 0.0
for X, y in testloader:
    X, y = V(X.cuda()), V(y.cuda())
    t_preds = teacher(X)
    t_loss = label_loss(t_preds, y)
    t_opt.zero_grad()
    test_teacher += t_loss.data[0]
print('*' * 20, 'Test Stats', '*' * 20)
print('teacher_loss:', test_teacher / len(testloader))
print('\n')

th.save(teacher.state_dict(), './teacher.pth')
