#!/usr/bin/env python3

import torch.nn as nn


class SobolevLoss(nn.Module):
    def __init__(self, loss=None, weight=1.0, order=2):
        super(SobolevLoss, self).__init__()
        self.order = order
        if loss is None:
            loss = nn.MSELoss()
        self.loss = loss
        self.weight = weight

    def forward(self, student_params, teacher_params):
        loss = 0.0
        for s, t in zip(student_params, teacher_params):
            s_grad = s
            t_grad = t
            for i in range(self.order - 1):
                s_grad = s_grad.grad
                t_grad = t_grad.grad
            s_grad.volatile = False
            t_grad.volatile = False
            s_grad.requires_grad = True
            t_grad.requires_grad = False
            sobolev = self.weight * self.loss(s_grad, t_grad)
            loss += sobolev
            sobolev.backward()
            s.grad.data += s_grad.grad.data
            s_grad.grad = None
            s_grad.volatile = True
            t_grad.volatile = True
            s_grad.requires_grad = False
        return sobolev
