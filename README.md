# Sobolev Training with Pytorch
Small scale replication of Sobolev Training for NNs.

# Overview

You can use the code by importing `SobolevLoss` from `sobolev.py`. In order to use it, checkout the example in `main.py`. The general guideline for distillation is:

```python
from sobolev import SobolevLoss

teacher = Net()
student = Net()
loss = SobolevLoss(loss=nn.MSELoss(), weight=1.0, order=2)

# compute the gradients of teacher and student

sobolev = loss(student.parameters(), teacher.parameters())

# At this point, the parameters' gradients of student look like:
# s.grad = s.original_grad + s.grad.grad
# where s.grad.grad comes from the Sobolov loss

```

Remarks: 

* Make sure that your teacher is well-trained.
* It works well towards the end of distillation.
* Instead of `student.parameters()` and `teacher.parameters()` you can pass an iterable of parameters whose nth order gradients have been computed.
* Theoretically should work for higher order, but I didn't test it.

## Benchmark results

The results obtained by distilling a LeNet-teacher (converged) into a LeNet-student with the same random architecture. The results are in the form `train / test` at the 100th epoch of training.

Vanilla    | Sobolev
-----------|------------
1.2 / 1.19 | 0.56 / 0.64
0.94 / 0.9 | 0.8 / 0.82
0.7 / 0.72 | 0.7 / 0.72
n / a      | 2e-4 / 4e-4


