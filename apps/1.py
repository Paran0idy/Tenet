import torch

a = torch.tensor([1., 2.], requires_grad = True)
b = torch.tensor([1., 2.], requires_grad = True)

c = a + b;
c.sum().backward();

print(a.grad)