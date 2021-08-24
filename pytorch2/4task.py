# 实现y = w1x^2 + w2x + b
import torch

x_data = torch.tensor([1, 2, 3])
y_data = torch.tensor([2, 4, 6])

w1 = torch.tensor([1.0], requires_grad=True)
w2 = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)
learning_rate = 0.01

print(f"before training: y(4) = {w1.item()*4**2+w2.item()*4+b.item()}")

for epoch in range(10000):
    for x, y in zip(x_data, y_data):
        y_pred = w1 * (x ** 2) + w2 * x + b
        loss = (y_pred - y) ** 2
        loss.backward()

        with torch.no_grad():
            w1 -= learning_rate*w1.grad
            w2 -= learning_rate*w2.grad
            b -= learning_rate*b.grad

            w1.grad = None
            w2.grad = None
            b.grad = None
    if epoch % 100 == 0:
        print("loss=", loss.item())
print(f"y={w1.item()}x^2+{w2.item()}x+{b.item()}")
print(f"after training: y(4) = {w1.item()*4**2+w2.item()*4+b.item()}")

