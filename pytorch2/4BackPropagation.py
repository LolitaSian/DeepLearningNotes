import torch

x_data = [1, 2, 3]
y_data = [2, 4, 6]

w = torch.tensor([1.0], requires_grad=True)

print(f"predict before training, y = {w.item()}*x")
# .item() 当Tensor中是单个值的时候将其取出

for epoch in range(50):
    for x, y in zip(x_data, y_data):
        y_pred = x * w
        loss = (y_pred - y) ** 2
        # 上边两步构建计算图，每次计算都会创建新的计算图
        loss.backward()
        # 这一步时候前边的计算图会释放，进行反向传播
        # 会将计算链路上所有需要梯度的地方都把梯度求出来，求完之后存到对应的张量中

        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data     # 这样不会建立计算图
        w.grad.data.zero_()     # 将梯度清零
    print("progress:", epoch, loss.item())
print(f"predict after training, y = {w.item()}*x")
