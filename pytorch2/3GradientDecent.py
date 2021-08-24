import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


def hypothesis(x):
    return x * w


def cost(X, Y):
    cost = 0
    for x, y in zip(X, Y):
        h = hypothesis(x)
        cost += (h - y) ** 2
    return cost / len(X)


def gradient(X, Y):
    grad = 0;
    for x, y in zip(X, Y):
        grad += 2 * x * (x * w - y)
    return grad / len(X)


e = []
loss_lis = []

print('Predict (brfore training)', 4, hypothesis(4))
for epoch in range(88):
    cos_val = cost(x_data, y_data)
    loss_lis.append(cos_val)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val
    e.append(epoch)
    print('Epoch:', epoch, 'w=', round(w, 3), 'loss=', round(cos_val, 3))

print('Predict (after training)', 4, hypothesis(4))

plt.plot(e, loss_lis)
plt.xlabel('w')
plt.ylabel('cost')
plt.show()
