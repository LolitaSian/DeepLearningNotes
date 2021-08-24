import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0  # w就是θ


def hypothesis(x):
    return x * w


def loss(x, y):
    h = hypothesis(x)
    return (h - y) ** 2


def gradient(x, y):
    return 2 * x * (x * w - y)


print('Predict (before training)', 4, hypothesis(4))
for epoch in range(100):
    print(epoch, " \t", round(w, 5), end="\t")
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w = w - 0.01 * grad
        l = loss(x, y)
    print(round(w, 5))

print('Predict (after training)', 4, hypothesis(4))
