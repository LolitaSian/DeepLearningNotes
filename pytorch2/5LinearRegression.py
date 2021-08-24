# 1.prepare dataset 数据集
# 2. design model using class y_pred
# 3. 使用pytorch 构建loss和optimizer
# 4. training cycle

# 使用mini-batch的时候注意xy一定要是矩阵

import torch

x_data = torch.tensor([1.0, 2.0, 3.0])
y_data = torch.tensor([2.0, 4.0, 6.0])


# pytorch 中不用人工求导了，只要考虑构造计算图即可
# 构造计算图最后loss一定要是标量，向量无法使用backward
# sum(loss) 再考虑要不要除n

class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
        # nn.Linear是一个类，包含weight和bias两部分。类后边加括号就是构造一个对象。nn是neural network

    def forward(self, x):
        y_pred = self.linear(x)
        # 函数后边加括号就是实现一个可调用的
        return y_pred

# 用Module实现的可以自己完成backward
# 如果有的计算pytorch无法实现，但是可以由pytorch的基本运算构成，那你可以将其封装为Module
# 如果你自用pytorch的计算图效率不如自定义高，那可以从function里边继承，需要自己实现反向传播

module = LinearRegression()

# 构造损失函数和优化器

criterion = torch.nn.Module(size_averge = False)
optimizer = torch.optim.SGD(module.parameters(),lr=0.01)

'''
class torch.nn.Linear(in_features, out_features, bias=True)
对输入数据做线性变换：y=Ax+b
参数：

in_features - 每个输入样本的大小
out_features - 每个输出样本的大小
bias - 若设置为False，这层不会学习偏置。默认值：True
形状：

输入: (N,in_features)
输出： (N,out_features)
变量：

weight -形状为(out_features x in_features)的模块中可学习的权值
bias -形状为(out_features)的模块中可学习的偏置
例子：

>>> m = nn.Linear(20, 30)
>>> input = autograd.Variable(torch.randn(128, 20))
>>> output = m(input)
>>> print(output.size())
'''