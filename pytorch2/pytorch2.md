梯度下降曲线可能并不平滑，可以加上个指数加权均值。

$c_0、c_1……c_n$变为$c_0' = c_0、c_i' = \beta c_i + (1-\beta)c'_{i-1}$



实际中用的更多的是随机梯度下降。

因为梯度下降可能会遇到鞍点问题，就是到了相对平滑的一点（局部最优点）之后就停滞不前了。

![image-20210925102734039](E:\Code\DeepLearning\image\3-1.png)

所以在实际生活中为了解决这个问题通常使用随机梯度下降（stochastic gradient descent）代替梯度下降（gradient descent），就是用单个样本的loss function来代替整个样本的cost function。

将$cost = \frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}$换成$loss = (h_{\theta}\left(x\right)-y)$

$\theta = \theta - \alpha \frac{\partial cost}{\partial\theta} = \theta -   \frac{\alpha}{2m} \sum^m_{i=1}(h_{\theta}(x)-y)x $

就变为：$\theta = \theta - \alpha \frac{\partial loss}{\partial\theta} = \theta -  2\alpha(h_{\theta}(x)-y)x $

使用随机梯度下降，由于每个样本都带有噪声，所以因为这些噪声的存在很有可能推动梯度下降跨越鞍点。性能比梯度下降高。



但是梯度下降相当于所有样本并行执行的，而随机梯度下降是对哦每个样本依次进行，每次的θ都是继承上一个的θ，时间复杂度较大。

现在取一个这种方法叫Batch，传统的Batch指的是梯度下降，所以标准叫法叫mini-batch。就是将数据总体分组，进行批量的随机梯度下降，对每组数据进行随机梯度下降。

反向传播就是挨个元素逆向链式求导的过程。

链式求导：![image-20211005104202109](E:\Code\DeepLearning\image\image-20211005104202109.png)

