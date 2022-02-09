这篇笔记是关于序列模型的，是吴恩达深度学习进阶课程的笔记，视频链接：[序列模型 - 网易云课堂 (163.com)](https://mooc.study.163.com/learn/2001280005?tid=2403042002#/learn/announce)

---

# 0 数学符号

本节介绍用到的数学符号

$x^{(i)}$：第i个时序序列样本

$x^{<i>}$：时序序列中第i个元素

$y^{(i)}$：第i个样本的输出

$y^{<i>}$：输出中的第i个元素

$T_y^{(i)}$：第i个输出序列的长度

$T_x^{(i)}$：第i个输入序列的长度

![](https://github.com/LolitaSian/DeepLearningNotes/raw/master/AndrewNg/image/2-1.png)

比如上边，红色框是$x^{(2)}$即第二个样本；其中$T_x^{(2)}=22$，即第二个样本的句子长度为22；绿色框是$x^{(1)<17>}$，即第一个样本的第17个元素。

处理序列的时候将其转化为one-hot编码。即一般采用一个词典，在对应位置将出现的字母设置为1。

课程中选用长

比如上边的$x^{(2)}$，有22个向量，每个向量的长度为10000。

对于单词表中没出现的使用$<unk>$标记。

---

# 1 Recurrent Neural Network

对于时序数列来说全连接神经网络有诸多缺点。

比如使用`hello world !`进行预测。

如果使用全连接神经网络，使用`hello`的时候输出`world`，使用`world`的时候输出`！`。但是会存在一个问题，每一项都和前一项没关系。语法根据前边的语境进行预测。你可以强行在`world`的时候加上一个`hello`，但是这样输入的长度就发生改变了。

![](https://github.com/LolitaSian/DeepLearningNotes/raw/master/AndrewNg/image/2-3.png)

因此在时序序列中使用循环神经网络（Recurrent Neural Network，RNN）。使用下面两幅图均可以表示循环神经网络。初始化时候给一个激活值$a^{<0>}$，一般初始化为0向量，当然也可以使用其他初始化方法。

$$
\begin{aligned}
&a^{<0>}=\overrightarrow{0} \\
&a^{<1>}=g_1\left(\omega_{a a} a^{<0>}+\omega_{a x} x^{\langle 1\rangle}+b_{a}\right) \\
&\hat{y}^{<1>}=g_2\left(\omega_{y a} a^{<1>}+b_{y}\right)
\end{aligned}
$$

比如进行一步简单的计算，初始化激活值（隐状态）为零向量；第一步往下传递的激活值就相当于第0步的激活值乘aa之间的权重加上xa之间的权重乘输入加上激活偏置单元；第一步的输出就是由第一步的激活值乘ay之间的权重再加上输出偏置单元。注意$w$是每层共享的，也就是说整个RNN不同时间步之间用的是同一个$w$。至于隐状态$a$激活函数，一般使用ReLu或者tanh，输出$\hat{y}$的激活值一般使用sigmoid。

![](https://github.com/LolitaSian/DeepLearningNotes/raw/master/AndrewNg/image/2-2.png)

通过上一步，我们可以归纳出通式：

$$
\begin{aligned}
&a^{<t>}=g\left(W_{a a} a^{<t>}+W_{a x} x^{<t>}+b_{a}\right) \\
&\hat{y}^{<t>}=g\left(W_{y a} a^{<t>}+b_{y}\right)
\end{aligned}
$$

上述式子由于矩阵计算的原因可以进行合并，化简为：

$$
\begin{aligned}
&a^{<t>}=g_1\left(w_{a}\left[a^{<t-1>}, x^{<t>}\right]+b_{a}\right)
\\
&\hat{y}^{<t>}=g_2\left(W_{y} a^{<t>}+b_{y}\right)
\end{aligned}
$$

$\hat y$和$y$之间的损失函数使用交叉熵损失：

$$
\mathcal{L}^{<t>}\left(\hat{y}^{<t>}, y^{<t>}\right)=-y^{(t)} \log \hat{y}^{\langle t\rangle}-\left(1-y^{<t>}\right) \log \left(1-\hat{y}^{<t>}\right)
$$

整体的loss：

$$
\mathcal{L} = \sum_{t=1}^{T_x}\mathcal{L} ^{<t>}
$$