# mechain 

# 1 监督学习和无监督学习

**Machine Learning definition**

Machine Learning: Field of study that gives computers the ability to learn without being explicitly programmed.

<p align="right">——Arthur Samuel（1959）</p>

Well-posed Learning Problem: A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by p, improves with experience E.

<p align="right">——Tom Mitchell（1998）</p>

一个适当的学习问题定义如下：计算机程序从经验E中学习，解决某一任务T，通过某一性能度量P测定在T上的表现随着经验E而提高。

## 1.1 监督学习 Supervised Learning

在监督学习中，给出一个数据集，且每个样本都指出正确答案，使用算法预测出“正确答案”。

Supervised Learning: We gave the algorithm a data set，in which the right answers were given. The task of the algorithm was to just produce more of these right answers.
- Regression: Predict continuous valued output（回归问题：预测一个连续值输出）
- Classification: Discrete valued output（分类问题：预测离散值输出）

### 1.1.1 回归问题"regression"

![image.png](E:\Code\DeepLearning\image\1-1.jpg)

已知数据元素学习时间和测试分数，简单拟合出一个一元函数。拟合出公式之后，随意给出一个数据即可预测出对应的另一个数据，比如已知考75分可以推测复习的3h左右，已知复习3h可以推测出考75分左右。



### 1.1.2 离散问题"classification"

![image.png](E:\Code\DeepLearning\image\1-2.png)
已知不同的身高体重和性别，图中明显看出由于身高体重不同而划分出的性别差异。之后给定一个身高体重数据便可以推测其性别。

**例题：**

You're running a company and you want to develop learning algorithms to address each of two problems 

Problem 1：You have a large inventory of identical items. You want to predict how many of these items will sell over the next 3 months 

Problem 2：Youd like software to examine individual customer accounts and for each account decide if it has been hacked /compromised

> 答案： Treat problem 1 as a regression problem, problem 2 as a classification problem.

p1预测销量，肯定是对历史销量数据进行拟合从而得到一个曲线模型而进行预测，因此是回归问题。

p2检测账号，数据是离散的，安全或者不安全，因此是分类问题。

## 1.2 无监督学习 Unsupervised Learning

Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

在无监督学习中，给出数据集，不进行数据区分。程序自动对输入的数据进行分类或分群，以寻找数据的模型和规律。

In Unsupervised Learning, the data that doesn't have any labels,or that all has the same labels or really no labels.
- clustering algorithm 聚类

  将数据进行聚类，作为一个曾经的生物学学生，我第一反应就是聚类在生信中应用及其广泛。生信中的聚类，给定DNA序列，就可以自动划分为不同的物种。下图为一个热图。

  ![image.png](E:\Code\DeepLearning\image\1-3.png)

- Non-clustering The "Cocktail Party Algorithm"



**例题：**

Of the following examples, which would you address using an unsupervised learning algorithm？

- Given email labeled as spam/ not spam learn a spam filter

- Given a set of news articles found on the web, group them into set of articles about the same story 

- Given a database of customer data, automatically discover market segments and group customers into different market segments 

- Given a dataset of patients diagnosed as either having diabetes or not, learn to classify new patients as having diabetes or not.

> 答案：
> - Given a set of news articles found on the web, group them into set of articles about the same story 
> - Given a database of customer data, automatically discover market segments and group customers into different market segments 

1.区分是否是垃圾邮件：监督学习-离散

4.区分病人是否患糖尿病：监督学习-离散

# 2 线性代数

## 2.1 向量和矩阵 matrices and vectors
- A matrix is a rectangular array of numbers written between square brackets. 
    - 矩阵是写在方括号中的数字矩形阵列。
    - Dimension of matrix： number of rows x number of columns
        - 矩阵的维数：行×列
        - $\R^{m×n}$
    - Matrix Elements（entries of matrix）（矩阵元素）：$A_{ij}$ = i,j entry in the ith row, jth column
    - 同济教材：![image.png](E:\Code\DeepLearning\image\1-4.png)

- A vector is a matrix that has only 1 columns.
    - 向量是只有一列的矩阵（狭义上的向量，是我们认知范围里的列向量，注意区分）
    - Dimension of vector: number of rows
        - 矩阵的维数：行
        - $\R^{m}$
    - element of the vectory（向量元素）：$y_{i}$
    - 同济教材定义：![image.png](E:\Code\DeepLearning\image\1-5.png)

        

**<font color=red>注意</font>**

这里涉及到一个**1-indexed 0-indexed**的问题。在计算机中为了方便处理通常是下标从0开始，但是学习习惯中我们通常是从1开始。本学习笔记中提到的矩阵和向量，如果不是做出特殊标注，都默认是1-indexed写法。

![image.png](E:\Code\DeepLearning\image\1-6.png)

---

## 2.2 加法 matrix addition

把两个矩阵对应位置上的元素相加即可。

矩阵的维数要相同（行数列数要相同），相同位置的两数相加。

$$
\begin{bmatrix}
1 & 0 \\
2 & 5 \\
3 & 1
\end{bmatrix}
+ 
\begin{bmatrix}
4 & 0.5 \\
2 & 5 \\
0 & 1
\end{bmatrix}

=
\begin{bmatrix}
5 & 0.5 \\
4 & 10 \\
3 & 2
\end{bmatrix}
$$

行列不相同的话不能相加。


$$
\begin{bmatrix}
1 & 0 \\
2 & 5 \\
3 & 1
\end{bmatrix}
+ 
\begin{bmatrix}
4 & 0.5 \\
2 & 5 
\end{bmatrix}
=Error
$$

## 2.3 标量乘法 Scalar Multiplication

就是数乘矩阵，把矩阵对应位置上的元素乘以给出的数即可。

$$
3 \times \begin{bmatrix}
1 & 0 \\
2 & 5 \\
3 & 1
\end{bmatrix}=
\begin{bmatrix}
3 & 0 \\
6 & 15 \\
9 & 3
\end{bmatrix}
$$

$$
\left[\begin{array}{ll}
4 & 0 \\
6 & 3
\end{array}\right] / 4=\frac{1}{4}\left[\begin{array}{cc}
4 & 0 \\
6 & 3
\end{array}\right]=\left[\begin{array}{cc}
1 & 0 \\
\frac{3}{2} & \frac{3}{4}
\end{array}\right]
$$

## 2.4 矩阵向量乘法
矩阵和向量相乘。
矩阵的列数要和向量的行数一样，即$\R_A^{m×n} \times \R_y^{n}$，结果为一个向量，行数和矩阵相同$\R_{result}^{m}$
矩阵k行的第i个元素和向量的第i个元素相乘然后求和，得出一个新数及结果向量的第k个元素。

m × n matrix（m rows n columns）  ×  n x 1 matrix（n-dimensional vector） = m-dimensional vector

![image.png](E:\Code\DeepLearning\image\1-7.png)

$$
\left[\begin{array}{ll}
1 & 3 \\
4 & 0 \\
2 & 1
\end{array}\right] \times {\left[\begin{array}{l}
1 \\
5
\end{array}\right]} = \left[\begin{array}{c}
16 \\
4 \\
7
\end{array}\right]
$$

$$
\begin{array}{l}
1 \times 1+3 \times 5=16 \\
4 \times 1+0 \times 5=4 \\
2 \times 1+1 \times 5=-7
\end{array}
$$

**应用**：
再回到预测房价那个问题，现在假设知道一组房屋大小的数据，并且拟合出了放假曲线为$h_{\theta}(x)=-40+0.25 x$

House sizes：
- 2104
- 1416
- 1534
- 852 

快速得出结论就可以使用向量矩阵相乘。
因为房屋价格为x，带入公式

$$
\left[\begin{array}{ll}1 & 2104 \\ 1 & 1416 \\ 1 & 1534 \\ 1 & 852\end{array}\right] 
\times
\left[\begin{array}{l}-40 \\ 0.25\end{array}\right]=
\left[\begin{array}{l}2104 \\ 1416  \\ ... \\ ...\end{array}\right]
$$
矩阵中第一列为1，因为有个常数项-40，要保留-40，所以第一列为1，第二列为房价x，向量第二行是0.25为x的系数。


## 矩阵乘法
前一个矩阵的列数和后一个矩阵的行数相同。即$\R_A^{m×n} \times \R_B^{n×s}$，结果矩阵行数和前一个相同弄，列数和后一个相同，$\R_{result}^{m×s}$

前一个矩阵i行和后一个矩阵的的第j列元素一次相乘后求和，得出一个新数为结果矩阵第i行第j列。

![image.png](E:\Code\DeepLearning\image\1-9.png)

$$
\left[\begin{array}{ll}
1 & 3 \\
2 & 5
\end{array}\right]\left[\begin{array}{ll}
0 & 1 \\
3 & 2
\end{array}\right] = \left[\begin{array}{l}
1 \times 0+3 \times 3 & 1 \times 1+3 \times 2 \\
2 \times 0+5 \times 3 & 2 \times 1+5 \times 2
\end{array}\right]
= 
\left[\begin{array}{ll}
9 & 4 \\
15 & 12
\end{array}\right]
$$

**应用**：从矩阵乘向量那里房价的例子已经熟悉了怎么应用，而矩阵相乘则可以同时计算好几条拟合曲线。

![image.png](E:\Code\DeepLearning\image\1-10.png)

**注意**：

矩阵相乘没有交换律，$A \times B \neq B \times A$

矩阵相乘可以用结合律，$A \times B \times C = A \times （B \times C）$


## 2.5 逆 matrix inverse

matrix inverse：If A is an m × m matrix（square matrix）, and if it has an inverse $AA^{-1}= A^{-1}A=I$
可逆矩阵的定义：A是方阵，并且A和A逆的乘积=A逆乘A=单位矩阵I

**备注**：还记得单位矩阵吗，就是除了对角线元素是1，其余元素都是0。

## 2.6 转置 Matrix transpose
转置就是行变列，列变行。A的转置记作$A^T$

# 3 单变量线性回归 Linear Regression with One Variable

首先继续说房价预测这个问题。

已知数据有房屋面积以及对应的价格。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/23610ff2e31644a8963ee75ee4c731a4~tplv-k3u1fbpfcp-watermark.image)

经过算法处理，这些数据更适合拟合成一条一元函数的直线。得出结果如下：

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a4956d362a5b446a860a30b0f86a8ae0~tplv-k3u1fbpfcp-watermark.image)

由上图可以预测，如果有一套房子的面积为1250feet²，那预测其价格应该在$220 000左右。

---

## 3.1 一些概念

In supervised learning, we have a data set and this data set is called a training set.

在监督学习中我们有一个数据集被称为训练集。

**Notation：**

- $m$=Number of training examples 训练样本的数量
- $x's$="input"variable/features 输入变量
- $y's$="output"variable/"target"variable 输出变量，即预测的结果
- $(x,y)$=one training example 一个训练样本
- $(x^{(i)},y^{(i)})$=ith training example 第i个训练样本
- Hypothesis：$h_{\theta}(x)$. 拟合出的假设函数，有时简写为$h(x)$

---

就上边房价的例子中，下边给出一个具体数据。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7674e8e90458439aa6741f06a771827d~tplv-k3u1fbpfcp-watermark.image)

预测房价训练集：左侧一列数据是房屋面积，右侧一列数据是房屋价格。

- 房屋面积及其对应的价格组成的数据集就是训练集。
- 假如训练集中房屋面积及其价格数据对一共有41对，那这个训练集中$m=41$。
- $x^{(3)}=1534$
- $y^{(1)}=460$
- 由于是一元线性回归，因此假设函数是$h_{\theta}(x)=\theta_0+\theta_1x$
- ……


How supervised learning algorithm works：

监督学习流程：

We saw that with the training set like our training set of housing prices and we feed that to our learning algorithm.

首先我们给学习算法提供训练集，比如给房价训练集。

Is the job of a learning algorithm to then output a function, which by convention is usually denoted lowercase h, and h stands for hypothesis.

学习算法输出一个函数，用常用小写h表示。h即假设函数。

And what the job of the hypothesis is a function that takes as input the size of a house. And it tries to output the estimated value ofy for the corresponding house. So h is a function that maps from x's to y's.

假设函数就是把房屋大小作为输入，并输出预测的房屋价格。因此假设函数就是引导x得到y的函数。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/3bfeac61b32f47b89145cea02eed6ff3~tplv-k3u1fbpfcp-watermark.image)

In univariate linear regression, the hypothesis is $h_{\theta}(x)=\theta_0+\theta_1x$. These theta i's are what I called the paraneters of the model. 

单变量线性回归中假设函数为$h_{\theta}(x)=\theta_0+\theta_1x$，其中$\theta_i$为模型参数

How to choose these two parameter values, theta zero and theta one. With different choices of parameters theta zero and theta one we get different hypotheses, different hypothesis functions.

选择不同的$\theta_i$会产生不同的假设函数。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b99a721a54fa42fa85261494e2d512f9~tplv-k3u1fbpfcp-watermark.image)

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/385fb521afbc413c9b54e4b7f909594e~tplv-k3u1fbpfcp-watermark.image)
依旧对于这个房价预测问题，如何选参数$\theta$？
毕竟选择不同的值对结果有很大的影响。

现在给出一个图：

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/eba560e04beb4b03a89699b228d334e3~tplv-k3u1fbpfcp-watermark.image)

给这个图找出适宜的$\theta_0$和$\theta_1$，使求出的预测函数能更好的和给出的这几个数据点拟合。


首先要解决一个最小化问题。尽量减小假设输出结果和真实房价 差的平方。

minimize：$\sum_{i=1}^{m}\left(h_{0}\left(x^{(i)}\right)-y^{(i)}\right)^{2}$

求取所有的样本点的$y$与$h_{0}(x)$差的平方和。

上一个公式我们可以得到平均误差：$\frac 1 m \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}$

而实际上我们所要求的$\theta_0$和$\theta_1$应该满足使$\frac 1 {2m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}$得到最小值。

对于上图给出的图像，使用的**代价函数就是**：
$J\left(\theta_{0}, \theta_1 \right)=\frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}$

$\stackrel{misimize}{\theta_{0}, \theta_{1}}$  $J\left(\theta_{0}, \theta_{1}\right)$

Cost function is also called the squared error function or sometimes called the square error cost function

代价函数也被称作平方误差函数有时也被称为平方误差代价函数。

**平方误差代价函数只是代价函数的一种，是解决回归问题最常用的手段之一。**



**举个栗子**

为了使代价函数J更好的可视化，我们先使用一个简化的假设函数。

假设现在的预测函数为$h_{\theta}(x)=\theta_{1} x$，即$\theta_0=0$。

那代价函数就可以简化为$J\left(\theta_{1}\right)=\frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}$

$\underset{\theta_{1}}{\operatorname{minimize}} J\left(\theta_{1}\right)$

假设现在我们的训练集如下，给出一个图：

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e55ea3a90d4e4901a70ae2f854b30120~tplv-k3u1fbpfcp-watermark.image)

$当\theta_{1}=1时候$

$h_{\theta}(1)-y_{1}=0\\
h_{\theta}(2)-y_{2}=0\\
h_{\theta}(3)-y_{3}=0$

$J\left(\theta_{1}\right)=\frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}=0$

$当\theta_{1}=0.5时候$

$h_{\theta}(1)-y_{1}=-0.5\\
h_{\theta}(2)-y_{2}=-1\\
h_{\theta}(3)-y_{3}=-1.5$

$J\left(\theta_{1}\right)=\frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}=\frac{1}{2 \times 3}(0.25+1+2.25)=0.583333$

……

依次计算下去，最后得到代价函数J是个一元二次函数：

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/5bb24e1b2cff44ccaf14b54c61607940~tplv-k3u1fbpfcp-watermark.image)

所以在这个例子中$当\theta_{1}=1时候$代价函数取得最小值，拟合函数最精确。

现在看一下没简化过的。

给出数据如图所示：

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/279791e7b0524438bf4a8d92d38a9cab~tplv-k3u1fbpfcp-watermark.image)

但是此时代价函数并不是二维图形了。因为多了一个参数$\theta_0$，所以变为三维图像，例如下图：

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/8fe5f76a8a4a4051ab4626dc37ce0857~tplv-k3u1fbpfcp-watermark.image)

但是为了便于表示，之后不会继续使用三维图像，而是使用等高图像，类似于下图：

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/de5437aafb454feda5cc47beb9bb0ee3~tplv-k3u1fbpfcp-watermark.image)

同一条线上的点代表虽然$\theta_0$和$\theta_1$不同，但是他们代价函数值相同。 

## 3.2 Gradient Descent 梯度下降

**梯度下降是什么？**

对于一个代价函数$J(\theta_0,\theta_1,\theta_2,…)$，我们想要使其最小化：
- 初始时候令$\theta_i$都为0
- 逐渐改变$\theta_i$的值
- 直到得到最小值或者局部最小值

用图解释一下梯度下降，对于代价函数$J(\theta_0,\theta_1)$，如下一个三维图形，你先想象成这是一篇山区。你从任意一个点开始下山，每次环顾四周，找一个同样步长的最低点向下移一步，直到最低点。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c2208af492904f1596ad615fc62a4bfb~tplv-k3u1fbpfcp-watermark.image)

但是如果你起始点不同，可能会找到另一个最低点。也就是说梯度下降算法起始点不同找到的最优解可能是不同的。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c18bc8f3ca324375ac5d080feece1ac6~tplv-k3u1fbpfcp-watermark.image)

**梯度下降算法定义：**

repeat until convergence \{ 

$$ \theta_{j}:=\theta_{j}-\alpha \frac{\partial}{\partial \theta_{j}} J\left(\theta_{0}, \theta_{1}\right) \quad(\text { for } j=0 \text { and } j=1) $$

}

Correct: Simultaneous update
$$
\begin{aligned}
&\text { temp0 }:=\theta_{0}-\alpha \frac{\partial}{\partial \theta_{0}} J\left(\theta_{0}, \theta_{1}\right) \\
&\text { temp } 1:=\theta_{1}-\alpha \frac{\partial}{\partial \theta_{1}} J\left(\theta_{0}, \theta_{1}\right) \\
&\theta_{0}:=\text { temp } 0 \\
&\theta_{1}:=\text { temp } 1
\end{aligned}
$$

- $:=$是赋值符号，比如$a:=b$就是给a赋上b的值
- $=$是判断符号，比如$a=b$如果两个值相等返回true，否则返回false
- $\alpha$学习速率，控制梯度下降的步长，如果$\alpha$较大那下降的就会很迅速
- $\frac{\partial}{\partial \theta_{j}} J\left(\theta_{0}, \theta_{1}\right)$求导
- 对于上述公式正确的执行步骤：
    - 使用$\theta_0,\theta_1$更新temp
    - 使用temp更新$\theta_0,\theta_1$
    

**解释一下$\alpha \frac{\partial}{\partial \theta_{j}} J\left(\theta_{0}, \theta_{1}\right)$有什么用**

- $\alpha$前边说了，它的大小控制梯度下降的步长大小，可以称之为**学习速率**，其取值永远为正数。
- 对于$\frac{\partial}{\partial \theta_{j}} J\left(\theta_{0}, \theta_{1}\right)$求导公式我们依旧先从最简单的例子开始。

**先解释导数部分**

假设现在的代价函数是$J(\theta_1)$其中$\theta_1 \in R$，那他的图像应该是个一元二次函数。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/dece6483dfee49c181738b9696abd642~tplv-k3u1fbpfcp-watermark.image)

现在从右边蓝色这个点进行梯度下降，那就会得到结果$\theta_{1}:=\theta_{1}-\alpha \frac{d}{d \theta_{1}} J\left(\theta_{1}\right)$

**你可能会注意到原来公式中的∂变为d了，**

> **符号 d 与符号 ∂ 的区别是什么？**
> $\partial$ 指偏微分。对于一个多元函数 $f(x, y), \frac{\partial f}{\partial x}$ 相当于固定 $y$ 变量, 对 $x$ 进行求导。
> $\mathrm{d}$ 指全微分。对于一个多元函数 $f(x, y), \mathrm{d} f=\frac{\partial f}{\partial x} \mathrm{~d} x+\frac{\partial f}{\partial y} \mathrm{~d} y$ 。
> 在复合函数中, 全微分的概念更加有用。设多元函数 $f(x(t), y(t))$, 则 $\frac{\mathrm{d} f}{\mathrm{~d} t}=\frac{\partial f}{\partial x} \frac{\mathrm{d} x}{\mathrm{~d} t}+\frac{\partial f}{\partial y} \frac{\mathrm{d} y}{\mathrm{~d} t}$ 。
> 对于单变量函数, 偏微分和全微分没有区别, 所以均用 $\frac{\mathrm{d}}{\mathrm{d} x}$ 表示。


我们都知道在上图中导数的几何意义是过蓝色点切线的斜率，而图中可知这个斜率一定是个正数。再加上前边的条件$\alpha$学习速率一定是正数，那原式就可以理解为$\theta_{1}:=\theta_{1}-一个正数$，那$\theta_{1}$一定会向左移，<s>从而使$J(\theta_{1})$减小</s>。

*为什么我把“从而使$J(\theta_{1})$减小”这句话划掉了呢，因为如果α取值过大，那它可能移动的太大而越过了最小值，而在最小值左右两侧反复横跳。*

同理看下图，如果初始$\theta_{1}$在最低点的左侧，那现在该点的斜率是个负数，此时原式就可以理解为$\theta_{1}:=\theta_{1}-一个负数$，导致$\theta_{1}$增大，使其右移。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/99ecaffb4a3f46f38ee33cbecda8e650~tplv-k3u1fbpfcp-watermark.image)



**再讨论一下α**如果过大或者过小会有什么影响。

If α is too small, gradient descent can be slow.
If a is too large, gradient descent can overshoot the minimum It may fail to converge or even diverge.

如果α太小，那梯度下降的速度就会很慢；如果太大，可能直接越过最小值，在最小值左右两边反复横跳，离得原来越远，看下边图2。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/3006d7c96c414cad9e33032698dc74da~tplv-k3u1fbpfcp-watermark.image)

> **提问！**
>
> 如果此时$\theta_{1}$已经处于一个局部最优点，那会如何移动？
>
> （？我也不知道为啥吴恩达老师会提这种问题，学过高中数学的就会回答这个问题了吧![66DABEE6.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/55fc22de995b4017a587f2452977a6cb~tplv-k3u1fbpfcp-watermark.image)）
>
> 最低点切线是水平的，斜率为0，所以公式就是$\theta_{1}:=\theta_{1}-\alpha \times 0 = \theta_1$ ，到达局部最优解之后就停在原地不动了。


Gradient descent can converge to a local minimum even with the learning rate α fixed.

（α取值合适的前提下）虽然α是固定值，但是梯度下降却可以取得局部最小值。

As we approach a local minimum, gradient descent will automatically take smaller steps.
So, no need to decrease α over time.

因为梯度下降越接近局部最小值，下降步长会越小，也就是说下降步长会随着逐渐靠近局部最优解而逐渐减小，因此我们无需改变α的值就可以顺利取到局部最小值。

看下图理解一下：

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f2a535f0b9114cd9bd7ac49d8ef1463a~tplv-k3u1fbpfcp-watermark.image)

从最右侧粉色的点开始进行梯度下降，下降以后到达绿色的点。
$\theta_{1}:=\theta_{1}-\alpha \times k_{pink}$

从绿色点进行梯度下降，$\theta_{1}:=\theta_{1}-\alpha \times k_{green}$

红色的点进行梯度下降：$\theta_{1}:=\theta_{1}-\alpha \times k_{red}$

**这三个点有什么区别？**

越接近最低点，曲线越平缓，下降点的斜率越小。

对于$\theta_{1}:=\theta_{1}-\alpha \frac{d}{d \theta_{1}} J\left(\theta_{1}\right)$，其中$\frac{d}{d \theta_{1}} J\left(\theta_{1}\right)$逐渐减小，会导致$\alpha \frac{d}{d \theta_{1}} J\left(\theta_{1}\right)$逐渐减小，也就是每次$\theta_{1}$减的值逐渐减小，使$\theta_{1}$移动逐渐减缓。



**既然我们已经弄懂了代价函数$J(\theta_1)$，那现在就回到$J(\theta_0,\theta_1)$。**

$\because J\left(\theta_{0}, \theta_{1}\right)=\frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(1)}\right)-y^{(i)}\right)^{2}$

$\therefore \frac{\partial}{\partial \theta_{j}} J\left(\theta_{0}, \theta_{1}\right)=\frac{\partial}{2 \theta_{j}} \cdot \frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}$

$\because h_{\theta}(x)=\theta_{0}+\theta_{1} x$

$\therefore \frac{\partial}{\partial \theta_{j}} J\left(\theta_{0}, \theta_{1}\right) =\frac{\partial}{2 \theta_{j}} \cdot \frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}=\frac{\partial}{\partial \theta_{j}} \frac{1}{2 m} \sum_{i=1}^{m}\left(\theta_{0}+\theta_{1} x^{(i)}-y^{(i)}\right)^{2}$

**所以此时接着求导**（求导过程在多变量里）：
$$
\begin{aligned}
&\theta_{0}: \frac{\partial}{\partial \theta_{0}} J\left(\theta_{0}, \theta_{1}\right)=\frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) \\
&\theta_{1}: \frac{\partial}{\partial \theta_{1}} J\left(\theta_{0}, \theta_{1}\right)=\frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) \cdot x^{(i)}
\end{aligned}
$$

推导过程写完了，得出结论，对于线性回归来说，梯度下降如下：

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/be51954aff15489584c291ffaccf71c5~tplv-k3u1fbpfcp-watermark.image)


还记得解释什么叫梯度下降时候使用的三维图吗（就是那个想象成下山那个图，不记得的往上翻一下。）
线性回归的代价函数一般不会那么复杂，always going to be a bow-shaped function. 碗状图。专业术语叫“convex function”。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/fbffa9e7d1b049fdbf5f0f841685c27e~tplv-k3u1fbpfcp-watermark.image)

这种图不仅看起来比那种图简单，实际操作也简单，没有局部最优解，只有全局最优解，也就是说不管你从哪里开始最后找到的最优解都是一样的。

再回到之前的例子，还是这个等高图，从点1开始进行梯度下降，最后找到最优解（此时已经找到代价函数的最小值）。每一步都对应一个回归函数图像，点1-4和最优解的回归函数都画出来了，肉眼可见的最后一张图拟合程度最高。

![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/57029619a097418c960a235d9eee477f~tplv-k3u1fbpfcp-watermark.image)

![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e2cf0cc6e00a4e219744fc25ef0d6add~tplv-k3u1fbpfcp-watermark.image)

**放一个详细点的最后一张图：**
![放一个详细点的最后一张图](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f4e7ed026a6f4b3ea26f1faec69ba567~tplv-k3u1fbpfcp-watermark.image)


## 3.3 Batch

这个梯度下降又称为“Batch” gradient Descent 

"Batch"：Each step of gradient descent uses all the training examples

指的是每一步都要遍历整个训练集。因为$\frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}$公式每一步都要对总体求和。

# 4 多变量线性回归 Multivariate linear regression

考虑一下之前的单变量线性回归，按照数学的说法，单变量线性回归是一元函数，那多变量线性回归就是多元函数。
之前例子是房屋面积及其对应的价格，拟合的曲线是已知房屋的面积对其价格进行预测。$h_{\theta}(x)=\theta_0+\theta_1x$，数学上一个标准的一元一次函数。而对于多变量线性回归，可以考虑影响房价的有多个因素，处理房屋大小还有楼层啊位置啊之类的……

依旧是预测房屋价格：
$$
\begin{array}{c|c|c|c|c}
\text { Size (feet²) } & \begin{array}{c}
\text { Number of } \\
\text { bedrooms }
\end{array} & \begin{array}{c}
\text { Number of } \\
\text { floors }
\end{array} & \begin{array}{c}
\text { Age of home } \\
\text { (years) }
\end{array} & \text { Price (\$1000) } \\
\hline 2104 & 5 & 1 & 45 & 460 \\
1416 & 3 & 2 & 40 & 232 \\
1534 & 3 & 2 & 30 & 315 \\
852 & 2 & 1 & 36 & 178 \\
\ldots & \ldots & \ldots & \ldots & \ldots
\end{array}
$$

Notation：
- n = number of features
    - 本例中n=4，有四个特点（面积、卧室数量、楼层、使用年数）
    - 注意区分m和n，m指的是训练样本数，有几行数据就有几个训练样本
- $x^{(i)}$=input（features）of $i_{th}$ training example
    - 指第几个训练样本，通俗说就是第几行数据
    - 比如$x^{(2)}$指的就是$x^{(2)}=\left[\begin{array}{c}1416 \\3 \\2 \\40\end{array}\right]$，将其作为偶一个四维向量看待。
- $x_{j}^{(i)}$=value of feature j in $i_{th}$ training example
    - 第i个训练样本中第j个特征量
    - 比如$x^{(2)}_3$=2
    
## 4.1 预测函数

Hypothesis：
$$
h_{\theta}(x)=\theta_{0}+\theta_{1} x_{1}+\theta_{2} x_{2}+\theta_{3} x_{3}+\theta_{4} x_{4}
$$

For convenience of notation, we always  

简化以后可以写为$h_{\theta}(x)=\theta_{0} x_{0}+\theta_{1} x_{1}+\cdots+\theta_{n} x_{n}=\theta^Tx$。对于这个预测函数，将其变量和系数都写成向量形式。

$$
x=\left[\begin{array}{l}
x_{0} \\
x_{1} \\
x_{2} \\
\vdots \\
x_{n}
\end{array}\right] \in \mathbb{R}^{n+1} \quad \theta=\left[\begin{array}{c}
\theta_{0} \\
\theta_{1} \\
\theta_{2} \\
\vdots \\
\theta_{n}
\end{array}\right] \in \mathbb{R}^{n+1}
$$

## 4.2 多元梯度下降算法
Hypothesis:

 define $x_0=1$
$$
 h_{\theta}(x)=\theta^{T} x=\theta_{0} x_{0}+\theta_{1} x_{1}+\theta_{2} x_{2}+\cdots+\theta_{n} x_{n} 
$$

Parameters：n+1-dimension vector
$$
\theta_{0}, \theta_{1}, \ldots, \theta_{n} 
$$

Cost function:
$$
\qquad J\left(\theta_{0}, \theta_{1}, \ldots, \theta_{n}\right)=\frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}
$$

Gradient descent:

$Repeat \{ \\$
$$
\theta_{j}:=\theta_{j}-\alpha \frac{\partial}{\partial \theta_{j}} J\left(\theta_{0}, \ldots, \theta_{n}\right)
$$
$\}$ (simultaneously update for every $j=0, \ldots, n$ )

对后边部分求导：
$$
\begin{aligned}
\frac{\partial J(\theta)}{\partial \theta_{j}} &=\frac{\alpha}{\partial \theta_{j}}\left(\frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}\right) \\
&=\frac{\partial}{\partial \theta_{j}}\left(\frac{1}{2 m} \sum_{i=1}^{m}\left(\theta_{0} x_{0}^{(i)}+\theta_{1} x_{1}^{(i)}+\cdots+\theta_{j} x_{j}^{(i)}+\cdots+\theta_{n} x_{n}^{(i)}\right)^{2}\right) \\
&=\frac{1}{2 m} \cdot \sum_{i=1}^{m} \frac{\partial}{\partial \theta_{k}}\left(\theta_{0} x_{0}^{(i)}+\theta_{1} x_{1}^{(i)}+\cdots+\theta_{j} x_{j}^{(i)}+\cdots+\theta_{n} x_{n}^{(i)}\right)^{2} \\
&=\frac{1}{2 m} \sum_{i=1}^{m} 2\left(\theta_{0} x_{0}^{(i)}+\cdots+\theta_{n} x_{n}^{(i)}\right) \cdot \frac{\partial}{\partial \theta_{j}}\left(\theta_{0} x_{0}^{(i)}+a_{1} x_{1}^{(i)}+\cdots+\theta_{j} x_{j}^{(i)}+\cdots+\theta_{n} x_{n}^{(i)}\right) \\
&=\frac{1}{m} \sum_{i=1}^{m}\left(\theta_{0} x_{0}^{(i)}+\cdots+\theta_{n} x_{n}^{(i)}\right) \cdot x_{k} . \\
&=\frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}-y^{(i)}\right) \cdot x_{k} .\right.
\end{aligned}
$$
**最后简化为**

$Repeat \{ \\$
$$
\theta_{j}:=\theta_{j}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}
$$
$\}$ (simultaneously update for every $j=0, \ldots, n$ )

对于上边这个梯度下降展开来看：
$$
\begin{array}{l}
\theta_{0}:=\theta_{0}-\alpha \frac{1}{m} \sum_{i=1}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{0}^{(i)} \\
\theta_{1}:=\theta_{1}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{1}^{(i)} \\
\theta_{2}:=\theta_{2}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{2}^{(i)}
\end{array}
$$

单变量线性回归的简化后公式长这样：
$$
\begin{aligned}
\theta_{0}: \frac{\partial}{\partial \theta_{0}} J\left(\theta_{0}, \theta_{1}\right) &=\frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) \\
\theta_{1}: \frac{\partial}{\partial \theta_{1}} J\left(\theta_{0}, \theta_{1}\right) &=\frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) \cdot x^{(i)}
\end{aligned}
$$
**对比之后发现多变量线性回归和单变量线性回归是完全一样的。**
- 因为$令x_0=1$，所以对$\theta_0$来说是完全一样的。
- 剩余的对$\theta_i$，只是因为多变量的缘故，所以给每个x多加了一个下标而已。
- 代价函数是对于整体样本而言的，而损失函数（loss function）是针对单个样本而言的。
- Mean Squared Error Loss是代价函数前边不/2

# 5 向量化

**举个简单栗子：**

$h_{\theta}(x)=\sum_{j=0}^{n} \theta_{j} x_{j}$

如果让你自己手算这个，你肯定会想，不就是个简单的一次函数吗，挨个带进去算就行了。但是如果“带进去算”这个方法让计算机来实现就会变成这样：

声明两个向量，遍历。

**Unvectorized implementation（没向量化）**

**C++：**
```c++
double prediction = 0.0; 
for(int j = 0; j<=n； j++)
{
    prediction += theta[j]*x[j];
}
```

**octave：**

```c
prediction =0.0； 
for j = 1:n+l
    prediction = prediction + theta(j) * x(j)
end
```

上边提到了“没向量化”。**什么是向量化呢？** 就是将你认知上的公式转化成矩阵和向量的乘法。这样计算机处理起来就会更迅速，也就是我前边提到的“更适合计算机的角度去处理问题”。


$h_{\theta}(x)=\sum_{j=0}^{n} \theta_{j} x_{j}$ 的向量化可以将其看做是$\theta^{T} x$而进行计算。

这样就转化为两个向量$\theta=\left[\begin{array}{l}\theta_{0} \\ \theta_{1} \\ \theta_{2} \\ ... \end{array}\right] \quad x=\left[\begin{array}{l}x_{0} \\ x_{1} \\ x_{2} \\ ...\end{array}\right]$的乘积了。

**Vectorized implementation（向量化之后）**

**octave：**
```c
prediction= theta' * x； 
```
**C++：**
```c++
double prediction = theta.transpose()*x;
```

向量化之后为什么比你自己用数组计算迅速呢？因为向量化之后可以更便捷的用到一些语言内置的库。比如上边octave中的`theta'`英文单引号 ' 表示求矩阵或向量的转置（[忘记的回去看octave语法](https://juejin.cn/post/6996500843130257438)）。C++中`theta.transpose()`也是用到了线性代数库的转置函数。

使用高级语言的库，更便捷了我们的操作。这些库函数都是各种计算机的大佬创造优化出来的，比我们自己写便捷千百倍。

用内置算法的好处：
- 速度更快
- 用更少代码实现
- 相比于你自己写的更不易出错
- 更好地配合硬件系统

**说了这么多，再来个栗子：**
$$
\begin{array}{l}
\theta_{j}:=\theta_{j}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}
    \end{array}  \text{        for all j}
$$

这个公式很熟悉，是梯度下降的公式。
在这个公式中x是数据矩阵，y是列向量。

对于多元线性回归：
$$
\begin{array}{l}
\theta_{0}:=\theta_{0}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{0}^{(i)} \\
\theta_{1}:=\theta_{1}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{1}^{(i)} \\
\theta_{2}:=\theta_{2}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{2}^{(i)}
\end{array}
\\...
$$

先简化一下梯度下降公式使其更容易编写代码：


$$
\theta:=\theta-\alpha \delta
$$

$$
\delta = \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x^{(i)}
$$

其中δ是个列向量：$$\delta=\left[\begin{array}{l}
\delta_{0} \\
\delta_{1} \\
\delta_{2} \\...
\end{array}\right]$$

而对于x来说，$$x^{(i)}=\left[\begin{array}{c}
x_{0}^{(i)} \\
x_{1}^{(i)} \\
x_{2}^{(i)} \\...
\end{array}\right]$$

$$
\delta = 数×(...)× x^{(i)}
$$

由此可是一定是 $\delta_{n \times 1} = 数×(...)× x_{1×n}^{(i)}$ 即$x_j^{(i)}$进行转置。

这个逻辑如果不向量化的话可能需要写好多循环才能完成。
我随手用c++写了一下，不保证对啊，你们大致看一下就行了，我也没运行这段代码。

```c++
for(int j=0;j<n;j++)
{
    //先计算出h(x)来
    for(int i=0;i<len;i++)
    {
        hx[i] += theta[i][j]*x[i][j];
    }
    //计算[h(x)-j]*x并求和
    for(int i=0;i<len;i++)
    {
        sum += (h[i] - y[i])*x[i][j]; 
    }
    //公式剩余部分
    theta[j] = theta[j] - alpha * (1/m) * sum;
}
```

但是如果你向量化以后就可以写为：
```c
% 假设现在是二元的
hx = X * theta;
theta(1) = theta(1) - alpha * (1/m) * sum((hx-y)*X(:,1:1))
theta(2) = theta(2) - alpha * (1/m) * sum((hx-y)*X(:,2:2))
%注意octave和C等其他语言不同，下标从1开始
```

```c
% 如果是n元的
hx = X * theta;
for i = 1:n
    theta(i) = theta(i) - alpha * (1/m) * sum((hx-y)*X(:,i:i))
    %注意octave和C等其他语言不同，下标从1开始
endfor
```

向量化之后怎么都好处理，但是如果不经过向量化，那你可能就要嵌套好多for循环了。所以要善用向量化减少工作量。

# 6 正规方程

这个正规方程是放在多元线性回归里边讲的。之前线性回归我们讲的是梯度下降，循环执行一个公式逐步下降，而正规方程与之相反，是直接对θ求最优解。基本上只需一步就可以完成。

**什么是正规方程？**

先举个简单的例子：

Intuition: 

If $1\mathrm{D}(\theta \in \R)$

$J(\theta)=a \theta^{2}+b \theta+c$

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/1d0c3ed636c441f9a7281db705d82802~tplv-k3u1fbpfcp-watermark.image)

现在假设θ只是一个实数，不是向量，函数J是关于θ的一个二次函数。 

对这个函数求最小值，怎么一步实现？只要你学过高中数学就会知道：求导。求出$\frac{\text d h(x)}{\text d x} = 0$那个x就是符合是函数最小化的值。

但是一般我们接触到的并不是这种函数，取值范围都是向量。在梯度下降中是循环执行对每一个θ求偏导，最后求出何时θ=0，那现在我们就可以直接求出等于0的这一步。


![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/60798b4fbaf84df3911b70e567917f33~tplv-k3u1fbpfcp-watermark.image)

现在我们有一个训练样本，在数据集中加上一列$x_0 = 1$把这个训练集变成一个系数矩阵：

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/2f700b33114849ea9daa811e88a7cdfd~tplv-k3u1fbpfcp-watermark.image)

$X=\left[\begin{array}{ccccc}1 & 2104 & 5 & 1 & 45 \\ 1 & 1416 & 3 & 2 & 40 \\ 1 & 1534 & 3 & 2 & 30 \\ 1 & 852 & 2 & 1 & 36\end{array}\right]$

同样把y列成一个向量：

$y=\left[\begin{array}{l}460 \\ 232 \\ 315 \\ 178\end{array}\right]$

矩阵X包含了所有的特征量，是一个m\*n+1的矩阵，y是一个m维矩阵。m是训练样本的数量。

现在只需要一步：$\theta=\left(X^{T} X\right)^{-1} X^{T} y$即可求出最优解。

Set theta to be equal to X transpose X inverse times X transpose y, this would give you the value of theta that minimizes your cost function.

特征量矩阵的转置乘自身，然后求逆，之后再乘特征量矩阵的转置，然后再乘y向量。

所以正规方程就是：

m examples $((x^1,y^1),...,(x^n,y^n))$
，n features.

假设现在我们的训练集有m个训练样本。一共有n个特征量。那特征量x的向量就是

$x = \begin{bmatrix} x^i_0  \\ x^i_1 \\ x^i_2\\ ...\\ x^i_n\end{bmatrix} \in \R^{n+1}$

而将x转化为矩阵X就变成

$X = \begin{bmatrix} ...(x^i_0)^T...  \\ ...(x^i_1)^T... \\ ...(x^i_2)^T...\\ ...\\ ...(x^i_n)^T...\end{bmatrix} \in  \R^{m \times n+1} $

而y则是：

$y = \begin{bmatrix} y^1  \\ y^2 \\ y^3 \\ ...\\ y^m \end{bmatrix} \in \R^m$

列出Xy以后:

$$
\theta=\left(X^{T} X\right)^{-1} X^{T} y
$$

在octave中只需要一句`pinv(X' * X) * x' * y`

并且这种方法也不需要进行特征量缩放。

**如果$X^{T} X$矩阵不可逆怎么办？**

其实在octave中，有两个求逆矩阵的方法，一个`pinv()`一个`inv()`。用前者，即使矩阵不可逆，你也可以得到卒子红正确的θ值。

> 可逆矩阵 AB = BA = I，对于矩阵A，能找到一个矩阵B与其相乘，使结果等于单位矩阵，那矩阵A就是可逆矩阵。

一般来说你遇到的不可逆矩阵有两情况：
1. 有多余的特征量
    比如给你
    
    $x_1 = size \quad in \quad feet^2  \\x_2 = size \quad in \quad m^2$
    一个面积单位是平方英尺，一个面积单位是平方米。这种情况下你可以舍弃一个特征量。
2. 特征量过多（m<=n）

    这种情况下删除某些特征量或者进行正则化。

    正则化之后会讲到。

    我有个不成熟的想法：为什么不可以将m循环一下，使其称为一个方阵哈哈哈哈哈，就比如$\begin{bmatrix} a_{11},a_{12},a_{13},a_{14},a_{15}\\a_{21},a_{22},a_{23},a_{24},a_{25}\\a_{31},a_{32},a_{33},a_{34},a_{35}\\a_{11},a_{12},a_{13},a_{14},a_{15}\\a_{21},a_{22},a_{23},a_{24},a_{25}\end{bmatrix}$

**对比正规方程和梯度下降**

| Gradient descent                           | Normal equation                                              |
| ------------------------------------------ | ------------------------------------------------------------ |
| Need to choose α <br>Needs many iterations | No need to choose α <br> Don't need to iterate               |
| Works well even when n is large.           | Need to compute$(X^TX)^{-1}$， Slow if n is very large<br>The normal equation method actually do not work for some more sophisticated learning algorithms. |


- 梯度下降：
    - 缺点：
        - 需要尝试多次选择一个合适的α
        - 需要进行多次迭代
    - 优点：
        - 当数据量很大的时候也依旧可用
- 正规方程：
    - 优点：
        - 不需要选择α
        - 不需要多迭代
        - 不先不要考虑取值范围进行缩放
    - 缺点：
        - 需要计算$(X^TX)^{-1}$而两个矩阵相乘，复杂度数量级是$O(n^3)$所以数据量比较大的时候运行会很慢
        - 对于一些复杂算法程无法使用
        

**总结**

数据量小的简单算法使用正规方程更迅速。数据量大或者算法更为复杂还是需要使用梯度下降。



# 7 逻辑回归 Logistic Regression

跟线性回归不同，在这里要预测的y是离散值。
用到的 Logistic Regression 算法是当今最流行最广泛使用的学习算法之一。

还记得监督学习和无监督学习的分类吗
Classification
Emai:Spam/ Not Spam？
Online Transactions:Fraudulent（Yes/No）？
Tumor:Malignant/ Benign

邮件：是否是垃圾邮件
网上交易：是否存在欺诈
肿瘤分类：良心恶性

记得的话就会知道上述都属于离散型监督学习。上述三个问题的共同之处：

y∈{0,1}

0：Negative Class"

1："Positive Class"


当然并不是所有的离散型问题都非黑即白只有两个结果，也有可能是y∈{0,1，3,...,n}可数有限多个。

## 举个例子

从简单的二类离散开始：

从肿瘤的大小预测其是良性还是恶性：

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ccc2ad6d9d6d47558dec02e350621d9e~tplv-k3u1fbpfcp-watermark.image)

如果我们还是用线性回归的方法拟合是无法生效的。比如像这样：

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6834921821074be7820e6b3108e156cf~tplv-k3u1fbpfcp-watermark.image)

现在我们对于这条拟合直线假定$h(x)>0.5$为恶性，反之为良性。你可能说这不是拟合的挺好的吗。

但是如果这样呢：

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/2d6bc3331c6f4a3385e2e00b7afe0a7e~tplv-k3u1fbpfcp-watermark.image)

就会有很多恶心的肿瘤被判断为良性。

所以说线性回归并不适合离散型问题。

## 逻辑回归

那对于离散回归怎么办呢。首先我们要确保:

$0 \leq h_{\theta}(x) \leq 1$

这样就不会出现线性回归那样的问题。线性回归中不管你怎么拟合。只要超出一定范围总会出现$h(x)>1$的情况。

那要如何改良呢？

这就借助到Sigmoid function又称Logistic function：

$g(z) = \frac{1}{1+e^{-z}}$

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/0b4e283655374ac9950f1242b8767ff7~tplv-k3u1fbpfcp-watermark.image)

再将线性回归的公式应用到Logistic函数中后得到：

$$
h_{\theta}(x) =g(\theta^Tx)= \frac{1}{1+e^{-\theta^Tx}}
$$

现在已经知道如何将$h_θ(x)$的值固定在0\~1之间，那对于输出结果怎么描述呢？

Estimated probability that y = 1 on input x。

描述输入x对于y = 1时结论的可能性。

比如y = 1代表恶性肿瘤，用户x输入得到的结果为0.7。

你不能说“恭喜你，你得恶性肿瘤了”，而是要说“你得恶性肿瘤的概率是70%”。

正规结果用概率的方式表示就是：
$$
h_θ(x) = P(y=1|x;θ)
$$

上面的式子我们还可以推出$$h_θ(x) = P(y=1|x;θ)+h_θ(x) = P(y=0|x;θ) = 1$$

## 决策边界

刚才上边说到描述结果，要说x的输入对于y=1时候的可能性。但是严格意义上来说并不是所有的都要说是对于y=1时候的可能性。

Suppose ：
- predict y=1 if $h_{\theta}(x) \geq 0.5$
- predict y=0 if $h_{\theta}(x) < 0.5$

一般来说，$h_{\theta}(x)$大于等于0.5就说相对于y=1的结论，小于0.5说相对于0的结论。

注意区分一下逻辑，比如x输入之后y=0.2，你不能说你有20%的几率是恶性肿瘤，而是说你有80%几率是良性肿瘤。

上边的说法还可以等价为

**Suppose ：**

- **predict y=1 if $\theta^Tx \geq 0$**
- **predict y=0 if $\theta^Tx < 0$**

因为上边的$g(z) = \frac{1}{1+e^{-z}}$的图像，z>0时候为正，z<0为负。而$h_{\theta}(x) =g(\theta^Tx)$，所以可以进行如上转换。

而**决策边界就是$\theta^Tx = 0$的时候**。

画个图像更直观的了解一下：

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6695d382a31e45fbb62ba07fd0f0cb44~tplv-k3u1fbpfcp-watermark.image)

上图假设我们已经找到预测函数$h_{\theta}(x) = \theta_0 + \theta_1x_1 + \theta_2x_2$，其中$θ =\begin{bmatrix} -3 \\ 1 \\ 1 \end{bmatrix}$，带入就是$h_{\theta}(x) = -3 + x_1 + x_2$

*你现在不用管预测函数是怎么出来的，文章下边会讲的*

其中$\theta^Tx = -3 + x_1 + x_2 =0$就是$x_1 + x_2 = 3$这条直线。这条直线就是决策边界。而这条线上边的红叉叉区域就是$ x_1 + x_2 > 3$的区域，被称为y=1区域。反之下边蓝圈圈就是y=0区域。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/1a4bac797076415da024eaaea30e8aa9~tplv-k3u1fbpfcp-watermark.image)

**The decision boundary is a property of the hypothesis.**

**决边界是假设函数的一个属性，跟数据无关。也就是说虽然我们需要数据集来确定θ的取值，但是一旦确定以后，我们的决策边界就确定了，剩下的就和数据集无关了，图像上也不一定非要把数据集可视化。**

现在来个更复杂的例子：

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/fe88a04c07774a388852e0f878982e88~tplv-k3u1fbpfcp-watermark.image)

对于这个图像我们的预测函数是$h_{\theta}(x)=g(\theta_{0}+\theta_{1} x_{1}+\theta_{2} x_{2}\left.+\theta_{3} x_{1}^{2}+\theta_{4} x_{2}^{2}\right)$
，其中$θ = \begin{bmatrix} -1 \\ 0 \\ 0 \\ 1 \\ 1 \end{bmatrix}$

y = 1 的时候就是$x_{1}^{2}+x_{2}^{2} \geq 1$

y = 0 的时候就是$x_{1}^{2}+x_{2}^{2} < 1$

这个例子中决策边界就是$x_{1}^{2}+x_{2}^{2}  = 1$

# 如何拟合Logistic Regression 

Training set: 

$\left\{\left(x^{(1)}, y^{(1)}\right),\left(x^{(2)}, y^{(2)}\right), \cdots,\left(x^{(m)}, y^{(m)}\right)\right\}$

m examples $\quad x \in\left[\begin{array}{c}x_{0} \\ x_{1} \\ \ldots \\ x_{n}\end{array}\right] \quad x_{0}=1, y \in\{0,1\}$

$h_{\theta}(x)=\frac{1}{1+e^{-\theta^{T} x}}$

How to choose parameters $\theta$ ?

先说我们的训练集，是m个点。跟之前一样把x列成一个矩阵，并且加上一个$x_0 = 1$，$\theta^Tx=\theta_0X_0+ \theta_1x_1+...+\theta_nx_n$

那如何选择θ？

要计算θ首先要找到代价函数。

## 代价函数

还记得线性回归的代价函数吗？

$J\left(\theta\right)=\frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}$

换一种写法，用
$$Cost(h_{\theta}(x), y)=\frac{1}{2}(h_{\theta}(x)-y)^{2}$$

$Cost(h_{\theta}(x), y)$代表预测函数和实际值的差，代价函数J则表示对所有训练样本和预测函数的代价求和之后取平均值。

所以线性回归代价函数还可以写为：

$J\left(\theta\right)=\frac{1}{m} \sum_{i=1}^{m}Cost(h_{\theta}(x), y)$


在离散值里如果你继续使用这个代价函数，那画出图像以后结果是个非凸函数。长下边这样，也就是说你没办法顺利的找到最优解。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/828fb390fc0c49bc8cad1c9905571401~tplv-k3u1fbpfcp-watermark.image)

所以我们需要找一个凸函数来作为逻辑回归的代价函数：

Logistic regression cost function
$$
\operatorname{Cost}\left(h_{\theta}(x), y\right)=\left\{\begin{aligned}
-\log \left(h_{\theta}(x)\right) & \text { if } y=1 \\
-\log \left(1-h_{\theta}(x)\right) & \text { if } y=0
\end{aligned}\right.
$$

如果y=1图像如下：

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/3078df478d4b4a4ea58708da66d599fb~tplv-k3u1fbpfcp-watermark.image)

我们知道h(x)的取值范围在0\~1之间，结合log图像的特点，我们就可理解上述图像是怎么出现的了。

这个函数有一些很有趣的优秀性质：

$Cost = 0:\quad if \quad y=1,h_θ(x)=1$

当代价函数等于0的时候，也就是我们的假设函数$h_θ(x)=1$，即我们预测是恶性肿瘤，并且实际数据$y=1$，即病人确实是恶性肿瘤。也就是说代价函数0，我们预测正确。

$But \quad as \quad h_θ(x)→0,Cost→∞$ 

但是如果我们的假设函数**趋于**0的时候，代价函数却趋于正无穷。

Captures intuition that if $h_θ(x)=0$
（predict $P(y=1|x;θ)=0$）， but y=1， we'll penalize learning algorithm by a very large cost.

如果假设函数**等于**0相当于说对于y=1即病人的恶性肿瘤这件事，我们预测的概率是0。

如果放到现实生活中就相当于我们对病人说：你完全不可能是恶性肿瘤！现实中如果肿瘤确实是恶性，那医生的话就是重大医疗事故。医生要付出很大代价。但是在这个函数中只能趋于0，不会等于0。

再看看y=0的情况：

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/851d43303117417885fea5efffe162a7~tplv-k3u1fbpfcp-watermark.image)

$Cost = 0:\quad if \quad y=0,h_θ(x)=0$

代价函数0，y=0代表病人是良性肿瘤，我们预测函数$h_θ(x)=0$说明我们预测的肿瘤是良性，预测完全正确，所以代价函数0.

$But \quad as \quad h_θ(x)→0,Cost→∞$ 

Captures intuition that if $h_θ(x)=1$
（predict $P(y=0|x;θ)=1$）

y=0代表病人良性肿瘤，但是我们预测函数**等于**1的话，说明我们预测的是恶性肿瘤，告诉病人：你不可能是良性肿瘤。在生活中万一人家是良性肿瘤，医生这句话又会造成不必要恐慌……

所以这个函数的有趣又优良之处在于不能把话说的太满。



上边已经说到代价函数了：

Logistic regression cost function 

$J(\theta)=\frac{1}{m} \sum_{i=1}^{m} \operatorname{Cost}\left(h_{\theta}\left(x^{(i)}\right), y^{(i)}\right)$

$\operatorname{Cost}\left(h_{\theta}(x), y\right)=\left\{\begin{aligned}-\log \left(h_{\theta}(x)\right) & \text { if } y=1 \\-\log \left(1-h_{\theta}(x)\right) & \text { if } y=0 \end{aligned}\right.$

Note: $y=0$ or 1 always

现在我们将其简化：

$$
cost(h_{θ}(x), y)=-y \log (h_{θ}(x))-(1-y) \log (1-h_{\theta}(x))
$$

为什么简化之后就是这个式子？

直接带数进上边式子就可以了。
- y = 1:$cost(h_{θ}(x), 1)=-1 \log (h_{θ}(x))-(1-1) \log (1-h_{\theta}(x)) = - \log (h_{θ}(x))$
- y = 0:$cost(h_{θ}(x), 0)=0 \log (h_{θ}(x))-(1-0) \log (1-h_{\theta}(x)) = -\log (1-h_{\theta}(x))$

所以就是直接将上边两个式子合并成一个式子，不需要再来判断y的取值情况分类进行了。

**现在我们就可以写出逻辑回归的代价函数：**

**Logistic regression cost function**
$$
\begin{aligned}
J(\theta) &=\frac{1}{m} \sum_{i=1}^{m} \operatorname{Cost}\left(h_{\theta}\left(x^{(i)}\right), y^{(i)}\right) \\
&=-\frac{1}{m}\left[\sum_{i=1}^{m} y^{(i)} \log h_{\theta}\left(x^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right]
\end{aligned}
$$


根据这个代价函数，我们要找到$\min_{\theta} J(\theta)$，即让代价函数取得最小值的参数θ。  

那就又要进行梯度下降了。

## 梯度下降

Gradient Descent
$$
J(\theta)=-\frac{1}{m}\left[\sum_{i=1}^{m} y^{(i)} \log h_{\theta}\left(x^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right]
$$
Want $\min _{\theta} J(\theta):$
Repeat \{
$$
\theta_{j}:=\theta_{j}-\alpha \frac{\partial}{\partial \theta_{j}} J(\theta)
$$
\}
(simultaneously update all $\theta_{j}$ )

对上述的$J(\theta)$求偏导之后带入梯度下降公式，最后得到如下形式：

Gradient Descent
$$
J(\theta)=-\frac{1}{m}\left[\sum_{i=1}^{m} y^{(i)} \log h_{\theta}\left(x^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right]
$$
Want $\min _{\theta} J(\theta):$
Repeat \{
$$
\theta_{j}:=\theta_{j}-\alpha \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}
$$
$$
\}
$$
(simultaneously update all $\theta_{j}$ )

emmmmm现在你有没有发现一个问题，逻辑回归的梯度下降公式和线性回归梯度下降公式**看起来**一模一样。

为什么加粗了**看起来**，因为毕竟他们的预测函数不同。

$\begin{aligned}
&h_{\theta}(x)=\theta^{T} x \\
&h_{\theta}(x)=\frac{1}{1+e^{-\theta^{T} x}}
\end{aligned}$

所以说虽然看起来一样，但实际还是天差地别。

## 高级优化

讲线性回归的时候我们说找最佳的θ除了使用梯度下降，还可以使用正规方程。对于逻辑回归也是如此，我们除了使用梯度下降，还可以使用别的算法。（下边列举的这三种都需要代价函数和代价函数的偏导。区别是跟梯度下降迭代部分不一样）
- Conjugate gradient 
- BFGS 
- L-BFGS

Advantages：
- No need to manually pick α 
- Often faster than gradient descent Disadvantages：
- More complex

这三个算法相对于梯度下降，有点事不需要选择学习速率α并且比梯度下降更快。缺点是算法更为复杂。

> 当然更为复杂这个根本不是缺点，因为你不需要知道原理，直接用别人已经写好的就行了。吴恩达老师原话“我用了十多年了，然而我前几年才搞清楚他们的一些细节。”
>
> 想起来一个好笑的梗：贵的东西只有一个缺点那就是贵，但这不是东西的缺点，是我的缺点。

octave和MATLAB有这种库，直接用就行了。至于你使用C，C++，python之类的，那你可能要多试几个库才能找到实现比较好的。

**如何应用到逻辑回归中？**

theta $=\left[\begin{array}{c}\theta_{0} \\ \theta_{1} \\ \vdots \\ \theta_{n}\end{array}\right]$

function [jVal, gradient] $=$ costFunction (theta)

$\mathrm{jVal}=[$ code to compute $J(\theta)] ;$

gradient $(1)=\left[\right.$ code to compute $\left.\frac{\partial}{\partial \theta_{0}} J(\theta)\right]$

gradient $(2)=\left[\right.$ code to compute $\left.\frac{\partial}{\partial \theta_{1}} J(\theta)\right]$

...

gradient $(n+1)=\left[\right.$ code to compute $\left.\frac{\partial}{\partial \theta_{n}} J(\theta) \quad\right] $

举个例子：

$$
\begin{aligned}
&\text { Example: }\\
&\theta=\left[\begin{array}{l}
\theta_{1} \\
\theta_{2}
\end{array}\right]\\
&J(\theta)=\left(\theta_{1}-5\right)^{2}+\left(\theta_{2}-5\right)^{2}\\
&\frac{\partial}{\partial \theta_{1}} J(\theta)=2\left(\theta_{1}-5\right)\\
&\frac{\partial}{\partial \theta_{2}} J(\theta)=2\left(\theta_{2}-5\right)
\end{aligned}
$$

现在有一个含两个参数的实例，肉眼可见当代价函数最小（等于0）的时候两个θ都等于5。好嘛，现在是为了学算法，我们假装不知道结果。

```c
function [j,gradient] = costFunction(theta)
  %代价函数J
  j = (theta(1)-5)^2 + (theta(2)-5)^2;
  
  gradient = zeros(2,1);
  
  %偏导
  gradient(1) = 2*(theta(1)-5);
  gradient(2) = 2*(theta(2)-5);
  
 endfunction
```
costFunction函数有两个返回值。一个是代价函数J，一个是对J求偏导，用于存储结果的向量。

```c
%octave中输入：
options = optimset ('GradObj','on','MaxIter','100');
initheta = zeros(2,1);
[Theta,J,Flag] = fminunc (@costFunction,initheta,options)
```
- optimset：进行设置，其中四个参数：
    - GradObj：设置梯度目标参数
    - 确认上一步设置开启
    - 最大迭代次数
    - 设置最大迭代次数值
- fminunc：octave的无约束最小化函数，需要传入三个参数
    - 你自己写的函数，前边一定要加@
    - 你预设的θ，必须是一个二维及以上的向量，如果是一个实数，该函数会失效。
    - 对该函数的设置


![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/99926d017cfc4ce4bb989e89914fd182~tplv-k3u1fbpfcp-watermark.image)

最后运行结果长这样，Theta存储最终的代价函数最小化时θ的取值。J表示代价函数最优解，Flag = true表示已经收敛。

# 多类分类 Multiclass classification

什么是多类分类？

比如你又一个邮件要对他自动分类为：工作、朋友、家庭、其他。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/5df2ca3a9f6a42dc899efd1735ac5994~tplv-k3u1fbpfcp-watermark.image)

对这种分类怎么处理？

Using an idea called one-versus-all classification, we can then take this and make it work for muti-class classification, as well.

利用一对多的分类思想，我们同样可以把二类分类的思想应用在多类别分类上。

Here's how one-versus-alll classiffication works. And, this is also sometimes called one-versus-rest.

现在介绍一下敌对多分类方法（一对余）：

Let's say, we have a training set

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b1c8f02b5f0241eca1033bebd18a8d49~tplv-k3u1fbpfcp-watermark.image)

用三角表示1，方块表示2，叉表示3

现在将其改为三个独立的二元分类：

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b8686ede6f494a099893d5cae0412c6e~tplv-k3u1fbpfcp-watermark.image)

$h_{\theta}^{(i)}(x)=P(y=i \mid x ; \theta) \quad(i=1,2,3)$

在这里表i=1的时候就是三角形做正类的时候。上述i个分类器对其中每一种情况都进行了训练。


**在一对多分类中**

One-vs-all
Train a logistic regression classifier $h_{\theta}^{(i)}(x)$ for each class $i$ to predict the probability that $y=i$

On a new input $x$, to make a prediction, pick the class $i$ that maximizes
$$
\max _{i} h_{\theta}^{(i)}(x)
$$

我们获得一个逻辑回归分类器，$h_{\theta}^{(i)}(x)$预测i类别在y=i时候的概率。最后做出预测，我们给出一个新的输入值x，想获得预测结果，我们要做的就是在每个分类器中运行输入x，最后选择预测函数最大的类别，就是我们要预测的结果y。

# 8 正则化

在某些回归问题中会遇到过度拟合的问题而导致他们的表现欠佳。

# 什么是过度拟合？

对于线性回归中的房价问题，我们给出一个数据集：

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/87f3d2f39e5c4866bed197626d7573cf~tplv-k3u1fbpfcp-watermark.image)

如果对其进行拟合：

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/8aa0a3a1f39c45efad206369adb59006~tplv-k3u1fbpfcp-watermark.image)

可以明显看出这个的拟合效果并不好，我们称之为： underfit 欠拟合，或者被称为 high bias 高偏差。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ab874e2f36b94d58a20165d7e1c5cc19~tplv-k3u1fbpfcp-watermark.image)

使用二阶拟合良好。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/84592c9bec994bd59ca04354f4aa2d5f~tplv-k3u1fbpfcp-watermark.image)

但是如果使用一个四阶多项式，这似乎完美地拟合了数据，因为这条曲线通过了所有的数据点，但是我们主观上知道这并没有很好地拟合数据，这种情况就被称为 overfit 过度拟合，或者说 high variance 高方差。 

逻辑回归也是如此：

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/9bb26f3b17b8460e93c54b9fcfc4e1f8~tplv-k3u1fbpfcp-watermark.image)

**Overfitting**: If we have too many features, the learned hypothesis may fit the training set very well $\left(J(\theta)=\frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2} \approx 0\right)$, but fail
to generalize to new examples (predict prices on new examples).

过渡拟合的问题概括的说，将会在变量x多的时候出现，这是训练的假设能很好地拟合训练集，所以你的代价函数非常接近于0。但你是你可能会得到这样的曲线：它努力拟合样本数据导致其无法泛化到新样本当中。

所谓泛化就是一个假设模型应用到新样本中的能力。

对于过度拟合有两个办法来解决：

1. Reduce number of features 
    - Manually select which features to keep 
    - Model selection algorithm （later in course）
    
1. 减小选取变量的数量
    - 人工选择保留哪些变量
    - 模型选择算法，该算法会自动选择哪些特征需要保留，哪些特征要舍弃
    

这个方法的缺点是你需要放弃一些特征量，也就意味着你放弃了一些关于问题的信息。例如也许所有的特征变量都是有用的，我们就不能随意舍弃。

2. Regularization 
    - Keep all the features but reduce magnitude /values of parameters $\theta_j$
    - Works well when we have a lot of features, each of which contributes a bit to predicting y

2. 正则化
    - 保留所有的特征变量，但是减少量级或者参数$\theta_j$的大小
    - 当我们特征量很多的时候这个方法非常有效，其中每一个变量都会对预测y值产生一点影响。
    
# 正则化

又回到这个问题：

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/87f3d2f39e5c4866bed197626d7573cf~tplv-k3u1fbpfcp-watermark.image)

二阶时候能对其进行拟合，效果良好。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ab874e2f36b94d58a20165d7e1c5cc19~tplv-k3u1fbpfcp-watermark.image)

但是更高阶却会出现过拟合现象。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/84592c9bec994bd59ca04354f4aa2d5f~tplv-k3u1fbpfcp-watermark.image)

对于这个例子我们知道$θ_3$和$θ_4$是没必要的，那我们在$\theta_{0}+\theta_{1} x+\theta_{2} x^{2}+\theta_{3} x^{3}+\theta_{4} x^{4}$的代价函数中加入惩罚项，使$θ_3$和$θ_4$变得非常小。

怎么让这两项变得特别小，怎么加一个“惩罚项”：在代价函数后边给这两项加上巨大的系数，比如加上1000。那这两项的代价函数就会变成：$\min _{\theta} \frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}+1000 \theta_{3}^{2}+1000 \theta_{4}^{2}$，后边多出来这一块就是“惩罚项”。

代价函数是干嘛用的？求最小值，将得到最小值时候θ的值作为预测函数$h(x)$的θ。

加上惩罚项以后，带着两个1000的系数，怎么才能让代价函数取得较小的值？那就需要$θ_3$和$θ_4$变得特别小，甚至于接近于0。既然现在$θ_3$和$θ_4$都是极小的数接近于0，那带回预测函数中，$h(x)=\theta_{0}+\theta_{1} x+\theta_{2} x^{2}+\theta_{3} x^{3}+\theta_{4} x^{4}≈\theta_{0}+\theta_{1} x+\theta_{2} x^{2}$，这样就变成了二阶函数加上一些也别小的项，那后两项特别小的就可以忽略不计，又回到了近似于二阶函数的样子。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4156dc25d202451b9be0365d1df957d2~tplv-k3u1fbpfcp-watermark.image)

参数值θ较小意味着一个更简单的假设模型。比如上边的例子中，加入惩罚项之后$θ_3$和$θ_4$接近于0，我们得到了一个更简单的假设模型（四次变二次）。

**正则化**：

Regularization.

Small values for parameters $\theta_{0}, \theta_{1}, \ldots, \theta_{n}$.
- "Simpler" hypothesis
- Less prone to overfitting

正则化思想就是当我们为所有参数项都加上惩罚项，那么就相当于尽量去简化这个假设模型。因为越多的参数值接近于0，我们得到的预测模型就会越平滑、简单。

$$
J(\theta)=\frac{1}{2 m}\left[\sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}+\lambda \sum_{j=1}^{n} \theta_{j}^{2}\right]
$$

- $\lambda \sum_{j=1}^{n} \theta_{j}^{2}$就是正则化项，其作用是来缩小每个参数的值。从$\theta_1$开始哦，因为测试证明不管你加不加$\theta_0$，最后结果差别都不大，所以一般都从$\theta_1$开始正则化。
- $\lambda$是正则化参数，作用是控制两个不同目标之间的取舍。
    - 第一个目标：与目标函数的第一项有关，就是如何更好的拟合训练集
    - 第二个目标：尽量保持参数值较小，保持预测模型简单，与正则化有关
- $\lambda$：注意选择，可以联想考虑一下之前的学习速率α
    - 如果太大：我们对这些参数惩罚程度太大，导致所有才参数值都有接近于0，那最后预测函数就会接近于一个常数函数$h(x)≈\theta_0$，会造成欠拟合。
    
    ![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/455bec3f07404cecaf7cc62cc404764e~tplv-k3u1fbpfcp-watermark.image)
    - 如果太小：惩罚了和没惩罚一个样。


之前举的例子只是借用了正则化的思想，告诉你如何缩小参数θ，现在举个正则化的例子：

Housing:
- Features: $x_{1}, x_{2}, \ldots, x_{100}$
- Parameters: $\theta_{0}, \theta_{1}, \theta_{2}, \ldots, \theta_{100}$

还是预测房屋价格，假设现在有100个特征值，有101个参数θ，假设测试数据有70组。

正则化之后其代价函数写为：
$$J(\theta)=\frac{1}{2 ×70}\left[\sum_{i=1}^{70}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}+\lambda \sum_{j=1}^{100} \theta_{j}^{2}\right]$$

# 线性回归的正则化

对于线性回归我们之前讨论过两种算法：
- 梯度下降
- 正规方程

Regularized linear regression
$$
\begin{aligned}
&J(\theta)=\frac{1}{2 m}\left[\sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}+\lambda \sum_{j=1}^{n} \theta_{j}^{2}\right] \\
&\min _{\theta} J(\theta)
\end{aligned}
$$

对于上面的代价函数我们想寻找合适的θ使其最小化。

## Gradient descent
还记得传统的梯度下降长这样：

Repeat {

$\theta_{j}:=\theta_{j}-\alpha \quad \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)} \quad\quad
(j=0,1,2,3, \ldots, n)$

}


现在把梯度下降里$\theta_0$单独摘出来再给剩下的单独加上惩罚项就可以了。因为之前讲过了，惩罚项是从$\theta_{1}$开始的。

Repeat {

$\theta_{0}:=\theta_{0}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{0}^{(i)}$

$\theta_{j}:=\theta_{j}-\alpha \lbrack\frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}+\frac{\lambda}{m} \theta_{j} \rbrack \quad\quad
(j=1,2,3, \ldots, n)$

}

化简之后可以写成这样：

Repeat {

$\theta_{0}:=\theta_{0}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{0}^{(i)}$

$\theta_{j}:=\theta_{j}\left(1-\alpha \frac{\lambda}{m}\right)-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)} \quad\quad
(j=0,1,2,3, \ldots, n)$

}

## Normal equaion


原来的正规方程：

$$\theta=\left(X^{T} X\right)^{-1} X^{T} y$$

其中
$X = \begin{bmatrix} ...(x^i_0)^T...  \\ ...(x^i_1)^T... \\ ...(x^i_2)^T...\\ ...\\ ...(x^i_n)^T...\end{bmatrix} \in  \R^{m \times n+1} \quad\quad\quad$
$y = \begin{bmatrix} y^1  \\ y^2 \\ y^3 \\ ...\\ y^m \end{bmatrix} \in \R^m$

使用正则化之后

$$
\theta=\left(x^{\top} x+\lambda\left[\begin{array}{lllll}
0 & & & \\
& 1 & & \\
& & 1 & \\
& & & \cdots \\
& & & & 1
\end{array}\right]\right)^{-1} x^T y
$$

其中这个对角矩阵$\left[\begin{array}{lllll}
0 & & & \\
& 1 & & \\
& & 1 & \\
& & & \cdots \\
& & & & 1
\end{array}\right]$的维度$\R_{n+1 \times n+1}$

# 逻辑回归的正则化

逻辑回归的代价函数：

$\begin{aligned}
J(\theta) =-\frac{1}{m}\left[\sum_{i=1}^{m} y^{(i)} \log h_{\theta}\left(x^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right]
\end{aligned}$

我们也要在后边添加一，添加之后的是：
$$
\begin{aligned}
J(\theta) =-\frac{1}{m}\left[\sum_{i=1}^{m} y^{(i)} \log h_{\theta}\left(x^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right]+\frac{\lambda}{2 m} \sum_{j=1}^{n} \theta_{j}^{2}
\end{aligned}
$$

添加之后即使你拟合的参数很多并且阶数很高，只要添加了这个正则化项，保持参数较小，你仍然可以得到一条合理的决策边界。

## 梯度下降

之前我们已经知道线性回归和逻辑回归看起来形式上是一样的，所以我们直接把线性回归的梯度下降搬过来：

Repeat {

$\theta_{0}:=\theta_{0}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{0}^{(i)}$

$\theta_{j}:=\theta_{j}-\alpha \lbrack\frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}+\frac{\lambda}{m} \theta_{j} \rbrack \quad\quad
(j=1,2,3, \ldots, n)$

}

为了使其正则化符合逻辑回归，我们需要将第二个式子也加上一个：

Repeat {

$\theta_{0}:=\theta_{0}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{0}^{(i)}$

$\theta_{j}:=\theta_{j}-\alpha \lbrack\frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}+\frac{\lambda}{m} \theta_{j} \rbrack \quad\quad
(j=1,2,3, \ldots, n)$

}

**虽然看起来是和线性回归一样，但是一定要记住二者的$h(x)$存在区别。**

## Advanced optimization

讲逻辑回归的时候，除了梯度下降我们还提到了其他高级算法，但是没有展开细说。那如何在高级算法中使用正则化

$function [jVal, gradient] = costFunction (theta)$

$jVal$=[ code to compute $J(\theta)$]

$gradient (1)=\left[\right.$ code to compute $\left.\frac{\partial}{\partial \theta_{0}} J(\theta)\right]$

$gradient (2)=\left[\right.$ code to compute $\left.\frac{\partial}{\partial \theta_{1}} J(\theta)\right]$

...

$gradient (n+1)=\left[\right.$ code to compute $\left.\frac{\partial}{\partial \theta_{n}} J(\theta) \quad\right] $

还是需要你自己写costFunction函数，在这个函数中：

- $function [jVal, gradient] = costFunction (theta)$ 需要传入theta，theta $=\left[\begin{array}{c}\theta_{0} \\ \theta_{1} \\ \vdots \\ \theta_{n}\end{array}\right]$
- $jVal$=[ code to compute $J(\theta)$]这一句就是写代价函数J的表达式
- $gradient (1)=\left[\right.$ code to compute $\left.\frac{\partial}{\partial \theta_{0}} J(\theta)\right]$是计算$\frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{0}^{(i)}$
- $gradient (n+1)=\left[\right.$ code to compute $\left.\frac{\partial}{\partial \theta_{n}} J(\theta) \quad\right] $是计算$\frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{n}^{(i)}+\frac{\lambda}{m}J(\theta_n)$



# 9 神经网络

## 神经网络的表示

都有线性回归和逻辑回归了为什么还要学神经网络呢？

现在举几个例子。


![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/794e9e69f2b54817bd89c26a8f12ee98~tplv-k3u1fbpfcp-watermark.image)

这是一个监督学习的分类问题。只有两个特征量。

如果你使用逻辑回归，想拟合出下面的曲线，那你可能需要很多项式才能将正负样本分开。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/1c0d04be64fb4143852697580bc93777~tplv-k3u1fbpfcp-watermark.image)

$$
\begin{aligned}
&g\left(\theta_{0}+\theta_{1} x_{1}+\theta_{2} x_{2}\right. +\theta_{3} x_{1} x_{2}+\theta_{4} x_{1}^{2} x_{2} \left.+\theta_{5} x_{1}^{3} x_{2}+\theta_{6} x_{1} x_{2}^{2}+\ldots\right)
\end{aligned}
$$

只是两个特征量就如此麻烦，那要是100个特征量呢？

就算只包含二次项，比如$x_1^2、x_2^2、...、x_{100}^2、x_1x_2、x_1x_3、...、x_{99}x_{100}$，这样最终也有5000多个多项式。
*大约是$\frac{n^2}{2}$个，$x_1$从$x_1$乘到$x_{100}$就是100项，以此类推最后$x_{100}$从$x_1$乘到$x_{100}$加起来就是10000项。但是$x_1x_{100}$和$x_{100}x_1$是一样的，再合并一下相同的。所以最后大概剩下一半。*

现在已经5000多项了，最后的结果很可能会过拟合，并且还会存在运算量过大的问题。

现在算一下包括三次项的。
$x_1^3、x_2^3、...、x_{100}^3、x_1^2x_2、x_1^2x_3、...、x_{99}x_{100}^2$，这样最后就有170000个三次项。（自己验证，我不推了）。

上边的例子可以知道，当特征量很大的时候就会造成空间急剧膨胀。因此特征量n很大的时候用构造多项式的方法来建立非线性分类器是很不明智的选择！

然而现实中对于很多机器学习的实际问题，特征量个数n一般都特别大。

比如对于计算机视觉的问题：现在你需要通过机器学习来判断图像是否是一个汽车。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7893fc45db564a0f9079589219ded95a~tplv-k3u1fbpfcp-watermark.image)

你肉眼看到之后就能知道这是一个汽车，但是及其肯定不能一下就知道这是汽车。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/0fef8e38c6c5492caf2aa1eeaa11df53~tplv-k3u1fbpfcp-watermark.image)
取这一小块。在你眼里是个门把手，但是及其只能识别出像素强度值的网格，知道每个像素的亮度值。因此计算机视觉的问题就是根据像素点的亮度矩阵告诉我们这个矩阵代表一个汽车的门把手。

当我们用机器学习后造一个汽车识别器的时候，我们要做的就是提供一个带标签的样本。其中一部分是车，另一部分样本不是车。将这样的样本集输入给学习算法，从而训练场出一个分类器，最后输入一个新图片，让分类器来判定这是个什么东西。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4119144c22c642f0ae6d56528f88e263~tplv-k3u1fbpfcp-watermark.image)

**引入非线性假设的必要性**：


![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ddebc2e16f3749949b46bfd12315f69b~tplv-k3u1fbpfcp-watermark.image)

一个车，取两个像素点，将其放到坐标系中。位置取决于像素点12的强度。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d88c775650b64ac9be3d9bceafb0e338~tplv-k3u1fbpfcp-watermark.image)


![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/259dcebb4cbd4358b9854fc17f3aa054~tplv-k3u1fbpfcp-watermark.image)

现在我们需要一个非线性假设来区分这两类样本。

假设车子图片是50\*50像素，现在用x向量存储每个像素点的亮度。那这样就有2500个像素点。向量的维度就是2500.如果图片是彩色的，用RGB存储，那向量的维度就是7500。

如果现在要想用包含所有的二次项来列出非线性假设，那Quadratic features $\left(x_{i} \times x_{j}\right): \approx 3$ million features。三百万，计算成本太高了。因此只包含平方项、立方项找偶记回归假设模型只适合n比较小的情况。n较大的情况下，即在复杂的非线性假设上神经网络的效果更为优异。

神经网络算法已经很久了，起初是为了制造能模拟大脑的机器。

## 模型展示

神经网络模仿了大脑中的神经元。神经元有轴突和树突。轴突相当于输入路线，树突相当于输出路线。神经元就可以看做是一个计算单元。神经元之间
用电流进行交流。（emmm作为一个生物学的学生我居然还清楚地记得这些知识……）

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/248231bf99bb4b40ad28fe34ee4af281~tplv-k3u1fbpfcp-watermark.image)

在一个人工实现的神经网络里，用下图一个简单的模型来模拟神经元工作。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f9ade2333ef643aab56be7bdf73f5b4b~tplv-k3u1fbpfcp-watermark.image)

$x_1-x_3$表示树突作为输入端，右边则作为轴突输出预测模型。
![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/af48631fbf7b4b46b9993ce07b737e43~tplv-k3u1fbpfcp-watermark.image)

输入是$x_1-x_3$但通常会加上额外节点$x_0$，成为偏置单元（bias unit）或者偏置神经元（bias neuron）。

**在神经网络的文献中如果看到权重（weight）这个词，他其实和摸清参数θ是同一个东西。**



人类的神经网络是一组神经元，下图是一个更复杂的图像，他的第一层蓝色圈圈被称为输入层，因为我们在第一层输入特征量。最后一层的黄色圈圈被称为输出层，因为这一层神经元输出假设的最终计算结果，中间的第二层被称位隐藏层，隐藏层可以不止一层。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b8b836a30d2048279b6c6f8657cca6e3~tplv-k3u1fbpfcp-watermark.image)

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/976b5d10893c4948811b38f4a34df615~tplv-k3u1fbpfcp-watermark.image)

$\begin{aligned} a_{i}^{(j)}=& \text { "activation" of unit } i \text { in layer } j \\ \Theta^{(j)}=& \text { matrix of weights controlling } \\ & \text { function mapping from layer } j \text { to } \\ & \text { layer } j+1 \end{aligned}$

$a_{i}^{(j)}$表示第j层第i个神经元或者单元的激活项，至于权重矩阵$\Theta^{(j)}$，控制第j层到j+1层的映射关系。

$\begin{aligned} 
a_{1}^{(2)}=g\left(\Theta_{10}^{(1)} x_{0}+\Theta_{11}^{(1)} x_{1}+\Theta_{12}^{(1)} x_{2}+\Theta_{13}^{(1)} x_{3}\right) \\ a_{2}^{(2)}=g\left(\Theta_{20}^{(1)} x_{0}+\Theta_{21}^{(1)} x_{1}+\Theta_{22}^{(1)} x_{2}+\Theta_{23}^{(1)} x_{3}\right) \\ a_{3}^{(2)}=g\left(\Theta_{30}^{(1)} x_{0}+\Theta_{31}^{(1)} x_{1}+\Theta_{32}^{(1)} x_{2}+\Theta_{33}^{(1)} x_{3}\right) \\ h_{\Theta}(x)=a_{1}^{(3)}=g\left(\Theta_{10}^{(2)} a_{0}^{(2)}+\Theta_{11}^{(2)} a_{1}^{(2)}+\Theta_{12}^{(2)} a_{2}^{(2)}+\Theta_{13}^{(2)} a_{3}^{(2)}\right) \end{aligned}$

$a_{1}^{(2)}$的激活函数$a_{1}^{(2)}=g\left(\Theta_{10}^{(1)} x_{0}+\Theta_{11}^{(1)} x_{1}+\Theta_{12}^{(1)} x_{2}+\Theta_{13}^{(1)} x_{3}\right)$

上一张小图里没有加上偏置单元，看上上个图加上偏置神经元的那个图。这里边$\Theta^{(1)}$矩阵代表的就是第一层（输入层x）到第二层隐藏层$a^{(2)}$的权重（参数）矩阵。

可以理解为$\Theta^{(1)}=\begin{bmatrix}\theta_{10}  \theta_{11} \theta_{12} \theta_{13} \\ \theta_{20}  \theta_{21} \theta_{22} \theta_{23} \\ \theta_{30}  \theta_{31} \theta_{32} \theta_{33} \end{bmatrix}$

对于上边那个公式，不加偏置单元的时候就是三个输入单元到三个隐藏单元。最终得到的$\Theta$矩阵是加上偏执单元的映射，所以权重矩阵是$\R^{3 \times 4}$。

**总结一下：**
**从第j层到j+1层，如果第j层有$s_j$个元素，第j+1层有$s_{j+1}$个元素，那矩阵$\Theta^{(j)}的维度是$s_{j+1} \times s_j+1$**。

那现在就可以知道隐藏层到后边的输出层的权重矩阵（参数矩阵）$\Theta^{(2)} = \begin{bmatrix} \theta_{10}  \theta_{11} \theta_{12} \theta_{13}  \end{bmatrix}$

现在把上边的那一堆式子改一下下：

$\begin{aligned} 
a_{1}^{(2)}=g\left(z_1^{(2)}\right) \\ a_{2}^{(2)}=g\left(z_2^{(2)}\right) \\ a_{3}^{(2)}=g\left(z_3^{(2)}\right) \\ h_{\Theta}(x)=a_{1}^{(3)}=g\left(z_1^{(3)}\right) \end{aligned}$

Z的上标表示它后一列是什么。

这样改完以后是不是感觉又能回到矩阵乘法了。

$x=\left[\begin{array}{l}x_{0} \\ x_{1} \\ x_{2} \\ x_{3}\end{array}\right] \quad z^{(2)}=\left[\begin{array}{c}z_{1}^{(2)} \\ z_{2}^{(2)} \\ z_{3}^{(2)}\end{array}\right]$

$z^{(2)}=\Theta^{(1)} x$

$a^{(2)}=g\left(z^{(2)}\right)$

Add $a_{0}^{(2)}=1$

 ${z^{(3)}=\Theta^{(2)} a^{(2)}}$

$h_{\Theta}(x)=a^{(3)}=g\left(z^{(3)}\right)$

上边这个计算$h(x)$的过程被称为**向前传播**。
因为我们从输入单元的激活项开始进行向前传播，给隐藏层计算隐藏层的激活项，然后继续向前传播计算输出层的激活项。

**这种依次计算激活项，从输入层到隐藏层再到输出层的过程叫前向传播。** 

我们刚才就是推导这一过程向量化的实现方法。

# 10 反向传播



![image-20210925145828087](E:\Code\DeepLearning\image\image-20210925145828087.png)

假设我们有一组训练集$\{ (x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),...(x^{(m)},y^{(m)}) \}$

- L：表示有多少层，图中layer4共四层
- $s_l$：number of unit in layer l，第l层有几个元素，上图中$s_1=3,s_2=s_3=5,s_4=4$



神经网络：

$h_{\Theta}(x) \in \mathbb{R}^{K} \quad\left(h_{\Theta}(x)\right)_{i}=i^{t h}$ output
$J(\Theta)=-\frac{1}{m}\left[\sum_{i=1}^{m} \sum_{k=1}^{K} y_{k}^{(i)} \log \left(h_{\Theta}\left(x^{(i)}\right)\right)_{k}+\left(1-y_{k}^{(i)}\right) \log \left(1-\left(h_{\Theta}\left(x^{(i)}\right)\right)_{k}\right)\right]$
$+\frac{\lambda}{2 m} \sum_{l=1}^{L-1} \sum_{i=1}^{s_{l}} \sum_{j=1}^{s_{l+1}}\left(\Theta_{j i}^{(l)}\right)^{2}$
$$
\min _{\Theta} J(\Theta)
$$
Need code to compute:
$$
\begin{aligned}
&J(\Theta) \\
&\frac{\partial}{\partial \Theta_{i j}^{(l)}} J(\Theta)=a_j^{(l)}\delta_j^{(l+1)}
\end{aligned}
$$
**Gradient computation**
Given one training example $(x, y)$ :
Forward propagation:
$$
\begin{aligned}
a^{(1)} &=x \\
z^{(2)} &=\Theta^{(1)} a^{(1)} \\
a^{(2)} &=g\left(z^{(2)}\right)\left(\operatorname{add} a_{0}^{(2)}\right) \\
z^{(3)} &=\Theta^{(2)} a^{(2)} \\
a^{(3)} &=g\left(z^{(3)}\right)\left(\text { add } a_{0}^{(3)}\right) \\
z^{(4)} &=\Theta^{(3)} a^{(3)} \\
a^{(4)} &=h_{\Theta}(x)=g\left(z^{(4)}\right)
\end{aligned}
$$
对于前向传播，假设现在只有一个训练样本$(x,y)$，最后计算出的$h(x)$就是$a^{(4)}$。

为了计算导数值我们需要后向传播。反向传播得名于我们要从输出层开始从后往前计算$\delta$的值。就是把输出层的误差进行反向传播给倒数第一层，然后再传给倒数第二层，以此类推。

Gradient computation: Backpropagation algorithm 

Intuition: $\delta_{j}^{(l)}=$ "error" of node $j$ in layer $l$.

$\delta_{j}^{(4)}={a_{j}^{(4)}}{-y_{j}}=h_{\Theta}(x)_j-y_i$

$\delta^{(3)} = (\Theta^{(3)})^T\delta^{(4)}·*g'(z^{(3)})=(\Theta^{(3)})^T\delta^{(4)}·*a^{(3)}.*(1-a^{(3)})$

$\delta^{(2)} = (\Theta^{(2)})^T\delta^{(3)}·*g'(z^{(2)})$

不用求$\delta^{(1)}$，因为第一列是我们观测到的值，输入值不存在误差。



> a\*b表示矩阵a与矩阵b进行矩阵相乘，a.\*b表示矩阵a中的元素与矩阵b中的元素按位置依次相乘



**当我们有大量的训练样本的时候**

$\text {Training set }\left\{\left(x^{(1)}, y^{(1)}\right), \ldots,\left(x^{(m)}, y^{(m)}\right)\right\}$

$Set \Delta_{ij}^{(l)}=0$

$for \quad i=1\quad to\quad m$

$\quad Set \quad a^{(1)}=x^{(i)}$

$\quad \text{Perform forward propagation to compute }a^{(l)} \text{ for } l = 1,2,3,...,L $

$\quad Using \quad y^{(i)},compute \quad \delta^{(L)}-y^{(i)}$

$\quad Conpute\quad \delta^{(L-1)},\delta^{(L-2)},...,\delta^{(2)}$

$\quad \Delta^{(l)}_{ij}:=\Delta^{(l)}_{ij}+a_j^{(l)}\delta_i^{(l+1)}$

$\begin{aligned}
&D_{i j}^{(l)} :=\frac{1}{m} \Delta_{i j}^{(l)}+\lambda \Theta_{i j}^{(l)}  \text { if } j \neq 0 \\
&D_{i j}^{(l)}:=\frac{1}{m} \Delta_{i j}^{(l)}  \text { if } j=0
\end{aligned}$



![image-20210928113246899](E:\Code\DeepLearning\image\image-20210928113246899.png)

## 10.1 梯度检测

因为反向传播的复杂性，所以有时候虽然你看着好像运行过程梯度下降是正常的，但是最后的结果可能却是错误的。那如何测试运行过程是否正确执行，那就需要进行梯度检测。

对于一个实数点我们这样求他的近似导数：$\frac{d}{d \theta} J(\theta) \approx \frac{J(\theta+\varepsilon)-J(\theta-\varepsilon)}{2 \varepsilon}$

当$\theta \in \mathbb{R}^{n} \quad$ (E.g. $\theta$ is "unrolled" version of $\left.\Theta^{(1)}, \Theta^{(2)}, \Theta^{(3)}\right)$作为一个向量的时候，

 $\theta=\theta_{1}, \theta_{2}, \theta_{3}, \ldots, \theta_{n}$

此时的近似求导就是：
$$
\begin{gathered}
\frac{\partial}{\partial \theta_{1}} J(\theta) \approx \frac{J\left(\theta_{1}+\epsilon, \theta_{2}, \theta_{3}, \ldots, \theta_{n}\right)-J\left(\theta_{1}-\epsilon, \theta_{2}, \theta_{3}, \ldots, \theta_{n}\right)}{2 \epsilon} \\
\frac{\partial}{\partial \theta_{2}} J(\theta) \approx \frac{J\left(\theta_{1}, \theta_{2}+\epsilon, \theta_{3}, \ldots, \theta_{n}\right)-J\left(\theta_{1}, \theta_{2}-\epsilon, \theta_{3}, \ldots, \theta_{n}\right)}{2 \epsilon} \\
\vdots \\
\frac{\partial}{\partial \theta_{n}} J(\theta) \approx \frac{J\left(\theta_{1}, \theta_{2}, \theta_{3}, \ldots, \theta_{n}+\epsilon\right)-J\left(\theta_{1}, \theta_{2}, \theta_{3}, \ldots, \theta_{n}-\epsilon\right)}{2 \epsilon}
\end{gathered}
$$
这样你就可以估计代价函数J关于所有参数的偏导数。

注意：梯度检验的主要作用是确定你反向传播求的D的正确性，计算量非常大，所以在你梯度检验完确定反向传播没问题之后记得关闭它。

## 10.2 随机初始化

在你开始梯度下降的时候对于第一组$\Theta$你要进行初始化，那怎么初始化呢？如果都给相同的值1之类的，这样是没意义的，因为你之后计算$\delta$发现也是一样的值，并且某一单元分出去的线权重相同，也会造成偏导数相同。

![image-20211004235834178](E:\Code\DeepLearning\image\image-20211004235834178.png)

所以为了防止上述情况就需要对theta进行随机初始化，Initialize each $\Theta_{i j}^{(l)}$ to a random value in $[-\epsilon, \epsilon]$ (i.e. $\left.-\epsilon \leq \Theta_{i j}^{(l)} \leq \epsilon\right)$
