简介

GNU Octave is a high-level language primarily intended for numerical computations. It is
typically used for such problems as solving linear and nonlinear equations, numerical linear
algebra, statistical analysis, and for performing other numerical experiments. It may also
be used as a batch-oriented language for automated data processing.

GNU Octave是一种高级语言，主要用于数值计算。它通常用于求解线性和非线性方程、数值线性代数、统计分析等问题，以及进行其他数值实验。它也可以用作自动化数据处理的面向批处理的语言。

**更多的功能可以看一下官方文档：[octave.pdf](https://octave.org/octave.pdf)**

# 基本
1. 注释 %
    - 类似于C++的//和python的#
    - ![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/14b7ed4beb084ce894c797ed878a073b~tplv-k3u1fbpfcp-watermark.image)
2. ans 用于存储输出结果
3. 如果你不喜欢输入行前缀，可以自定义修改`PS1('你想改成的内容');`
    - ![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6c7f99aa753f4ecaa55c5057bb679433~tplv-k3u1fbpfcp-watermark.image)
    这个不是永久的，你电脑重启之后他又恢复默认了。
4. `help` 帮助
    如果你忘了某一个语句的用法了，可以用这个查询。
        
    比如输入`help eye`就会显示eye的用法。
    ![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/79a5e56ab87b4577864b5bdbbe6ffaa9~tplv-k3u1fbpfcp-watermark.image)
    

5.多个语句同时执行用逗号分开即可。
    ```c
    octave:226> a=4,b=5,c=6
    a = 4
    b = 5
    c = 6
    ```
6. `quit`，`exit`可以关闭Octave（如果你不想点右上角的×的话）
7. `addpath('路径')`添加路径
   
# 符运算

## 算术运算符 
加减乘除乘方
```c
octave:1> 1+5
ans = 6
octave:2> 1*5
ans = 5
octave:3> 1/5
ans = 0.2000
octave:4> 1-5
ans = -4
octave:5> 5^2
ans = 25
```

## 逻辑运算符
- 等于 `==`
- 不等于`~=`
- 或 `||`
- 与 `&&`
- 异或 `xor(a,b)`
- 大于等于` >=`
- 大于 `>`

```c
ans = 0
>>1==1
ans = 1
>>1 ~=2
ans = 1
>>1&&2
ans = 1
>>1&&0
ans = 0
>>1||0
ans = 1
>>xor(1,1)
ans = 0
>>xor(1,2)
ans = 0
>>xor(0,1)
ans = 1
>>1>=2
ans = 0
>>1<2
ans = 1
```

## 赋值运算符
- `=`赋值运算符
    - 语句后边加分号 不直接打印赋值语句
    - 语句后边不加分号 直接打印赋值语句

```c
>>a=1   %a=1后边没分号，所以输入回车之后下一行会直接打印a=1
a = 1
>>a     %再输入a回车看看是否赋值成功
a = 1
>>b=2;  %b=2后边加上分号了，所以下一行不会直接打印b的赋值
>>b     %输入b回车，查看给b的赋值
b = 2   

>>c=pi
c = 3.1416 
```

## 格式化输出
- `disp(x)`：打印x的值
- `disp(sprintf('%0.2f',c))`：格式化输出保留两位小数，这个和C语言一样
- `format long/short`：对整数无影响，对小数输出格式有影响
```c
>>disp(a)
1
>>disp(c)
3.1416
>>disp(sprintf('%0.2f',c))
3.14

>>format long
>>a
a = 1
>>c
c = 3.141592653589793
>>format short
>>a
a = 1
>>c
c = 3.1416
```

## 向量和矩阵
声明矩阵官方原话写的是：
> Vectors and matrices are the basic building blocks for numerical analysis. To create a new matrix and store it in a variable so that you can refer to it later, type the command
>
> octave:1> A = [ 1, 1, 2; 3, 5, 8; 13, 21, 34 ]
>
> Octave will respond by printing the matrix in neatly aligned columns. 

```c
>>martix = [1,2,3;4,5,6;7,8,9]
martix =

   1   2   3
   4   5   6
   7   8   9
```
当然还有很多其他写法，虽然不规范也能声明矩阵，这里就不列举了。

声明行向量，各个元素之间不用加符号，或者加逗号。

```c
martix =

   1   2
   2   3
   3   4

>>vector=[1 2 3]
vector =

   1   2   3

>>vector = [4;5;6]
vector =

   4
   5
   6
```

**其他特殊写法**：
- `变量 = 起始:终止` ，生成一个向量，并且步长为1
- `变量 = 起始:步长:终止` ，就会生成一个向量
    ```c
    > v = 1:5
    v =
    
       1   2   3   4   5
    
    > m = 1:0.5:1
    m = 1
    octave:3> m = 1:0.5:5
    m =
    
        1.0000    1.5000    2.0000    2.5000    3.0000    3.5000    4.0000    4.5000    5.0000
    ```
- `ones(a,b)` 生成a行b列只有1的矩阵
    - 除了`ones`还有`zeros`，生成的元素都是0，但是没有twos、threes……
    - ```c
        >ones(2,3)
        ans =
      
           1   1   1
           1   1   1
      ```
- rand(n) 形成n阶方阵，元素都是0-1之间随机数
- rand(a,b) 生成a行b列的0-1之间随机数的矩阵
    - ```c
        >rand(2,3)
        ans =
      
           0.4046   0.4508   0.8021
           0.6986   0.8620   0.2631
      ```
    - `max(rand(n),rand(n))`生成量个方阵，取其中较大值形成一个新方阵
        ```c
        octave:124> c = max(rand(4),rand(4))
        c =
        
           0.8010   0.9926   0.7241   0.9053
           0.9007   0.3026   0.9856   0.5710
           0.7679   0.2630   0.5200   0.9615
           0.7333   0.5113   0.6957   0.5683
        ```
- `randn(a,b)` 也是生成a行b列随机数矩阵，但是符合高斯分布（正态分布）。返回一个具有正态分布随机元素的矩阵，这些随机元素的平均值为零，方差为1。
    - 正态分布：   
    ![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/00dbe23431754bb0b8cd28554eeeea22~tplv-k3u1fbpfcp-watermark.image)
- `randi()`
    - randi(max,m,n)：生成1~max的m行n列矩阵
    - randi(\[min,max\],m,n)：生成min~max的m行n列矩阵
    - ```c
        >randi(5,3,3)
        ans =
      
           1   5   5
           5   2   1
           5   1   1
      ```
    - ```c
        >randi([5,10],5,5)
        ans =
            6    8    6    8    5
           10    9   10    8    7
            5    7    7    9   10
           10    8   10    8    8
           10    9    7    6    7
      
      ```
- `eye(n)` 生成n维单位矩阵
    ```c
    >eye(5)
    ans =
    
    Diagonal Matrix
    
       1   0   0   0   0
       0   1   0   0   0
       0   0   1   0   0
       0   0   0   1   0
       0   0   0   0   1
    ```
- `matrixB = matrixA(x:y,v:w) `:将matrixA的x\~y行v\~w列形成一个新矩阵
    - `:`表示所有元素
    - `a:b`表示a到b的元素
    - `[a b c]`表示abc三行或列
    - 以上三种表示方法在矩阵别的方法中也能使用，比如赋值、打印
- 矩阵拼接
    - 在矩阵右侧添加一列
        ```c
        octave:30> A
        A =
        
            4    2    7    5
            1   10    7   10
            5   10    9    9
        
        octave:31> A = [A,[2;4;5]]
        A =
        
            4    2    7    5    2
            1   10    7   10    4
            5   10    9    9    5
        ```
    - `matrixC=[matrixA matrixB]` 横向拼接
        写`[matrixA, matrixB]`也一样，加不加逗号都行
        ```c
        octave:33> A = randi(5,2,3)
        A =
        
           4   1   3
           2   3   5
        
        octave:34> B = randi(5,2,3)
        B =
        
           5   4   5
           1   1   3
        
        octave:35> c = [A B]
        c =
        
           4   1   3   5   4   5
           2   3   5   1   1   3
        ```
    - `matrixC=[matrixA; matrixB]`纵向拼接
        ```c
        octave:33> A = randi(5,2,3)
        A =
        
           4   1   3
           2   3   5
        
        octave:34> B = randi(5,2,3)
        B =
        
           5   4   5
           1   1   3
        
        octave:36> d=[A;B]
        d =
        
           4   1   3
           2   3   5
           5   4   5
           1   1   3
        ```
    
- `matrix(:)` 将矩阵中的所有元素放入一个列向量
    ```c
    octave:30> A
    A =
    
        4    2    7    5
        1   10    7   10
        5   10    9    9
        
    octave:31> A(:)
    ans =
    
        4
        1
        5
        2
       10
       10
        7
        7
        9
        5
       10
        9
        2
        4
        5
    ```
    
## 其他
- `hist(变量)`

    可以画出变量的图像
    ![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/dad9f76d27dd4808a1d2233a7a05a9d6~tplv-k3u1fbpfcp-watermark.image)

    如果你觉得这样太少了。那可以使用`hist(变量，条纹数)`，下图就是画出50条纹后的图像。
    ![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/cd6bd9e1f2944613b2650e7530073e44~tplv-k3u1fbpfcp-watermark.image)

- **检测大小**
    - `size()`
    - `length()`

    下图可以看出，length显示矩阵的长度。size显示矩阵有几行几列。size(矩阵名称，1/2)其中1代表行参数，2代表列参数。

    ```c
    v =
    
        9    9    5    8    8
        5   10    7    6    8
        6    7    7    7    6
        7   10    8    7    9
        7   10    7    5    6
        
    >length (v)
    ans = 5
    >size(v)
    ans =
    
       5   5
    >size(v,1)
    ans = 5
    ```

- 索引：
    - `matrix(i,j)` 显示第i行第j列的元素
        ```c
        octave:26> A = randi(10,3,4) %生成一个1~10之间随机数的 3×4 的矩阵
        A =
            4    2    7    5
            1   10    7   10
            5   10    9    9
        octave:27> A(2,3)            %打印这个矩阵2行3列的元素
        ans = 7
        ```
    - `matrix(:,[a,b,c...])`
        - `:`：代表全部
        - `[]`：方括号中有哪几个就选中哪几个
        - ```c
            octave:26> A = randi(10,3,4)
            A =
          
                4    2    7    5
                1   10    7   10
                5   10    9    9
          
            octave:27> A(:,[1,2,4])   %显示A所有行的1,2,4列
            ans =
          
                4    2    5
                1   10   10
                5   10    9
          ```

- 显示现有变量
    - `who`：显示目前所有变量
    - `whos`：显示目前所有变量和详细信息
    - ![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/cf458727e09f4e0fb596d6050694923d~tplv-k3u1fbpfcp-watermark.image)
    
- 删除某个变量 `clear`
    - `clear 变量` 删除三个变量
    - `clear   删除所有变量
    ```c
    octave:12> a = 15   % 给a赋值15
    a = 15
    octave:13> a        % 打印a
    a = 15              % 显示a的值
    octave:14> clear a
    octave:15> a        % 再次打印a
    error: 'a' undefined near line 1, column 1    %报错
    ```

# 移动数据

用过其他命令行的应该能适应，octave也同样支持某些命令。

- `pwd`：列出当前目录
- `cd xxx文件夹`：进入xxx文件夹
- `ls`：列出当前目录信息
- 加载文件：
    - `load 文件名`
    - `load('文件名')`
- save 文件名 变量; ：将变量保存到文件中，比如`save data.txt num`，把num变量保存到data.txt中（如果操作目录中没有这个文件会创建一个新文件，如果目录中有这个文件会重写）

下边给出两个文件的数据，featureX和priceY，自己搞两个文件存起来。
featureX:
```
2104  3
1600  3
2400  3
1416  2
3000  4
1985  4
1534  3
1427  3
1380  3
1494  3
1940  4
2000  3
1890  3
4478  5
1268  3
2300  4
1320  2
1236  3
2609  4 
3031  4
1767  3
1888  2
1604  3
1962  4
3890  3
1100  3
1458  3
2526  3
2200  3
2637  3
1839  2
1000  1
2040  4
3137  3
1811  4
1437  3
1239  3
2132  4
4215  4
2162  4
1664  2
2238  3
2567  4
1200  3 
852  2 
1852  4 
1203  3 
```
priceY:
```
399900
329900
369000
232000
539900
299900
314900
198999
212000
242500
239999
347000
329999
699900
259900
449900
299900
199900
499998
599000
252900
255000
242900
259900
573900
249900
464500
469000
475000
299900
349900
169900
314900
579900
285900
249900
229900
345000
549000
287000
368500
329900
314000
299000
179900
299900
239500
```

首先进入对应的操作目录。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b5653c6812f94d26b4bd94cd5df1384c~tplv-k3u1fbpfcp-watermark.image)

用who查看当前有的变量。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d81cda2763354716a99881bc08ae652b~tplv-k3u1fbpfcp-watermark.image)

打印依稀下X6price确实已经成功读入文件

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/269f3b705e5f4c0b825fb5b897bfce95~tplv-k3u1fbpfcp-watermark.image)

用size查看一下，导入的数据是存在矩阵中的。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/8d37ace081fd443783b5c7d59f33150e~tplv-k3u1fbpfcp-watermark.image)

把矩阵X6feature的1~5行，第2列 赋值给矩阵x，并将矩阵x保存到data.txt中

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/3fdfb72ef8a54253b057fe7174ec7326~tplv-k3u1fbpfcp-watermark.image)

# 计算
**除了最开始提到的算术运算符，还有以下运算。**

注意一个特殊的`.*`，矩阵同位相乘，即矩阵相同位置上的元素相乘。
```c
octave:37> A
A =

   4   1   3
   2   3   5

octave:38> B
B =

   5   4   5
   1   1   3

octave:39> A * B    %AB无法相乘
error: operator *: nonconformant arguments (op1 is 2x3, op2 is 2x3)
octave:40> A .* B   %AB可以点乘
ans =

   20    4   15
    2    3   15
```
`.^`给矩阵每个元素做乘方
```c
octave:43> A
A =

   4   1   3
   2   3   5

octave:44> A .^2
ans =

   16    1    9
    4    9   25

octave:45>
```
`矩阵/数`即可让矩阵的每个元素都除以该数字，但是如果你想让一个数除以矩阵的每个元素，结果仍旧是矩阵，那就需要`./`
```c
octave:38> B
B =

   5   4   5
   1   1   3
octave:50> 2 ./B
ans =

   0.4000   0.5000   0.4000
   2.0000   2.0000   0.6667
```

如果想让矩阵每个元素都加或减去一个数，直接用`+-`即可。
```c
octave:71> C
C =

   2  -1  -4   1
  -5  -2  -1  -1
   1  -1  -5  -4

octave:72> C+1
ans =

   3   0  -3   2
  -4  -1   0   0
   2   0  -4  -3

octave:73> C-2
ans =

   0  -3  -6  -1
  -7  -4  -3  -3
  -1  -3  -7  -6

octave:74> 1+C
ans =

   3   0  -3   2
  -4  -1   0   0
   2   0  -4  -3
```

`log(n)`这里的log底数是e，即$\log_en$或$\ln n$
```c
octave:57> log(2)
ans = 0.6931
octave:58> log(e)
ans = 1
octave:59> A
A =

   4   1   3
   2   3   5

octave:60> log(A)
ans =

   1.3863        0   1.0986
   0.6931   1.0986   1.6094
```
`exp(n)`相当于执行$e^n$
```c
octave:61> exp(1)
ans = 2.7183
octave:62> exp(2)
ans = 7.3891
octave:63> A
A =

   4   1   3
   2   3   5

octave:64> exp(A)
ans =

    54.5982     2.7183    20.0855
     7.3891    20.0855   148.4132
```

`abs()`求绝对值
```c
octave:66> C = randi([-5,2],3,4)    %随机生成一个3×4的矩阵，取值在-5\~2之间
C =

   2  -1  -4   1
  -5  -2  -1  -1
   1  -1  -5  -4

octave:67> abs(-1)
ans = 1
octave:68> abs(C)
ans =

   2   1   4   1
   5   2   1   1
   1   1   5   4
```

`matrix'`转置，矩阵名称后加英文单引号
```c
octave:75> C
C =

   2  -1  -4   1
  -5  -2  -1  -1
   1  -1  -5  -4

octave:76> C'
ans =

   2  -5   1
  -1  -2  -1
  -4  -1  -5
   1  -1  -4
```

`flipud`使矩阵垂直翻转。就是左边变右边，右边变左边。
```c
result1 =

   17    0    0    0    0
    0    5    0    0    0
    0    0   13    0    0
    0    0    0   21    0
    0    0    0    0    9

octave:140> flipud(result1)
ans =

    0    0    0    0    9
    0    0    0   21    0
    0    0   13    0    0
    0    5    0    0    0
   17    0    0    0    0
```

`矩阵<数值`大于啊等于啊都同理。会返回一个同纬度的矩阵，每个元素都是对应位置元素进行比较返回的布尔值
```c
a =

   1   5   3   4

octave:85> a<2
ans =

  1  0  0  0
```

`magic(n)`生成n阶幻方，所谓幻方就是行、列、对角线加起来都是相同的值
```c
octave:95> magic(4)
ans =

   16    2    3   13
    5   11   10    8
    9    7    6   12
    4   14   15    1

octave:96> magic(3)
ans =

   8   1   6
   3   5   7
   4   9   2
```

`find(公式)`
- 找到**向量**中符合的数据并返回其索引
    ```c
    a =
    
       1   5   3   4
    
    octave:88> find(a<3)
    ans = 1
    octave:89> find(a>2)
    ans =
    
       2   3   4
    ```
- 找到**矩阵**中符合的数据
    ```c
    C =
    
    9    7    3    5
    4    2    5   10
    4    1    9   10
    
    octave:114> [r,c]=find(C==1)
    r = 3
    c = 2
    octave:115> [r,c]=find(C>7)
    r =
    
       1
       3
       2
       3
    
    c =
    
       1
       3
       4
       4
    ```

`sum()`求和`prod()`求积
```c
c =

   1
   3
   4
   4

octave:117> sum(c)
ans = 12
octave:118> prod(c)
ans = 48
```

`floor()`向下取整`ceil()`向上取整
```c
a =

   0.5000
   1.5000
   2.0000
   2.0000

octave:120> floor(a)
ans =

   0
   1
   2
   2
   
octave:121> ceil(a)
ans =

   1
   2
   2
   2

```

`max(矩阵)`取最大值
如果是矩阵，那就显示每一列的最大值；如果是向量，那就只显示一个最大值
- `v=max(matrix)`：将矩阵的最大值赋给v，不是硬性要求，v可以换别的名称
- `[v,i]=max(matrix)`：将矩阵最大值及其位置赋值给v和i
- `max(max(matrix))`：取得整个矩阵最大值
- `max(matrix(:))`：取得整个矩阵最大值，这个是将矩阵转化成响亮之后再求最大值
  
```c
c =

   2   1   4   1
   5   2   1   1
   1   1   5   4

octave:78> val = max(c)
val =

   5   2   5   4

octave:79> [val,ind]=max(c)
val =

   5   2   5   4

ind =

   2   2   3   3
```

`max(矩阵,[],1/2)` 取得矩阵每一行或者列的最大值形成一个向量
- 1 取每一列
- 2 取每一行
```c
c =

   8   9   7   9
   9   3   9   5
   7   2   5   9
   7   5   6   5

octave:129> max(c,[],1)
ans =

   9   9   9   9

octave:130> max(c,[],2)
ans =

   9
   9
   9
   7
```

**小练习：**
生成一个幻方，然后测试其每行，每列，对角线之和都是同一个数

```c
octave:131>  m = magic(5)     %生成一个5×5的幻方
m =

   17   24    1    8   15
   23    5    7   14   16
    4    6   13   20   22
   10   12   19   21    3
   11   18   25    2    9

octave:132> sum(m,1)          %每列相加都是65
ans =

   65   65   65   65   65

octave:133> sum(m,2)          %每行相加都是65
ans =

   65
   65
   65
   65
   65
   
octave:135> e=eye(5)          %生成一个单位矩阵
e =

Diagonal Matrix

   1   0   0   0   0
   0   1   0   0   0
   0   0   1   0   0
   0   0   0   1   0
   0   0   0   0   1

octave:136> result1 = e .* m   %单位矩阵同位相乘原矩阵，新矩阵只剩主对角线元素
result1 =

   17    0    0    0    0
    0    5    0    0    0
    0    0   13    0    0
    0    0    0   21    0
    0    0    0    0    9
octave:139> sum(result1(:))   %主对角线相加为65
ans = 65

octave:141> result2 = m .* flipud(e)
result2 =

    0    0    0    0   15
    0    0    0   14    0
    0    0   13    0    0
    0   12    0    0    0
   11    0    0    0    0

octave:142> sum(result2(:))   %副加为65
ans = 65
```

`pinv()`求逆矩阵
```c
m =

   8   1   6
   3   5   7
   4   9   2

octave:145> pinv(m)
ans =

   0.147222  -0.144444   0.063889
  -0.061111   0.022222   0.105556
  -0.019444   0.188889  -0.102778
```

# 绘图
`plot(A,B)`A是横坐标向量，B纵坐标

```c
octave:173> t=[-1:0.01:1];
octave:174> p1 = sin(2*pi*t);
octave:175> plot(t,p1)
```

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f017b0b477a44ea8a124827236b48954~tplv-k3u1fbpfcp-watermark.image)

```c
octave:176> p2 = cos(2*pi*t);
octave:177> plot(t,p2)
```
![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/dac7020a9770470cb84dd10d829ce37a~tplv-k3u1fbpfcp-watermark.image)

画完一张图以后想要画另一张就会重新绘制，用`hold on`可以保留，在原图基础上继续画，并且会自动给你切换颜色。
```c
octave:177> plot(t,p2)
octave:178> hold on
octave:179> plot(t,p1)
```
![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d61c7851e1554173840b1269e7ce58a4~tplv-k3u1fbpfcp-watermark.image)

给横轴纵轴添加标签。
```c
octave:183> xlabel('time')
octave:184> ylabel('speed')
```

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7999d55289454679aa83d468ebc0b31b~tplv-k3u1fbpfcp-watermark.image)

添加图例
```c
legend('cos','sin','y')
```

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7f0a62d183694ca9a2b656e42e2d7ff2~tplv-k3u1fbpfcp-watermark.image)

添加标题
```c
title('my demo')
```

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6efc1d18ec0d44ff846978d97cd80946~tplv-k3u1fbpfcp-watermark.image)

保存文件
```c
print -dpng '文件名'
```
保存之后你的当前目录下边就会多出一个图片。如过你不知道当前目录是啥，输入pwd查看当前目录的路径。

如果你想同时生成多个图片，那在前边加上`figure(n);`则会按顺序生成图片。
```c
octave:194> figure(1);plot(t,p1)
octave:195> figure(2);plot(t,p2)
```

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/377ab6716ef740c798ccc5031c3f1838~tplv-k3u1fbpfcp-watermark.image)

合并显示图像`subplot(n,m,index)`，将图像分成n行m列，取index位置绘图。
```c
octave:197> subplot(2,3,4)
octave:198> plot(t,p1)
octave:199> subplot(2,3,2)
octave:200> plot(t,p2)
```

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/5b540b1a32ff48d891551caeddb70be6~tplv-k3u1fbpfcp-watermark.image)

改变纵横轴取值范围`axis([横轴起始 终止 纵轴起始 终止])`
```c
octave:203> subplot(1,2,1)
octave:204> plot(t,p2)
octave:205> subplot(1,2,2)
octave:206> plot(t,p2)
octave:207> axis([0 1 -2 2])
```
下图两个是同一副图，右边是改变坐标轴范围之后的图像。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/93558a94ceb3495fbbe39377a6a01e3d~tplv-k3u1fbpfcp-watermark.image)

`clf`清空整幅图，`close`不用点右上角×关闭图像

**矩阵可视化**
`imagesc()`

```c
a =

   8   1   6
   3   5   7
   4   9   2

octave:217> imagesc (a)
```

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/090d4ac70bdc488c8d5d807e0af6c5f9~tplv-k3u1fbpfcp-watermark.image)

> 说个题外话，作为一个曾经的生物学生，我好像突然get到了我们以前的热图是怎么出来的。

```c
octave:218> load 6feature.txt   % 还记得这组数据吗，就是我上边给的那两组数据的第一组。
octave:219> imagesc (X6feature)
```
画出来以后图像长这样，恩我真的get到热图怎么画的了。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/76bc23edd1904b0cacb0723a911708df~tplv-k3u1fbpfcp-watermark.image)

现在让我们生成一个热图（假的，不是热图，看起来像而已）
```c
octave:220> r = rand(5,50);
octave:221> imagesc (r)
```

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/2ecf28cdf42844d8befdf3137ac37703~tplv-k3u1fbpfcp-watermark.image)

`colorbar`添加比色的图例。
```c
octave:223> colorbar
```

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/57137c708a2a446887f1e6e2b777457d~tplv-k3u1fbpfcp-watermark.image)

` colormap gray`使其变为灰度图
```c
 colormap gray
```

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/435666a5010d4314b8fcc1c534f8539f~tplv-k3u1fbpfcp-watermark.image)

# 基础语法

- **循环**
    ```c
    for var = expression 
    body 
    endfor
    ```
    ```c
    while (condition) 
    body 
    endwhile
    ```
    - 其实上边的endwhile，endfor还有下边的endif可以直接写为end
    - break、continue也可以用
 - **判断**
     ```c
     if (condition) 
         then-body 
     endif
     ```
     ```c
     if (condition) 
         then-body 
     elseif (condition) 
         elseif-body 
     else 
         else-body 
     endif
     ```
     - 举个栗子
     
         ```c
         v =
        
            8    4    8    2    5
            7    4    6    1    3
           10    6    6    4   10
            6    6    8    8   10
            9    1   10    5    7
        
        > for i = 1:5
        >     for j = 1:5
        >         if v(i,j)<5
        >             continue
        >         else
        >             disp(v(i,j))
        >         endif
        >     endfor
        > endfor
        8
        8
        5
        7
        6
        10
        6
        6
        10
        6
        6
        8
        8
        10
        9
        10
        5
        7
        ```
- **函数**
    - 普无返回值的函数
        ```c
        function name 
            body 
        endfunctio
        ```
    - 带返回值的函数
        ```c
        function [ret-list] = name (arg-list) 
            body 
        endfunction
        ```
- **在octave中写函数：**
    ```c
    >function say_hi(name)
    > str = ['hello' name];
    > disp(str)
    > endfunction
    >say_hi ('Sian')
    helloSian
    ```
- **让octave使用外部的函数。**
    - 在当前目录或者path中放你的函数文件
    - 文件名就是函数名
    - 后缀`.m`

- 举个栗子
    现在写一个函数show_matrix：

    ![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/46580ea166064c82aaffd31c9e915b2f~tplv-k3u1fbpfcp-watermark.image)

    存放在path中，并且命名为show_matrix.m

    ![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c59d8fa46df747b8ba19222f8953ba2a~tplv-k3u1fbpfcp-watermark.image)

    回到octave中声明一个矩阵，看看是否能用该函数成功打印

    ![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/1f675cfce1c645e594d4db2d4ddc2f9d~tplv-k3u1fbpfcp-watermark.image)
- 再整一个带返回值的


    ![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/66ebbbdf5c7e4aa9b9cc3d6c52cb5dc0~tplv-k3u1fbpfcp-watermark.image)
    
    ![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/12a4bff7312b494b837f5f69eb83bc80~tplv-k3u1fbpfcp-watermark.image)


​    