# 机器学习--Lab1
## 1. 基本介绍
### 1.1 目的
本次Lab主要是实现两种方法的SVM模型并对比其准确度和时间。其中，两种方法分别是：使用sklearn包中的线性SVM模型和使用梯度下降法的手动实现的SVM模型。
### 1.2 环境
#### 1.2.1 python环境
本次使用的python版本为3.7.9 64bit。
#### 1.2.2 包环境
通过pip freeze将环境生成requirements.txt文件，如下所示：
- numpy\==1.21.4
- scikit_learn\==1.1.3
#### 1.2.3 机器环境
理论上，本代码在所有64bit的机器上均能正常运行。本次实验所用的环境为：
- CPU: Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz   2.50 GHz
- RAM: 4GB 1867MHZ
## 2. 使用方法
1. 使用pip或者conda配置环境，安装上述所述包环境。
2. 在svm_gradient.ipynb中，是使用gradient decent的脚本，在开头设置了相关配置以及打印输出，可以进行修改。
3. 在svm_sklearn.ipynb中，是使用sklearn的脚本，在开头设置了相关配置以及打印输出，可以进行修改。
