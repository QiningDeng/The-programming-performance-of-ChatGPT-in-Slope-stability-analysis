# 简介

[In English](README.md)

三维边坡稳定性分析作为案例研究，用于评估 ChatGPT 在协助工程计算任务中的表现。采用 Python 编程语言，通过标准化的编程实践实现了两种计算边坡稳定性系数的方法，即极限平衡法和强度折减法。此外，提出了一种系统的 ChatGPT 辅助编程方法，用于三维边坡稳定性分析。

极限平衡法基于三维简化 Janbu 方法。通过迭代计算潜在滑动面区域内的几何参数和力学参数，随后计算每个滑动面的稳定性系数。通过比较，识别出最小的稳定性系数，并确定相应的滑动面为潜在的临界滑动面。

强度折减法通过以下三个关键部分实现：有限元基本方程的计算、强度折减过程以及基于牛顿迭代法的非线性弹塑性本构模型的建立。

强度折减法的代码测试和验证与以下参考文献中的计算理论保持一致：

S. Sysala, E. Hrubešová, Z. Michalec, F. Tschuchnigg: “Optimization and variational principles for the shear strength reduction method,” International Journal for Numerical and Analytical Methods in Geomechanics, 2021, 45(16): 2388-2407. 

S. Sysala, F. Tschuchnigg, E. Hrubešová, Z. Michalec: “Optimization variant of the shear strength reduction method and its usage for stability of embankments with unconfined seepage,” Computers and Structures, 2023, 281: 107033.

该案例研究假设边坡三维几何形状为梯形，重点分析均质土体的稳定性，未考虑地下水影响。未来工作将扩展该方法，以分析多层土、地下水渗流、地震作用及其他复杂工况下的场景，从而提升方法的适用性。

# 运行环境

代码测试运行所使用的环境如下，供参考：

Python = 3.12.3

极限平衡法下的Python运行环境需求程序包 numpy=1.26.4、pandas=2.2.2、matplotlib=3.9.2、tkinter 及 sys。

强度折减法下的Python运行环境需求程序包 numpy=1.26.4、scipy=1.13.1、cupy-cuda12x=13.3.0。其中，cupy-cuda12x为实现GPU加速稀疏矩阵线性方程组迭代求解过程的必要程序包。依据不同的计算机显卡类型及驱动程序版本需要进行不同的CUDA环境配置，具体可参见CUDA项目“[https://github.com/cupy/cupy](https://github.com/cupy/cupy)”。

请下载完整的存储库的ZIP文件进行使用。
