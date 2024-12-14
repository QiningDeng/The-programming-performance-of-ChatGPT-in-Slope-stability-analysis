# 简介

[In English](README.md)

三维边坡稳定性分析被选为案例研究，以评估 ChatGPT 在辅助解决工程计算问题中的性能。坡体稳定系数的计算采用两种方法，即极限平衡法和强度折减法，通过标准化编程实践使用 Python 实现。本研究提出了一种系统化的 ChatGPT 辅助三维边坡稳定性分析编程方法。

极限平衡法基于三维简化 Janbu 方法。通过对每个潜在滑动区域的几何和力学参数进行迭代计算，随后得出各滑动面对应的稳定系数。通过比较确定最小稳定系数，并将其对应的滑动面视为潜在的临界滑动面。

强度折减法的实现包括三个关键部分：有限元基本公式的计算、强度折减过程的执行，以及非线性弹塑性本构模型的建立与牛顿迭代求解。强度折减法的计算逻辑与理论框架，经许可后，参考了 https://github.com/sysala/SSRM 上的 MATLAB 源代码。

代码测试与验证进一步结合了以下文献中描述的计算理论：

S. Sysala, E. Hrubešová, Z. Michalec, F. Tschuchnigg: "Optimization and variational principles for the shear strength reduction method," International Journal for Numerical and Analytical Methods in Geomechanics, 45 (2021), pp. 2388–2407.

S. Sysala, F. Tschuchnigg, E. Hrubešová, Z. Michalec: "Optimization variant of the shear strength reduction method and its usage for stability of embankments with unconfined seepage," Computers and Structures, 281 (2023), Article 107033.

本案例研究假设三维边坡几何形状固定，重点分析均质土体的稳定性，且不考虑地下水影响。未来的研究将扩展该方法，以分析涉及多层土、地下水渗流、地震等复杂工况下的边坡稳定性问题，从而提高方法的适用性。
