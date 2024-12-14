# Introduction

[中文版本 (In Chinese)](README_zh.md)

Three-dimensional slope stability analysis has been selected as a case study to evaluate ChatGPT's performance in assisting with engineering computation tasks. Two methods for calculating slope stability coefficients, namely the limit equilibrium method and the strength reduction method, are implemented using Python through standardized programming practices. A systematic ChatGPT-assisted programming approach for 3D slope stability analysis is proposed.

The limit equilibrium method is based on the three-dimensional simplified Janbu method. Geometric and mechanical parameters within each potential sliding surface region are calculated iteratively, followed by the computation of stability coefficients for each sliding surface. The minimum stability coefficient is identified through comparison, and the corresponding sliding surface is determined as the potential critical surface.

The strength reduction method is realized through three key components: the computation of finite element basic equations, the strength reduction process, and the establishment of a nonlinear elastic-plastic constitutive model with Newton's iterative solution. The computational logic and theoretical framework of the strength reduction method, with the owner's consent, reference the MATLAB source code available at https://github.com/sysala/SSRM.

Code testing and validation are further aligned with the computational theory described in the following references:

S. Sysala, E. Hrubešová, Z. Michalec, F. Tschuchnigg: "Optimization and variational principles for the shear strength reduction method," International Journal for Numerical and Analytical Methods in Geomechanics, 45 (2021), pp. 2388–2407.

S. Sysala, F. Tschuchnigg, E. Hrubešová, Z. Michalec: "Optimization variant of the shear strength reduction method and its usage for stability of embankments with unconfined seepage," Computers and Structures, 281 (2023), Article 107033.

The case study assumes a fixed three-dimensional slope geometry, focusing on the stability analysis of homogeneous soil without considering groundwater effects. Future work will extend the methodology to analyze scenarios involving multi-layered soil, groundwater seepage, seismic effects, and other complex conditions to enhance applicability.

# Operating Environment

The environment used for testing the code is as follows for reference:

Python = 3.12.3

For the Python operating environment required under the Limit Equilibrium Method, the necessary packages are: numpy=1.26.4, pandas=2.2.2, matplotlib=3.9.2, tkinter, and sys.

For the Python operating environment required under the Strength Reduction Method, the necessary packages are: numpy=1.26.4, scipy=1.13.1, cupy-cuda12x=13.3.0. Among them, cupy-cuda12x is an essential package for implementing GPU-accelerated iterative solving of sparse matrix linear equations. Depending on the type of computer graphics card and the version of the drivers, different CUDA environment configurations are required. For more details, refer to the CUDA project at “https://github.com/NVIDIA/cuda-samples”.

Please download the ZIP file of the complete repository for use.
