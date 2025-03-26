# 📚Introduction

Three-dimensional slope stability analysis has been selected as a case study to evaluate ChatGPT's performance in assisting with engineering computation tasks. Two methods for calculating slope stability coefficients, namely the limit equilibrium method and the strength reduction method, are implemented using Python through standardized programming practices. A systematic ChatGPT-assisted programming approach for 3D slope stability analysis is proposed.

The limit equilibrium method is based on the three-dimensional simplified Janbu method. Geometric and mechanical parameters within each potential sliding surface region are calculated iteratively, followed by the computation of stability coefficients for each sliding surface. The minimum stability coefficient is identified through comparison, and the corresponding sliding surface is determined as the potential critical surface.

The strength reduction method is realized through three key components: the computation of finite element basic equations, the strength reduction process, and the establishment of a nonlinear elastic-plastic constitutive model with Newton's iterative solution. 

Code testing and validation under strength reduction method are further aligned with the computational theory described in the following references:

S. Sysala, E. Hrubešová, Z. Michalec, F. Tschuchnigg: "Optimization and variational principles for the shear strength reduction method," International Journal for Numerical and Analytical Methods in Geomechanics, 45 (2021), pp. 2388–2407.

S. Sysala, F. Tschuchnigg, E. Hrubešová, Z. Michalec: "Optimization variant of the shear strength reduction method and its usage for stability of embankments with unconfined seepage," Computers and Structures, 281 (2023), Article 107033.

The case study assumes that the three-dimensional geometry of the slope is trapezoid, focusing on the stability analysis of homogeneous soil without considering groundwater effects. Future work will extend the methodology to analyze scenarios involving multi-layered soil, groundwater seepage, seismic effects, and other complex conditions to enhance applicability.

# ⚙️Operating Environment

The environment used for testing the code is as follows for reference:

Python = 3.12.3

For the Python operating environment required under the Limit Equilibrium Method, the necessary packages are: numpy=1.26.4, pandas=2.2.2, matplotlib=3.9.2, tkinter and sys.

For the Python operating environment required under the Strength Reduction Method, the necessary packages are: numpy=1.26.4, scipy=1.13.1 and cupy-cuda12x=13.3.0. Among them, cupy-cuda12x is an essential package for implementing GPU-accelerated iterative solving of sparse matrix linear equations. Depending on the type of computer graphics card and the version of the drivers, different CUDA environment configurations are required. For more details, refer to the CUDA project at “[https://github.com/cupy/cupy](https://github.com/cupy/cupy)”.

Please download the ZIP file of the complete repository for use.

# 📊Code accuracy verification

The strength reduction method can accurately simulate the three-dimensional stress distribution and nonlinear behavior of slope, and deeply reveal the key characteristics of slope stability. Therefore, the strength reduction method is temporarily chosen as the representative method to evaluate the generalization ability of the scheme.

Based on the general range of slope parameters, a small validation dataset of 70 input parameter combinations was built by random sampling.The statistical distribution diagram of the accuracy quantification evaluation indicators of the validation dataset is shown as follows:
<p float="left">
  <img src="https://github.com/user-attachments/assets/c31fec7d-6aa4-48ae-8dc2-56bb492e56cc" width="45%" />
  <img src="https://github.com/user-attachments/assets/bcd7a594-4016-4c33-8e84-6747e50585b5" width="45%" />
</p>

In the future, the number of validation data sets will continue to be expanded to better evaluate the calculation accuracy under the strength reduction method.

# ✨Contact

If you have any questions, please email `1974739605ngyx@gmail.com`.
