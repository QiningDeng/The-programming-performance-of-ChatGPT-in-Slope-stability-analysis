import numpy as np
from E_reduction import RED
from F_newton import NEW
from G_potential import POT

# 定义函数：用于使用强度折减和牛顿求解器计算位移场和控制变量
def OME(lambda_ini,U_ini,eps,d_lambda,it_newt_max,it_damp_max,tol,r_min,
               r_damp,WEIGHT,B,ESM,Q,f,c0,phi,psi,matrix_G, matrix_K, matrix_lamlda):
    
    omega = 0 # 定义初始控制变量
    n_strain = 6 # 定义应力（应变）分量数
    
    # 计算在位移场U下的对应强度折减系数λ的能量函数J_λ​(U)最小值
    
    c_bar, sin_phi = RED(c0, phi, psi, lambda_ini)
    U, flag= NEW(U_ini, tol, it_newt_max, it_damp_max, r_min, r_damp, WEIGHT, B, ESM,
               Q, f, c_bar, sin_phi, matrix_G, matrix_K, matrix_lamlda)
    
    if flag == 1:
        return U, omega, flag
    
    E = (B @ U.flatten(order='F')[:, np.newaxis]).reshape(n_strain, -1,order='F')
    
    # 调用POT函数，计算每个积分点下的势能值
    Psi = POT(E, c_bar, sin_phi,matrix_G, matrix_K, matrix_lamlda)
    
    J = WEIGHT @ Psi.T - f.flatten(order='F')[Q.flatten(order='F')].T @ \
        U.flatten(order='F')[Q.flatten(order='F')]
    J = J.item()
    
    # 计算在在位移场U下的对应强度折减系数λ-eps的能量函数J_λ-eps​(U)最小值
    
    beta = min(1, eps / d_lambda)
    U_beta = beta * U_ini + (1 - beta) * U
    
    c_bar, sin_phi = RED(c0, phi, psi, lambda_ini - eps)
    
    U_eps, flag = NEW(U_beta, tol, it_newt_max, it_damp_max, r_min, r_damp,WEIGHT,
                      B, ESM, Q, f, c_bar, sin_phi, matrix_G, matrix_K, matrix_lamlda)
    
    if flag == 1:
        return U, omega, flag
    
    E = (B @ U_eps.flatten(order='F')[:, np.newaxis]).reshape(n_strain, -1,order='F')
    
    # 调用POT函数，计算每个积分点下的势能值
    Psi = POT(E, c_bar, sin_phi,matrix_G, matrix_K, matrix_lamlda)
    
    J_eps = WEIGHT @ Psi.T - f.flatten(order='F')[Q.flatten(order='F')].T @ \
            U_eps.flatten(order='F')[Q.flatten(order='F')]
    J_eps = J_eps.item()
    
    # 计算控制变量
    omega = (J_eps - J) / eps
    
    return U, omega, flag
