import numpy as np

# 定义函数：利用强度折减系数调整强度参数
def RED(c0,phi,psi,lambda_val):
    
    # 折减粘聚力和内摩擦角
    c01 = c0 / lambda_val
    phi1 = np.arctan(np.tan(phi) / lambda_val)
    psi1 = np.arctan(np.tan(psi) / lambda_val)
    
    # 计算beta系数
    beta = np.cos(phi1) * np.cos(psi1) / (1 - np.sin(phi1) * np.sin(psi1))
    
    # 利用beta更新粘聚力和内摩擦角
    c0_lambda = beta * c01
    phi_lambda = np.arctan(beta * np.tan(phi1))
    
    # 计算折减后强度参数
    c_bar = 2 * c0_lambda * np.cos(phi_lambda)
    sin_phi = np.sin(phi_lambda)
    
    return c_bar, sin_phi