import numpy as np
import math
from scipy.sparse import csr_matrix
from H2_constitutive_problem_b import CON_b

# 定义函数：计算阻尼系数
def DAM(it_damp_max, U_it, dU, F, f, B, WEIGHT,c_bar, sin_phi, matrix_G, matrix_K, matrix_lamlda):
    
    dim = 3  
    n_strain = dim * (dim + 1) // 2  # 应变（应力）分量的数量
    
    # 计算初始的 decrease
    F_sparse= csr_matrix(F.flatten(order='F')[:,np.newaxis])
    f_sparse = csr_matrix(f.flatten(order='F')[:,np.newaxis])
    dU_sparse = csr_matrix(dU.flatten(order='F')[:,np.newaxis])
    decrease = np.ndarray.item(((F_sparse - f_sparse).T @ dU_sparse).todense())
    
    # 检查条件
    if math.isnan(decrease) or np.linalg.norm(dU.flatten()) == np.inf or decrease >= 0:
        alpha = 0
        
    else:
        alpha = 1.0
        alpha_min = 0.0
        alpha_max = 1.0
        it_damp = 0
    
        while it_damp < it_damp_max:
            it_damp += 1
            
            # 计算 U_alpha
            U_alpha = U_it + alpha * dU
            E_alpha = (B @ (U_alpha.flatten(order='F')[:,np.newaxis]))
            E_alpha = E_alpha.reshape(n_strain, -1, order='F')
    
            # 处理三维本构问题
            S_alpha = CON_b(E_alpha, c_bar, sin_phi,matrix_G, matrix_K, matrix_lamlda)
    
            # 计算 F_alpha
            F_alpha = B.T @ ((np.tile(WEIGHT, (n_strain, 1)) * S_alpha[0:n_strain, :]).flatten(order='F')).reshape(-1, 1)
            
            # 计算新的 decrease
            F_alpha_sparse = csr_matrix(F_alpha)
            decrease = np.ndarray.item(((F_alpha_sparse - f_sparse).T @ dU_sparse).todense())
            
            # 更新 alpha
            if decrease < 0:
                if alpha == 1.0:
                    break
                alpha_min = alpha
            else:
                alpha_max = alpha
            
            alpha = (alpha_min + alpha_max) / 2
    
    return alpha