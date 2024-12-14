import cupyx.scipy.sparse.linalg
import numpy as np
import cupy as cp

# 定义GMRES（广义最小残差法）迭代求解器（使用GPU并行计算优化）
def pyamg_GMRES(K_r, Q, f, F, tol,  maxiter):
    
    # 计算牛顿增量
    K_r_QQ = K_r[np.ix_(Q.flatten(order='F'),Q.flatten(order='F'))]
    f_Q = f.flatten(order='F')[Q.flatten(order='F')][:, np.newaxis]
    F_Q = F.flatten(order='F')[Q.flatten(order='F')][:, np.newaxis]
    f_Q_F_Q = (f_Q - F_Q).ravel()
    K_r_QQ_gpu = cupyx.scipy.sparse.csr_matrix(K_r_QQ)
    f_Q_F_Q_gpu = cp.array(f_Q_F_Q).reshape(-1, 1)
    
    # 使用GMRES求解线性方程组 K_r_QQ*X=f_Q_F_Q
    X, info = cupyx.scipy.sparse.linalg.gmres(K_r_QQ_gpu, f_Q_F_Q_gpu, tol=tol,  maxiter =maxiter)
    X = X.get()
    
    if info == 0:
        
        print(f"GMRES迭代求解牛顿增量达到预定求解精度 {tol}")
    else :
        final_residual = np.linalg.norm(f_Q_F_Q - K_r_QQ @ X) # 计算求解基于残差的容差值
        print(f"GMRES迭代求解牛顿增量达到最大迭代次数 {info} 最大迭代次数下的基于残差的容差值{final_residual:.4g}")
    
    # 清理 GPU 内存
    del K_r_QQ_gpu, f_Q_F_Q_gpu  # 删除变量
    cp.get_default_memory_pool().free_all_blocks()  # 释放未使用的内存块
    
    # 释放当前所有未使用的内存块
    cp.get_default_memory_pool().free_all_blocks()
    return X