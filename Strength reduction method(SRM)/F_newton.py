import numpy as np
from scipy.sparse import csc_matrix
from H1_constitutive_problem_a import CON_a
from I_damping import DAM
from GMRES_GPU import pyamg_GMRES

# 定义函数：牛顿迭代法，用于求解非线性问题
def NEW(U_ini, tol, it_newt_max, it_damp_max, r_min, r_damp, WEIGHT, B, ESM,Q, f,
        c_bar, sin_phi, matrix_G, matrix_K, matrix_lamlda):
    
    # 初始化计算参数
    n_n = U_ini.shape[1] # 节点数
    n_int = WEIGHT.shape[1] # 积分点数
    dim = 3 # 维度 3D
    dU = np.zeros((dim, n_n)) # 牛顿法增量
    U_it = U_ini # 迭代位移初始化
    F = np.zeros((dim, n_n)) # 内力向量
    n_strain = int(dim * (dim + 1) / 2) # 应变（应力）分量个数
    E = np.zeros((n_strain, n_int))  # 应变张量
    
    # 构造本构算子及其导数
    E_flat = E.flatten(order='F')[:, np.newaxis]
    E_flat = B @ (U_it.flatten(order='F'))[:, np.newaxis]
    E = E_flat.reshape(n_strain, n_int, order='F')
    
    # 调用函数CON，建立本构模型以计算应力S和刚度矩阵DS
    S, DS = CON_a(E, c_bar, sin_phi, matrix_G, matrix_K, matrix_lamlda)
    
    # 计算内力
    F_flat = F.flatten(order='F')[:,np.newaxis]
    F_flat = B.T @ np.reshape(np.tile(WEIGHT, (n_strain, 1)) * S[:n_strain, :], (n_strain * n_int, 1), order='F')
    F = F_flat.reshape(dim, n_n, order = 'F')
    
    # 迭代次数初始化
    it = 0
    flag_N = 0
    r = r_damp
    
    # 切向刚度矩阵的构造与正则化
    while True:
        
        it = it + 1
        
        # 构造稀疏矩阵的索引
        AUX = np.arange(1, n_strain * n_int +1 )[:, np.newaxis].reshape(n_strain, n_int, order='F')
        iD = np.tile(AUX, (n_strain, 1))
        jD = np.kron(AUX, np.ones((n_strain, 1)))
        vD = np.tile(WEIGHT, (n_strain**2, 1)) * DS  # 加权材料刚度矩阵导数
        
        # 组装稀疏矩阵
        D_p = csc_matrix((vD.flatten(order='F'), (iD.flatten(order='F') - 1,
                                                  jD.flatten(order='F') - 1)),
                         shape=(n_strain * n_int, n_strain * n_int))
        
        # 计算切向刚度矩阵并保持对称性
        K_tangent = B.T @ D_p @ B
        K_tangent = (K_tangent + K_tangent.T) / 2
        
        # 正则化的刚度矩阵
        K_r = r_min * ESM + (1 - r_min) * K_tangent
        
        # 设置求解精度和最大迭代次数
        tol = 1e-6       # 设置求解精度
        maxiter = 5000   # 设置最大迭代次数
        
        # 调用迭代求解器进行求解
        X =pyamg_GMRES(K_r, Q, f, F, tol,  maxiter)
        
        # 获取 Q 中为 True 的元素的索引
        row_indices, col_indices = np.where(Q)
        
        # 将行索引和列索引组合并按列顺序排序
        sorted_indices = sorted(zip(col_indices, row_indices))
        col_indices_sorted, row_indices_sorted = zip(*sorted_indices)
        
        row_indices = np.array(row_indices_sorted)
        col_indices = np.array(col_indices_sorted)
        
        dU[row_indices, col_indices] = X[:len(row_indices)].flatten()
        
        alpha = DAM(it_damp_max, U_it, dU, F, f, B, WEIGHT,
                        c_bar, sin_phi, matrix_G, matrix_K, matrix_lamlda)
        
        if alpha == 0 :
            while alpha == 0 :
                
                print("阻尼系数为 0 ，刚度矩阵需要重新正则化")
                
                r = r * 10
                
                # 正则化的刚度矩阵
                K_r = r * ESM + (1 - r) * K_tangent
                
                # 调用迭代求解器进行求解
                X =pyamg_GMRES(K_r, Q, f, F, tol,  maxiter)
                
                # 获取 Q 中为 True 的元素的索引
                row_indices, col_indices = np.where(Q)
                
                # 将行索引和列索引组合并按列顺序排序
                sorted_indices = sorted(zip(col_indices, row_indices))
                col_indices_sorted, row_indices_sorted = zip(*sorted_indices)
                
                row_indices = np.array(row_indices_sorted)
                col_indices = np.array(col_indices_sorted)
                
                dU[row_indices, col_indices] = X[:len(row_indices)].flatten()
                
                alpha = DAM(it_damp_max, U_it, dU, F, f, B, WEIGHT,
                                c_bar, sin_phi, matrix_G, matrix_K, matrix_lamlda)
                
            r = r / 20  # 正则化后减小 r 值
        
        # 更新位移
        U_it += alpha * dU
        
        # 重新计算应变
        E_flat = E.flatten(order='F')[:, np.newaxis]
        E_flat = B @ (U_it.flatten(order='F'))[:, np.newaxis]
        E = E_flat.reshape(n_strain, n_int, order='F')
        
        # 再次计算应力
        S, DS = CON_a(E, c_bar, sin_phi, matrix_G, matrix_K, matrix_lamlda)
        
        # 再次计算内力
        F_flat = F.flatten(order='F')[:,np.newaxis]
        F_flat = B.T @ np.reshape(np.tile(WEIGHT, (n_strain, 1)) * S[:n_strain, :], (n_strain * n_int, 1), order='F')
        F = F_flat.reshape(dim, n_n, order = 'F')
        
        # 利用欧几里得范数收敛性判断
        f_Q = csc_matrix(f.flatten(order='F')[Q.flatten(order='F')][:, np.newaxis])
        F_Q = csc_matrix(F.flatten(order='F')[Q.flatten(order='F')][:, np.newaxis])
        f_Q_F_Q = (f_Q - F_Q).toarray().ravel()
        
        criterion = np.linalg.norm(f_Q_F_Q) / np.linalg.norm(f_Q.toarray())
        
        if criterion < tol:
            print(f'牛顿法收敛: 迭代次数={it}, 停止准则={criterion:.6e}')
            break
        
        # 检查是否达到最大迭代次数
        if np.isnan(criterion) or it == it_newt_max:
            flag_N = 1
            print(f'牛顿法不收敛: 停止准则={criterion:.6e} 将调整折减系数增量重新此步长下的计算')
            break
    
    return U_it, flag_N