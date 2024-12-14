import numpy as np
from D_omega import OME

# 定义函数：用于计算第一步迭代数据信息
def IP(lambda1,d_lambda1,d_lambda_min,it_newt_max,it_damp_max,tol,eps,r_min,r_damp,
       WEIGHT,B,ESM,Q,f,c0,phi,psi, matrix_G, matrix_K, matrix_lamlda) :
    
    n_n = Q.shape[1]
    U_ini = np.zeros((3, n_n)).astype(np.float64)
    print("\n")
    print('  迭代步数：{}'.format(1))
    
    # 调用OME函数，计算第一步下的位移场函数和强度折减系数
    U_1, omega_1, flag = OME(lambda1,U_ini,eps,1000,it_newt_max,it_damp_max,tol,r_min,r_damp,
                          WEIGHT,B,ESM,Q,f,c0,phi,psi, matrix_G, matrix_K, matrix_lamlda)
    
    if flag == 1:
        raise ValueError('初始强度折减系数过大,考虑降低初始强度折减系数.')
    
    print(f'强度折减系数={lambda1:.4g}, 强度折减系数增量={d_lambda1:.4g}, 控制变量={omega_1:.4g}')
    print("\n")
    print('  迭代步数：{}'.format(2))
    
    d_lambda = d_lambda1
    
    while True:
        
        lambda_it = lambda1 + d_lambda
        
        U_it, omega_it, flag = OME(lambda_it,U_1,eps,d_lambda,it_newt_max,it_damp_max,tol,
                                   r_min,r_damp,WEIGHT,B,ESM,Q,f,c0,phi,psi,matrix_G,
                                   matrix_K, matrix_lamlda)
        
        if (flag == 1) or (omega_it <= omega_1):
            d_lambda = d_lambda / 2
        else:
            U_2 = U_it
            omega_2 = omega_it
            lambda2 = lambda_it
            break
    
        if d_lambda < d_lambda_min:
            raise ValueError('三维边坡稳定性系数FoS可能等同于初始强度折减系数，请考虑降低初始折减系数重新进行计算.')
    
    # 打印结果
    print(f'强度折减系数={lambda2:.4g}, 强度折减系数增量={lambda2 - lambda1:.4g}, 控制变量={omega_2:.4g}, 控制变量增量={omega_2 - omega_1:.4g}')
    
    return U_2,omega_1,omega_2,lambda2