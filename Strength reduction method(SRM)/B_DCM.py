import numpy as np
from C_initial_phase import IP
from D_omega import OME

# 定义函数：用于使用直接延续法求解强度折减系数
def DCM(lambda_init, d_lambda_init, d_lambda_min,step_max, it_newt_max,
        it_damp_max, tol,r_min, r_damp, WEIGHT, B, ESM, Q, f, c0, phi,
        psi, matrix_G, matrix_K, matrix_lamlda):
    
    omega_hist = np.zeros(1000).reshape(1, -1) # 控制变量的存储矩阵
    lambda_hist = np.zeros(1000).reshape(1, -1) # 初始化强度折减系数的存储矩阵
    eps = tol * 10 # 初始化数值计算过程的精度存储矩阵
    
    # 调用 IP 函数，计算第一步迭代数据信息
    U, omega_old, omega, lambda_value = IP(lambda_init, d_lambda_init, d_lambda_min, it_newt_max,
                                    it_damp_max, tol, eps, r_min, r_damp, WEIGHT, B, ESM, Q, f,
                                    c0, phi, psi, matrix_G, matrix_K, matrix_lamlda)
    
    # 存储第1步迭代前后的折减系数及增量
    omega_hist[0,0] = omega_old
    lambda_hist[0,0] = lambda_init
    
    omega_hist[0,1] = omega
    lambda_hist[0,1] = lambda_value
    
    d_omega = omega-omega_old
    d_lambda= lambda_value - lambda_init
    
    # 初始迭代结束，开始后续迭代计算求解折减系数
    
    step = 2
    
    while True:
        print(f"\n  迭代步数：{step + 1}")
        
        lambda_it = lambda_value + d_lambda
        
        # 调用OME函数，计算位移场函数及折减系数
        U_it, omega_it, flag = OME(lambda_it, U, eps, d_lambda, it_newt_max, it_damp_max, tol,
                                   r_min, r_damp, WEIGHT, B, ESM, Q, f, c0, phi, psi,
                                   matrix_G, matrix_K, matrix_lamlda)
        
        d_omega_test = omega_it - omega_hist[0,step - 1]
        
        if (flag == 1) or (d_omega_test < 0):
            print('强度折减系数增量过大，将减少强度折减系数增量后重新开始此步长下的计算')
            d_lambda = d_lambda / 2
        else:
            U = U_it
            omega = omega_it
            lambda_value = lambda_it
            step += 1
            lambda_hist[0,step - 1] = lambda_value
            omega_hist[0,step - 1] = omega
            
            # 显示输出
            print(f"强度折减系数={lambda_value:.4g}, 强度折减系数增量={d_lambda:.4g}, 控制变量={omega:.4g}, 控制变量增量={d_omega:.4g}")
            
            d_omega = d_omega_test
        
        if d_lambda < d_lambda_min:
            print('求解完成，达到最小折减系数增量.')
            break
        if step >= step_max:
            print('求解完成，达到最大计算步长.')
            break
    
    # 剪裁输出数组
    lambda_hist = lambda_hist[:,:step]
    omega_hist = omega_hist[:,:step]
    
    return U, lambda_hist, omega_hist
