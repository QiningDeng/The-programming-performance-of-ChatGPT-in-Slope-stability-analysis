import numpy as np

# 定义函数：建立基于摩尔库伦屈服准则的本构模型
def CON_b(E, c_bar, sin_phi, shear, bulk, lame):
    
    sin_psi = sin_phi

    # 线性本构算子
    IDENT = np.diag([1, 1, 1, 1/2, 1/2, 1/2])
    iota = np.array([1, 1, 1, 0, 0, 0]).reshape(6, 1)
    VOL = iota @ iota.T
    DEV = np.diag([1, 1, 1, 1/2, 1/2, 1/2]) - VOL / 3
    ELAST = 2 * DEV.flatten(order='F')[:,np.newaxis] @ shear + VOL.flatten(order='F')[:,np.newaxis] * bulk
    
    # 计算积分点数量
    n_int = E.shape[1]
    E_trial = E  # 应变张量的表示
    E_tr = IDENT @ E_trial  # 应力的表示
    
    # 计算应力表示下试应变的平方
    E_square = np.array([
        E_tr[0, :]**2 + E_tr[3, :]**2 + E_tr[5, :]**2,
        E_tr[1, :]**2 + E_tr[3, :]**2 + E_tr[4, :]**2,
        E_tr[2, :]**2 + E_tr[4, :]**2 + E_tr[5, :]**2,
        E_tr[0, :] * E_tr[3, :] + E_tr[1, :] * E_tr[3, :] + E_tr[4, :] * E_tr[5, :],
        E_tr[3, :] * E_tr[5, :] + E_tr[1, :] * E_tr[4, :] + E_tr[2, :] * E_tr[4, :],
        E_tr[0, :] * E_tr[5, :] + E_tr[3, :] * E_tr[4, :] + E_tr[2, :] * E_tr[5, :]
    ])
    
    # 计算不变量
    I1 = (E_tr[0, :] + E_tr[1, :] + E_tr[2, :]).reshape(1, -1)
    I2 = (E_tr[0, :] * E_tr[1, :] + E_tr[0, :] * E_tr[2, :] + E_tr[1, :] * E_tr[2, :] - \
         E_tr[3, :]**2 - E_tr[4, :]**2 - E_tr[5, :]**2).reshape(1, -1)
    I3 = (E_tr[0, :] * E_tr[1, :] * E_tr[2, :] - E_tr[2, :] * E_tr[3, :]**2 - \
         E_tr[1, :] * E_tr[5, :]**2 - E_tr[0, :] * E_tr[4, :]**2 + \
         2 * E_tr[3, :] * E_tr[4, :] * E_tr[5, :]).reshape(1, -1)
    
    # 计算Q和R不变量
    Q = np.maximum(0, (1/9) * (I1**2 - 3 * I2))
    R = (1/54) * (-2 * I1**3 + 9 * (I1 * I2) - 27 * I3)
    theta0 = np.zeros(n_int).reshape(1, -1)
    mask = Q != 0
    theta0[mask] = R[mask] / np.sqrt(Q[mask]**3)
    theta = np.arccos(np.clip(theta0, -1, 1)) / 3  # Lode角
    
    # 计算特征值
    eig_1 = -2 * np.sqrt(Q) * np.cos(theta + 2 * np.pi / 3) + I1 / 3
    eig_2 = -2 * np.sqrt(Q) * np.cos(theta - 2 * np.pi / 3) + I1 / 3
    eig_3 = -2 * np.sqrt(Q) * np.cos(theta) + I1 / 3
    
    # 决策标准的临界值
    f_tr = 2 * shear * ((1 + sin_phi) * eig_1 - (1 - sin_phi) * eig_3) + \
           2 * lame * sin_phi * I1 - c_bar
    gamma_sl = (eig_1 - eig_2) / (1 + sin_psi)
    gamma_sr = (eig_2 - eig_3) / (1 - sin_psi)
    gamma_la = (eig_1 + eig_2 - 2 * eig_3) / (3 - sin_psi)
    gamma_ra = (2 * eig_1 - eig_2 - eig_3) / (3 + sin_psi)
    
    # 计算塑性乘子候选值
    denom_s = 4 * lame * sin_phi * sin_psi + 4 * shear * (1 + sin_phi * sin_psi)
    denom_l = 4 * lame * sin_phi * sin_psi + shear * (1 + sin_phi) * (1 + sin_psi) + \
              2 * shear * (1 - sin_phi) * (1 - sin_psi)
    denom_r = 4 * lame * sin_phi * sin_psi + 2 * shear * (1 + sin_phi) * (1 + sin_psi) + \
              shear * (1 - sin_phi) * (1 - sin_psi)
    denom_a = 4 * bulk * sin_phi * sin_psi
    
    lambda_s = f_tr / denom_s
    lambda_l = (shear * ((1 + sin_phi) * (eig_1 + eig_2) - 2 * (1 - sin_phi) * eig_3) + \
                2 * lame * sin_phi * I1 - c_bar) / denom_l
    lambda_r = (shear * (2 * (1 + sin_phi) * eig_1 - (1 - sin_phi) * (eig_2 + eig_3)) + \
                2 * lame * sin_phi * I1 - c_bar) / denom_r
    lambda_a = (2 * bulk * sin_phi * I1 - c_bar) / denom_a
    
    S = np.zeros((6, n_int))  # 初始化应力矩阵
    
    # 弹性响应
    test_el = (f_tr <= 0).astype(bool)
    lame_el = lame.flatten(order='F')[test_el.flatten(order='F')]
    shear_el = shear.flatten(order='F')[test_el.flatten(order='F')]
    
    if lame_el.size > 0 and shear_el.size > 0:  # 检查lame_el和shear_el是否为空
        term1 = np.tile(lame_el, (6, 1)) * (VOL @ E_trial[:, test_el.ravel(order='F')])
        term2 = 2 * np.tile(shear_el, (6, 1)) * (IDENT @ E_trial[:, test_el.ravel(order='F')])
        result = term1 + term2
    else:
        result = np.array([])  # 如果lame_el和shear_el为空，结果为空矩阵
    
    # 如果结果不是空矩阵，则赋值给S
    if result.size > 0:
        S[:, test_el.ravel(order='F')] = result
    
    # 回到屈服面的光滑部分
    test_s = (lambda_s <= np.minimum(gamma_sl, gamma_sr)) & (~test_el)
    lame_s = lame.flatten(order='F')[test_s.flatten(order='F')]
    shear_s = shear.flatten(order='F')[test_s.flatten(order='F')]
    sin_phi_s = sin_phi.flatten(order='F')[test_s.flatten(order='F')]
    sin_psi_s = sin_psi.flatten(order='F')[test_s.flatten(order='F')]
    eig_1_s = eig_1.flatten(order='F')[test_s.flatten(order='F')]
    eig_2_s = eig_2.flatten(order='F')[test_s.flatten(order='F')]
    eig_3_s = eig_3.flatten(order='F')[test_s.flatten(order='F')]
    I1_s = I1.flatten(order='F')[test_s.flatten(order='F')]
    E_square_s = E_square[:, test_s.ravel(order='F')]
    E_tr_s = E_tr[:, test_s.ravel(order='F')]
    lambda_s = lambda_s.flatten(order='F')[test_s.flatten(order='F')]
    
    # 计算特征投影
    denom_s1 = (eig_1_s - eig_2_s) * (eig_1_s - eig_3_s)
    denom_s2 = (eig_2_s - eig_1_s) * (eig_2_s - eig_3_s)
    denom_s3 = (eig_3_s - eig_1_s) * (eig_3_s - eig_2_s)
    
    Eig_1_s = (np.ones((6,1)) @ (1 / denom_s1[:,np.newaxis].T)) * (E_square_s - np.ones((6,1)) @ \
              (eig_2_s + eig_3_s)[:,np.newaxis].T * E_tr_s + iota @ (eig_2_s * eig_3_s)[:,np.newaxis].T)
    
    Eig_2_s = (np.ones((6,1)) @ (1 / denom_s2[:,np.newaxis].T)) * (E_square_s - np.ones((6,1)) @ \
              (eig_1_s + eig_3_s)[:,np.newaxis].T * E_tr_s + iota @ (eig_1_s * eig_3_s)[:,np.newaxis].T)
    
    Eig_3_s = (np.ones((6,1)) @ (1 / denom_s3[:,np.newaxis].T)) * (E_square_s - np.ones((6,1)) @ \
              (eig_1_s + eig_2_s)[:,np.newaxis].T * E_tr_s + iota @ (eig_1_s * eig_2_s)[:,np.newaxis].T)
    
    # 计算主应力
    sigma_1_s = lame_s * I1_s + 2 * shear_s * eig_1_s - lambda_s * (2 * lame_s * sin_psi_s + 2 * shear_s * (1 + sin_psi_s))
    sigma_2_s = lame_s * I1_s + 2 * shear_s * eig_2_s - lambda_s * (2 * lame_s * sin_psi_s)
    sigma_3_s = lame_s * I1_s + 2 * shear_s * eig_3_s - lambda_s * (2 * lame_s * sin_psi_s - 2 * shear_s * (1 - sin_psi_s))
    
    # 计算未知的应力张量
    S[:, test_s.ravel(order='F')] = (np.ones((6, 1)) @ sigma_1_s[:,np.newaxis].T) * Eig_1_s + (np.ones((6, 1)) @ \
                                    sigma_2_s[:,np.newaxis].T)* Eig_2_s + (np.ones((6, 1)) @ sigma_3_s[:,np.newaxis].T)*\
                                    Eig_3_s
    
    # 返回屈服面左边缘
    test_l = (gamma_sl < gamma_sr) & (lambda_l >= gamma_sl) & \
             (lambda_l <= gamma_la) & ~(test_el | test_s)
    lame_l = lame.flatten(order='F')[test_l.flatten(order='F')]
    shear_l = shear.flatten(order='F')[test_l.flatten(order='F')]
    sin_phi_l = sin_phi.flatten(order='F')[test_l.flatten(order='F')]
    sin_psi_l = sin_psi.flatten(order='F')[test_l.flatten(order='F')]
    eig_1_l = eig_1.flatten(order='F')[test_l.flatten(order='F')]
    eig_2_l = eig_2.flatten(order='F')[test_l.flatten(order='F')]
    eig_3_l = eig_3.flatten(order='F')[test_l.flatten(order='F')]
    I1_l = I1.flatten(order='F')[test_l.flatten(order='F')]
    lambda_l = lambda_l.flatten(order='F')[test_l.flatten(order='F')]
    E_square_l = E_square[:, test_l.ravel(order='F')]
    E_tr_l = E_tr[:, test_l.ravel(order='F')]
    
    if lame_l.size == 0:
        nt_l = 0
    else:
        nt_l = lame_l.shape[0]
    
    # 特征投影
    denom_l3 = (eig_3_l - eig_1_l) * (eig_3_l - eig_2_l)
    Eig_3_l = (np.ones((6, 1)) * (1 / denom_l3[:,np.newaxis].T)) * (E_square_l - 
              (np.ones((6, 1)) * (eig_1_l + eig_2_l)[:,np.newaxis].T) * E_tr_l + 
              iota * (eig_1_l * eig_2_l)[:,np.newaxis].T)
    Eig_12_l = np.vstack([np.ones((3, nt_l)), np.zeros((3, nt_l))]) - Eig_3_l
    
    # 主应力
    sigma_1_l = lame_l * I1_l + shear_l * (eig_1_l + eig_2_l) - \
                lambda_l * (2 * lame_l * sin_psi_l + shear_l * (1 + sin_psi_l))
    sigma_3_l = lame_l * I1_l + 2 * shear_l * eig_3_l - \
                lambda_l * (2 * lame_l * sin_psi_l - 2 * shear_l * (1 - sin_psi_l))
    
    # 未知应力张量
    S[:, test_l.ravel(order='F')] = (np.ones((6, 1)) @ sigma_1_l[:,np.newaxis].T) * Eig_12_l + \
                                    (np.ones((6, 1)) @ sigma_3_l[:,np.newaxis].T) * Eig_3_l
    
    # 返回屈服面右边缘
    test_r = (gamma_sl > gamma_sr) & (lambda_r >= gamma_sr) & \
             (lambda_r <= gamma_ra) & ~(test_el | test_s)
    lame_r = lame.flatten(order='F')[test_r.flatten(order='F')]
    shear_r = shear.flatten(order='F')[test_r.flatten(order='F')]
    sin_phi_r = sin_phi.flatten(order='F')[test_r.flatten(order='F')]
    sin_psi_r = sin_psi.flatten(order='F')[test_r.flatten(order='F')]
    eig_1_r = eig_1.flatten(order='F')[test_r.flatten(order='F')]
    eig_2_r = eig_2.flatten(order='F')[test_r.flatten(order='F')]
    eig_3_r = eig_3.flatten(order='F')[test_r.flatten(order='F')]
    I1_r = I1.flatten(order='F')[test_r.flatten(order='F')]
    lambda_r = lambda_r.flatten(order='F')[test_r.flatten(order='F')]
    
    if lame_r.size == 0:
        nt_r = 0
    else:
        nt_r = lame_r.shape[0]
    
    # 特征投影
    denom_r1 = (eig_1_r - eig_2_r) * (eig_1_r - eig_3_r)
    Eig_1_r = (np.ones((6, 1)) @ (1 / denom_r1[:,np.newaxis].T)) * (E_square[:, test_r.ravel(order='F')] - 
              (np.ones((6, 1)) @ (eig_2_r + eig_3_r)[:,np.newaxis].T) * E_tr[:, test_r.ravel(order='F')] + 
              iota @ (eig_2_r * eig_3_r)[:,np.newaxis].T)
    Eig_23_r = np.vstack([np.ones((3, nt_r)), np.zeros((3, nt_r))]) - Eig_1_r
    
    # 主应力
    sigma_1_r = lame_r * I1_r + 2 * shear_r * eig_1_r - \
                lambda_r * (2 * lame_r * sin_psi_r + 2 * shear_r * (1 + sin_psi_r))
    sigma_3_r = lame_r * I1_r + shear_r * (eig_2_r + eig_3_r) - \
                lambda_r * (2 * lame_r * sin_psi_r - shear_r * (1 - sin_psi_r))
    
    # 未知应力张量
    S[:, test_r.ravel(order='F')] = (np.ones((6, 1)) * sigma_1_r[:,np.newaxis].T) * Eig_1_r + \
                                    (np.ones((6, 1)) * sigma_3_r[:,np.newaxis].T) * Eig_23_r
    
    # 返回屈服面顶点
    test_a = ~(test_el | test_s | test_l | test_r)
    lambda_a = lambda_a.flatten(order='F')[test_a.flatten(order='F')]
    
    if lambda_a.size == 0:
        nt_a = 0
    else:
        nt_a = lambda_a.shape[0]
    
    sigma_1_a = c_bar.flatten(order='F')[test_a.flatten(order='F')] / \
                (2 * sin_phi.flatten(order='F')[test_a.flatten(order='F')])
    
    S[:, test_a.ravel(order='F')] = iota @ sigma_1_a[:,np.newaxis].T
    
    return S