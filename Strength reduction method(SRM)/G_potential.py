import numpy as np

# 定义函数:计算每个积分点下的势能值
def POT(E, c_bar, sin_phi, matrix_G, matrix_K, matrix_lamlda):
    
    # 积分点的数量
    n_int = E.shape[1]

    # 试应变和辅助数组
    E_trial = E         # 应变表示
    IDENT = np.diag([1, 1, 1, 1/2, 1/2, 1/2]) # 恒等算子
    E_tr = IDENT @ E_trial  # 应力表示
    
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
    f_tr = 2 * matrix_G * ((1 + sin_phi) * eig_1 - (1 - sin_phi) * eig_3) + \
           2 * matrix_lamlda * sin_phi * I1 - c_bar
    gamma_sl = (eig_1 - eig_2) / (1 + sin_phi)
    gamma_sr = (eig_2 - eig_3) / (1 - sin_phi)
    gamma_la = (eig_1 + eig_2 - 2 * eig_3) / (3 - sin_phi)
    gamma_ra = (2 * eig_1 - eig_2 - eig_3) / (3 + sin_phi)
    
    # 计算塑性乘子候选值
    denom_s = 4 * matrix_lamlda * sin_phi * sin_phi + 4 * matrix_G * (1 + sin_phi * sin_phi)
    denom_l = 4 * matrix_lamlda * sin_phi * sin_phi + matrix_G * (1 + sin_phi) * (1 + sin_phi) + \
              2 * matrix_G * (1 - sin_phi) * (1 - sin_phi)
    denom_r = 4 * matrix_lamlda * sin_phi * sin_phi + 2 * matrix_G * (1 + sin_phi) * (1 + sin_phi) + \
              matrix_G * (1 - sin_phi) * (1 - sin_phi)
    denom_a = 4 * matrix_K * sin_phi * sin_phi
    
    lambda_s = f_tr / denom_s
    lambda_l = (matrix_G * ((1 + sin_phi) * (eig_1 + eig_2) - 2 * (1 - sin_phi) * eig_3) + \
                2 * matrix_lamlda * sin_phi * I1 - c_bar) / denom_l
    lambda_r = (matrix_G * (2 * (1 + sin_phi) * eig_1 - (1 - sin_phi) * (eig_2 + eig_3)) + \
                2 * matrix_lamlda * sin_phi * I1 - c_bar) / denom_r
    lambda_a = (2 * matrix_K * sin_phi * I1 - c_bar) / denom_a
    
    # 确定潜能 Psi
    Psi = np.zeros((1, n_int))
    trace_E = eig_1 + eig_2 + eig_3  # 试应变的迹
    
    # 弹性响应
    test_el = (f_tr <= 0).astype(bool)
    if np.any(test_el):  # 检查是否包含True元素
        a0, b0 = np.where(test_el)
        sorted_indices = sorted(zip(b0,a0))
        b0_sorted, a0_sorted = zip (*sorted_indices)
        a0 = np.array(a0_sorted)
        b0 = np.array(b0_sorted)
        Psi[a0,b0] = 0.5 * matrix_lamlda.flatten(order='F')[test_el.flatten(order='F')] * trace_E.flatten(order='F')[test_el.flatten(order='F')]**2 + \
                      matrix_G.flatten(order='F')[test_el.flatten(order='F')] * (eig_1.flatten(order='F')[test_el.flatten(order='F')]**2 + \
                      eig_2.flatten(order='F')[test_el.flatten(order='F')]**2 + eig_3.flatten(order='F')[test_el.flatten(order='F')]**2)
    
    # 返回到屈服面的平滑部分
    test_s = (lambda_s <= np.minimum(gamma_sl, gamma_sr)) & (~test_el)
    if np.any(test_s):
        a0, b0 = np.where(test_s)
        sorted_indices = sorted(zip(b0,a0))
        b0_sorted, a0_sorted = zip (*sorted_indices)
        a0 = np.array(a0_sorted)
        b0 = np.array(b0_sorted)
        Psi[a0,b0] = 0.5 * matrix_lamlda.flatten(order='F')[test_s.flatten(order='F')] * \
                      trace_E.flatten(order='F')[test_s.flatten(order='F')]**2 + \
                      matrix_G.flatten(order='F')[test_s.flatten(order='F')] * \
                      (eig_1.flatten(order='F')[test_s.flatten(order='F')]**2 + \
                      eig_2.flatten(order='F')[test_s.flatten(order='F')]**2 + \
                      eig_3.flatten(order='F')[test_s.flatten(order='F')]**2) - \
                      0.5 * denom_s.flatten(order='F')[test_s.flatten(order='F')] * \
                      (lambda_s.flatten(order='F')[test_s.flatten(order='F')]**2)
    
    # 返回到屈服面的左边缘
    test_l = (gamma_sl < gamma_sr) & (lambda_l >= gamma_sl) & (lambda_l <= gamma_la) & \
             (~(test_el | test_s))
    if np.any(test_l):
        a0, b0 = np.where(test_l)
        sorted_indices = sorted(zip(b0,a0))
        b0_sorted, a0_sorted = zip (*sorted_indices)
        a0 = np.array(a0_sorted)
        b0 = np.array(b0_sorted)
        Psi[a0,b0] = 0.5 * matrix_lamlda.flatten(order='F')[test_l.flatten(order='F')] * trace_E.flatten(order='F')[test_l.flatten(order='F')]**2 + \
                     matrix_G.flatten(order='F')[test_l.flatten(order='F')] * (eig_3.flatten(order='F')[test_l.flatten(order='F')]**2 + \
                     0.5 * (eig_1.flatten(order='F')[test_l.flatten(order='F')] + eig_2.flatten(order='F')[test_l.flatten(order='F')])**2) - \
                     0.5 * denom_l.flatten(order='F')[test_l.flatten(order='F')] * (lambda_l.flatten(order='F')[test_l.flatten(order='F')]**2)
    
    # 返回到屈服面的右边缘
    test_r = (gamma_sl > gamma_sr) & (lambda_r >= gamma_sr) & (lambda_r <= gamma_ra) & \
             (~(test_el | test_s))
    if np.any(test_r):
        a0, b0 = np.where(test_r)
        sorted_indices = sorted(zip(b0,a0))
        b0_sorted, a0_sorted = zip (*sorted_indices)
        a0 = np.array(a0_sorted)
        b0 = np.array(b0_sorted)
        Psi[a0,b0] = 0.5 * matrix_lamlda.flatten(order='F')[test_r.flatten(order='F')] * trace_E.flatten(order='F')[test_r.flatten(order='F')]**2 + \
                     matrix_G.flatten(order='F')[test_r.flatten(order='F')] * (eig_1.flatten(order='F')[test_r.flatten(order='F')]**2 + \
                     0.5 * (eig_2.flatten(order='F')[test_r.flatten(order='F')] + eig_3.flatten(order='F')[test_r.flatten(order='F')])**2) - \
                     0.5 * denom_r.flatten(order='F')[test_r.flatten(order='F')] * (lambda_r.flatten(order='F')[test_r.flatten(order='F')]**2)
    
    # 返回到屈服面的顶点
    test_a = ~(test_el | test_s | test_l | test_r)
    if np.any(test_a):
        a0, b0 = np.where(test_a)
        sorted_indices = sorted(zip(b0,a0))
        b0_sorted, a0_sorted = zip (*sorted_indices)
        a0 = np.array(a0_sorted)
        b0 = np.array(b0_sorted)
        Psi[a0,b0] = 0.5 * matrix_K.flatten(order='F')[test_a.flatten(order='F')] * trace_E.flatten(order='F')[test_a.flatten(order='F')]**2 - \
                     0.5 * denom_a.flatten(order='F')[test_a.flatten(order='F')] * (lambda_a.flatten(order='F')[test_a.flatten(order='F')]**2)
    
    return Psi