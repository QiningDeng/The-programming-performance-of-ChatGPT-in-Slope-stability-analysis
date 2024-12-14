import numpy as np

# 定义函数：用于二次形函数下四面体网格单元划分
def mesh(N_h,x1,x2,x3,y1,y2,z):

    # 计算网格段数
    N_x = int((x1 + x2 + x3) * N_h)  # x方向的总段数
    N_y = int((y1 + y2) * N_h)       # y方向的总段数
    N_z = int(z * N_h)               # z方向的总段数
    N1_x = int(x1 * N_h)             # 边坡前段平台X方向段数
    N2_x = N_x - N1_x                # 除前段平台外的x方向段数
    N1_y = int(y1 * N_h)             # 边坡前段平台y方向段数
    N2_y = N_y - N1_y                # 除前段平台外的y方向段数
    
    # 计算总结点数
    n_node_xy = (2 * N_x + 1) * (2 * N1_y + 1) + (2 * N2_x + 1) * 2 * N2_y # XY平面内的总结点数
    n_n = n_node_xy * (2 * N_z + 1)  # 三维几何模型下的总节点数
    n_cell_xy = N_x * N1_y + N2_x * N2_y  # XY平面内的单元数量
    n_e = n_cell_xy * N_z * 6  # 三维几何模型下的总单元数量
    
    # 创建辅助数组，用于记录每个单元的节点编号
    C = np.zeros((2 * N_x + 1, 2 * N_y + 1, 2 * N_z + 1), dtype=int) # 全局网格编号
    C1 = np.reshape(
        np.arange(1, (2 * N_x + 1) * (2 * N1_y + 1) * (2 * N_z + 1) + 1),
        (2 * N_x + 1, 2 * N1_y + 1, 2 * N_z + 1),order='F') # 前段平台网格编号
    C2 = np.reshape(
        np.arange((2 * N_x + 1) * (2 * N1_y + 1) * (2 * N_z + 1) + 1, n_n + 1),
        (2 * N2_x + 1, 2 * N2_y, 2 * N_z + 1),order='F') # 后段平台网格编号
    C[:2 * N_x + 1, :2 * N1_y + 1, :] = C1 
    C[2 * N1_x: 2*N_x+1, 2 * N1_y + 1: 2*N_y+1, :] = C2 
    
    # 初始化节点坐标数组
    coord_x = np.linspace(0, x1 + x2 + x3, 2 * N_x + 1)
    coord_y = np.linspace(0, y1 + y2, 2 * N_y + 1)
    coord_z = np.linspace(0, z, 2 * N_z + 1)
    
    # 计算斜坡区域节点的x坐标 (cy_x)
    
    linspace_array = np.tile( np.linspace(-x2 - x3, 0, 2 * N2_x + 1)[:,np.newaxis], ( 1, 2 * N2_y))
    coord_y_subset = coord_y[:,np.newaxis].T[:,1:(2 * N2_y + 1)]
    second_part = np.tile( (1 - x2 / ((x2 + x3) * y2) * coord_y_subset), ( 2 * N2_x + 1, 1))
    
    cy_x = x1 + x2 + x3 + linspace_array * second_part
    
    # 生成节点坐标数组
    
    c_x = np.hstack([
        np.tile(coord_x, (2 * N1_y + 1) * (2 * N_z + 1))[:,np.newaxis].T,
        np.tile(cy_x.flatten(order='F')[:,np.newaxis].T, 2 * N_z + 1)
    ])
    c_y = np.hstack([
        np.tile(np.kron(coord_y[:2 * N1_y + 1], np.ones(( 1, 2 * N_x + 1))),( 1, 2 * N_z + 1)),
        np.tile(np.kron(coord_y[2 * N1_y + 1:2 * N_y + 1], np.ones(( 1, 2 * N2_x + 1))),( 1, 2 * N_z + 1))
    ])
    c_z = np.hstack([
        np.kron(coord_z, np.ones((1, (2 * N1_y + 1) * (2 * N_x + 1)))),
        np.kron(coord_z, np.ones((1, (2 * N2_x + 1) * 2 * N2_y)))
    ])
    
    coord = np.vstack([c_x, c_y, c_z])
    
    # 初始化单位立方体各个顶点位置的逻辑数组
    V1 = np.zeros((2*N_x+1, 2*N_y+1, 2*N_z+1), dtype=int)
    V2 = np.zeros_like(V1)
    V3 = np.zeros_like(V1)
    V4 = np.zeros_like(V1)
    V5 = np.zeros_like(V1)
    V6 = np.zeros_like(V1)
    V7 = np.zeros_like(V1)
    V8 = np.zeros_like(V1)
    
    V1[0:2*N_x-1:2, 0:2*N1_y-1:2, 0:2*N_z-1:2] = 1
    V1[2*N1_x:2*N_x-1:2, 2*N1_y:2*N_y-1:2, 0:2*N_z-1:2] = 1
    
    V2[2:2*N_x+1:2, 0:2*N1_y-1:2, 0:2*N_z-1:2] = 1
    V2[2*N1_x+2:2*N_x+1:2, 2*N1_y:2*N_y-1:2, 0:2*N_z-1:2] = 1
    
    V3[2:2*N_x+1:2, 2:2*N1_y+1:2, 0:2*N_z-1:2] = 1
    V3[2*N1_x+2:2*N_x+1:2, 2*N1_y+2:2*N_y+1:2, 0:2*N_z-1:2] = 1
    
    V4[0:2*N_x:2, 2:2*N1_y+1:2, 0:2*N_z:2] = 1
    V4[2*N1_x:2*N_x-1:2, 2*N1_y+2:2*N_y+1:2, 0:2*N_z-1:2] = 1
    
    V5[0:2*N_x:2, 0:2*N1_y:2, 2:2*N_z+1:2] = 1
    V5[2*N1_x:2*N_x-1:2, 2*N1_y:2*N_y-1:2, 2:2*N_z+1:2] = 1
    
    V6[2:2*N_x+1:2, 0:2*N1_y-1:2, 2:2*N_z+1:2] = 1
    V6[2*N1_x+2:2*N_x+1:2, 2*N1_y:2*N_y-1:2, 2:2*N_z+1:2] = 1
    
    V7[2:2*N_x+1:2, 2:2*N1_y+1:2, 2:2*N_z+1:2] = 1
    V7[2*N1_x+2:2*N_x+1:2, 2*N1_y+2:2*N_y+1:2, 2:2*N_z+1:2] = 1
    
    V8[0:2*N_x:2, 2:2*N1_y+1:2, 2:2*N_z+1:2] = 1
    V8[2*N1_x:2*N_x-1:2, 2*N1_y+2:2*N_y+1:2, 2:2*N_z+1:2] = 1
    
    # 初始化单位立方体各个中点位置的逻辑数组
    V12 = np.zeros_like(V1)
    V14 = np.zeros_like(V1)
    V15 = np.zeros_like(V1)
    V16 = np.zeros_like(V1)
    V23 = np.zeros_like(V1)
    V24 = np.zeros_like(V1)
    V26 = np.zeros_like(V1)
    V34 = np.zeros_like(V1)
    V36 = np.zeros_like(V1)
    V37 = np.zeros_like(V1)
    V45 = np.zeros_like(V1)
    V46 = np.zeros_like(V1)
    V47 = np.zeros_like(V1)
    V48 = np.zeros_like(V1)
    V56 = np.zeros_like(V1)
    V58 = np.zeros_like(V1)
    V67 = np.zeros_like(V1)
    V68 = np.zeros_like(V1)
    V78 = np.zeros_like(V1)
    
    V12[1:2*N_x:2, 0:2*N1_y-1:2, 0:2*N_z-1:2] = 1
    V12[2*N1_x+1:2*N_x:2, 2*N1_y:2*N_y-1:2, 0:2*N_z-1:2] = 1
    
    V14[0:2*N_x-1:2, 1:2*N1_y:2, 0:2*N_z-1:2] = 1
    V14[2*N1_x:2*N_x-1:2, 2*N1_y+1:2*N_y:2, 0:2*N_z-1:2] = 1
    
    V15[0:2*N_x-1:2, 0:2*N1_y-1:2, 1:2*N_z:2] = 1
    V15[2*N1_x:2*N_x-1:2, 2*N1_y:2*N_y-1:2, 1:2*N_z:2] = 1
    
    V16[1:2*N_x:2, 0:2*N1_y-1:2, 1:2*N_z:2] = 1
    V16[2*N1_x+1:2*N_x:2, 2*N1_y:2*N_y-1:2, 1:2*N_z:2] = 1
    
    V23[2:2*N_x+1:2, 1:2*N1_y:2, 0:2*N_z-1:2] = 1
    V23[2*N1_x+2:2*N_x+1:2, 2*N1_y+1:2*N_y:2, 0:2*N_z-1:2] = 1
    
    V24[1:2*N_x:2, 1:2*N1_y:2, 0:2*N_z-1:2] = 1
    V24[2*N1_x+1:2*N_x:2, 2*N1_y+1:2*N_y:2, 0:2*N_z-1:2] = 1
    
    V26[2:2*N_x+1:2, 0:2*N1_y-1:2, 1:2*N_z:2] = 1
    V26[2*N1_x+2:2*N_x+1:2, 2*N1_y:2*N_y-1:2, 1:2*N_z:2] = 1
    
    V34[1:2*N_x:2, 2:2*N1_y+1:2, 0:2*N_z-1:2] = 1
    V34[2*N1_x+1:2*N_x:2, 2*N1_y+2:2*N_y+1:2, 0:2*N_z-1:2] = 1
    
    V36[2:2*N_x+1:2, 1:2*N1_y:2, 1:2*N_z:2] = 1
    V36[2*N1_x+2:2*N_x+1:2, 2*N1_y+1:2*N_y:2, 1:2*N_z:2] = 1
    
    V37[2:2*N_x+1:2, 2:2*N1_y+1:2, 1:2*N_z:2] = 1
    V37[2*N1_x+2:2*N_x+1:2, 2*N1_y+2:2*N_y+1:2, 1:2*N_z:2] = 1
    
    V45[0:2*N_x-1:2, 1:2*N1_y:2, 1:2*N_z:2] = 1
    V45[2*N1_x:2*N_x-1:2, 2*N1_y+1:2*N_y:2, 1:2*N_z:2] = 1
    
    V46[1:2*N_x:2, 1:2*N1_y:2, 1:2*N_z:2] = 1
    V46[2*N1_x+1:2*N_x:2, 2*N1_y+1:2*N_y:2, 1:2*N_z:2] = 1
    
    V47[1:2*N_x:2, 2:2*N1_y+1:2, 1:2*N_z:2] = 1
    V47[2*N1_x+1:2*N_x:2, 2*N1_y+2:2*N_y+1:2, 1:2*N_z:2] = 1
    
    V48[0:2*N_x-1:2, 2:2*N1_y+1:2, 1:2*N_z:2] = 1
    V48[2*N1_x:2*N_x-1:2, 2*N1_y+2:2*N_y+1:2, 1:2*N_z:2] = 1
    
    V56[1:2*N_x:2, 0:2*N1_y-1:2, 2:2*N_z+1:2] = 1
    V56[2*N1_x+1:2*N_x:2, 2*N1_y:2*N_y-1:2, 2:2*N_z+1:2] = 1
    
    V58[0:2*N_x-1:2, 1:2*N1_y:2, 2:2*N_z+1:2] = 1
    V58[2*N1_x:2*N_x-1:2, 2*N1_y+1:2*N_y:2, 2:2*N_z+1:2] = 1
    
    V67[2:2*N_x+1:2, 1:2*N1_y:2, 2:2*N_z+1:2] = 1
    V67[2*N1_x+2:2*N_x+1:2, 2*N1_y+1:2*N_y:2, 2:2*N_z+1:2] = 1
    
    V68[1:2*N_x:2, 1:2*N1_y:2, 2:2*N_z+1:2] = 1
    V68[2*N1_x+1:2*N_x:2, 2*N1_y+1:2*N_y:2, 2:2*N_z+1:2] = 1
    
    V78[1:2*N_x:2, 2:2*N1_y+1:2, 2:2*N_z+1:2] = 1
    V78[2*N1_x+1:2*N_x:2, 2*N1_y+2:2*N_y+1:2, 2:2*N_z+1:2] = 1
    
    # 单位立方体可划分为6个四面体单元，每个单元包含顶点中点共计10个个节点
    
    # 利用逻辑索引提取数组元素，构造四面体的连接性
    C_V1 = np.ravel(C, order='F')[V1.ravel(order='F') == 1]
    C_V2 = np.ravel(C, order='F')[V2.ravel(order='F') == 1]
    C_V3 = np.ravel(C, order='F')[V3.ravel(order='F') == 1]
    C_V4 = np.ravel(C, order='F')[V4.ravel(order='F') == 1]
    C_V5 = np.ravel(C, order='F')[V5.ravel(order='F') == 1]
    C_V6 = np.ravel(C, order='F')[V6.ravel(order='F') == 1]
    C_V7 = np.ravel(C, order='F')[V7.ravel(order='F') == 1]
    C_V8 = np.ravel(C, order='F')[V8.ravel(order='F') == 1]
    C_V12 = np.ravel(C, order='F')[V12.ravel(order='F') == 1]
    C_V14 = np.ravel(C, order='F')[V14.ravel(order='F') == 1]
    C_V15 = np.ravel(C, order='F')[V15.ravel(order='F') == 1]
    C_V16 = np.ravel(C, order='F')[V16.ravel(order='F') == 1]
    C_V23 = np.ravel(C, order='F')[V23.ravel(order='F') == 1]
    C_V24 = np.ravel(C, order='F')[V24.ravel(order='F') == 1]
    C_V26 = np.ravel(C, order='F')[V26.ravel(order='F') == 1]
    C_V34 = np.ravel(C, order='F')[V34.ravel(order='F') == 1]
    C_V36 = np.ravel(C, order='F')[V36.ravel(order='F') == 1]
    C_V37 = np.ravel(C, order='F')[V37.ravel(order='F') == 1]
    C_V45 = np.ravel(C, order='F')[V45.ravel(order='F') == 1]
    C_V46 = np.ravel(C, order='F')[V46.ravel(order='F') == 1]
    C_V47 = np.ravel(C, order='F')[V47.ravel(order='F') == 1]
    C_V48 = np.ravel(C, order='F')[V48.ravel(order='F') == 1]
    C_V56 = np.ravel(C, order='F')[V56.ravel(order='F') == 1]
    C_V58 = np.ravel(C, order='F')[V58.ravel(order='F') == 1]
    C_V67 = np.ravel(C, order='F')[V67.ravel(order='F') == 1]
    C_V68 = np.ravel(C, order='F')[V68.ravel(order='F') == 1]
    C_V78 = np.ravel(C, order='F')[V78.ravel(order='F') == 1]
    
    aux_elem = np.vstack([
        np.array(C_V1), np.array(C_V2), np.array(C_V4), np.array(C_V6), np.array(C_V12),
        np.array(C_V24), np.array(C_V14), np.array(C_V26), np.array(C_V46), np.array(C_V16),
        np.array(C_V1), np.array(C_V4), np.array(C_V5), np.array(C_V6), np.array(C_V14),
        np.array(C_V45), np.array(C_V15), np.array(C_V46), np.array(C_V56), np.array(C_V16),
        np.array(C_V4), np.array(C_V5), np.array(C_V6), np.array(C_V8), np.array(C_V45),
        np.array(C_V56), np.array(C_V46), np.array(C_V58), np.array(C_V68), np.array(C_V48),
        np.array(C_V2), np.array(C_V3), np.array(C_V4), np.array(C_V6), np.array(C_V23),
        np.array(C_V34), np.array(C_V24), np.array(C_V36), np.array(C_V46), np.array(C_V26),
        np.array(C_V3), np.array(C_V6), np.array(C_V7), np.array(C_V4), np.array(C_V36),
        np.array(C_V67), np.array(C_V37), np.array(C_V46), np.array(C_V47), np.array(C_V34),
        np.array(C_V4), np.array(C_V6), np.array(C_V7), np.array(C_V8), np.array(C_V46),
        np.array(C_V67), np.array(C_V47), np.array(C_V68), np.array(C_V78), np.array(C_V48)
    ])
    
    elem = np.reshape(aux_elem, (10, n_e), order='F')
    
    # y=0表面信息进行提取，并分解为三角形网格
    
    C_s = np.zeros((2 * N_x + 1, 2 * N_z + 1))
    C_s[:, :] = C[:, 0, :]
    
    V1_s = np.zeros((2 * N_x + 1, 2 * N_z + 1), dtype=int)
    V1_s[0:2 * N_x-1:2, 0:2 * N_z-1:2] = 1
    
    V2_s = np.zeros_like(V1_s)
    V2_s[2:2 * N_x + 1:2, 0:2 * N_z-1:2] = 1
    
    V5_s = np.zeros_like(V1_s)
    V5_s[0:2 * N_x-1:2, 2:2 * N_z + 1:2] = 1
    
    V6_s = np.zeros_like(V1_s)
    V6_s[2:2 * N_x + 1:2, 2:2 * N_z + 1:2] = 1
    
    V12_s = np.zeros_like(V1_s)
    V12_s[1:2 * N_x:2, 0:2 * N_z-1:2] = 1
    
    V15_s = np.zeros_like(V1_s)
    V15_s[0:2 * N_x-1:2, 1:2 * N_z:2] = 1
    
    V16_s = np.zeros_like(V1_s)
    V16_s[1:2 * N_x:2, 1:2 * N_z:2] = 1
    
    V26_s = np.zeros_like(V1_s)
    V26_s[2:2 * N_x + 1:2, 1:2 * N_z:2] = 1
    
    V56_s = np.zeros_like(V1_s)
    V56_s[1:2 * N_x:2, 2:2 * N_z + 1:2] = 1
    
    # 提取相应的节点值
    C_s_V1 = np.ravel(C_s, order='F')[V1_s.ravel(order='F') == 1]
    C_s_V2 = np.ravel(C_s, order='F')[V2_s.ravel(order='F') == 1]
    C_s_V5 = np.ravel(C_s, order='F')[V5_s.ravel(order='F') == 1]
    C_s_V6 = np.ravel(C_s, order='F')[V6_s.ravel(order='F') == 1]
    C_s_V12 = np.ravel(C_s, order='F')[V12_s.ravel(order='F') == 1]
    C_s_V15 = np.ravel(C_s, order='F')[V15_s.ravel(order='F') == 1]
    C_s_V16 = np.ravel(C_s, order='F')[V16_s.ravel(order='F') == 1]
    C_s_V26 = np.ravel(C_s, order='F')[V26_s.ravel(order='F') == 1]
    C_s_V56 = np.ravel(C_s, order='F')[V56_s.ravel(order='F') == 1]
    
    aux_surf1 = np.vstack([
        np.array(C_s_V2), np.array(C_s_V1), np.array(C_s_V6), np.array(C_s_V16), np.array(C_s_V26),
        np.array(C_s_V12), np.array(C_s_V5), np.array(C_s_V6), np.array(C_s_V1), np.array(C_s_V16),
        np.array(C_s_V15), np.array(C_s_V56)
    ])
    
    surf1 = aux_surf1.reshape(6, 2 * N_x * N_z, order='F')
    
    # y=y1表面信息进行提取，并分解为三角形网格
    
    C_s = np.zeros((2 * N_x + 1, 2 * N_z + 1))
    C_s[:, :] = C[:, 2 * N1_y, :]
    
    V3_s = np.zeros_like(V1_s)
    V3_s[2:2 * N1_x + 1:2, 0:2 * N_z - 1:2] = 1
    
    V4_s = np.zeros_like(V1_s)
    V4_s[0:2 * N1_x - 1:2, 0:2 * N_z - 1:2] = 1
    
    V7_s = np.zeros_like(V1_s)
    V7_s[2:2 * N1_x + 1:2, 2:2 * N_z + 1:2] = 1
    
    V8_s = np.zeros_like(V1_s)
    V8_s[0:2 * N1_x - 1:2, 2:2 * N_z + 1:2] = 1
    
    V34_s = np.zeros_like(V1_s)
    V34_s[1:2 * N1_x:2, 0:2 * N_z - 1:2] = 1
    
    V37_s = np.zeros_like(V1_s)
    V37_s[2:2 * N1_x + 1:2, 1:2 * N_z:2] = 1
    
    V47_s = np.zeros_like(V1_s)
    V47_s[1:2 * N1_x:2, 1:2 * N_z:2] = 1
    
    V48_s = np.zeros_like(V1_s)
    V48_s[0:2 * N1_x - 1:2, 1:2 * N_z:2] = 1
    
    V78_s = np.zeros_like(V1_s)
    V78_s[1:2 * N1_x:2, 2:2 * N_z + 1:2] = 1
    
    C_s_V3 = np.ravel(C_s, order='F')[V3_s.ravel(order='F') == 1]
    C_s_V4 = np.ravel(C_s, order='F')[V4_s.ravel(order='F') == 1]
    C_s_V7 = np.ravel(C_s, order='F')[V7_s.ravel(order='F') == 1]
    C_s_V8 = np.ravel(C_s, order='F')[V8_s.ravel(order='F') == 1]
    C_s_V34 = np.ravel(C_s, order='F')[V34_s.ravel(order='F') == 1]
    C_s_V37 = np.ravel(C_s, order='F')[V37_s.ravel(order='F') == 1]
    C_s_V47 = np.ravel(C_s, order='F')[V47_s.ravel(order='F') == 1]
    C_s_V48 = np.ravel(C_s, order='F')[V48_s.ravel(order='F') == 1]
    C_s_V78 = np.ravel(C_s, order='F')[V78_s.ravel(order='F') == 1]
    
    aux_surf2 = np.vstack([
        np.array(C_s_V3), np.array(C_s_V7), np.array(C_s_V4), np.array(C_s_V47), np.array(C_s_V34),
        np.array(C_s_V37), np.array(C_s_V8), np.array(C_s_V4), np.array(C_s_V7), np.array(C_s_V47),
        np.array(C_s_V78), np.array(C_s_V48)
    ])
    
    surf2 = aux_surf2.reshape(6, 2 * N1_x * N_z, order='F')
    
    # y=y1+y2表面信息进行提取，并分解为三角形网格
    C_s = np.zeros((2 * N_x + 1, 2 * N_z + 1))
    C_s[:, :] = C[:, -1, :]
    
    V3_s = np.zeros_like(V1_s)
    V3_s[2 * N1_x + 2:2 * N_x + 1:2, 0:2 * N_z - 1:2] = 1
    
    V4_s = np.zeros_like(V1_s)
    V4_s[2 * N1_x :2 * N_x - 1:2, 0:2 * N_z - 1:2] = 1
    
    V7_s = np.zeros_like(V1_s)
    V7_s[2 * N1_x + 2:2 * N_x + 1:2, 2:2 * N_z + 1:2] = 1
    
    V8_s = np.zeros_like(V1_s)
    V8_s[2 * N1_x :2 * N_x - 1:2, 2:2 * N_z + 1:2] = 1
    
    V34_s = np.zeros_like(V1_s)
    V34_s[2 * N1_x + 1:2 * N_x:2, 0:2 * N_z - 1:2] = 1
    
    V37_s = np.zeros_like(V1_s)
    V37_s[2 * N1_x + 2:2 * N_x + 1:2, 1:2 * N_z:2] = 1
    
    V47_s = np.zeros_like(V1_s)
    V47_s[2 * N1_x + 1:2 * N_x:2, 1:2 * N_z:2] = 1
    
    V48_s = np.zeros_like(V1_s)
    V48_s[2 * N1_x :2 * N_x - 1:2, 1:2 * N_z:2] = 1
    
    V78_s = np.zeros_like(V1_s)
    V78_s[2 * N1_x + 1:2 * N_x:2, 2:2 * N_z + 1:2] = 1
    
    C_s_V3 = np.ravel(C_s, order='F')[V3_s.ravel(order='F') == 1]
    C_s_V4 = np.ravel(C_s, order='F')[V4_s.ravel(order='F') == 1]
    C_s_V7 = np.ravel(C_s, order='F')[V7_s.ravel(order='F') == 1]
    C_s_V8 = np.ravel(C_s, order='F')[V8_s.ravel(order='F') == 1]
    C_s_V34 = np.ravel(C_s, order='F')[V34_s.ravel(order='F') == 1]
    C_s_V37 = np.ravel(C_s, order='F')[V37_s.ravel(order='F') == 1]
    C_s_V47 = np.ravel(C_s, order='F')[V47_s.ravel(order='F') == 1]
    C_s_V48 = np.ravel(C_s, order='F')[V48_s.ravel(order='F') == 1]
    C_s_V78 = np.ravel(C_s, order='F')[V78_s.ravel(order='F') == 1]
    
    aux_surf3 = np.vstack([
        np.array(C_s_V3), np.array(C_s_V7), np.array(C_s_V4), np.array(C_s_V47), np.array(C_s_V34),
        np.array(C_s_V37), np.array(C_s_V8), np.array(C_s_V4), np.array(C_s_V7), np.array(C_s_V47),
        np.array(C_s_V78), np.array(C_s_V48)
    ])
    
    surf3 = aux_surf3.reshape(6, 2 * N2_x * N_z, order='F')
    
    # x=0表面信息进行提取，并分解为三角形网格
    C_s = np.zeros((2 * N_y + 1, 2 * N_z + 1))
    C_s[:, :] = C[0, :, :]
    
    V1_s = np.zeros((2 * N_y + 1, 2 * N_z + 1), dtype=int)
    V1_s[0:2 * N1_y - 1:2, 0:2 * N_z - 1:2] = 1
    
    V4_s = np.zeros_like(V1_s)
    V4_s[2:2 * N1_y + 1:2, 0:2 * N_z - 1:2] = 1
    
    V5_s = np.zeros_like(V1_s)
    V5_s[0:2 * N1_y - 1:2, 2:2 * N_z + 1:2] = 1
    
    V8_s = np.zeros_like(V1_s)
    V8_s[2:2 * N1_y + 1:2, 2:2 * N_z + 1:2] = 1
    
    V14_s = np.zeros_like(V1_s)
    V14_s[1:2 * N1_y:2, 0:2 * N_z - 1:2] = 1
    
    V15_s = np.zeros_like(V1_s)
    V15_s[0:2 * N1_y - 1:2, 1:2 * N_z:2] = 1
    
    V45_s = np.zeros_like(V1_s)
    V45_s[1:2 * N1_y:2, 1:2 * N_z:2] = 1
    
    V48_s = np.zeros_like(V1_s)
    V48_s[2:2 * N1_y + 1:2, 1:2 * N_z:2] = 1
    
    V58_s = np.zeros_like(V1_s)
    V58_s[1:2 * N1_y:2, 2:2 * N_z + 1:2] = 1
    
    C_s_V1 = np.ravel(C_s, order='F')[V1_s.ravel(order='F') == 1]
    C_s_V4 = np.ravel(C_s, order='F')[V4_s.ravel(order='F') == 1]
    C_s_V5 = np.ravel(C_s, order='F')[V5_s.ravel(order='F') == 1]
    C_s_V8 = np.ravel(C_s, order='F')[V8_s.ravel(order='F') == 1]
    C_s_V14 = np.ravel(C_s, order='F')[V14_s.ravel(order='F') == 1]
    C_s_V15 = np.ravel(C_s, order='F')[V15_s.ravel(order='F') == 1]
    C_s_V45 = np.ravel(C_s, order='F')[V45_s.ravel(order='F') == 1]
    C_s_V48 = np.ravel(C_s, order='F')[V48_s.ravel(order='F') == 1]
    C_s_V58 = np.ravel(C_s, order='F')[V58_s.ravel(order='F') == 1]
    
    aux_surf4 = np.vstack([
        np.array(C_s_V1), np.array(C_s_V4), np.array(C_s_V5), np.array(C_s_V45), np.array(C_s_V15),
        np.array(C_s_V14), np.array(C_s_V8), np.array(C_s_V5), np.array(C_s_V4), np.array(C_s_V45),
        np.array(C_s_V48), np.array(C_s_V58)
    ])
    
    surf4 = aux_surf4.reshape(6, 2 * N1_y * N_z, order='F')
    
    # x=x1表面信息进行提取，并分解为三角形网格
    C_s = np.zeros((2 * N_y + 1, 2 * N_z + 1))
    C_s[:, :] = C[2 * N1_x, :, :]
    
    V1_s = np.zeros((2 * N_y + 1, 2 * N_z + 1), dtype=int)
    V1_s[2 * N1_y:2 * N_y - 1:2, 0:2 * N_z - 1:2] = 1
    
    V4_s = np.zeros_like(V1_s)
    V4_s[2 * N1_y + 2:2 * N_y + 1:2, 0:2 * N_z - 1:2] = 1
    
    V5_s = np.zeros_like(V1_s)
    V5_s[2 * N1_y:2 * N_y - 1:2, 2:2 * N_z + 1:2] = 1
    
    V8_s = np.zeros_like(V1_s)
    V8_s[2 * N1_y + 2:2 * N_y + 1:2, 2:2 * N_z + 1:2] = 1
    
    V14_s = np.zeros_like(V1_s)
    V14_s[2 * N1_y + 1:2 * N_y:2, 0:2 * N_z - 1:2] = 1
    
    V15_s = np.zeros_like(V1_s)
    V15_s[2 * N1_y:2 * N_y - 1:2, 1:2 * N_z:2] = 1
    
    V45_s = np.zeros_like(V1_s)
    V45_s[2 * N1_y + 1:2 * N_y:2, 1:2 * N_z:2] = 1
    
    V48_s = np.zeros_like(V1_s)
    V48_s[2 * N1_y + 2:2 * N_y + 1:2, 1:2 * N_z:2] = 1
    
    V58_s = np.zeros_like(V1_s)
    V58_s[2 * N1_y + 1:2 * N_y:2, 2:2 * N_z + 1:2] = 1
    
    C_s_V1 = np.ravel(C_s, order='F')[V1_s.ravel(order='F') == 1]
    C_s_V4 = np.ravel(C_s, order='F')[V4_s.ravel(order='F') == 1]
    C_s_V5 = np.ravel(C_s, order='F')[V5_s.ravel(order='F') == 1]
    C_s_V8 = np.ravel(C_s, order='F')[V8_s.ravel(order='F') == 1]
    C_s_V14 = np.ravel(C_s, order='F')[V14_s.ravel(order='F') == 1]
    C_s_V15 = np.ravel(C_s, order='F')[V15_s.ravel(order='F') == 1]
    C_s_V45 = np.ravel(C_s, order='F')[V45_s.ravel(order='F') == 1]
    C_s_V48 = np.ravel(C_s, order='F')[V48_s.ravel(order='F') == 1]
    C_s_V58 = np.ravel(C_s, order='F')[V58_s.ravel(order='F') == 1]
    
    aux_surf5 = np.vstack([
        np.array(C_s_V1), np.array(C_s_V4), np.array(C_s_V5), np.array(C_s_V45), np.array(C_s_V15),
        np.array(C_s_V14), np.array(C_s_V8), np.array(C_s_V5), np.array(C_s_V4), np.array(C_s_V45),
        np.array(C_s_V48), np.array(C_s_V58)
    ])
    
    surf5 = aux_surf5.reshape(6, 2 * N2_y * N_z, order='F')
    
    
    # x=x1+x2+x3 表面信息进行提取，并分解为三角形网格
    C_s = np.zeros((2 * N_y + 1, 2 * N_z + 1))
    C_s[:, :] = C[-1, :, :]
    
    V2_s = np.zeros((2 * N_y + 1, 2 * N_z + 1), dtype=int)
    V2_s[0:2 * N_y - 1:2, 0:2 * N_z - 1:2] = 1
    
    V3_s = np.zeros_like(V2_s)
    V3_s[2:2 * N_y + 1:2, 0:2 * N_z - 1:2] = 1
    
    V6_s = np.zeros_like(V2_s)
    V6_s[0:2 * N_y - 1:2, 2:2 * N_z + 1:2] = 1
    
    V7_s = np.zeros_like(V2_s)
    V7_s[2:2 * N_y + 1:2, 2:2 * N_z + 1:2] = 1
    
    V23_s = np.zeros_like(V2_s)
    V23_s[1:2 * N_y:2, 0:2 * N_z - 1:2] = 1
    
    V26_s = np.zeros_like(V2_s)
    V26_s[0:2 * N_y - 1:2, 1:2 * N_z:2] = 1
    
    V36_s = np.zeros_like(V2_s)
    V36_s[1:2 * N_y:2, 1:2 * N_z:2] = 1
    
    V37_s = np.zeros_like(V2_s)
    V37_s[2:2 * N_y + 1:2, 1:2 * N_z:2] = 1
    
    V67_s = np.zeros_like(V2_s)
    V67_s[1:2 * N_y:2, 2:2 * N_z + 1:2] = 1
    
    C_s_V2 = np.ravel(C_s, order='F')[V2_s.ravel(order='F') == 1]
    C_s_V3 = np.ravel(C_s, order='F')[V3_s.ravel(order='F') == 1]
    C_s_V6 = np.ravel(C_s, order='F')[V6_s.ravel(order='F') == 1]
    C_s_V7 = np.ravel(C_s, order='F')[V7_s.ravel(order='F') == 1]
    C_s_V23 = np.ravel(C_s, order='F')[V23_s.ravel(order='F') == 1]
    C_s_V26 = np.ravel(C_s, order='F')[V26_s.ravel(order='F') == 1]
    C_s_V36 = np.ravel(C_s, order='F')[V36_s.ravel(order='F') == 1]
    C_s_V37 = np.ravel(C_s, order='F')[V37_s.ravel(order='F') == 1]
    C_s_V67 = np.ravel(C_s, order='F')[V67_s.ravel(order='F') == 1]
    
    aux_surf6 = np.vstack([
        np.array(C_s_V2), np.array(C_s_V6), np.array(C_s_V3), np.array(C_s_V36), np.array(C_s_V23),
        np.array(C_s_V26), np.array(C_s_V7), np.array(C_s_V3), np.array(C_s_V6), np.array(C_s_V36),
        np.array(C_s_V67), np.array(C_s_V37)
    ])
    
    surf6 = aux_surf6.reshape(6, 2 * N_y * N_z, order='F')
    
    
    # z=0 表面信息进行提取，并分解为三角形网格
    C_s = np.zeros((2 * N_x + 1, 2 * N_y + 1))
    C_s[:, :] = C[:, :, 0]
    
    V1_s = np.zeros((2 * N_x + 1, 2 * N_y + 1), dtype=int)
    V1_s[0:2 * N_x - 1:2, 0:2 * N1_y - 1:2] = 1
    V1_s[2 * N1_x :2 * N_x - 1:2, 2 * N1_y:2 * N_y - 1:2] = 1
    
    V2_s = np.zeros_like(V1_s)
    V2_s[2:2 * N_x + 1:2, 0:2 * N1_y - 1:2] = 1
    V2_s[(2 * N1_x + 2):2 * N_x + 1:2, 2 * N1_y :2 * N_y - 1:2] = 1
    
    V3_s = np.zeros_like(V1_s)
    V3_s[2:2 * N_x + 1:2, 2:2 * N1_y + 1:2] = 1
    V3_s[(2 * N1_x + 2):2 * N_x + 1:2, (2 * N1_y + 2):2 * N_y + 1:2] = 1
    
    V4_s = np.zeros_like(V1_s)
    V4_s[0:2 * N_x - 1:2, 2:2 * N1_y + 1:2] = 1
    V4_s[2 * N1_x :2 * N_x - 1:2, (2 * N1_y + 2):2 * N_y + 1:2] = 1
    
    V12_s = np.zeros_like(V1_s)
    V12_s[1:2 * N_x:2, 0:2 * N1_y - 1:2] = 1
    V12_s[(2 * N1_x + 1):2 * N_x:2, 2 * N1_y :2 * N_y - 1:2] = 1
    
    V14_s = np.zeros_like(V1_s)
    V14_s[0:2 * N_x - 1:2, 1:2 * N1_y:2] = 1
    V14_s[2 * N1_x :2 * N_x - 1:2, (2 * N1_y + 1):2 * N_y:2] = 1
    
    V23_s = np.zeros_like(V1_s)
    V23_s[2:2 * N_x + 1:2, 1:2 * N1_y:2] = 1
    V23_s[(2 * N1_x + 2):2 * N_x + 1:2, (2 * N1_y + 1):2 * N_y:2] = 1
    
    V24_s = np.zeros_like(V1_s)
    V24_s[1:2 * N_x:2, 1:2 * N1_y:2] = 1
    V24_s[(2 * N1_x + 1):2 * N_x:2, (2 * N1_y + 1):2 * N_y:2] = 1
    
    V34_s = np.zeros_like(V1_s)
    V34_s[1:2 * N_x:2, 2:2 * N1_y + 1:2] = 1
    V34_s[(2 * N1_x + 1):2 * N_x:2, (2 * N1_y + 2):2 * N_y + 1:2] = 1
    
    C_s_V1 = np.ravel(C_s, order='F')[V1_s.ravel(order='F') == 1]
    C_s_V2 = np.ravel(C_s, order='F')[V2_s.ravel(order='F') == 1]
    C_s_V3 = np.ravel(C_s, order='F')[V3_s.ravel(order='F') == 1]
    C_s_V4 = np.ravel(C_s, order='F')[V4_s.ravel(order='F') == 1]
    C_s_V12 = np.ravel(C_s, order='F')[V12_s.ravel(order='F') == 1]
    C_s_V14 = np.ravel(C_s, order='F')[V14_s.ravel(order='F') == 1]
    C_s_V23 = np.ravel(C_s, order='F')[V23_s.ravel(order='F') == 1]
    C_s_V24 = np.ravel(C_s, order='F')[V24_s.ravel(order='F') == 1]
    C_s_V34 = np.ravel(C_s, order='F')[V34_s.ravel(order='F') == 1]
    
    aux_surf7 = np.vstack([
        np.array(C_s_V1), np.array(C_s_V2), np.array(C_s_V4), np.array(C_s_V24), np.array(C_s_V14),
        np.array(C_s_V12), np.array(C_s_V3), np.array(C_s_V4), np.array(C_s_V2), np.array(C_s_V24),
        np.array(C_s_V23), np.array(C_s_V34)
    ])
    
    surf7 = aux_surf7.reshape(6, 2 * N_x * N1_y + 2 * N2_x * N2_y, order='F')
    
    # z=zmax 表面信息进行提取，并分解为三角形网格
    C_s = np.zeros((2 * N_x + 1, 2 * N_y + 1))
    C_s[:, :] = C[:, :, -1]
    
    V5_s = np.zeros((2 * N_x + 1, 2 * N_y + 1), dtype=int)
    V5_s[0:2 * N_x - 1:2, 0:2 * N1_y - 1:2] = 1
    V5_s[2 * N1_x :2 * N_x - 1:2, 2 * N1_y:2 * N_y - 1:2] = 1
    
    V6_s = np.zeros_like(V5_s)
    V6_s[2:2 * N_x + 1:2, 0:2 * N1_y - 1:2] = 1
    V6_s[2 * N1_x + 2:2 * N_x + 1:2, 2 * N1_y:2 * N_y - 1:2] = 1
    
    V8_s = np.zeros_like(V5_s)
    V8_s[0:2 * N_x - 1:2, 2:2 * N1_y + 1:2] = 1
    V8_s[2 * N1_x :2 * N_x - 1:2, 2 * N1_y + 2:2 * N_y + 1:2] = 1
    
    V7_s = np.zeros_like(V5_s)
    V7_s[2:2 * N_x + 1:2, 2:2 * N1_y + 1:2] = 1
    V7_s[2 * N1_x + 2:2 * N_x + 1:2, 2 * N1_y + 2:2 * N_y + 1:2] = 1
    
    V56_s = np.zeros_like(V5_s)
    V56_s[1:2 * N_x :2, 0:2 * N1_y - 1:2] = 1
    V56_s[2 * N1_x + 1:2 * N_x :2, 2 * N1_y :2 * N_y - 1:2] = 1
    
    V58_s = np.zeros_like(V5_s)
    V58_s[0:2 * N_x - 1:2, 1:2 * N1_y :2] = 1
    V58_s[2 * N1_x :2 * N_x - 1:2, 2 * N1_y + 1:2 * N_y :2] = 1
    
    V68_s = np.zeros_like(V5_s)
    V68_s[1:2 * N_x :2, 1:2 * N1_y :2] = 1
    V68_s[2 * N1_x + 1:2 * N_x :2, 2 * N1_y + 1:2 * N_y :2] = 1
    
    V67_s = np.zeros_like(V5_s)
    V67_s[2:2 * N_x + 1:2, 1:2 * N1_y :2] = 1
    V67_s[2 * N1_x + 2:2 * N_x + 1:2, 2 * N1_y + 1:2 * N_y :2] = 1
    
    V78_s = np.zeros_like(V5_s)
    V78_s[1:2 * N_x :2, 2:2 * N1_y + 1:2] = 1
    V78_s[2 * N1_x + 1:2 * N_x :2, 2 * N1_y + 2:2 * N_y + 1:2] = 1
    
    C_s_V5 = np.ravel(C_s, order='F')[V5_s.ravel(order='F') == 1]
    C_s_V6 = np.ravel(C_s, order='F')[V6_s.ravel(order='F') == 1]
    C_s_V7 = np.ravel(C_s, order='F')[V7_s.ravel(order='F') == 1]
    C_s_V8 = np.ravel(C_s, order='F')[V8_s.ravel(order='F') == 1]
    C_s_V56 = np.ravel(C_s, order='F')[V56_s.ravel(order='F') == 1]
    C_s_V58 = np.ravel(C_s, order='F')[V58_s.ravel(order='F') == 1]
    C_s_V67 = np.ravel(C_s, order='F')[V67_s.ravel(order='F') == 1]
    C_s_V68 = np.ravel(C_s, order='F')[V68_s.ravel(order='F') == 1]
    C_s_V78 = np.ravel(C_s, order='F')[V78_s.ravel(order='F') == 1]
    
    aux_surf8 = np.vstack([
        np.array(C_s_V5), np.array(C_s_V8), np.array(C_s_V6), np.array(C_s_V68), np.array(C_s_V56),
        np.array(C_s_V58), np.array(C_s_V7), np.array(C_s_V6), np.array(C_s_V8), np.array(C_s_V68),
        np.array(C_s_V78), np.array(C_s_V67)
    ])
    
    surf8 = aux_surf8.reshape(6, 2 * N_x * N1_y + 2 * N2_x * N2_y, order='F')
    
    # 将所有表面进行整合，得到所有表面的三角形网格信息
    surf = np.hstack((surf1, surf2, surf3, surf5, surf4, surf6, surf7, surf8))
    
    # Dirichlet边界条件的设定
    Q = coord > 0
    Q[0, coord[0, :] == (x1 + x2 + x3)] = 0
    Q[2, coord[2, :] == z] = 0
    
    # 将布尔值转换为整数 1 和 0
    Q = Q.astype(int)


    return coord, elem, surf, Q
