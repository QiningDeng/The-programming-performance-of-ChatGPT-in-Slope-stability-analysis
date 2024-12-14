import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.font_manager as fm
import random
import scipy.sparse
from A_mesh import mesh
from B_DCM import DCM

# 定义几何参数
x1 = 10  # 坡体前段平台长度
x2 = 7 # 坡体自身长度
x3 = 10  # 坡体后段平台长度
y1 = 10  # 坡体前段平台高度
y2 = 8  # 坡体自身高度
z = 10  # 坡体宽度

# 土体材料物理参数及网格划分参数
gamma = 17 # 土体自重
c0 = 30 # 土体的粘聚力
phi = 20 # 土体内摩擦角
E = 37500 # 材料弹性模量
v = 0.30 # 材料泊松比
psi = 0.10 # 材料剪胀角
G = E / ( 2 * ( 1 + v )) # 材料剪切模量
K = E / ( 3 * ( 1 - 2 * v )) # 材料体积模量
lamlda = K - 2 * G / 3 # 材料第一拉梅参数 λ

N_h = 1 # 网格划分密度参数

# 四节点四面体单元二次形函数下高斯积分法的积分点及权重值

# 高斯点坐标矩阵
Xi = np.array([
    [0.2500, 0.0714285714285714, 0.785714285714286, 0.0714285714285714, 0.0714285714285714, 0.399403576166799,
    0.100596423833201, 0.100596423833201, 0.399403576166799, 0.399403576166799, 0.100596423833201],
    [0.2500, 0.0714285714285714, 0.0714285714285714, 0.785714285714286, 0.0714285714285714, 0.100596423833201,
    0.399403576166799, 0.100596423833201, 0.399403576166799, 0.100596423833201, 0.399403576166799],
    [0.2500, 0.0714285714285714, 0.0714285714285714, 0.0714285714285714, 0.785714285714286, 0.100596423833201,
    0.100596423833201, 0.399403576166799, 0.100596423833201, 0.399403576166799, 0.399403576166799]
]) 
# 权重因子数组
WF = np.array([-0.013155555555555, 0.007622222222222, 0.007622222222222, 0.007622222222222, 0.007622222222222, 0.024888888888888,
               0.024888888888888, 0.024888888888888, 0.024888888888888, 0.024888888888888, 0.024888888888888]).reshape(1, -1) 

# 计算局部基函数及各点的方向导数

xi_1 = Xi[0, :] # 提取高斯点坐标矩阵的第一行，参考单元第二个顶点的重心坐标
xi_2 = Xi[1, :] # 提取高斯点坐标矩阵的第二行，参考单元第三个顶点的重心坐标
xi_3 = Xi[2, :] # 提取高斯点坐标矩阵的第三行，参考单元第四个顶点的重心坐标
xi_0 = 1 - xi_1 -xi_2 - xi_3 # 参考单元第一个顶点的重心坐标

n_q = len(xi_1)
HatP = np.array([
    xi_0 * (2 * xi_0 - 1), xi_1 * (2 * xi_1 - 1),
    xi_2 * (2 * xi_2 - 1), xi_3 * (2 * xi_3 - 1),
    4 * xi_0 * xi_1, 4 * xi_1 * xi_2,
    4 * xi_0 * xi_2, 4 * xi_1 * xi_3,
    4 * xi_2 * xi_3, 4 * xi_0 * xi_3
]) # 基函数的值

DHatP1 = np.array([
    -4 * xi_0 + 1, 4 * xi_1 - 1,
    np.zeros(n_q), np.zeros(n_q),
    4 * (xi_0 - xi_1), 4 * xi_2,
    -4 * xi_2, 4 * xi_3,
    np.zeros(n_q), -4 * xi_3
]) # 基函数在参考坐标系下第一方向上的偏导数

DHatP2 = np.array([
    -4 * xi_0 + 1, np.zeros(n_q),
    4 * xi_2 - 1, np.zeros(n_q),
    -4 * xi_1, 4 * xi_1,
    4 * (xi_0 - xi_2), np.zeros(n_q),
    4 * xi_3, -4 * xi_3
]) # 基函数在参考坐标系下第二方向上的偏导数

DHatP3 = np.array([
    -4 * xi_0 + 1, np.zeros(n_q),
    np.zeros(n_q), 4 * xi_3 - 1,
    -4 * xi_1, np.zeros(n_q),
    -4 * xi_2, 4 * xi_1,
    4 * xi_2, 4 * (xi_0 - xi_3)
]) # 基函数在参考坐标系下第三方向上的偏导数

# 调用 mesh 函数生成四节点四面体网格
coord, elem, surf, Q = mesh(N_h, x1, x2, x3, y1, y2, z)
Q = Q.astype(bool)
coord_Q = np.ravel(coord, order='F')[Q.ravel(order='F') == 1]

# 存储网格信息
n_n = coord.shape[1]  # 节点数
n_unknown = len(np.array(coord_Q))  # 未知数的数量
n_p = elem.shape[0] # 每个单元的顶点数量
n_e = elem.shape[1]  # 单元数
n_q = WF.shape[1]  # 积分点的数量
n_int = n_e * n_q  # 总积分点数

# 打印网格统计信息
print(f'网格数据: 节点数 = {n_n}, 未知数 = {n_unknown}, 单元数 = {n_e}, 积分点数 = {n_int}')

# 更新材料参数于积分点
matrix_c0 = c0 * np.ones(n_int).reshape(1, -1)
matrix_phi = np.deg2rad(phi) * np.ones(n_int).reshape(1, -1)
matrix_psi = np.deg2rad(psi) * np.ones(n_int).reshape(1, -1)
matrix_G = G * np.ones(n_int).reshape(1, -1)
matrix_K = K * np.ones(n_int).reshape(1, -1)
matrix_lamlda = lamlda * np.ones(n_int).reshape(1, -1)
matrix_gamma = gamma * np.ones(n_int).reshape(1, -1)

# 沿列复制 DHatP1, DHatP2, DHatP3
DHatPhi1 = np.tile(DHatP1, (1, n_e))
DHatPhi2 = np.tile(DHatP2, (1, n_e))
DHatPhi3 = np.tile(DHatP3, (1, n_e))

# 根据 elem 重塑 coord
coorde1 = np.reshape(coord[0, elem.flatten(order='F')-1], (n_p, n_e),order='F')
coorde2 = np.reshape(coord[1, elem.flatten(order='F')-1], (n_p, n_e),order='F')
coorde3 = np.reshape(coord[2, elem.flatten(order='F')-1], (n_p, n_e),order='F')

# 每个积分点周围节点的坐标
coordint1 = np.kron(coorde1, np.ones((1, n_q)))
coordint2 = np.kron(coorde2, np.ones((1, n_q)))
coordint3 = np.kron(coorde3, np.ones((1, n_q)))

# 雅可比矩阵的分量
J11 = np.sum(coordint1 * DHatPhi1, axis=0).reshape(1, -1)
J12 = np.sum(coordint2 * DHatPhi1, axis=0).reshape(1, -1)
J13 = np.sum(coordint3 * DHatPhi1, axis=0).reshape(1, -1)
J21 = np.sum(coordint1 * DHatPhi2, axis=0).reshape(1, -1)
J22 = np.sum(coordint2 * DHatPhi2, axis=0).reshape(1, -1)
J23 = np.sum(coordint3 * DHatPhi2, axis=0).reshape(1, -1)
J31 = np.sum(coordint1 * DHatPhi3, axis=0).reshape(1, -1)
J32 = np.sum(coordint2 * DHatPhi3, axis=0).reshape(1, -1)
J33 = np.sum(coordint3 * DHatPhi3, axis=0).reshape(1, -1)

# 雅可比行列式
DET = (J11 * (J22 * J33 - J32 * J23) - J12 * (J21 * J33 - J23 * J31) +
       J13 * (J21 * J32 - J22 * J31))

# 雅可比矩阵分量的逆
Jinv11 = (J22 * J33 - J23 * J32) / DET
Jinv12 = -(J12 * J33 - J13 * J32) / DET
Jinv13 = (J12 * J23 - J13 * J22) / DET
Jinv21 = -(J21 * J33 - J23 * J31) / DET
Jinv22 = (J11 * J33 - J13 * J31) / DET
Jinv23 = -(J11 * J23 - J13 * J21) / DET
Jinv31 = (J21 * J32 - J22 * J31) / DET
Jinv32 = -(J11 * J32 - J12 * J31) / DET
Jinv33 = (J11 * J22 - J12 * J21) / DET

# 沿行复制 Jinv 数组
Jinv11_rep = np.tile(Jinv11, (n_p, 1))
Jinv12_rep = np.tile(Jinv12, (n_p, 1))
Jinv13_rep = np.tile(Jinv13, (n_p, 1))

Jinv21_rep = np.tile(Jinv21, (n_p, 1))
Jinv22_rep = np.tile(Jinv22, (n_p, 1))
Jinv23_rep = np.tile(Jinv23, (n_p, 1))

Jinv31_rep = np.tile(Jinv31, (n_p, 1))
Jinv32_rep = np.tile(Jinv32, (n_p, 1))
Jinv33_rep = np.tile(Jinv33, (n_p, 1))

# 计算 DPhi 数组
DPhi1 = Jinv11_rep * DHatPhi1 + Jinv12_rep * DHatPhi2 + Jinv13_rep * DHatPhi3
DPhi2 = Jinv21_rep * DHatPhi1 + Jinv22_rep * DHatPhi2 + Jinv23_rep * DHatPhi3
DPhi3 = Jinv31_rep * DHatPhi1 + Jinv32_rep * DHatPhi2 + Jinv33_rep * DHatPhi3

# 应变-位移矩阵 B
n_b = 18 * n_p
vB = np.zeros((n_b, n_int))
vB[0:n_b-17:18, :] = DPhi1
vB[9:n_b-8:18, :] = DPhi1
vB[17:n_b:18, :] = DPhi1
vB[3:n_b-14:18, :] = DPhi2
vB[7:n_b-10:18, :] = DPhi2
vB[16:n_b-1:18, :] = DPhi2
vB[5:n_b-12:18, :] = DPhi3
vB[10:n_b-7:18, :] = DPhi3
vB[14:n_b-3:18, :] = DPhi3

# 稀疏矩阵 B 的索引
AUX = np.reshape(np.arange(1, 6 * n_int+1 ), (6, n_int),order='F')
iB = np.tile(AUX, (3 * n_p, 1))

AUX1 = np.array([1, 1, 1]).reshape(3, 1) @ np.arange(1, n_p + 1).reshape(1, n_p)
AUX2 = np.array([2, 1, 0]).reshape(3, 1) @ np.ones((1, n_p))

AUX3 = 3 * elem[AUX1.flatten(order='F').T - 1, :] - np.kron(np.ones((1, n_e)), AUX2.flatten(order='F')).reshape(len(AUX2.flatten(order='F')), n_e, order='F')
jB = np.kron(AUX3, np.ones((6, n_q)))

# 计算稀疏矩阵B
B = scipy.sparse.csr_matrix((vB.flatten(), (iB.flatten() - 1, jB.flatten() - 1)), shape=(6*n_int, 3*n_n))
B = B.tocsc()

# 弹性应力-应变矩阵 D
IOTA = np.array([1, 1, 1, 0, 0, 0])
VOL = np.outer(IOTA, IOTA)
DEV = np.diag([1, 1, 1, 0.5, 0.5, 0.5]) - VOL / 3
ELAST = 2 * np.outer(DEV.flatten(), matrix_G) + np.outer(VOL.flatten(), matrix_K)
WEIGHT = np.abs(DET) * np.tile(WF, n_e)

iD = np.tile(AUX, (6, 1))
jD = np.kron(AUX, np.ones((6, 1)))
vD = ELAST * WEIGHT

D = scipy.sparse.csr_matrix((vD.flatten(), (iD.flatten()-1, jD.flatten()-1)), shape=(6 * n_int , 6 * n_int ))
D = D.tocsc()

# 弹性刚度矩阵 K
K_elast = B.T.dot(D.dot(B))

# 组装体积力密度向量 f_V_int
f_V_int = np.array([np.zeros(n_int),-gamma * np.ones(n_int),np.zeros(n_int)])

# 总体积力向量 f_V
HatPhi = np.tile(HatP, (1, n_e))

vF1 = HatPhi * (WEIGHT * f_V_int[0, :])
vF2 = HatPhi * (WEIGHT * f_V_int[1, :])
vF3 = HatPhi * (WEIGHT * f_V_int[2, :])
iF = np.ones((n_p, n_int))
jF = np.kron(elem, np.ones((1, n_q)))

f_V = np.vstack([
    scipy.sparse.coo_matrix((vF1.flatten(), (iF.flatten()-1, jF.flatten()-1)), shape=(1, n_n)).toarray(),
    scipy.sparse.coo_matrix((vF2.flatten(), (iF.flatten()-1, jF.flatten()-1)), shape=(1, n_n)).toarray(),
    scipy.sparse.coo_matrix((vF3.flatten(), (iF.flatten()-1, jF.flatten()-1)), shape=(1, n_n)).toarray()
])

f = f_V

# 强度折减法计算参数
lambda_init = 0.8 # 初始折减因子
d_lambda_init = 0.1 # 初始增量
d_lambda_min = 1e-3 # 最小增量
step_max = 50 # 最大步数

# 牛顿迭代求解参数
it_newt_max = 30 # 最大牛顿迭代次数
it_damp_max = 10 # 线搜索最大迭代次数
tol = 1e-6 # 牛顿迭代的相对容限
r_min = tol / 100 # 刚度矩阵的基本正则化
r_damp = tol * 100 # 刚度矩阵的线搜索正则化

# 利用直接求解法DCM (Direct continuation method)计算每一迭代步骤下的位移场、强度折减系数及控制变量
U2, lambda_hist2, omega_hist2 = DCM (lambda_init, d_lambda_init, d_lambda_min, step_max, it_newt_max,
                                     it_damp_max, tol, r_min, r_damp, WEIGHT, B, K_elast, Q, f,
                                     matrix_c0, matrix_phi, matrix_psi, matrix_G, matrix_K, matrix_lamlda)

# 设置中文字体路径（Windows宋体路径）
font_path = "C:\\Windows\\Fonts\\simsun.ttc"  # 使用宋体字体
my_font = fm.FontProperties(fname=font_path)

# 提取数据
x_values = omega_hist2[0]
y_values = lambda_hist2[0]

# 定义逻辑函数模型
def logistic(x, a, b, c):
    return c / (1 + a * np.exp(-b * x))

# 曲线拟合
x_values = np.maximum(x_values, 0)  # 确保 x_values 中没有负值或零
weights = np.sqrt(x_values / max(x_values))  # 计算权重

# 曲线拟合，添加 sigma 参数
popt, pcov = curve_fit(
    logistic, x_values, y_values, p0=(1, 0.01, 1), sigma=1 / weights
)

# 使用拟合参数计算拟合曲线的 y 值
y_fit = logistic(x_values, *popt)

# 绘制散点图，使用随机颜色和较小的点
colors = [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(len(x_values))]
for i in range(len(x_values)):
    plt.scatter(x_values[i], y_values[i], color=colors[i], marker='o', s=15)

# 绘制拟合曲线
fit_label = r'拟合曲线'.format(*popt)
x_fine = np.linspace(min(x_values), max(x_values), 500)  # 增加点密度以提高拟合曲线平滑性
y_fine_fit = logistic(x_fine, *popt)
plt.plot(x_fine, y_fine_fit, color='red', label=fit_label)

# 绘制水平极限线（y 的最大值）
y_limit = popt[2]  # 逻辑函数中收敛值即为参数 c
plt.axhline(y=y_limit, color='blue', linestyle='--', zorder=0)  # zorder=0 保证水平线在曲线底层

# 动态计算文本位置，使其与 Y 轴标签对齐
x_range = max(x_values) - min(x_values)  # 计算 x 值的范围
text_x_position = min(x_values) - 0.06 * x_range  # 在 x 最小值基础上向左偏移 6% 的范围

# 计算 y 轴的刻度范围
y_ticks = plt.gca().get_yticks()

# 定义调整幅度
offset = 0.03 * (max(y_ticks) - min(y_ticks))

# 判断是否与 y 轴刻度值重叠并调整位置
for tick in y_ticks:
    if abs(y_limit - tick) < offset:
        # 根据相对位置调整
        if y_limit > tick:
            adjusted_y_limit = y_limit + offset  # 向上调整
        else:
            adjusted_y_limit = y_limit - offset  # 向下调整
        break  # 找到重叠时立即退出
else:
    # 如果没有重叠，保持原位置
    adjusted_y_limit = y_limit


# 更新水平线标注
plt.text(
    text_x_position, adjusted_y_limit, f'{y_limit:.4f}', 
    color='blue', va='center', ha='right', fontsize=10
)

# 图形设置，使用中文字体显示注释
plt.xlabel('系统总势能变化率', fontproperties=my_font)
plt.ylabel('强度折减系数', fontproperties=my_font)
plt.title('强度折减系数与系统总势能变化率关系曲线', fontproperties=my_font)

# 将拟合曲线的图例移至右下角
plt.legend(loc='lower right', prop=my_font)

# 打开网格
plt.grid(True)

# 保存高分辨率图像
plt.savefig("high_res_plot.png", dpi=1000, bbox_inches='tight')  # 保存图像到文件，分辨率为1000 PPI
plt.show()

