import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 定义几何参数
H = 8  # 坡面高度
alpha = 50  # 坡面倾角
W = 10  # 坡面宽度
T1 = 10  # 坡面上平台长度
T2 = 10  # 坡面下平台长度
S = 10  # 土层深度

# 计算坡面的x坐标
x_val = -T2 - H / np.tan(np.radians(alpha))

# 定义各点的坐标
B = np.array([0, 0, 0])
A = np.array([0, W, 0])
O = np.array([T1, W, 0])
P = np.array([T1, 0, 0])
B_ = np.array([0, 0, -S])
A_ = np.array([0, W, -S])
O_ = np.array([T1, W, -S])
P_ = np.array([T1, 0, -S])
M = np.array([x_val, 0, H])
N = np.array([x_val, W, H])
D = np.array([-H / np.tan(np.radians(alpha)), W, H])
C = np.array([-H / np.tan(np.radians(alpha)), 0, H])
M_ = np.array([x_val, 0, -S])
N_ = np.array([x_val, W, -S])
D_ = np.array([-H / np.tan(np.radians(alpha)), W, -S])
C_ = np.array([-H / np.tan(np.radians(alpha)), 0, -S])

# 创建图形
fig = plt.figure(figsize=(12, 10), dpi=600)
ax = fig.add_subplot(111, projection='3d')

# 定义各个六面体的面
faces = [
    [B, A, O, P],       # BAOP
    [P_, O_, A_, B_],   # P'O'A'B'
    [P, O, O_, P_],     # POO'P'
    [P, B, B_, P_],     # PBB'P'
    [O, A, A_, O_],     # OAA'O'
    [B, A, A_, B_],     # BAA'B'
    
    [B, C, D, A],       # BCDA
    [B_, C_, D_, A_],   # B'C'D'A'
    [C, D, D_, C_],     # CDD'C'
    [C, B, B_, C_],     # CBB'C'
    [A, D, D_, A_],     # ADD'A'
    
    [C, M, N, D],       # CMND
    [C_, M_, N_, D_],   # C'M'N'D'
    [M, N, N_, M_],     # MNN'M'
    [M, C, C_, M_],     # MCC'M'
    [N, D, D_, N_],     # NDD'N'
]

# 绘制各面
ax.add_collection3d(Poly3DCollection(faces, facecolors='goldenrod', linewidths=1, edgecolors='black', alpha=.6))

# 标注各点及其坐标
for point, name in zip([B, A, O, P, B_, A_, O_, P_, M, N, D, C, M_, N_, D_, C_], 
                       ['B', 'A', 'O', 'P', "B'", "A'", "O'", "P'", 'M', 'N', 'D', 'C', "M'", "N'", "D'", "C'"]):
    ax.text(point[0], point[1], point[2], f'{name}\n({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})', 
            color='red', fontsize=10, fontweight='bold', style='italic')

# 添加土体地质参数信息
textstr = '\n'.join((
    r'$\gamma=17\text{kN/m}^3$',
    r'$c=10\text{kN/m}^2$',
    r'$\phi=35^\circ$'
))

# 在图的左下角添加带有标题的文本框，并设置矩形边框
props = dict(boxstyle='square', facecolor='lightgrey', alpha=0.5, edgecolor='black')

# 设置文本框标题并显示地质参数
textbox = (r'边坡地质参数' + '\n\n' + textstr)
ax.text2D(0.05, 0.05, textbox, transform=ax.transAxes, fontsize=16,  # 文本框字体大小调整至16
        verticalalignment='bottom', bbox=props, fontfamily='SimSun', fontweight='bold')

# 添加标题
ax.set_title('三维边坡工程几何示意图', fontsize=18, fontfamily='SimSun', fontweight='bold')  # 标题字号加2，并加粗

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 设置坐标轴范围
ax.set_xlim([-20, 20])
ax.set_ylim([0, W+5])
ax.set_zlim([-S-5, H+5])

# 调整坐标轴视角
ax.view_init(elev=30, azim=45)

# 显示图形
plt.tight_layout()
plt.show()
