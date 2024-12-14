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

# 定义 x 和 y 的平面位置
x_planes = np.linspace(-16.71, 10, 28)  # x方向平面
y_planes = np.linspace(0, W, 11)  # y方向平面

# 创建图形
fig = plt.figure(figsize=(12, 10), dpi=600)
ax = fig.add_subplot(111, projection='3d')

# 创建原始坡体的多面体
original_faces = [
    [B, A, O, P], [P_, O_, A_, B_], [P, O, O_, P_], [P, B, B_, P_],
    [O, A, A_, O_], [B, A, A_, B_], [B, C, D, A], [B_, C_, D_, A_],
    [C, D, D_, C_], [C, B, B_, C_], [A, D, D_, A_], [C, M, N, D],
    [C_, M_, N_, D_], [M, N, N_, M_], [M, C, C_, M_], [N, D, D_, N_]
]

# 绘制原坡体（透明化）
ax.add_collection3d(Poly3DCollection(original_faces, facecolors='goldenrod', linewidths=1, edgecolors='black', alpha=0.2))

# 创建条柱并标记编号
pillar_params = []
pillar_counter = 0

# 函数：计算条柱的顶面高度
def calculate_z_top(x1, x2):
    """计算给定 x1 和 x2 位置的坡面顶面高度"""
    z_top1 = 0 if x1 >= 0 else (8 if x1 < -6.71 else -8 * x1 / 6.71)
    z_top2 = 0 if x2 >= 0 else (8 if x2 < -6.71 else -8 * x2 / 6.71)
    return z_top1, z_top2

# 计算两点间的距离
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

# 计算条柱体积
def calculate_volume(corners):
    """使用顶点坐标计算条柱的体积"""
    base_area = distance(corners[0], corners[1]) * distance(corners[0], corners[3])
    avg_height = (distance(corners[0], corners[4]) + distance(corners[1], corners[5]) + 
                  distance(corners[2], corners[6]) + distance(corners[3], corners[7])) / 4
    return base_area * avg_height

# 遍历x和y方向的平面位置，创建条柱
for i in range(len(x_planes) - 1):
    for j in range(len(y_planes) - 1):
        x1, x2 = x_planes[i], x_planes[i+1]
        y1, y2 = y_planes[j], y_planes[j+1]

        # 计算条柱的顶面和底面高度
        z_bottom = -S
        z_top1, z_top2 = calculate_z_top(x1, x2)
        
        if z_top1 > z_bottom or z_top2 > z_bottom:  # 有效条柱
            pillar_counter += 1
            
            # 条柱的顶点
            corners = [
                [x1, y1, z_bottom],
                [x2, y1, z_bottom],
                [x2, y2, z_bottom],
                [x1, y2, z_bottom],
                [x1, y1, z_top1],
                [x2, y1, z_top2],
                [x2, y2, z_top2],
                [x1, y2, z_top1]
            ]
            
            # 计算条柱体积
            volume = calculate_volume(corners)
            
            # 记录条柱的顶点坐标和体积
            pillar_params.append({
                '编号': pillar_counter,
                'corners': corners,
                'volume': volume
            })
            
            # 绘制条柱
            faces = [[corners[0], corners[1], corners[5], corners[4]],
                     [corners[7], corners[6], corners[2], corners[3]],
                     [corners[0], corners[3], corners[7], corners[4]],
                     [corners[1], corners[2], corners[6], corners[5]],
                     [corners[0], corners[1], corners[2], corners[3]],
                     [corners[4], corners[5], corners[6], corners[7]]]
            ax.add_collection3d(Poly3DCollection(faces, facecolors='lightblue', linewidths=0.1, edgecolors='black', alpha=.6))

# 添加标题和轴标签
ax.set_title('3D Slope with Corrected Pillar Heights', fontsize=18, fontweight='bold')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 设置坐标轴范围
ax.set_xlim([x_planes[0], x_planes[-1]])
ax.set_ylim([y_planes[0], y_planes[-1]])
ax.set_zlim([-S, H])

# 调整视角
ax.view_init(elev=30, azim=45)

# 显示图形
plt.tight_layout()
plt.show()

# 打印条柱顶点坐标及体积
for pillar in pillar_params:
    print(f"Pillar No.: {pillar['编号']}, Volume: {pillar['volume']:.2f}")
    print(f"Lower Base Coordinates:")
    for corner in pillar['corners'][:4]:
        print(f"\t({corner[0]:.2f}, {corner[1]:.2f}, {corner[2]:.2f})")
    print(f"Upper Base Coordinates:")
    for corner in pillar['corners'][4:]:
        print(f"\t({corner[0]:.2f}, {corner[1]:.2f}, {corner[2]:.2f})")
    print("\n")
