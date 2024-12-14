import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from tkinter import Tk
from tkinter import filedialog
import sys  # 用于终止程序

# 创建一个文件选择器窗口
def choose_excel_file():
    root = Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(
        title='选择 Excel 文件',
        filetypes=[('Excel 文件', '*.xlsx *.xls')]
    )
    return file_path

# 调用文件选择器，获取 Excel 文件路径
file_path = choose_excel_file()

# 简化后的检查：如果未选择文件，终止程序
if not file_path:
    print("未选择文件，程序终止。")
    sys.exit()  # 终止程序

# 读取 Excel 文件并提取数据
print(f"已选择文件: {file_path}")
df = pd.read_excel(file_path)

sphere_data = [
    (np.array([row['球心坐标_X'], row['球心坐标_Y'], row['球心坐标_Z']]), row['对应球半径'], row['最小边坡稳定性系数'])
    for _, row in df.iterrows()
]

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
M = np.array([x_val, 0, H])
N = np.array([x_val, W, H])
D = np.array([-H / np.tan(np.radians(alpha)), W, H])
C = np.array([-H / np.tan(np.radians(alpha)), 0, H])

# 球心坐标 Q 和半径 R
Q = np.array([-0.77, 5.00, 5.88])
R = 5.634

# 函数定义：计算平面方程
def plane_from_points(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)  # 法向量
    a0, b0, c0 = normal
    d0 = -np.dot(normal, p1)
    return v1, v2, normal, a0, b0, c0, d0

def intersection_sphere_plane(Q, R, a0, b0, c0, d0):
    # 球面与平面交线计算
    dist = abs(a0 * Q[0] + b0 * Q[1] + c0 * Q[2] + d0) / np.sqrt(a0**2 + b0**2 + c0**2)
    if dist > R:
        return None
    else:
        r = np.sqrt(R**2 - dist**2)
        t = (a0 * Q[0] + b0 * Q[1] + c0 * Q[2] + d0) / (a0**2 + b0**2 + c0**2)
        proj_center = Q - t * np.array([a0, b0, c0])
        return proj_center, r

# 平面方程
_, _, _, A1, B1, C1, D1 = plane_from_points(P, B, O)  # 平面 PBAO
_, _, _, A2, B2, C2, D2 = plane_from_points(B, C, A)  # 平面 BCDA
_, _, _, A3, B3, C3, D3 = plane_from_points(C, M, D)  # 平面 CMND

# 闭合曲线中心及半径
results = {
    'PBAO': intersection_sphere_plane(Q, R, A1, B1, C1, D1),
    'BCDA': intersection_sphere_plane(Q, R, A2, B2, C2, D2),
    'CMND': intersection_sphere_plane(Q, R, A3, B3, C3, D3)
}

# 创建3D绘图
fig = plt.figure(figsize=(12, 10), dpi=600)
ax = fig.add_subplot(111, projection='3d')

# 定义各个六面体的面
faces = [
    [B, A, O, P],       # BAOP
    [B, C, D, A],       # BCDA
    [C, M, N, D],       # CMND
]

# 绘制边坡模型
ax.add_collection3d(Poly3DCollection(faces, facecolors='goldenrod', linewidths=1, edgecolors='black', alpha=0.2))

# 函数定义：计算平面方程取值范围
def get_plane_bounds(p1, p2, p3, p4):
    x_min = min(p1[0], p2[0], p3[0], p4[0])
    x_max = max(p1[0], p2[0], p3[0], p4[0])
    
    y_min = min(p1[1], p2[1], p3[1], p4[1])
    y_max = max(p1[1], p2[1], p3[1], p4[1])
    
    z_min = min(p1[2], p2[2], p3[2], p4[2])
    z_max = max(p1[2], p2[2], p3[2], p4[2])
    
    return x_min, x_max, y_min, y_max, z_min, z_max

# 修改 plot_circle_within_bounds 函数以支持自定义 linewidth
def plot_circle_within_bounds(ax, center, radius, normal, v1, v2, bounds, color='b', label='', linewidth=1.0):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_points = np.array([center + radius * (np.cos(t) * v1 + np.sin(t) * v2) for t in theta])
    
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    valid_points = [p for p in circle_points if (x_min <= p[0] <= x_max and y_min <= p[1] <= y_max and z_min <= p[2] <= z_max)]
    
    if valid_points:
        valid_points = np.array(valid_points)
        ax.plot(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2], color=color, linewidth=linewidth)

# 获取每个平面的四个顶点并计算取值范围
bounds_PBAO = get_plane_bounds(P, B, O, A)
bounds_BCDA = get_plane_bounds(B, C, A, D)
bounds_CMND = get_plane_bounds(C, M, D, N)

# 平面的定义信息
planes = [
    (A1, B1, C1, D1, bounds_PBAO, P - B, A - B, 'PBAO', 'b'),
    (A2, B2, C2, D2, bounds_BCDA, B - C, D - C, 'BCDA', 'g'),
    (A3, B3, C3, D3, bounds_CMND, C - M, N - M, 'CMND', 'r')
]

# 函数定义：交线绘制
def plot_intersection_for_sphere(ax, Q, R, planes, colors):
    for (a0, b0, c0, d0, bounds, v1, v2, label, color) in planes:
        # 计算交线
        result = intersection_sphere_plane(Q, R, a0, b0, c0, d0)
        if result is not None:
            proj_center, r = result
            plot_circle_within_bounds(ax, proj_center, r, np.array([a0, b0, c0]), v1, v2, bounds, color=color, label=label)

# 提取稳定性系数并计算最小和最大值
stability_values = [data[2] for data in sphere_data]
min_stability, max_stability = min(stability_values), max(stability_values)

# 创建颜色映射（红 -> 黄 -> 绿 -> 蓝），并分为10个等级
cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['red', 'yellow', 'green', 'blue'], N=10)

# 定义分级边界值，按照间隔0.100分成10个区间
boundaries = np.linspace(min_stability, min_stability+1.000, 11)  # 生成10级边界
norm = mcolors.BoundaryNorm(boundaries, cmap.N)

# 创建标准化对象
norm = mcolors.BoundaryNorm(boundaries, cmap.N)

# 修改 plot_intersection_for_sphere 函数，按照稳定性系数优先级倒序绘制
def plot_intersection_for_sphere(ax, Q, R, stability_value, planes):
    # 根据稳定性系数获取颜色
    color = cmap(norm(stability_value))
    linewidth = 1.0  # 默认线宽

    for (a0, b0, c0, d0, bounds, v1, v2, label, _) in planes:
        # 计算交线
        result = intersection_sphere_plane(Q, R, a0, b0, c0, d0)
        if result is not None:
            proj_center, r = result
            # 使用加粗线条绘制最小稳定性系数的交线
            if stability_value == min_stability:
                linewidth = 2.5  # 最小稳定性系数对应的线条加粗
                color = 'white'  # 最小值交线用白色绘制
            plot_circle_within_bounds(ax, proj_center, r, np.array([a0, b0, c0]), v1, v2, bounds, color=color, linewidth=linewidth)

sphere_data_sorted = sorted(sphere_data, key=lambda x: x[2], reverse=True)

for Q, R, stability_value in sphere_data_sorted:
    plot_intersection_for_sphere(ax, Q, R, stability_value, planes)

# 设置颜色条
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, boundaries=boundaries, ticks=boundaries[:-1])
cbar.set_label('Stability Coefficient')

# 设置标题和坐标轴
ax.set_title('基于ChatGPT-Python工作流程的滑移面稳定性系数分布图', fontsize=18, fontfamily='SimSun', fontweight='bold')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 设置坐标轴范围
ax.set_xlim([-16.71, 10])
ax.set_ylim([0, W])
ax.set_zlim([-2, H])

# 调整视角
ax.view_init(elev=45, azim=45)

# 显示图形
plt.tight_layout()
plt.show()
