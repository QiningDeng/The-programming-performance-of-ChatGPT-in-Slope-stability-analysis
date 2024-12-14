import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 边坡几何参数
H = 8  # 坡面高度
alpha = 50  # 坡面倾角
W = 10  # 坡面宽度
T1 = 10  # 坡面上平台长度
T2 = 10  # 坡面下平台长度
S = 10  # 土层深度

# 坡体长度
x_val = -T2 - H / np.tan(np.radians(alpha))

# 三维建模的坡体关键点坐标
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

# 切分平面的定义
x_planes = np.linspace(-16.71, 10, 28)  # x方向平面
y_planes = np.linspace(0, W, 11)  # y方向平面

# 图形参数初始设定
fig = plt.figure(figsize=(12, 10), dpi=600)
ax = fig.add_subplot(111, projection='3d')

# 三维边坡坡体建模
original_faces = [
    [B, A, O, P], [P_, O_, A_, B_], [P, O, O_, P_], [P, B, B_, P_],
    [O, A, A_, O_], [B, A, A_, B_], [B, C, D, A], [B_, C_, D_, A_],
    [C, D, D_, C_], [C, B, B_, C_], [A, D, D_, A_], [C, M, N, D],
    [C_, M_, N_, D_], [M, N, N_, M_], [M, C, C_, M_], [N, D, D_, N_]
]

# 建模可视化绘制
ax.add_collection3d(Poly3DCollection(original_faces, facecolors='goldenrod', linewidths=1, edgecolors='black', alpha=0.2))

# 建立条柱数据库
pillar_params = []
pillar_counter = 0

# 函数定义：计算条柱的顶面高度
def calculate_z_top(x1, x2):
    """计算给定 x1 和 x2 位置的坡面顶面高度"""
    z_top1 = 0 if x1 >= 0 else (8 if x1 < -6.71 else -8 * x1 / 6.71)
    z_top2 = 0 if x2 >= 0 else (8 if x2 < -6.71 else -8 * x2 / 6.71)
    return z_top1, z_top2

# 函数定义：计算两点间的距离
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


# 依据切分平面进行条柱划分
for i in range(len(x_planes) - 1):
    for j in range(len(y_planes) - 1):
        x1, x2 = x_planes[i], x_planes[i+1]
        y1, y2 = y_planes[j], y_planes[j+1]

        # 计算条柱的顶面和底面高度
        z_bottom = -S
        z_top1, z_top2 = calculate_z_top(x1, x2)
        
        if z_top1 > z_bottom or z_top2 > z_bottom:  # 有效条柱判定
            pillar_counter += 1
            
            # 完整条柱的顶点
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
            
            # 存储条柱的顶点坐标
            pillar_params.append({
                '编号': pillar_counter,
                'corners': corners
            })

# 滑动球面参数
Q_0 = np.array([-3, 5, 5])  # 球心
R_0 = 5  # 球半径

# 函数定义：计算球面方程和棱方程的交点
def line_plane_intersection(p1, p2, Q_0, R_0):
    """计算直线p1p2与球面Q_0, R_0的交点"""
    A = p2 - p1
    B = p1 - Q_0
    a = np.dot(A, A)
    b = 2 * np.dot(B, A)
    c = np.dot(B, B) - R_0 ** 2
    
    delta = b ** 2 - 4 * a * c
    if delta < 0:
        return []  # 没有交点
    
    sqrt_delta = np.sqrt(delta)
    t1 = (-b - sqrt_delta) / (2 * a)
    t2 = (-b + sqrt_delta) / (2 * a)
    
    intersections = []
    if t1 >= 0:
        intersections.append(p1 + t1 * A)
    if t2 >= 0 and t2 != t1:
        intersections.append(p1 + t2 * A)
    
    return intersections

# 函数定义：检查交点是否在棱的端点范围内
def is_within_bounds(intersection, p1, p2):
    """检查交点是否在棱的端点范围内"""
    min_x, max_x = min(p1[0], p2[0]), max(p1[0], p2[0])
    min_y, max_y = min(p1[1], p2[1]), max(p1[1], p2[1])
    min_z, max_z = min(p1[2], p2[2]), max(p1[2], p2[2])
    
    return (min_x <= intersection[0] <= max_x and
            min_y <= intersection[1] <= max_y and
            min_z <= intersection[2] <= max_z)

# 条柱棱的相对位置注记
face_info = {
    1: "下顶面棱",
    2: "侧面棱",
    3: "上顶面棱",
    4: "侧面棱",
    5: "下顶面棱",
    6: "侧面棱",
    7: "上顶面棱",
    8: "下顶面棱",
    9: "侧面棱",
    10: "上顶面棱",
    11: "下顶面棱",
    12: "上顶面棱"
}

# 确定相交条柱的上顶面
intersecting_pillars = set()  # 存储含有交点的条柱编号

# 遍历条柱并计算每条棱与球面的交点
for pillar in pillar_params:
    corners = np.array(pillar['corners'])
    edges = [
        (corners[0], corners[1]), (corners[1], corners[5]), (corners[5], corners[4]), (corners[4], corners[0]),
        (corners[1], corners[2]), (corners[2], corners[6]), (corners[6], corners[5]), (corners[2], corners[3]),
        (corners[3], corners[7]), (corners[7], corners[6]), (corners[3], corners[0]), (corners[7], corners[4])
    ]
    
    side_edge_with_intersections_count = 0  # 含有交点的侧面棱数量计数器
    
    # 遍历每条棱，并根据与球面的交点计算输出
    for i, (p1, p2) in enumerate(edges):
        intersections = line_plane_intersection(p1, p2, Q_0, R_0)
        valid_intersections = [pt for pt in intersections if is_within_bounds(pt, p1, p2)]
        
        if valid_intersections:
            # 获取棱的相对位置注记
            edge_index = i + 1
            relative_position = face_info[edge_index]
            
            # 如果该棱是侧面棱（编号为2, 4, 6, 9），增加计数
            if edge_index in [2, 4, 6, 9]:
                side_edge_with_intersections_count += 1
            
            # 如果交点的侧面棱数量符合条件，标记条柱编号
            if side_edge_with_intersections_count >= 3:
                intersecting_pillars.add(pillar['编号'])
                break  # 不再检查其他棱，直接标记该条柱

# 绘制相交条柱的上顶面
for pillar in pillar_params:
    corners = np.array(pillar['corners'])
    if pillar['编号'] in intersecting_pillars:
        # 只绘制相交条柱的上顶面
        top_face = [corners[4], corners[5], corners[6], corners[7]]
        ax.add_collection3d(Poly3DCollection([top_face], facecolors='red', linewidths=0.1, edgecolors='black', alpha=0.7))

# 添加标题和轴标签
ax.set_title('3D Slope with Highlighted Top Faces of Intersecting Pillars', fontsize=18, fontweight='bold')
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
