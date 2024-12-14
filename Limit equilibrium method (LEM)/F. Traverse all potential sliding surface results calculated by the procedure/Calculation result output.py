import numpy as np
import pandas as pd

# 潜在滑动面的球心坐标的网格点
x_0_values = np.arange(-4, 4, 0.4)
y_0_values = np.arange(0, 8, 0.4)
z_0_values = np.arange(0, 8, 0.4)

# 边坡几何参数定义
H = 8  # 坡面高度
alpha = 50  # 坡面倾角
W = 10  # 坡面宽度
T1 = 10  # 坡面上平台长度
T2 = 10  # 坡面下平台长度
S = 10  # 土层深度

# 坡体长度计算
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

# 函数定义：计算点到平面的距离
def point_to_plane_distance(Q, P1, P2, P3):
    normal_vector = np.cross(P2 - P1, P3 - P1)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    return np.abs(np.dot(Q - P1, normal_vector))

# 函数定义：计算点到直线的距离
def point_to_line_distance(Q, P1, P2):
    return np.linalg.norm(np.cross(Q - P1, Q - P2)) / np.linalg.norm(P2 - P1)

#函数定义：计算给定球心下的球半径范围计算及有效值判定
def calculate_R_0_values(Q_0, A, B, C, D, step_size=0.01):
    # 计算球心到坡面ABD的距离
    a = point_to_plane_distance(Q_0, A, B, D)
    
    # 计算球心到棱线AD、BC的距离
    distance_to_AD = point_to_line_distance(Q_0, A, D)
    distance_to_BC = point_to_line_distance(Q_0, B, C)
    b = min(distance_to_AD, distance_to_BC)
    
    # 定义 R_0 的取值范围
    R_0_min = a + 1.8
    R_0_max = b
    
    # 检查 R_0_min 是否大于 R_0_max
    if R_0_min > R_0_max:
        return None  # 如果 R_0_min > R_0_max，返回 None

    num_steps = int((R_0_max - R_0_min) / step_size)
    R_0_values = np.linspace(R_0_min, R_0_max, num_steps)
    
    return R_0_values

# 切分平面的定义
x_planes = np.linspace(-16.71, 10, 28)  # x方向平面
y_planes = np.linspace(0, W, 11)  # y方向平面

# 三维边坡坡体建模
original_faces = [
    [B, A, O, P], [P_, O_, A_, B_], [P, O, O_, P_], [P, B, B_, P_],
    [O, A, A_, O_], [B, A, A_, B_], [B, C, D, A], [B_, C_, D_, A_],
    [C, D, D_, C_], [C, B, B_, C_], [A, D, D_, A_], [C, M, N, D],
    [C_, M_, N_, D_], [M, N, N_, M_], [M, C, C_, M_], [N, D, D_, N_]
]

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
            
            # 绘制条柱
            faces = [[corners[0], corners[1], corners[5], corners[4]],
                     [corners[7], corners[6], corners[2], corners[3]],
                     [corners[0], corners[3], corners[7], corners[4]],
                     [corners[1], corners[2], corners[6], corners[5]],
                     [corners[0], corners[1], corners[2], corners[3]],
                     [corners[4], corners[5], corners[6], corners[7]]]

# 边坡土体力学参数
gamma = 17  # 土体重度 (KN/m³)
c = 30      # 土体粘聚力 (KN/㎡)
phi = np.radians(20)  # 土体内摩擦角 

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

# 函数定义：空间三角形面积计算
def calculate_area_from_points(p1, p2, p3):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    cross_product = np.cross(v1, v2)
    area = 0.5 * np.linalg.norm(cross_product)  # 三角形面积
    return area

# 函数定义：法向量与坐标轴的夹角计算
def calculate_angle_between_vectors(v1, v2):
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    if angle_deg > 90:
        angle_deg = 180 - angle_deg  # 确保是锐角
    return angle_deg

# 函数定义：切后条柱几何参数计算
def calculate_geometric_params(pillar_data, S_pou=0.99):
    results = []
    
    for pillar in pillar_data:
        # 提取包含交点的侧面棱的最小z值的交点坐标
        extracted_points = []
        for edge in pillar['side_edges']:
            if edge['intersection_points']:
                min_z_point = min(edge['intersection_points'], key=lambda p: p[2])
                extracted_points.append(min_z_point)
        
        # 计算提取交点z值的平均数
        extracted_z_values = [p[2] for p in extracted_points]
        extracted_z_avg = sum(extracted_z_values) / len(extracted_z_values)
        
        # 计算剩余交点坐标
        remaining_points = [p for p in pillar['all_intersection_points'] if not any(np.array_equal(p, ep) for ep in extracted_points)]

        # 如果剩余交点个数为0，使用上顶面坐标补充
        if len(remaining_points) == 0:
            remaining_points = pillar['upper_base_coordinates']
        
        # 计算剩余交点z值的平均数
        remaining_z_values = [p[2] for p in remaining_points]
        remaining_z_avg = sum(remaining_z_values) / len(remaining_z_values)
        
        # 计算平均高度 H_mean_n
        H_mean_n = abs(extracted_z_avg - remaining_z_avg)
        
        # 计算体积 V_n
        V_n = H_mean_n * S_pou
        
        # 计算底面面积 A_n
        if len(extracted_points) >= 3:
            # 如果有4个点，取任意3个点组成三角形，然后面积乘以2
            if len(extracted_points) == 4:
                A_n = 2 * calculate_area_from_points(extracted_points[0], extracted_points[1], extracted_points[2])
            else:
                A_n = calculate_area_from_points(extracted_points[0], extracted_points[1], extracted_points[2])
        else:
            A_n = 0  # 如果不足3个点，无法计算面积
        
        # 计算法向量与x轴、y轴、z轴夹角
        if len(extracted_points) >= 3:
            v1 = np.array(extracted_points[1]) - np.array(extracted_points[0])
            v2 = np.array(extracted_points[2]) - np.array(extracted_points[0])
            normal_vector = np.cross(v1, v2)  # 计算法向量
            
            # 底面与x轴(1,0,0)夹角 α_x_n
            x_axis = np.array([1, 0, 0])
            α_x_n = 90 - calculate_angle_between_vectors(normal_vector, x_axis)
            
            # 底面与y轴(0,1,0)夹角 α_y_n
            y_axis = np.array([0, 1, 0])
            α_y_n = 90 - calculate_angle_between_vectors(normal_vector, y_axis)
            
            # 底面法向量与z轴的夹角 γ_z_n
            z_axis = np.array([0, 0, 1])
            γ_z_n = calculate_angle_between_vectors(normal_vector, z_axis) 
        else:
            α_x_n = None
            α_y_n = None
            γ_z_n = None
        
        # 将结果存储并包含提取坐标和剩余坐标信息
        results.append({
            'pillar_no': pillar['pillar_no'],
            'H_mean_n': H_mean_n,
            'V_n': V_n,
            'A_n': A_n,
            'α_x_n': α_x_n,
            'α_y_n': α_y_n,
            'γ_z_n': γ_z_n,
            'extracted_points': extracted_points,
            'remaining_points': remaining_points
        })
    
    return results

# 函数定义：切后条柱力学参数计算
def calculate_mechanical_params(geometric_results):
    mechanical_results = []
    
    for result in geometric_results:
        # 提取几何参数
        V_n = result['V_n']
        A_n = result['A_n']
        alpha_x_n = np.radians(result['α_x_n']) if result['α_x_n'] is not None else 0
        gamma_z_n = np.radians(result['γ_z_n']) if result['γ_z_n'] is not None else 0
        
        # 计算自重力 W_n
        W_n = gamma * V_n
        
        # 计算法向力 N_n
        if (cos_gamma_z_n := np.cos(gamma_z_n)) + (tan_phi := np.tan(phi)) * np.sin(alpha_x_n) != 0:
            N_n = (W_n - c * A_n * np.sin(alpha_x_n)) / (cos_gamma_z_n + tan_phi * np.sin(alpha_x_n))
        else:
            N_n = 0
        
        # 计算切向力 T_n
        T_n = N_n * np.tan(phi) + c * A_n
        
        # 存储结果
        mechanical_results.append({
            'pillar_no': result['pillar_no'],
            'W_n': W_n,
            'N_n': N_n,
            'T_n': T_n
        })
    
    return mechanical_results

# 函数定义：给定滑动面下的整体边坡稳定性系数计算
def calculate_slope_stability(geometric_results, mechanical_results):
    numerator_sum = 0  # 分子累加
    denominator_sum = 0  # 分母累加

    for i, (geom, mech) in enumerate(zip(geometric_results, mechanical_results)):
        # 提取几何参数
        A_n = geom['A_n']
        alpha_x_n = np.radians(geom['α_x_n']) if geom['α_x_n'] is not None else 0
        gamma_z_n = np.radians(geom['γ_z_n']) if geom['γ_z_n'] is not None else 0
        
        # 提取力学参数
        N_n = mech['N_n']
        
        # 计算分子和分母
        numerator = c * A_n * np.cos(alpha_x_n) + N_n * np.tan(phi) * np.cos(alpha_x_n)
        denominator = N_n * np.cos(gamma_z_n) * np.tan(alpha_x_n)
        
        # 累加分子和分母
        numerator_sum += numerator
        denominator_sum += denominator

    # 计算稳定性系数 Fs
    if denominator_sum != 0:
        Fs = numerator_sum / denominator_sum
    else:
        Fs = float('inf')  # 避免除以0
    
    return Fs

# 函数定义：给定球心下的最小稳定性系数搜索
def calculate_min_slope_stability(Q_0, R_0_values, pillar_params):

    # 初始化最小边坡稳定性系数及对应R值
    min_Fs = float('inf')
    optimal_R = None

    # 遍历每个R_0值
    for R_0 in R_0_values:
        pillar_data_list = []  # 存储每个条柱的几何信息

        for pillar in pillar_params:
            corners = np.array(pillar['corners'])
            edges = [
                (corners[0], corners[1]), (corners[1], corners[5]), (corners[5], corners[4]), (corners[4], corners[0]),
                (corners[1], corners[2]), (corners[2], corners[6]), (corners[6], corners[5]), (corners[2], corners[3]),
                (corners[3], corners[7]), (corners[7], corners[6]), (corners[3], corners[0]), (corners[7], corners[4])
            ]

            side_edge_with_intersections_count = 0  # 含有交点的侧面棱数量计数器
            side_edges_with_intersections = []  # 存储含有交点的侧面棱信息
            all_intersection_points = []  # 存储所有交点坐标
            upper_base_coordinates = [tuple(corner) for corner in corners[4:8]]  # 上顶面坐标

            # 遍历每条棱，并根据与球面的交点计算输出
            for i, (p1, p2) in enumerate(edges):
                intersections = line_plane_intersection(p1, p2, Q_0, R_0)
                valid_intersections = [pt for pt in intersections if is_within_bounds(pt, p1, p2)]

                if valid_intersections:
                    # 获取棱的相对位置注记
                    edge_index = i + 1

                    # 存储交点信息
                    all_intersection_points.extend(valid_intersections)

                    # 如果该棱是侧面棱（编号为2, 4, 6, 9），增加计数并存储该棱
                    if edge_index in [2, 4, 6, 9]:
                        side_edge_with_intersections_count += 1
                        side_edges_with_intersections.append({
                            'edge_index': edge_index,
                            'intersection_points': valid_intersections
                        })

            # 如果含有交点的侧面棱的数量小于3，不存储该条柱信息，跳过该条柱
            if side_edge_with_intersections_count < 3:
                continue

            # 存储该条柱的几何信息，用于后续几何参数计算
            pillar_data_list.append({
                'pillar_no': pillar['编号'],
                'side_edges': side_edges_with_intersections,
                'all_intersection_points': all_intersection_points,
                'upper_base_coordinates': upper_base_coordinates
            })

        # 计算几何和力学参数并求出边坡稳定性系数
        geometric_results = calculate_geometric_params(pillar_data_list)
        mechanical_results = calculate_mechanical_params(geometric_results)
        Fs = calculate_slope_stability(geometric_results, mechanical_results)
        
        if Fs < 0:
            Fs = float('inf')
            R_0 = None

        # 更新最小的稳定性系数及对应的R值
        if Fs < min_Fs:
            min_Fs = Fs
            optimal_R = R_0

    # 返回最小边坡稳定性系数及对应的R值
    return min_Fs, optimal_R

# 初始化Excel数据记录
results = []

# 遍历球心坐标,计算三维边坡模型的整体边坡稳定性系数
for x_0 in x_0_values:
    for y_0 in y_0_values:
        for z_0 in z_0_values:
            Q_0 = np.array([x_0, y_0, z_0])  # 当前球心坐标
            R_0_values = calculate_R_0_values(Q_0, A, B, C, D)  # 计算球半径范围
            
            if R_0_values is None:
                min_Fs = float('inf')  # 如果没有可行的 R_0 范围，最小稳定性系数设为无穷大
                optimal_R = None
            else:
                # 调用计算最小边坡稳定性系数的函数
                min_Fs, optimal_R = calculate_min_slope_stability(Q_0, R_0_values, pillar_params)

            if min_Fs != float('inf'):
                results.append({
                    '球心坐标_X': x_0,
                    '球心坐标_Y': y_0,
                    '球心坐标_Z': z_0,
                    '最小边坡稳定性系数': min_Fs,
                    '对应球半径': optimal_R
                })
            if min_Fs != float('inf'):
                print(f"球心坐标: ({x_0:.2f}, {y_0:.2f}, {z_0:.2f}), 最小边坡稳定性系数: {min_Fs:.4f}, 对应球半径: {optimal_R:.4f}" if optimal_R is not None else f"球心坐标: ({x_0:.2f}, {y_0:.2f}, {z_0:.2f}), 最小边坡稳定性系数: {min_Fs:.4f}, 对应球半径: None")

# 将结果保存到Excel文件，便于存档查阅
df = pd.DataFrame(results)
df.to_excel('slope_stability_results.xlsx', index=False)

# 计算结束语
print("计算完成")
