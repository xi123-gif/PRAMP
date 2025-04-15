import multiprocessing
import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import LineString
# 导入excle数据
import json
import networkx as nx
from haversine import haversine
import math
from warnings import simplefilter
from scipy.spatial import KDTree
simplefilter(action="ignore", category=FutureWarning)
import re
import ast
import xiangshi as xs
import time
import pandas as pd
import glob
from merge_mat2s_martix import *

# ---------------------01构建需要计算地理相似度的函数------------------------------------
def build_roi_data(geographic_similarity_data, node_coordinates):
    geographic_similarity_data_noduplicates = pd.DataFrame()
    # 准备构建无向图中全部区域的roi数据
    # 针对中心点经纬度寻找无向图G中最近节点，并保存到nearest_node字段
    keys_list = list(node_coordinates.keys())
    # keys_list中的节点信息
    keys_list_info = [eval(node_str) for node_str in keys_list]
    # 构建KD树
    kdtree = KDTree(keys_list_info)
    # 计算每个区域roi的中心点，并保存到geographic_similarity_data['user_id', 'roi', 'centre_lat', 'centre_lon']
    # for user_id in unique_users:
    #     unique_roi = data[data['user_id'] == user_id][['roi']]
    unique_roi = data['roi'].unique()
    temp_data = pd.DataFrame()
    for user_roi in unique_roi:
        # 计算每个roi的中心点，作为区域中心点
        user_data = data[(data['roi'] == user_roi)][
            ['user_id', 'roi', 'LAT', 'LON', 'check_in_freq', 'category', 'active_region','TIME']]

        # 计算每个roi区域中心点的经纬度
        centre_lat = user_data['LAT'].mean()
        centre_lon = user_data['LON'].mean()

        # formatted_coords = [(centre_lon, centre_lat)]
        formatted_coords = [( centre_lat,centre_lon)]

        # 查询最近邻点
        nearest_dist, nearest_idx = kdtree.query(formatted_coords)
        nearest_idx1 = nearest_idx[0]
        # 最近邻点的坐标
        nearest_node1, nearest_node2 = keys_list_info[nearest_idx1]

        nearest_node = [(nearest_node1, nearest_node2)]

        # 向user-data增加新列并赋值
        user_data['centre_lat'] = centre_lat
        user_data['centre_lon'] = centre_lon
        user_data['nearest_node'] = [nearest_node] * len(user_data)
        # temp_data.append(user_data)
        temp_data = temp_data._append(user_data, ignore_index=True)
    #保存原始数据
    geographic_similarity_data_noduplicates = geographic_similarity_data_noduplicates._append(temp_data, ignore_index=True)
    # geographic_similarity_data_noduplicates.to_csv('process_data//foursquare_roi_finaldata.csv', index=False)
    geographic_similarity_data_noduplicates.to_csv('process_data//shanghai_roi_finaldata.csv', index=False
                                                   ,encoding='utf-8-sig')

    # 去除用户重复值
    for user_id in unique_users:
        user_data = temp_data[(temp_data['user_id'] == user_id)][
            ['user_id', 'roi', 'centre_lon', 'centre_lat', 'check_in_freq', 'category', 'active_region',
             'nearest_node']]
        use_data = user_data.drop_duplicates(subset=['roi'])
        geographic_similarity_data = geographic_similarity_data._append(use_data, ignore_index=True)

    geographic_similarity_data.to_csv('process_data//shanghai_geographic_similarity_data.csv', index=False ,encoding='utf-8-sig')
    return geographic_similarity_data_noduplicates,geographic_similarity_data


# -----------------------02暂存道路数据的无向图格式-----------------

# 计算无向图中每条边的夹角信息
def calculate_direction_weight(G, edge):
    # 假设有函数 get_edge_direction 获取边的方向信息
    # 具体实现可能需要使用 GIS 工具或其他方法获取道路方向信息
    # 获取边的起点和终点坐标
    start_coords = G.nodes[edge[0]]['pos']
    end_coords = G.nodes[edge[1]]['pos']
    # 计算相对方向，可以使用角度差来表示
    relative_direction = calculate_relative_direction(start_coords, end_coords)
    return relative_direction


# 生成虚拟图时，需要将节点转换为字符串及节点属性为字典（带有经纬度）
def extract_float_values(input_str):
    pattern = r"\((-?\d+\.\d+), (-?\d+\.\d+)\)"
    match = re.search(pattern, input_str)

    if match:
        value1 = float(match.group(1))
        value2 = float(match.group(2))
        return value1, value2
    else:
        return None


def get_coordinate_from_node(node):
    if isinstance(node, tuple) and len(node) == 2:
        # 如果节点是元组类型且包含 2 个元素
        if isinstance(node[1], dict):
            # 如果第二个元素是字典类型
            if 'pos' in node[1]:
                # 如果包含 'pos' 字段
                pos_data = node[1]['pos']
                if isinstance(pos_data, str):
                    # 如果 'pos' 字段是字符串，则尝试将其解析为字典
                    try:
                        pos_data = ast.literal_eval(pos_data)
                    except (ValueError, SyntaxError):
                        return None
                return pos_data.get('lat'), pos_data.get('lon')
            elif 'lat' in node[1] and 'lon' in node[1]:
                # 如果包含 'lat' 和 'lon' 字段
                return node[1].get('lat'), node[1].get('lon')
    return None


def calculate_relative_direction(start_coords, end_coords):
    # 计算角度差，作为相对方向的表示
    start_lon = start_coords['lon']
    start_lat = start_coords['lat']
    end_lon = end_coords['lon']
    end_lat = end_coords['lat']

    delta_lon = end_lon - start_lon
    delta_lat = end_lat - start_lat

    # 计算角度
    angle = math.degrees(math.atan2(delta_lat, delta_lon))

    # 将角度调整到 [0, 360) 范围内
    relative_direction = (angle + 360) % 360

    return relative_direction


def road_graph(shp_path):
    gdf = gpd.read_file(shp_path)
    # 创建无向图
    G = nx.Graph()
    # 将线状几何对象添加为图的边
    for index, row in gdf.iterrows():
        geometry = row['geometry']
        if isinstance(geometry, LineString):
            coords = list(geometry.coords)
            for i in range(len(coords) - 1):
                # 提取起点和终点坐标
                start_coord, end_coord = coords[i], coords[i + 1]
                start_lon, start_lat = start_coord
                end_lon, end_lat = end_coord

                # 投影坐标转地理坐标
                start_lon = start_lon / 20037508.34 * 180
                start_lat = start_lat / 20037508.34 * 180
                start_lat = 180 / math.pi * (2 * math.atan(math.exp(start_lat * math.pi / 180)) - math.pi / 2)
                start_lon = round(start_lon, 5)
                start_lat = round(start_lat, 5)
                # start_coord = (start_lon, start_lat)
                start_coord = (start_lat,start_lon)

                # 投影坐标转地理坐标
                end_lon = end_lon / 20037508.34 * 180
                end_lat = end_lat / 20037508.34 * 180
                end_lat = 180 / math.pi * (2 * math.atan(math.exp(end_lat * math.pi / 180)) - math.pi / 2)
                end_lon = round(end_lon, 5)
                end_lat = round(end_lat, 5)
                # end_coord = (end_lon, end_lat)
                end_coord = (end_lat,end_lon)

                # 添加节点到图中，将经纬度坐标以字典形式存储在节点属性中
                G.add_node(str(start_coord), pos={'lon': start_lon, 'lat': start_lat})
                G.add_node(str(end_coord), pos={'lon': end_lon, 'lat': end_lat})

                # 计算地理距离作为权重
                distance = haversine(start_coord, end_coord)

                # 添加边到图中
                G.add_edge(str(start_coord), str(end_coord), weight=distance)

    # 在构建图时添加地理方向权重
    for edge in G.edges():
        # 计算地理方向权重
        direction_weight = calculate_direction_weight(G, edge)
        # 添加边的地理方向权重属性
        G[edge[0]][edge[1]]['direction_weight'] = direction_weight

    # 获取连通子图列表
    connected_subgraphs = list(nx.connected_components(G))
    # 添加虚拟边，连接不同的连通子图
    for i in range(len(connected_subgraphs) - 1):
        start_node_subgraph = list(connected_subgraphs[i])[0]
        end_node_subgraph = list(connected_subgraphs[i + 1])[0]
        # 确保虚拟边的起始节点和终止节点存在于图中
        if start_node_subgraph not in G.nodes or end_node_subgraph not in G.nodes:
            continue
        # 将节点转换为字符串及节点属性为字典（带有经纬度）
        start_node_str_lon, start_node_str_lat = extract_float_values(start_node_subgraph)
        end_node_str_lon, end_node_str_lat = extract_float_values(end_node_subgraph)

        G.add_node(str(start_node_subgraph),
                   pos={'lon': start_node_str_lon, 'lat': start_node_str_lat})  # 添加虚拟边的起点节点
        G.add_node(str(end_node_subgraph), pos={'lon': end_node_str_lon, 'lat': end_node_str_lat})  # 添加虚拟边的终点节点
        G.add_edge(str(start_node_subgraph), str(end_node_subgraph), weight=float('inf'))  # 添加虚拟边
    # 保存经纬度信息
    node_coordinates = {node_id: get_coordinate_from_node(node_data) for node_id, node_data in G.nodes(data=True)}

    # 存储图为GraphML文件
    for node in G.nodes(data=True):
        node_id, node_data = node
        node_data['pos'] = json.dumps(node_data['pos'])  # 将字典转为 JSON 字符串
    nx.write_graphml(G, "process_data//shanghai_graph.graphml")
    # # # 绘制图（可选）
    # pos = nx.get_node_attributes(G, 'pos')  # 获取节点的位置信息
    # nx.draw(G, pos,with_labels=True, font_weight='bold')
    # plt.show()
    # print('道路无向图保存成功！')
    # 检查图的连通性
    def is_graph_connected(graph):
        return nx.is_connected(graph)
    print("Is the graph connected?", is_graph_connected(G))
    return G, node_coordinates


# ------------------------03计算区域与区域之间的最短距离（dijstra 和A*）-----------------------------
# 调用函数计算从 ROI区域'unvisited_region' 到visited_region区域的最短路径
# 计算无向图中的最近节点


def calculate_shortest_paths_length(data, start_nearest, end_nearest):
    start_nearest = str(start_nearest[0])
    end_nearest = str(end_nearest[0])
    # 使用A*算法实现无向图中的最短路径
    path_length = nx.astar_path_length(data, source=start_nearest, target=end_nearest, weight='weight')
    # path_length = nx.shortest_path_length(data, source=start_nearest, target=end_nearest, weight='weight')
    # 检查目标边是否存在于图中的边中
    # 检查两个节点之间是否存在直接边
    if data.has_edge(start_nearest, end_nearest):
        direct_edge_data = data.get_edge_data(start_nearest, end_nearest)
        haversine_weight = direct_edge_data['weight']
        # print(f"两个节点之间存在直接边，权重为: {haversine_weight}")
    else:
        if nx.has_path(data, start_nearest, end_nearest):
            # 获取两节点间的最短路径
            shortest_path = nx.shortest_path(data, source=start_nearest, target=end_nearest)
            # 计算间接边的权重，可以根据具体需求选择权重的计算方式
            haversine_weight = sum(
                data.get_edge_data(u, v).get('weight', 0) for u, v in zip(shortest_path, shortest_path[1:]))
            if haversine_weight == 'inf':
                haversine_weight = 0
            # print(f"两个节点之间存在间接边，最短路径权重为: {haversine_weight}")
        else:
            haversine_weight = 0

    return path_length, haversine_weight


# -----------------------------04计算地理偏好模型公式------------------------


# 解析节点数据中的位置信息字符串为字典格式
def parse_position(pos_str):
    # 判断输入变量是否为字典型
    if isinstance(pos_str, dict):
        # 将字典转化为字符串
        pos_str = "{" + f'"lon":{pos_str.get("lon", 0.0)},"lat":{pos_str.get("lat", 0.0)}' + "}"

    parts = pos_str.split(",")
    lon = float(parts[0].split(":")[1])
    lat = float(parts[1].split(":")[1].strip().rstrip("'}"))
    return lon, lat


def calculate_direction_weight_function( start_nearest, end_nearest, angle):
    # 定义原点和目标节点的经纬度坐标
    lat0, lon0 = math.radians(0), math.radians(0)  # 原点经纬度
    start_coords = start_nearest[0]
    end_coords = end_nearest[0]

    start_lon = start_coords[0] - lon0
    start_lat = start_coords[1] - lat0
    end_lon = end_coords[0] - lon0
    end_lat = end_coords[1] - lat0

    start_angle = math.degrees(math.atan2(start_lat, start_lon))
    end_angle = math.degrees(math.atan2(end_lat, end_lon))

    # angle = math.degrees(math.atan2(delta_lat, delta_lon))
    start_relative_direction = (start_angle + 360) % 360
    end_relative_direction = (end_angle + 360) % 360

    # 查看初始角和起始角的角度重合,得出弧度之比
    # 角度转弧度
    angle_radians = angle * (math.pi / 180)
    diff = end_relative_direction - start_relative_direction
    if diff >= -angle_radians and diff <= angle_radians:
        angle_ratio = start_relative_direction / end_relative_direction
        if angle_ratio > 1:
            angle_ratio = end_relative_direction / start_relative_direction
    else:
        angle_ratio = 0
    return angle_ratio


def calculate_geographical_preference_weight(start_nearest, end_nearest, alpha, beta, gama, shortest_paths,
                                             haversine_distance, check_in_freq):
    c = shortest_paths + haversine_distance
    w_distance = 1 / (1 + np.exp(-c))
    if w_distance == 1:
        w_distance = 0

    # 获取活跃程度权重
    w_activity = check_in_freq

    # 计算相对方向权重
    # 设置角度阈值
    angle = 45
    w_direction = calculate_direction_weight_function( start_nearest, end_nearest, angle)

    # 计算地理偏好权重,加权平均
    preference_weight = (alpha * w_distance + beta * w_activity + gama * w_direction) / (alpha + beta + gama)

    return preference_weight


def calculate_preference_category_weight(start_category, end_category):
    # 将start_roi,end_roi由list变为符合要求的string
    # 去重并转换回列表
    start_roi = list(set(start_category))
    end_roi = list(set(end_category))
    # 将所有值转换为字符串
    start_roi_str = ', '.join(start_roi) + ', '
    end_roi_str = ', '.join(end_roi) + ', '

    # 添加检查，确保输入字符串不为空
    if not start_roi_str.strip() or not end_roi_str.strip():
        text_similarity_Minhash = 0
        text_similarity_Jaccard = 0
    else:
        # 确保传入的字符串不是空字符串
        start_roi_str = start_roi_str.strip()
        end_roi_str = end_roi_str.strip()

        # Minhash
        text_similarity_Minhash = xs.minhash(start_roi_str, end_roi_str)

        # Jaccard
        # 确保传入的集合不为空
        if start_roi_str.strip() and end_roi_str.strip()and start_roi_str != '0' and end_roi_str != '0':
            text_similarity_Jaccard = xs.jaccard(start_roi_str.strip(), end_roi_str.strip())
        else:
            # 如果集合为空或为0，设置Jaccard相似度为0
            text_similarity_Jaccard = 0

    return text_similarity_Minhash, text_similarity_Jaccard


# 计算每个用户的地理相似度，最终返回final_df中
def calculate_geographical_similarity(user_id, geographic_similarity_data, G,alpha,
                                                                        beta, gama):
    user_start_time = time.localtime()
    print("开始构建%s用户咯,开始时间%s！" % (user_id, user_start_time))
    st = time.time()
    # 计算每个用户的地理相似度，最终返回final_df中
    unvisited_data = geographic_similarity_data[geographic_similarity_data['user_id'] != user_id]
    unvisited_roi = unvisited_data['roi'].unique()

    visited_data = geographic_similarity_data[geographic_similarity_data['user_id'] == user_id]
    visited_roi = visited_data['roi'].unique()

    temp_dfs = []
    final_df = pd.DataFrame()
    # 对每个 unvisited_roi 进行并行处理
    for unvisited_roi_num in unvisited_roi:
        result1 = process_unvisited_roi(user_id, unvisited_roi_num, unvisited_data, visited_data,
                                               visited_roi, G,alpha,beta, gama,temp_dfs)

        final_df = pd.concat(result1,ignore_index=True)

    user_end_time = time.time()
    print("已找到%s用户的地理相似性、类别相似性咯！" % user_id + "耗时: {:.2f}秒".format(
        user_end_time - st))
    # 写入Excel文件
    # excel_file_path = f'process_data\\geographical_similarity_data_preference_region_category_{user_id}.xlsx'
    excel_file_path = f'process_data\\shanghai_geographical\\geographical_similarity_data_preference_region_category_{user_id}.csv'
    final_df.to_csv(excel_file_path, index=False, header=True)
    return excel_file_path





# 并行处理 visited_roi

def process_unvisited_roi(user_id, unvisited_roi_num, unvisited_data, visited_data, visited_roi, G,alpha,
                                                                        beta, gama,temp_dfs):
    end_list = unvisited_data[unvisited_data['roi'] == unvisited_roi_num][['centre_lat',
                                                                           'centre_lon', 'roi', 'check_in_freq',
                                                                           'nearest_node', 'category', 'active_region']]


    end_all_value = [end_list.iloc[i].tolist() for i in range(len(end_list))]
    end_category = [row[5] for row in end_all_value]


    end_nearest_node_list = end_list['nearest_node'].apply(tuple)
    end_nearest_node_list = end_nearest_node_list.drop_duplicates().tolist()
    end_nearest_node = end_nearest_node_list[0]

    for visited_roi_num in visited_roi:
        result = process_visited_roi(visited_roi_num, visited_data, user_id,
                                     end_nearest_node, end_category, unvisited_roi_num, G,alpha,
                                                                        beta, gama)
        temp_dfs.append(result)

    # 返回整合后的 DataFrame
    return temp_dfs


def process_visited_roi(visited_roi_num, visited_data, user_id, end_neareat_value, end_category, end_roi, G,alpha,
                                                                        beta, gama):
    start_list = visited_data[visited_data['roi'] == visited_roi_num][
        ['centre_lat', 'centre_lon', 'roi', 'check_in_freq', 'nearest_node', 'category', 'active_region']]
    start_node_list = start_list.iloc[0].tolist()
    start_category = [start_node_list[5]]

    check_in_freq = start_node_list[3]

    shortest_paths, haversine_distance = calculate_shortest_paths_length(G, start_nearest=start_node_list[4],
                                                                         end_nearest=end_neareat_value)
    preference_region_weight = calculate_geographical_preference_weight(start_node_list[4], end_neareat_value, alpha,
                                                                        beta, gama, shortest_paths, haversine_distance,
                                                                        check_in_freq)

    preference_category_Minhashweight, preference_category_Jaccardweight = calculate_preference_category_weight(
        start_category, end_category)
    # 检查 preference_category_Jaccardweight 是否为 None，如果是，则赋值为 0
    if preference_category_Jaccardweight is None:
        preference_category_Jaccardweight = 0
    category_preference = (preference_category_Minhashweight+preference_category_Jaccardweight)/2




    result = {
        'user_id': user_id ,
        'visited_roi': visited_roi_num ,
        'unvisited_roi': end_roi,
        'shortest_path_value': shortest_paths,
        'check_in_freq': check_in_freq,
        'preference_region_weight': preference_region_weight,
        'category_preference': category_preference
    }

    # 将新的行数据构建成一个 DataFrame
    result_df = pd.DataFrame([result])

    return result_df




def process_user_geographic_in_parallel(unique_users, geographic_similarity_data, G,alpha,
                                                                    beta, gama, worker_num):
   # #中途出现错误，可以使用这段代码
   #  # folder_path = 'process_data/geographical/'
   #  folder_path = 'process_data/shanghai_geographical/'
   #  csv_files = glob.glob(folder_path + '/*.csv')
   #  # 检查是否找到了任何匹配的文件
   #  if not csv_files:
   #      print("未找到任何匹配的csv文件。")
   #  else:
   #      numbers = []
   #      for file_name in csv_files:
   #          matches = re.findall(r'\d+', file_name)
   #          numbers.extend(map(int, matches))
   #  # 构建要传递给 starmap 函数的元组列表
   #  task_list = []
   #  for user_id in unique_users:
   #      if user_id in numbers:
   #          continue  # 如果 user_id 存在于 numbers 中，则跳过当前循环
   #      task_list.append(user_id)
   #
   #  with multiprocessing.Pool(worker_num) as pool:
   #      excel_files = pool.starmap(
   #          calculate_geographical_similarity,
   #          [(user_id, geographic_similarity_data, G,alpha, beta, gama) for user_id in task_list],
   #      )
   with multiprocessing.Pool(worker_num) as pool:
    excel_files = pool.starmap(
       calculate_geographical_similarity,
       [(user_id, geographic_similarity_data, G,alpha, beta, gama) for user_id in unique_users])

   return excel_files

if __name__ == '__main__':
    print('---------------开始生成地理相似性咯--------------------------------！')
    # poi_data = "process_data/foursquare_roi_poi-user_cluster.xlsx"
    poi_data = "process_data/shanghai_roi_poi-user_cluster.xlsx"
    # poi_data = "F://dataset//data//temp//test.xlsx"

    # 存储道路无向图的路径
    # shp_path = "F://dataset//data//foursquare_road.shp"
    shp_path = "F://dataset//data//shanghai_road2.shp"
    print('现在构建无向图啦')
    start_time = time.time()
    G, node_coordinates = road_graph(shp_path)
    end_time = time.time()
    print("无向图耗时: {:.2f}秒".format(end_time - start_time))

    # 挑出每个用户的详细数据
    data = pd.read_excel(poi_data)
    unique_users = data['user_id'].unique()

    # 存储用户的roi，roi的中心经纬度
    geographic_similarity_data = pd.DataFrame()

    # 构建需要计算地理相似度的数据
    print('现在开始构建数据啦！')
    # start_time = time.time()
    foursquare_roi_finaldata,geographic_similarity_data = build_roi_data(geographic_similarity_data, node_coordinates)
    # end_time = time.time()
    # print("构建数据耗时: {:.2f}秒".format(end_time - start_time))



    # 设置参数
    # 假设 alpha 、 beta和gama 是权衡地理距离和地理方向的系数
    alpha = 0.8
    beta = 0.1
    gama = 0.1

    # 使用multiprocessing进行并发
    excel_files  = process_user_geographic_in_parallel(unique_users, geographic_similarity_data, G,alpha,
                                                                        beta, gama, 15)
    # # 合并所有Excel文件
    # combined_df = pd.concat([pd.read_csv(file) for file in excel_files], ignore_index=True)
    # combined_df.to_csv('process_data/geographical_similarity_data_combined.csv', index=False)
    print('--------------------全部用户的空间区域相似性计算完毕！----------------------------------')
    merge_mat2s_martix()
