import os

os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import time
import csv
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix


def kmeans_plus_process(filepath):
    data = pd.read_excel(filepath)
    k = 5
    # 开始记录时间
    stat_time = time.time()
    # 设置每个用户的标签起始值
    label_start = 0
    best_label = 0
    # 存储用户返回的特征值及标签
    best_labels = []
    scaled_datas = []
    # 存储用户聚类结果
    user_cluster = pd.DataFrame({})
    # 针对每个用户进行独立的聚类,使用多次运行 K-means++ 算法，并选择其中惯性最小的聚类结果------------------
    unique_users = data['user_id'].unique()
    for user_id in unique_users:
        # 从原始 data 中提取给定 user_id 的签到数据，仅包括 'latitude' 和 'longitude' 两列。
        user_data = data[data['user_id'] == user_id][['LAT', 'LON']]

        if len(user_data) <= 50:
            continue  # 数量不足 100，跳过当前 user_id，继续下一个
        # user_data = user_data.drop_duplicates()
        user_data_info = data[data['user_id'] == user_id]
        # 对用户数据进行标准化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(user_data)
        # #检查唯一点的数量
        # unique_points = np.unique(scaled_data,axis=0)
        # num_unique_points = len(unique_points)
        # if num_unique_points < k:
        #     k = num_unique_points
        # else:
        #     k = k
        # 初始化K-means模型
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        # ------------------多次运行k-means算法------------------------
        best_inertia = np.inf
        num_runs = 10
        for _ in range(num_runs):
            # 随机初始化聚类中心
            kmeans.init = 'random'
            # 训练模型
            kmeans.fit(scaled_data)
            # 获取聚类结果和惯性（inertia）
            labels = kmeans.labels_
            inertia = kmeans.inertia_
            # 选择惯性最小的聚类结果
            if inertia < best_inertia:
                best_label = labels
                best_inertia = inertia
        cluster_data = user_data_info.copy()
        cluster_data['cluster'] = best_label + label_start
        # print('现在根据街区编号进行cluster编号约束')
        cluster_data = roi_constraint_cluster(user_id, cluster_data)
        # print('现在计算该用户的活跃区域！')
        cluster_data = calculate_active_regions(user_id, cluster_data)
        user_cluster = pd.concat([user_cluster, cluster_data], ignore_index=True)
        label_start += k
        # print("用户%s的Length of best_label:"%(user_id), len(best_label))
        # print("用户%s的Length of label_start:"%(user_id), label_start)
        # print("用户%s的Length of cluster_data:"%(user_id), len(cluster_data))
        # scaled_datas.append(scaled_data)
        # best_labels.append(best_label+label_start)
        scaled_datas.extend(scaled_data)
        best_labels.extend(best_label + label_start)

        # print('现在根据街区编号进行cluster编号约束')
        # user_cluster = roi_constraint_cluster(user_id, user_cluster)
        # print('现在计算该用户的活跃区域！')
        # user_cluster = calculate_active_regions(user_id, user_cluster)
    # 记录结束时间
    end_time = time.time()
    # 计算执行时间
    execution_time = end_time - stat_time
    print("执行时间：", execution_time, "秒")
    return user_cluster, scaled_datas, best_labels


# 对聚类中的cluster列进行约束，确保每个用户中相同roi编号对应相同的cluster
def roi_constraint_cluster(user_id, cluster_data):
    # 检查每个用户中是否有相同的 roi 对应不同的 cluster
    user_roi_clusters = cluster_data.groupby(['user_id', 'roi'])['cluster'].nunique()
    # 获取需要更新的行索引
    rows_to_update = user_roi_clusters[user_roi_clusters > 1].reset_index().set_index(['user_id', 'roi']).index
    # 将需要更新的行的 cluster 列值设置为相同的值
    for user_id, roi in rows_to_update:
        unique_cluster = \
            cluster_data[(cluster_data['user_id'] == user_id) & (cluster_data['roi'] == roi)]['cluster'].iloc[0]
        cluster_data.loc[
            (cluster_data['user_id'] == user_id) & (cluster_data['roi'] == roi), 'cluster'] = unique_cluster
    # 打印更新后的数据框
    # print(cluster_data)
    return cluster_data


def calculate_active_regions(user_id, cluster_data):
    cluster_data['active_region'] = None  # 初始化为 None，可以根据需要修改初始值
    cluster_data['check_in_freq'] = None  # 初始化为 None，可以根据需要修改初始值
    # 计算每个用户的所有roi签到数量
    user_count = len(cluster_data)
    # 提取 'roi' 列并计算不同数字的数量
    unique_rois = cluster_data['roi'].unique()
    roi_datas = {'roi': [], 'num': []}
    for _ in unique_rois:
        # 统计每个 roi 编号中的签到数量
        lable_count = cluster_data[cluster_data['roi'] == _]
        lable_count = len(lable_count)
        num = lable_count / user_count
        roi_datas['roi'].append(_)
        roi_datas['num'].append(num)
        #num赋值到check_in_freq作为freq
        # 根据条件将对应行的 check_in_freq 置为 num
        cluster_data.loc[cluster_data['roi'] == _, 'check_in_freq'] = num
    roi_datas_df = pd.DataFrame(roi_datas)
    # 找到 num 列中大于等于 0.5 的索引值
    activate_index = roi_datas_df[roi_datas_df['num'] >= 0.5].index
    # cluster_data.loc[activate_index, 'active_region'] = activate_roi
    # 将activate_roi列表中的值全部赋给active_region列
    if len(activate_index) > 0:
        # 返回对应的 roi 值，以列表形式返回
        activate_roi = roi_datas_df.loc[activate_index, 'roi'].tolist()
        # 在 cluster_data 中新增一列 'active_region'，并将对应的值填入
        # 将activate_roi列表中的值全部赋给active_region列
        cluster_data['active_region'] = activate_roi * (len(cluster_data) // len(activate_roi)) + activate_roi[:len(
            cluster_data) % len(activate_roi)]
    else:
        # 取出num最高的roi区域，当作该用户的活跃区域
        # 找到num列中的最大值的索引
        num_list = roi_datas['num']
        max_num = max(num_list)
        max_index = roi_datas['num'].index(max_num)
        # 返回对应的roi值
        max_roi = roi_datas['roi'][max_index]
        # 在cluster_data添加active_region，并将该号码填入此区域
        cluster_data['active_region'] = max_roi
        # print("Error: activate_roi列表为空")
    # print('%s用户的活跃区域计算完毕' % (user_id))
    return cluster_data


def evaluate_cluster(scaled_datas, best_labels):
    # ------------------使用 Calinski-Harabasz指标对聚类进行评价,类别之间的分离度-------------
    X = scaled_datas
    y_pred = best_labels
    # print("X的类型为：%s" % type(X))
    # print("y_pred的类型为：%s" % type(y_pred))
    calinski_index = metrics.calinski_harabasz_score(X, y_pred)
    print('kmeans的Calinski-Harabasz Index评估的聚类分数为：%s' % calinski_index)
    # -------------------使用Silhouette Score指标对聚类进行评价，类别之间的紧密度-----------------
    silhouette_index = silhouette_score(X, y_pred)
    print('kmeans的silhouette评估的聚类分数为：%s' % silhouette_index)
    return calinski_index, silhouette_index



if __name__ == '__main__':
    print("开始聚类咯！")
    # poi_data = "process_data/foursquare_roi_poi-user.xlsx"
    poi_data = "process_data/shanghai_roi_poi-user.xlsx"
    # poi_data = "F://dataset//data//foursquare_roi_poi-user.xlsx"
    user_cluster, scaled_datas, best_labels = kmeans_plus_process(poi_data)
    evaluate_cluster(scaled_datas, best_labels)
    # 将user_cluster导出到Excel文件中
    # user_cluster.to_excel('F://dataset//data//foursquare_roi_poi-user_cluster.xlsx', index=False)
    # user_cluster.to_excel('process_data/foursquare_roi_poi-user_cluster.xlsx', index=False)
    user_cluster.to_excel('process_data/shanghai_roi_poi-user_cluster.xlsx', index=False)
    print('foursquare_roi_poi-user_cluster.xlsx已经导出！')
# #----------------测试不同的k值，得到的聚类结果及评价指标--------------------
#     k_range = range(1,11)
#     result = pd.DataFrame(columns=['k','calinski_harabasz_score','silhouette_score','user_cluster'])
#     for k in k_range:
#         user_cluster, scaled_datas, best_labels = kmeans_plus_process_backup(poi_data,k)
#         calinski_index, silhouette_index = evaluate_cluster_backup(scaled_datas, best_labels)
#         # 使用 loc 方法将新的行添加到 DataFrame 中
#         result.loc[len(result)] = [k, calinski_index, silhouette_index, user_cluster]
#         print('%s聚类已完成！'%k)
#     # 存储结果到文件
#     result.to_excel('F://code//paper1//clustering_results.xlsx', index=False)
#     print('已实现k=1-10的聚类结果！快去看看哪个k值的聚类效果更好~')
# visualization_cluster(best_label, centroids)

    # ------------生成用户轨迹.NPY数据----------------

