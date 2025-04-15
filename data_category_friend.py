from scipy.sparse import csr_matrix
from data_geofeature_categroy import *
def find_similar_friend():
    similar_users = []

    for user_id in unique_users:
        # 创建有向图
        friend_G = nx.DiGraph()

        # 根据用户ID过滤数据
        user_data = data[data['user_id'] == user_id]

        # 取出当前用户的 roi 值
        user_roi = user_data['roi'].values.tolist()

        # 添加边到图中
        for roi in user_roi:
            friend_G.add_edge(user_id, roi)

        # 计算PageRank
        pagerank_values = nx.pagerank(friend_G)

        # 排序PageRank值，找到最相似的用户
        sorted_users = sorted(pagerank_values, key=pagerank_values.get, reverse=True)

        # 选择相似用户，包括用户本身
        similar_users_for_user = []
        for i, similar_user in enumerate(sorted_users):
            if i == 0:  # 跳过用户本身
                continue
            similar_users_for_user.append(similar_user)

        # 将最相似用户添加到similar_users列表中
        similar_users.append((user_id, similar_users_for_user))

    # 将similar_users转换为DataFrame
    similar_users_roi_df = pd.DataFrame(similar_users, columns=['user', 'similar_users_roi'])
    return similar_users_roi_df


def load_rating_train_as_matrix(data):
    print('现在开始生成用户-兴趣区域交互矩阵啦！')
    #必须生成符合交互矩阵的数据格式
    ##计算每个用户对每个ROI区域的签到次数
    check_ins_count = data.groupby(['user_id', 'roi']).size().reset_index(name='check_ins_count')
    interactions_data = check_ins_count.values.tolist()
    # 将列表转换为DataFrame
    interactions_df = pd.DataFrame(interactions_data, columns=['user_id', 'roi','count'])

    # 组成新的数据集，仅包含user_id、roi
    train_data = interactions_df[['user_id', 'roi','count']].copy()
    # 获取用户和兴趣区域的数量
    user_num = train_data['user_id'].max() + 1
    item_num = train_data['roi'].max() + 1
    # 转换成列表形式
    train_data = train_data.values.tolist()
    # 初始化稀疏矩阵
    user_roi_train_mat = csr_matrix((user_num, item_num), dtype=np.int32)
    # 遍历数据，将交互设置为1
    for x in train_data:
        user_roi_train_mat[x[0], x[1]] = x[2]
    user_roi_train_mat_dense = user_roi_train_mat.toarray()
    user_roi_train_df = pd.DataFrame(user_roi_train_mat_dense)
    print("用户-兴趣区域交互矩阵生成完成！")
    return user_roi_train_df
if __name__ == '__main__':
        print("开始寻找社交关系啦！")
        # poi_data = 'process_data/foursquare_roi_poi-user_cluster.xlsx'
        poi_data = 'process_data/shanghai_roi_poi-user_cluster.xlsx'
        # 挑出每个用户的详细数据
        data = pd.read_excel(poi_data)
        unique_users = data['user_id'].unique()
        similar_users_roi_df = find_similar_friend()
        # similar_users_roi_path = 'process_data\\similarity_data_users_roi.xlsx'
        similar_users_roi_path = 'process_data\\shanghai_similarity_data_users_roi.xlsx'
        # 将DataFrame导出到Excel文件
        similar_users_roi_df.to_excel(similar_users_roi_path, index=False)
        print(f"社交关系数据已成功导出到Excel文件: {similar_users_roi_path}")

        user_roi_train_df = load_rating_train_as_matrix(data)
        # user_roi_train_df_path = 'process_data\\user_roi_train_mat.csv'
        user_roi_train_df_path = 'process_data\\shanghai_user_roi_train_mat.csv'
        # 将DataFrame保存为CSV文件
        user_roi_train_df.to_csv(user_roi_train_df_path, index=False)
        print("所有特征值构建完毕！")

