import pandas as pd
import numpy as np
max_len=200
# 构建社交信息
def expand_rows(u_id):
    # 筛选出指定user_id的行
    user_data = users_roi_data[users_roi_data['user'] == u_id]
    if not user_data.empty:
        similar_users_roi = user_data['similar_users_roi'].iloc[0]
        # 去除字符串两端的方括号，并根据逗号分割成列表
        roi_list = similar_users_roi.strip('[]').split(',')
        roi_array = np.zeros(max_len, dtype=int)
        for i, roi in enumerate(roi_list):
            if i < max_len:
                roi_array[i] = int(roi.strip())
        return roi_array

# 构建类别信息
def expand_category(u_id):
    # 筛选出指定user_id的行
    # 筛选出 user_id 为 268 并且 category_preference 大于 0.5 的所有行
    filtered_data = user_category_data[
        (user_category_data['user_id'] == u_id)].sort_values(by='category_preference', ascending=False)
    # 对unvisited_roi列进行去重，并保留排序后的第一项
    filtered_data = filtered_data.drop_duplicates(subset='unvisited_roi')
    # 最终获得user_id和去重后的unvisited_roi
    # 判断unique_unvisited_roi长度是否大于max_len，如果是，则截取前max_len个
    if len(filtered_data) > max_len:
        unique_unvisited_roi = filtered_data[['unvisited_roi']].head(max_len)
    else:
        padding = max_len - len(filtered_data)
        unique_unvisited_roi = np.pad(filtered_data[['unvisited_roi']], (0, padding), mode='constant')
    return unique_unvisited_roi




# users_roi_data_path = 'process_data/similarity_data_users_roi.xlsx'
users_roi_data_path = 'process_data/shanghai_similarity_data_users_roi.xlsx'
users_roi_data = pd.read_excel(users_roi_data_path)

# user_roi_category_path = 'process_data/user_category_preference.csv'
user_roi_category_path = 'process_data/shanghai_user_category_preference.csv'
user_category_data =pd.read_csv(user_roi_category_path)







