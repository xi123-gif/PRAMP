from data_geofeature_categroy import *
data = pd.read_csv('process_data/geographical_similarity_data_combined.csv')
# 按照指定列取出需要的数据
selected_data = data[['user_id', 'visited_roi', 'unvisited_roi',
                      'preference_category_Minhashweight', 'preference_category_Jaccardweight']]
# # 计算category_preference的平均值
# selected_data.loc[:, 'category_preference'] = (selected_data['preference_category_Minhashweight'] +
#                                                 selected_data['preference_category_Jaccardweight']) / 2
# 计算 'preference_category_Minhashweight' 和 'preference_category_Jaccardweight' 的平均值
selected_data['category_preference'] = selected_data[['preference_category_Minhashweight',
                                                      'preference_category_Jaccardweight']].mean(axis=1)
# 选择所需的列
result_df = selected_data[['user_id', 'visited_roi', 'unvisited_roi', 'category_preference']]
# 打印结果
print(result_df)
# 仅保留需要的列，并保存到新的csv文件
selected_data.to_csv('process_data/user_category_preference.csv', index=False)