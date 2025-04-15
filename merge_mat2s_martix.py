import gc
import os

import pandas as pd

def merge_mat2s_martix():
    # 读取CSV文件
    # initial_df = pd.read_csv('process_data/foursquare_roi_finaldata.csv')
    initial_df = pd.read_csv('process_data/shanghai_roi_finaldata.csv')
    # 获取某一列的值，去除重复值
    column_name = 'roi'
    unique_values = initial_df[column_name].unique()
    # 创建矩阵，并将去重后的值作为行和列名称
    matrix = pd.DataFrame(index=unique_values, columns=unique_values, dtype='float64')
    # 将矩阵初始化为0（或其他默认值）
    matrix.fillna(0.0, inplace=True)
    print("空矩陣創建完畢！")
    # 读取包含多个CSV文件的文件夹
    # folder_path = 'process_data/geographical'
    folder_path = 'process_data/shanghai_geographical'
    # 读取文件夹中的每个CSV文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            # 使用chunksize参数分块读取CSV文件
            chunk_size = 10000  # 根据你的内存情况调整块大小
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                # 提取指定的两列和值列
                col1 = 'visited_roi'
                col2 = 'unvisited_roi'
                value_col = 'preference_region_weight'

                if col1 in chunk.columns and col2 in chunk.columns and value_col in chunk.columns:
                    for index, row in chunk.iterrows():
                        val1 = row[col1]
                        val2 = row[col2]
                        value = row[value_col]

                        # 检查值是否在矩阵的行和列中
                        if val1 in matrix.index and val2 in matrix.columns:
                            # 如果匹配，更新矩阵中的值
                            matrix.at[val1, val2] = value
            # 显式删除块对象并调用垃圾回收
            del chunk
            gc.collect()
        print("%s讀取完畢" % file_name)
    # 遍历矩阵，若行、列值都一样，则将对应矩阵的值设为1
    for index in matrix.index:
        for col in matrix.columns:
            if index == col:
                matrix.at[index, col] = 1.0
    # output_file_path = 'process_data/geographical_similarity_data_matrix.csv'
    output_file_path = 'process_data/shanghai_geographical_similarity_data_matrix.csv'
    matrix.to_csv(output_file_path)
    print("時空外部矩陣創建完畢！！！")
    return matrix


