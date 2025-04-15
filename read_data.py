# 数据清洗
import numpy as np
import pandas as pd
import os
from datetime import datetime


# -----------------------------------------------数据处理-------------------------------------
# open excel
def open_excle():
    if os.path.exists(filepath):
        data_file = pd.read_excel(filepath)
        print("excle数据已成功读入！")
        return data_file
    else:
        print("数据路径不存在，请检查文件！")


# data process
def data_process(data_file):
    # data drop duplicate
    subset = ["user_id", "category", "location_id", "LAT", "LON", "TIME","roi"]
    data_file = data_file.drop_duplicates(subset=subset)
    # data_file.to_excel("F:\\dataset\\data\\%s_dup.xlsx" % (filename), index=False)
    # print("%s.xlsx去重完成" % (filename))
    # data delete NA row
    data_file = data_file.dropna()
    # data_file.to_excel("F:\\dataset\\data\\%s_dropna.xlsx" % (filename), index=False)
    # print("%s.xlsx去除空余值完成" % (filename))

    # # 重新组织列表中的时间数据
    def time_process(date_string):
        # 去掉+0000部分
        data_string_for = date_string[:-10]
        data_string_bac = date_string[-4:]
        date_string_all = data_string_for + " " + data_string_bac
        # 定义日期字符串的格式
        date_format = "%a %b %d %H:%M:%S  %Y"
        # 使用datetime模块解析日期字符串
        date_object = datetime.strptime(date_string_all, date_format)
        # 以2012一月一日0点为起始时间
        start_time = datetime(date_object.year, month=1, day=1, hour=0, minute=0, second=0)
        # 计算时间间隔
        time_delta_d = date_object - start_time
        time_delta = time_delta_d * 24 * 60
        return time_delta

    data_file['TIME'] = data_file['TIME'].apply(time_process)


    # # 删除用户和poi签到数量小于5----生成内嵌函数
    def dele_user_poi(data_file):
        # 删除用户和roi签到数量小于5/20
        poi_counts = data_file['location_id'].value_counts()
        user_counts = data_file['user_id'].value_counts()
        # 筛选出符合条件的POI和用户
        valid_pois = poi_counts[poi_counts >= 100].index
        valid_users = user_counts[user_counts >= 100].index
        # 根据条件删除
        data_file = data_file[data_file['location_id'].isin(valid_pois)]
        data_file = data_file[data_file['user_id'].isin(valid_users)]
        data_file_path = "process_data/%s_poi-user.xlsx" % (filename)
        data_file.to_excel(data_file_path,index=False)

        return data_file

    data_file = dele_user_poi(data_file)
    print("%s.xlsx数据清洗已完成" % (filename))
    return data_file

# ------------主函数---------
if __name__ == '__main__':


    filename = "shanghai_roi"
    filepath = "F:\\dataset\\data\\%s.xlsx" % (filename)

    # ---------------------------------------------------------
    data_file = open_excle()
    data_process(data_file)

