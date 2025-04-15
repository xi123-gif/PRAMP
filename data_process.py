from read_data import *


# .xlsx文件保存为.npy-----生成用户活动轨迹和GPS活动轨迹--------------
def save_NPY_data(usercol, npyname_traU):
    # filename_traU = "foursquare_roi_finaldata"
    filename_traU = "shanghai_roi_finaldata"
    filepath_traU = "process_data/%s.csv" % (filename_traU)

    # if (npyname_traU == "foursquare_roi_userTrac.npy"):
    if (npyname_traU == "shanghai_roi_userTrac.npy"):
        # 保存用户活动轨迹npy文件
        data_file_traU = pd.read_csv(filepath_traU, usecols=usercol)
        npypath = "data/%s" % (npyname_traU)
        np.save(npypath, data_file_traU)
        print("%s数据格式转换成功！" % npyname_traU)
    else:
        # 保存GPS活动轨迹 npy文件
        data_file_traU = pd.read_csv(filepath_traU, usecols=usercol)
        data_file_traU_dup = data_file_traU.drop_duplicates(subset='roi')
        npypath = "data/%s" % (npyname_traU)
        np.save(npypath, data_file_traU_dup)
        print("%s数据格式转换成功！" % npyname_traU)

def build_categroycsv(data):
    # 按照指定列取出需要的数据
    # selected_data = data[['user_id', 'visited_roi', 'unvisited_roi',
    #                       'preference_category_Minhashweight', 'preference_category_Jaccardweight']]

    # # 计算 'preference_category_Minhashweight' 和 'preference_category_Jaccardweight' 的平均值
    # selected_data['category_preference'] = selected_data[['preference_category_Minhashweight',
    #                                                       'preference_category_Jaccardweight']].mean(axis=1)
    # 选择所需的列
    result_df = data[['user_id', 'visited_roi', 'unvisited_roi', 'category_preference']]
    # 仅保留需要的列，并保存到新的csv文件
    # result_df.to_csv('process_data/user_category_preference.csv', index=False)
    result_df.to_csv('process_data/shanghai_user_category_preference.csv', index=False)
    print("shanghai_user_category_preference.csv'成功生成！")

def build_geographical_similarity_h5(geographical_data_path):
    csv_files = [os.path.join(geographical_data_path, file) for file in os.listdir(geographical_data_path) if
                 file.endswith('.csv')]
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    # output_file = 'process_data/geographical_similarity_data_combined.h5'
    output_file = 'process_data/shanghai_geographical_similarity_data_combined.h5'
    combined_df.to_hdf(output_file, key='data', mode='w')
    # print('geographical_similarity_data_combined成功生成！')
    print('shanghai_geographical_similarity_data_combined成功生成！')

usercol = ['user_id', 'roi', 'TIME']
# npyname_traU = "foursquare_roi_userTrac.npy"
npyname_traU = "shanghai_roi_userTrac.npy"
save_NPY_data(usercol, npyname_traU)
# -----------生成GPS.NPY数据--------------------
gpscol = ['roi', 'centre_lat', 'centre_lon']
# npyname_gps = "foursquare_roi_GPSTrac.npy"
npyname_gps = "shanghai_roi_GPSTrac.npy"
save_NPY_data(gpscol, npyname_gps)

#生成總的距離文件
# geographical_data_path = 'process_data/geographical'
geographical_data_path = 'process_data/shanghai_geographical'
build_geographical_similarity_h5(geographical_data_path)

#生成類別csv文件
# data = pd.read_hdf('process_data/geographical_similarity_data_combined.h5')
data = pd.read_hdf('process_data/shanghai_geographical_similarity_data_combined.h5')
build_categroycsv(data)