import gc


from input_friend_categroy import *
import torch
from math import radians, cos, sin, asin, sqrt
import joblib
import random
from torch.nn.utils.rnn import pad_sequence
max_len = 200  # max traj len; i.e., M

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

def euclidean(point, each):
    lon1, lat1, lon2, lat2 = point[2], point[1], each[2], each[1]
    return np.sqrt((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2)

# 用于计算用户轨迹的所有位置点之间时间、空间关系
# 其中poi为所有位置信息
def rst_mat1(traj, roi):
    # traj (*M, [u, l, t]), poi(L, [l, lat, lon])
    # 用于存储距离-时间矩阵
    mat = np.zeros((len(traj), len(traj), 2))
    for i, item in enumerate(traj):

        for j, term in enumerate(traj):
            # 位置的loc_id从roi数据中检索出对应的POI信息
            #获取对应的roi索引值
            c = item[1]
            d = term[1]
            roi_item, roi_term = [row for row in roi if int(row[0]) == int(item[1])], \
                                [row for row in roi if int(row[0]) == int(term[1])]# retrieve poi by loc_id
            # 使用haversine函数计算位置i和位置j之间的距离，并将结果存储在距离-时间矩阵的第一个维度上。
            lon1 = roi_item[0][2]
            lat1 = roi_item[0][1]
            lon2 = roi_term[0][2]
            lat2 = roi_term[0][1]
            mat[i, j, 0] = haversine(lon1=roi_item[0][2], lat1=roi_item[0][1], lon2=roi_term[0][2], lat2=roi_term[0][1])

            # 这行代码计算位置i和位置j之间的时间差，并将结果存储在距离-时间矩阵的第二个维度上。
            time1=item[2]
            time2=term[2]
            mat[i, j, 1] = abs(item[2] - term[2])
    # # 将距离归一化，提取距离矩阵的第一列
    # distance_column = mat[:, :, 0]
    # if len(distance_column)!=0:
    #     # 计算第一列的最小值和最大值
    #     min_val = np.min(distance_column)
    #     max_val = np.max(distance_column)
    #     # 对第一列进行归一化
    #     normalized_distance_column = (distance_column - min_val) / (max_val - min_val)
    #     # 将归一化后的第一列替换回原始距离矩阵
    #     mat[:, :, 0] = normalized_distance_column


    return mat  # (*M, *M, [dis, tim])


# 需要把用户i未访问点的筛选出来进行比较，需要加入user-id
def rs_mat2s(geographical_df,l_max):

    l_max=int(l_max)
    data = geographical_df

    # 提取行和列名
    row_list = data.columns.tolist()
    rows = row_list[1:]

    #設置新行名
    data.index = rows
    cols_index = row_list[0]

    # 直接在原 DataFrame 中删除列 'B'
    data.drop(columns=[cols_index], inplace=True)

    return data  # (L, L)


def rt_mat2t(traj_time,u_id):  # traj_time (*M+1) triangle matrix

    mat = np.zeros((len(traj_time)-1, len(traj_time)-1))
    for i, item in enumerate(traj_time):  # label
        if i == 0:
            continue
        for j, term in enumerate(traj_time[:i]):  # data
            mat[i - 1, j] = np.abs(item - term)
    return mat  # (*M, *M)





def process_traj(dname):  # start from 1
    # userTrac (?, [u, l, t]), GPS (L, [l, lat, lon])
    userTrac = np.load('data/' + dname + '_userTrac.npy')
    GPS = np.load('data/' + dname + '_GPSTrac.npy')

    num_user = list(set(userTrac[:, 0]))
    # 使用列表推导式和int()函数将列表中的所有值转换为整数
    num_user_int = [int(num) for num in num_user]# max id of users, i.e. NUM
    min_num_user = np.min(userTrac[:, 0])
    data_user = userTrac[:, 0]  # user_id sequence in userTrac
    trajs, labels, mat1, mat2t, lens,mat3f,mat4cat  =  [], [], [], [],[],[],[]

    u_max, l_max = np.max(userTrac[:, 0]), np.max(userTrac[:, 1])

    for u_id in num_user_int:
        if u_id < min_num_user:  # skip u_id == 0
            continue

        init_mat1 = np.zeros((max_len, max_len, 2))  # first mat (M, M, 2)
        init_mat2t = np.zeros((max_len, max_len))  # second mat of time (M, M)
        user_traj = userTrac[np.where(data_user == u_id)]  # find all check-ins of u_id
        user_traj = user_traj[np.argsort(user_traj[:, 2])].copy()  # sort traj by time
        print(u_id, len(user_traj)) if u_id % 100 == 0 else None

        if len(user_traj) > max_len + 1:  # consider only the M+1 recent check-ins
            # 0:-3 are training userTrac, 1:-2 is training label;
            # 1:-2 are validation userTrac, 2:-1 is validation label;
            # 2:-1 are test userTrac, 3: is the label for test.
            # *M would be the real length if <= max_len + 1
            user_traj = user_traj[-max_len - 1:]  # (*M+1, [u, l, t])

        # spatial and temporal intervals
        user_len = len(user_traj[:-1])  # the len of userTrac, i.e. *M
        # 构建轨迹时空矩阵
        user_mat1 = rst_mat1(user_traj[:-1], GPS)  # (*M, *M, [dis, tim])
        # 构建候选时间矩阵(user_traj[:, 2]第三列的所有行，即时间)
        user_mat2t = rt_mat2t(user_traj[:, 2],u_id)  # (*M, *M)
        # #构建轨迹空间矩阵
        # user_candidate_mat2s = rs_mat2s(u_id, l_max)
        #构建社交关系矩阵
        user_mat3f = expand_rows(u_id)

        #构建类别偏好矩阵
        user_mat4cat = expand_category(u_id)
        # user_mat1的数据复制到init_mat1的左上角的子矩阵中，以实现数据的复制和替换操作。
        init_mat1[0:user_len, 0:user_len] = user_mat1
        init_mat2t[0:user_len, 0:user_len] = user_mat2t

        # 用户轨迹数据转换为PyTorch张量，并且去掉最后一个元素后添加到trajs列表中。
        trajs.append(torch.LongTensor(user_traj)[:-1])  # (NUM, *M, [u, l, t])
        mat1.append(init_mat1)  # (NUM, M, M, 2)
        mat2t.append(init_mat2t)  # (NUM, M, M)

        # del user_candidate_mat2s,user_mat1,user_mat2t
        del user_mat1,user_mat2t
        gc.collect()
        # 用户轨迹数据中的location部分提取出来并转换为PyTorch，添加到labels列表
        labels.append(torch.LongTensor(user_traj[1:, 1]))  # (NUM, *M)
        mat3f.append(torch.LongTensor(user_mat3f))
        mat4cat.append(torch.LongTensor(user_mat4cat.iloc[:, 0].values))
        lens.append(user_len - 2)  # (NUM), the real *M for every user


    # geographical_df= pd.read_csv('process_data/geographical_similarity_data_matrix.csv')
    geographical_df= pd.read_csv('process_data/shanghai_geographical_similarity_data_matrix.csv')
    mat2s = rs_mat2s(geographical_df, l_max)  # contains dis of all locations, (L, L)
    print("用户的候选时空、轨迹时空矩、社交、类别矩阵构建完毕！")
    zipped = zip(*sorted(zip(trajs, mat1, mat2t, labels, lens,mat3f,mat4cat), key=lambda x: len(x[0]), reverse=True))
    print("输入数据转换已完成~")
    # 解压ziiped，将原始列表按照特定规则排序后的内容解压缩，并将解压缩后的内容转换为列表类型
    trajs, mat1, mat2t, labels, lens,mat3f,mat4cat = zipped
    trajs, mat1, mat2t, labels, lens,mat3f ,mat4cat = list(trajs), list(mat1), list(mat2t), list(labels), list(lens),list(mat3f),list(mat4cat) # # padding zero to the vacancies in the right，trajs,labels 的右下角补全0
    trajs = pad_sequence(trajs, batch_first=True, padding_value=0)  # (NUM, M, 3)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)  # (NUM, M)

    # 创建了一个名为userTrac的列表，其中包含用户历史轨迹，轨迹时空矩阵、候选轨迹时间矩阵，候选轨迹空间矩阵，标签，长度，用户最大值，地点编号最大值
    userTrac = [trajs, np.array(mat1),mat2s, np.array(mat2t), labels, np.array(lens), int(u_max), int(l_max),np.array(mat3f),np.array(mat4cat)]
    data_pkl = 'data\\' + dname + '_data.pkl'
    # data_pkl = 'data\\' + dname + '_testdata.pkl'
    open(data_pkl, 'a')
    with open(data_pkl, 'wb') as pkl:
        joblib.dump(userTrac, pkl)
    print("已成功保存输入数据！")


if __name__ == '__main__':
    # name = 'foursquare_roi'
    name = 'shanghai_roi'
    process_traj(name)
