import torch
from torch.utils.tensorboard import SummaryWriter
from input_model import max_len
import time
import random
from torch import optim
import torch.utils.data as data
from tqdm import tqdm
from models import *
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
import os
torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark=True

#返回top-1，top5，top10，top20的准确率，每次迭代中累加并求平均准确率
def calculate_acc(prob, label):
    # log_prob (N, L), label (N), batch_size [*M]
    # acc_train ={'top1': 0, 'top5': 0, 'top10': 0, 'top20': 0}
    acc_train = [0, 0, 0, 0]
    recall_train = [0, 0, 0, 0]
    f1_train = [0, 0, 0, 0]
    ndcg_train = [0, 0, 0, 0]
    total_relevant = len(label)
    for i, k in enumerate([1, 5, 10, 20]):
        _, topk_predict_batch = torch.topk(prob, k=k)
        topk_predict_batch = to_npy(topk_predict_batch)
        correct_predictions = 0  # Track correct predictions for precision
        relevant_in_topk = 0  # Track relevant items in top-k for recall
        ndcg = 0

        for j in range(len(label)):
            label_j = to_npy(label[j])  # Ensure label_j is a numpy array
            label_j = label_j.item() if label_j.size == 1 else label_j[0]  # Ensure label_j is a single integer
            topk_predict = topk_predict_batch[j]

            if label_j in topk_predict:
                acc_train[i] += 1
                relevant_in_topk += 1
                correct_predictions += 1
                rank = np.where(topk_predict == label_j)[0][0] + 1
                ndcg += 1 / np.log2(rank + 1)

        # # Calculate accuracy for k
        # acc_train[i] = correct_predictions / total_relevant

        # Calculate recall for k
        recall_train[i] = relevant_in_topk / total_relevant

        # Calculate precision for k
        precision = correct_predictions / (k * len(label))

        # Calculate F1 Score
        if precision + recall_train[i] > 0:
                f1_train[i] = 2 * (precision * recall_train[i]) / (precision + recall_train[i])
        else:
                f1_train[i] = 0
        ndcg_train[i] = ndcg / total_relevant
    return acc_train, recall_train, f1_train,ndcg_train




def sampling_prob(prob, label, num_neg):
    num_label, l_m = prob.shape[0], prob.shape[1]-1  # prob (N, L)
    label = label.view(-1)  # label (N)
    init_label = np.linspace(0, num_label-1, num_label)  # (N), [0 -- num_label-1]
    init_prob = torch.zeros(size=(num_label, num_neg+len(label)))  # (N, num_neg+num_label)

    random_ig = random.sample(range(1, l_m+1), num_neg)  # (num_neg) from (1 -- l_max)
    while len([lab for lab in label if lab in random_ig]) != 0:  # no intersection
        random_ig = random.sample(range(1, l_m+1), num_neg)

    global global_seed
    random.seed(global_seed)
    global_seed += 1

    # place the pos labels ahead and neg samples in the end
    for k in range(num_label):
        for i in range(num_neg + len(label)):
            if i < len(label):
                init_prob[k, i] = prob[k, label[i]]
            else:
                init_prob[k, i] = prob[k, random_ig[i-len(label)]]

    return torch.FloatTensor(init_prob), torch.LongTensor(init_label)  # (N, num_neg+num_label), (N)


class DataSet(data.Dataset):
    # def __init__(self, traj, m1,mat2s, v, label, length,mat3f,mat4cat):
    #     # (NUM, M, 3), (NUM, M, M, 2), (L, L), (NUM, M), (NUM), (NUM)
    #     self.traj, self.mat1, self.mat2s,self.vec, self.label, self.length,self.mat3f,self.mat4cat\
    #         = traj, m1,mat2s, v, label, length,mat3f,mat4cat
    def __init__(self, traj, m1, v, label, length,mat3f,mat4cat):
        # (NUM, M, 3), (NUM, M, M, 2), (L, L), (NUM, M), (NUM), (NUM)
        self.traj, self.mat1, self.vec, self.label, self.length,self.mat3f,self.mat4cat = traj, m1, v, label, length,mat3f,mat4cat


    def __getitem__(self, index):
        traj = self.traj[index].to(device)
        mats1 = self.mat1[index].to(device)
        # mat2s = self.mat2s[index].to(device)
        vector = self.vec[index].to(device)
        label = self.label[index].to(device)
        length = self.length[index].to(device)
        mat3f = self.mat3f[index].to(device)
        mat4cat = self.mat4cat[index].to(device)
        return traj, mats1,vector, label, length,mat3f,mat4cat
        # return traj, mats1,mat2s,vector, label, length,mat3f,mat4cat

    def __len__(self):  # no use
        return len(self.traj)

#用于训练模型和评估模型性能
class Trainer:
    # def __init__(self, model, record,writer, batch_size,epoch):
    def __init__(self, model, record, batch_size,epoch):
        # load other parameters
        self.model = model.to(device)
        self.records = record
        self.start_epoch = record['epoch'][-1] if load else 1
        self.num_neg = 10#负样本数
        self.interval = 1000#间隔
        self.batch_size = batch_size # N = 1
        self.learning_rate = 3e-3#学习率
        self.num_epoch = epoch#训练轮数
        self.threshold = np.mean(record['acc_valid'][-1]) if load else 0  # 0 if not update阈值
        # self.writer = writer
        # (NUM, M, 3), (NUM, M, M, 2), (L, L), (NUM, M, M), (NUM, M), (NUM) i.e. [*M]
        self.traj, self.mat1, self.mat2s, self.mat2t, self.label, self.len,self.mat3f,self.mat4cat = \
            trajs, mat1, mat2s, mat2t, labels, lens,mat3f,mat4cat
        # nn.cross_entropy_loss counts target from 0 to C - 1, so we minus 1 here.
        self.dataset = DataSet(self.traj, self.mat1, self.mat2t, self.label-1, self.len,self.mat3f ,self.mat4cat)
        # self.dataset = DataSet(self.traj, self.mat1,self.mat2s, self.mat2t, self.label-1, self.len,self.mat3f ,self.mat4cat)
        self.data_loader = data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)

    def train(self):
        # set optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0)#adam优化器更新model参数
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1)#stepLR调度器，动态调整学习率
        for t in range(int(self.num_epoch)):
            # settings or validation and test
            valid_size, test_size = 0, 0
            cum_valid_acc = np.zeros(4)
            cum_valid_recall = np.zeros(4)
            cum_valid_f1 = np.zeros(4)
            cum_valid_ndcg = np.zeros(4)
            cum_test_acc = np.zeros(4)
            cum_test_recall = np.zeros(4)
            cum_test_f1 = np.zeros(4)
            cum_test_ndcg = np.zeros(4)

            bar = tqdm(total=part)
            for step, item in enumerate(self.data_loader):
                # get batch data, (N, M, 3), (N, M, M, 2), (N, M, M), (N, M), (N)
                person_input_orgin, person_m1 ,person_m2t, person_label, person_traj_len,person_mat3f,person_mat4cat = item
                # person_input_orgin, person_m1,person_m2s ,person_m2t, person_label, person_traj_len,person_mat3f,person_mat4cat = item
                # first, try batch_size = 1 and mini_batch = 1
                input_mask = torch.zeros((self.batch_size, max_len, 3), dtype=torch.long).to(device)
                # 创建一个与 input_mask 相同形状的零张量
                person_input = torch.zeros_like(input_mask, dtype=torch.float).to(device)
                # 计算需要填充的长度
                padding_length = person_input.shape[1] - person_input_orgin.shape[1]
                # 使用 torch.cat() 在第二个维度上将 person_input_orgin 和零张量拼接起来
                if padding_length > 0:
                    padding = torch.zeros((self.batch_size, padding_length, 3), dtype=torch.float).to(device)
                    # 在原始张量上添加两个维度，使其形状变为（1，100，169，3）
                    # person_input_orgin = person_input_orgin.unsqueeze(0)
                    # # 在填充张量上添加一个维度，使其形状变为（1，1000，31，3）
                    # padding = padding.unsqueeze(0)
                    person_input = torch.cat((person_input_orgin, padding), dim=1)
                else:
                    person_input = person_input_orgin
                m1_mask = torch.zeros((self.batch_size, max_len, max_len, 2), dtype=torch.float32).to(device)
                for mask_len in range(1, person_traj_len[0]+1):  # from 1 -> len
                    input_mask[:, :mask_len] = 1.
                    m1_mask[:, :mask_len, :mask_len] = 1.
                    train_input = person_input * input_mask
                    train_m1 = person_m1 * m1_mask
                    train_m2t = person_m2t[:, mask_len - 1]
                    # train_m2s = person_m2s.squeeze()

                    train_label = person_label[:, mask_len - 1]  # (N)
                    train_len = torch.zeros(size=(self.batch_size,), dtype=torch.long).to(device) + mask_len
                    train_mat3f = person_mat3f
                    train_mat4cat = person_mat4cat

                    prob = self.model(train_input, train_m1, self.mat2s,
                                      train_m2t, train_len,train_mat3f,train_mat4cat)  # (N, L)




                    if mask_len <= person_traj_len[0] - 2:  # only training
                        # nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                        prob_sample, label_sample = sampling_prob(prob, train_label, self.num_neg)
                        loss_train = F.cross_entropy(prob_sample, label_sample)
                        loss_train.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                    elif mask_len == person_traj_len[0] - 1:  # only validation
                        valid_size += person_input.shape[0]
                        acc_valid, recall_valid, f1_valid,ndcg_valid= calculate_acc(prob, train_label)
                        cum_valid_acc += acc_valid
                        cum_valid_recall += recall_valid
                        cum_valid_f1 += f1_valid
                        cum_valid_ndcg += ndcg_valid

                    elif mask_len == person_traj_len[0]:  # only test
                        test_size += person_input.shape[0]
                        acc_test, recall_test, f1_test,ndcg_test = calculate_acc(prob, person_label)
                        cum_test_acc += acc_test
                        cum_test_recall += recall_test
                        cum_test_f1 += f1_test
                        cum_test_ndcg += ndcg_test

                bar.update(self.batch_size)


            bar.close()

            # Normalize metrics
            acc_valid = cum_valid_acc / valid_size
            recall_valid = cum_valid_recall / valid_size
            f1_valid = cum_valid_f1 / valid_size
            ndcg_valid = cum_valid_ndcg / valid_size
            acc_test = cum_test_acc / test_size
            recall_test = cum_test_recall / test_size
            f1_test = cum_test_f1 / test_size
            ndcg_test = cum_test_ndcg / test_size

            print(
                f'epoch:{self.start_epoch + t}, time:{time.time() - start}, valid_acc:{acc_valid}, valid_recall:{recall_valid}, valid_f1:{f1_valid},valid_ndcg:{ndcg_valid}')
            print(
                f'epoch:{self.start_epoch + t}, time:{time.time() - start}, test_acc:{acc_test}, test_recall:{recall_test}, test_f1:{f1_test},test_ndcg:{ndcg_test}')
            acc_valid_str,recall_valid_str,f1_valid_str,ndcg_valid_str = str(acc_valid),str(recall_valid),str(f1_valid),str(ndcg_valid)
            acc_test_str,recall_test_str,f1_test_str,ndcg_test_str = str(acc_test),str(recall_test),str(f1_test),str(ndcg_test)
            model_result = {'achieve_epoch':str(self.start_epoch + t),'time:':str((time.time() - start)/60),'acc_valid':acc_valid_str,'recall_valid':recall_valid_str,'f1_valid':f1_valid_str,'ndcg_valid':ndcg_valid_str
                 ,'acc_test':acc_test_str,'recall_test':recall_test_str,'f1_test':f1_test_str,'ndcg_test':ndcg_test_str}
            model_result_pd = pd.DataFrame(model_result,index=[0])
            # path_file_result = 'para_result/'
            path_file_result = 'data/result/'
            # name =  'foursquare_roi_data-result'
            # name =  'shanghai_roi_data-spacetime_threefeature_result'
            name =  'FourSquare_roi_data-spacetime_result'
            write_csv_test_end(model_result_pd, path_file_result, name,self.batch_size,self.num_epoch)
            # Update records
            self.records['acc_valid'].append(acc_valid)
            self.records['acc_test'].append(acc_test)
            self.records['recall_valid'].append(recall_valid)
            self.records['recall_test'].append(recall_test)
            self.records['f1_valid'].append(f1_valid)
            self.records['f1_test'].append(f1_test)
            self.records['ndcg_valid'].append(ndcg_valid)
            self.records['ndcg_test'].append(ndcg_test)
            self.records['epoch'].append(self.start_epoch + t)



            if self.threshold < np.mean(acc_valid):
                self.threshold = np.mean(acc_valid)
                # save the model
                torch.save({'state_dict': self.model.state_dict(),
                            'records': self.records,
                            'time': time.time() - start},
                           'best_pramp_win_1000_' + dname + '.pth')


    def inference(self):
        user_ids = []
        for t in range(self.num_epoch):
            # settings or validation and test
            valid_size, test_size = 0, 0
            acc_valid, recall_valid, f1_valid,ndcg_valid = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],[0, 0, 0, 0]
            acc_test, recall_test, f1_test ,ndcg_test= [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],[0, 0, 0, 0]

            cum_acc_valid, cum_acc_test = [0, 0, 0, 0], [0, 0, 0, 0]
            cum_recall_valid, cum_recall_test = [0, 0, 0, 0], [0, 0, 0, 0]
            cum_f1_valid, cum_f1_test = [0, 0, 0, 0], [0, 0, 0, 0]
            cum_ndcg_valid, cum_ndcg_test = [0, 0, 0, 0], [0, 0, 0, 0]

            for step, item in enumerate(self.data_loader):
                # get batch data, (N, M, 3), (N, M, M, 2), (N, M, M), (N, M), (N)
                person_input, person_m1,person_m2s, person_m2t, person_label, person_traj_len = item

                # first, try batch_size = 1 and mini_batch = 1

                input_mask = torch.zeros((self.batch_size, max_len, 3), dtype=torch.long).to(device)
                m1_mask = torch.zeros((self.batch_size, max_len, max_len, 2), dtype=torch.float32).to(device)
                for mask_len in range(1, person_traj_len[0] + 1):  # from 1 -> len
                    # if mask_len != person_traj_len[0]:
                    #     continue
                    input_mask[:, :mask_len] = 1.
                    m1_mask[:, :mask_len, :mask_len] = 1.

                    train_input = person_input * input_mask
                    train_m1 = person_m1 * m1_mask
                    train_m2t = person_m2t[:, mask_len - 1]
                    # train_m2s = person_m2s[:, mask_len - 1]
                    train_label = person_label[:, mask_len - 1]  # (N)
                    train_len = torch.zeros(size=(self.batch_size,), dtype=torch.long).to(device) + mask_len

                    prob = self.model(train_input, train_m1, self.mat2s, train_m2t, train_len)  # (N, L)

                    if mask_len <= person_traj_len[0] - 2:  # only training
                        continue

                    elif mask_len == person_traj_len[0] - 1:  # only validation
                        acc_valid,recall_valid,f1_valid,ndcg_valid= calculate_acc(prob, train_label)
                        cum_acc_valid += acc_valid
                        cum_recall_valid += recall_valid
                        cum_f1_valid += f1_valid
                        cum_ndcg_valid += ndcg_valid

                    elif mask_len == person_traj_len[0]:  # only test
                        acc_test,recall_test,f1_test,ndcg_test = calculate_acc(prob, train_label)
                        cum_acc_test += acc_test
                        cum_recall_test += recall_test
                        cum_f1_test += f1_test
                        cum_ndcg_test += ndcg_test

                print(step, acc_valid,recall_valid,f1_valid,ndcg_valid, acc_test,recall_test,f1_test,ndcg_test)

                if acc_valid.sum() == 0 and acc_test.sum() == 0:
                    user_ids.append(step)


#保存結果文件
def write_csv_test_end(model_result,path_file_result,name,batch_size,epoch):

   print("==========开始写入csv文件============")
   datapath1 = path_file_result+name+'.csv'
   CNN_result = pd.DataFrame(model_result)
   CNN_result['batch_size'] = batch_size
   CNN_result['epoch'] = epoch
   CNN_result=CNN_result.round(4)

   # 尝试打开文件，检查是否存在。如果存在，则追加写入
   try:
       with open(datapath1, 'x') as f:
           CNN_result.to_csv(f, index=False)
           print(name + "csv文件已保存")
   except FileExistsError:
       CNN_result.to_csv(datapath1, mode='a', header=False, index=False)
       print(name + "csv文件已追加保存")


if __name__ == '__main__':
    # load data
    dname = 'foursquare_roi'
    # dname = 'shanghai_roi'
    file = open('data/' + dname + '_data.pkl', 'rb')
    file_data = joblib.load(file)
    # tensor(NUM, M, 3), np(NUM, M, M, 2), np(L, L), np(NUM, M, M), tensor(NUM, M), np(NUM)
    [trajs, mat1, mat2s_orginal, mat2t, labels, lens, u_max, l_max,mat3f,mat4cat] = file_data
    # 将 dataframe 转换为以列名作为行列名称的矩阵,並擴張矩陣大小l_max
    mat2s_next = mat2s_orginal.values
    # 计算需要的填充尺寸
    mat2s_next_lenth =len(mat2s_next)
    padding_size = l_max - mat2s_next_lenth

    mat1, mat2t, lens,mat3f,mat4cat,mat2s_next = torch.FloatTensor(mat1), torch.FloatTensor(mat2t),torch.LongTensor(lens)\
        ,torch.LongTensor(mat3f),torch.LongTensor(mat4cat),torch.FloatTensor(mat2s_next).to(device)
    # 扩展张量的尺寸到 4824x4824，填充值为0
    mat2s = F.pad(mat2s_next, (0, padding_size, 0, padding_size), 'constant', 0)

    # the run speed is very flow due to the use of location matrix (also huge memory cost)
    # please use a partition of the data (recommended)
    part = 100
    trajs, mat1, mat2t, labels, lens,mat3f,mat4cat = \
        trajs[:part], mat1[:part], mat2t[:part], labels[:part], lens[:part],mat3f[:part],mat4cat[:part]
    print(mat1[:, :, :, 0])
    print(mat1[:, :, :, 0].max())
    ex = mat1[:, :, :, 0].max(), mat1[:, :, :, 0].min(), mat1[:, :, :, 1].max(), mat1[:, :, :, 1].min()
    u_max=int(u_max)
    pramp = Model(t_dim=hours+1, l_dim=l_max+1, u_dim=u_max+1, embed_dim=50, ex=ex, dropout=0 )
    num_params = 0
    #展示权重名
    for name in pramp.state_dict():
        print(name)
    #展示参数
    for param in pramp.parameters():
        num_params += param.numel()
    print('num of params', num_params)

    load = False

    if load:
        checkpoint = torch.load('best_pramp_win_' + dname + '.pth')
        pramp.load_state_dict(checkpoint['state_dict'])
        start = time.time() - checkpoint['time']
        records = checkpoint['records']
    else:
        records = {'epoch': [], 'acc_valid': [], 'acc_test': [], 'recall_valid': [], 'recall_test': [], 'f1_valid': [],
                   'f1_test': [],'ndcg_valid': [],'ndcg_test': []}
        start = time.time()
    # writer = SummaryWriter()
    #训练过程中的训练指标会保存到records中


    # batch_list=[10]
    # epoch_list = [50]
    # for i in batch_list:
    batch_size = 5
        # for j in epoch_list:
    epoch = 50
    # trainer = Trainer(pramp, records, writer, batch_size=batch_size, epoch=epoch)
    trainer = Trainer(pramp, records, batch_size=batch_size, epoch=epoch)
    # trainer.train()
    # 记录训练时间
    start_time = time.time()
    trainer.train()
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f" 模型训练总耗时：{elapsed_time:.2f} 秒（约 {elapsed_time / 60:.2f} 分钟）")

