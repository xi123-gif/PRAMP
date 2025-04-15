from input_model import *
import torch
from torch import nn
from torch.nn import functional as F
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True
seed = 0
global_seed = 0
hours = 24*7
torch.manual_seed(seed)
device = 'cuda'



def to_npy(x):
    return x.cpu().data.numpy() if device == 'cuda' else x.detach().numpy()


class Attn(nn.Module):
    def __init__(self, emb_loc, loc_max):
        super(Attn, self).__init__()
        self.value = nn.Linear(max_len, 1, bias=False)
        self.emb_loc = emb_loc
        self.loc_max = loc_max



    def forward(self, self_attn, self_delta, traj_len):
        # self_attn (N, M, emb), candidate (N, L, emb), self_delta (N, M, L, emb), len [N]
        self_delta = torch.sum(self_delta, -1).transpose(-1, -2)  # squeeze the embed dimension
        [N, L, M] = self_delta.shape
        candidates = torch.linspace(1, int(self.loc_max), int(self.loc_max)).long()  # (L)
        candidates = candidates.unsqueeze(0).expand(N, -1).to(device)  # (N, L)
        emb_candidates = self.emb_loc(candidates)  # (N, L, emb)
        attn = torch.mul(torch.bmm(emb_candidates, self_attn.transpose(-1, -2)), self_delta)  # (N, L, M)
        # pdb.set_trace()
        attn_out = self.value(attn).view(N, L)  # (N, L)

        return attn_out  # (N, L)


class SelfAttn(nn.Module):
    def __init__(self, emb_size, output_size, dropout=0.1):
        super(SelfAttn, self).__init__()
        self.query = nn.Linear(emb_size, output_size, bias=False)
        self.key = nn.Linear(emb_size, output_size, bias=False)
        self.value = nn.Linear(emb_size, output_size, bias=False)

    def forward(self, joint, delta, traj_len):
        delta = torch.sum(delta, -1)  # squeeze the embed dimension
        # joint (N, M, emb), delta (N, M, M, emb), len [N]
        # construct attention mask
        mask = torch.zeros_like(delta, dtype=torch.float32)
        for i in range(mask.shape[0]):
            mask[i, 0:traj_len[i], 0:traj_len[i]] = 1

        attn = torch.add(torch.bmm(self.query(joint), self.key(joint).transpose(-1, -2)), delta)  # (N, M, M)
        attn = F.softmax(attn, dim=-1) * mask  # (N, M, M)

        attn_out = torch.bmm(attn, self.value(joint))  # (N, M, emb)

        return attn_out  # (N, M, emb)


class Embed(nn.Module):
    def __init__(self, ex, emb_size, loc_max, embed_layers):
        super(Embed, self).__init__()
        _, _, _, self.emb_su, self.emb_sl, self.emb_tu, self.emb_tl = embed_layers
        self.su, self.sl, self.tu, self.tl = ex
        self.emb_size = emb_size
        self.loc_max = loc_max


    def forward(self, traj_loc, mat2, vec, traj_len):

        # traj_loc (N, M), mat2 (L, L), vec (N, M), delta_t (N, M, L)
        delta_t = vec.unsqueeze(-1).expand(-1, -1, int(self.loc_max))
        delta_s = torch.zeros_like(delta_t, dtype=torch.float32)
        mask = torch.zeros_like(delta_t, dtype=torch.long)
        for i in range(mask.shape[0]):  # N
            mask[i, 0:traj_len[i]] = 1
            delta_s[i, :traj_len[i]] = torch.index_select(mat2, 0, (traj_loc[i]-1)[:traj_len[i]])
            # mask[i, 0:traj_len[i]] = 1
            # a =mat2
            # c=(traj_loc[i] - 1)
            # d=c[:traj_len[i]]
            # delta_s[i, :traj_len[i]] = torch.index_select(mat2, 0, d)
            # mat2 = mat2[0]
            # index = (traj_loc[i] - 1).long()[:traj_len[i]]
            # index = index.unsqueeze(1)
            # # 进行索引操作
            # c=index[0]
            # # selected_values = torch.index_select(mat2, 0, index[0])
            # # 将选择的结果赋值给 delta_s 张量的前 traj_len[i] 个元素
            # delta_s[i, :traj_len[i]] = selected_values.squeeze()
        esl, esu, etl, etu = self.emb_sl(mask), self.emb_su(mask), self.emb_tl(mask), self.emb_tu(mask)
        vsl, vsu, vtl, vtu = (delta_s - self.sl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
                             (self.su - delta_s).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
                             (delta_t - self.tl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
                             (self.tu - delta_t).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)

        space_interval = (esl * vsu + esu * vsl) / (self.su - self.sl)
        time_interval = (etl * vtu + etu * vtl) / (self.tu - self.tl)
        delta = space_interval + time_interval  # (N, M, L, emb)

        return delta


class MultiEmbed(nn.Module):
    def __init__(self, ex, emb_size, embed_layers):
        super(MultiEmbed, self).__init__()
        self.emb_t, self.emb_l, self.emb_u, \
        self.emb_su, self.emb_sl, self.emb_tu, self.emb_tl = embed_layers
        self.su, self.sl, self.tu, self.tl = ex
        self.emb_size = emb_size

    def forward(self, traj, mat, traj_len):
        # traj (N, M, 3), mat (N, M, M, 2), len [N]
        traj[:, :, 2] = (traj[:, :, 2]-1) % hours + 1  # segment time by 24 hours * 7 days
        time = self.emb_t(traj[:, :, 2].long())  # (N, M) --> (N, M, embed)
        loc = self.emb_l(traj[:, :, 1].long())  # (N, M) --> (N, M, embed)
        user = self.emb_u(traj[:, :, 0].long())  # (N, M) --> (N, M, embed)
        joint = time + loc + user  # (N, M, embed)

        delta_s, delta_t = mat[:, :, :, 0], mat[:, :, :, 1]  # (N, M, M)
        mask = torch.zeros_like(delta_s, dtype=torch.long)
        for i in range(mask.shape[0]):
            mask[i, 0:traj_len[i], 0:traj_len[i]] = 1

        esl, esu, etl, etu = self.emb_sl(mask), self.emb_su(mask), self.emb_tl(mask), self.emb_tu(mask)
        test_sl =self.sl
        test_su =self.su

        vsl, vsu, vtl, vtu = (delta_s - self.sl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
                             (self.su - delta_s).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
                             (delta_t - self.tl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
                             (self.tu - delta_t).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)

        space_interval = (esl*vsu+esu*vsl) / (self.su-self.sl)
        time_interval = (etl*vtu+etu*vtl) / (self.tu-self.tl)
        delta = space_interval + time_interval  # (N, M, M, emb)

        return joint, delta

class SocialCNN(nn.Module):
    def __init__(self, embed_dim, dropout=0.5):
        super(SocialCNN, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 3), padding=1)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), padding=1)
        # # self.pool = nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))
        # self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2),padding=(0,0))
        # # 计算卷积和池化后的输出大小
        # self.conv_output_size = embed_dim // 4
        # self.fc1 = nn.Linear(64 * self.conv_output_size, 128)
        # self.fc2 = nn.Linear(128, embed_dim)
        # self.dropout = nn.Dropout(dropout)
        # 卷积层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # 全连接层
        self.fc1 = nn.Linear(16 * 100, 64)  # 根据输入长度和卷积池化后的输出计算
        self.fc2 = nn.Linear(64, 10)  # 输出类别数

    def forward(self, social_info):
        social_info = social_info.float()
        social_info = social_info.unsqueeze(1)  # 在通道维度上添加一个维度，变成 (batch_size, 1, length)
        # social_info_conv1 = self.conv1(social_info)
        # print(social_info_conv1.shape)
        # x = F.relu(social_info_conv1)  # 卷积层后应用ReLU激活函数
        # print(x.shape)
        x = F.relu(self.conv1(social_info))  # 卷积层后应用ReLU激活函数
        x = self.pool(x)  # 池化层
        # print(x.shape)
        x = x.view(-1, 16 * 100)  # 展平特征图，这里假设卷积池化后长度为100
        # print(x.shape)
        x = F.relu(self.fc1(x))  # 全连接层 + ReLU激活函数
        # print(x.shape)
        x = self.fc2(x)  # 输出层
        # print(x.shape)
        return x

class CategoryMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(CategoryMLP, self).__init__()
        # self.fc1 = nn.Linear(200, 64)
        # self.fc2 = nn.Linear(64, 10)
        # self.dropout = nn.Dropout(dropout)
        # self.relu = nn.ReLU()
        self.fc1 = nn.Linear(200, 64)  # 输入维度为200，输出维度为64
        self.fc2 = nn.Linear(64, 10)   # 输入维度为64，输出维度为10

    def forward(self, category_info):
        category_info = category_info.float()
        # x = self.relu(self.fc1(category_info))
        # x = self.dropout(x)
        # category_output = self.fc2(x)
        # return category_output
        x = category_info.view(category_info.size(0), -1)
        x = F.relu(self.fc1(x))  # 第一个全连接层 + ReLU激活函数
        x = self.fc2(x)  # 第二个全连接层，输出层
        return x


class DeepFM(nn.Module):
    def __init__(self,input_dim, embedding_dim, hidden_dims, dropout):
        super(DeepFM, self).__init__()
        # Linear part
        #注意！！！！記得同步batch——size
        self.linear = nn.Linear(input_dim, 1)
        # FM part
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        # DNN part
        self.dnn = nn.Sequential(
            nn.Linear(input_dim*embedding_dim,hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1])
        )

    def forward(self, x):
        x=x.view(x.size(0),-1)
        # Linear part
        linear_part = self.linear(x)
        # 获取数据中唯一值的数量
        #將輸入數據調整爲非著證書
        # 找到最小的负数索引
        min_index = torch.min(x)
        # 将所有索引都加上最小的负数索引的绝对值，使得所有索引都非负,embedding才能正確運行
        x = x + abs(min_index)+1
        # #獲取combined_output以定義DEEPfm的輸入特徵維度
        # num_unique_values = int(len(torch.unique(x)))
        # print("特征值的数量为:", num_unique_values)
        # FM part
        x =x.long()
        # print(x)
        fm_embedding = self.embedding(x)
        square_of_sum = torch.sum(fm_embedding, dim=1) ** 2
        sum_of_square = torch.sum(fm_embedding ** 2, dim=1)
        fm_part = 0.5 * (square_of_sum - sum_of_square).sum(dim=1, keepdim=True)

        # DNN part
        # t= fm_embedding.size(0)
        # f = fm_embedding.view(t, -1)
        # dnn_part = self.dnn(f)
        dnn_input = fm_embedding.view(fm_embedding.size(0), -1)
        dnn_part = self.dnn(dnn_input)


        # Sum up the results
        output = linear_part + fm_part + dnn_part

        return torch.sigmoid(output)