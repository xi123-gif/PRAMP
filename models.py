from layers import *
CUDA_LAUNCH_BLOCKING=1
def adaptive_weighted_preference(spacetime_output, social_output, category_output, region_vectors):

    N, D = spacetime_output.shape

    # Step 1-2: Compute cosine similarity confidence scores
    def cosine_confidence(user_vecs, region_vecs):
        norm_user = F.normalize(user_vecs, p=2, dim=1)        # (N, D)
        norm_region = F.normalize(region_vecs, p=2, dim=1)     # (R, D)
        sim = torch.matmul(norm_user, norm_region.T)          # (N, R)
        confidence = sim.mean(dim=1)                          # (N,)
        return confidence

    conf_s = cosine_confidence(spacetime_output, region_vectors)
    conf_r = cosine_confidence(social_output, region_vectors)
    conf_c = cosine_confidence(category_output, region_vectors)

    # Step 3: Normalize confidence scores
    total_conf = conf_s + conf_r + conf_c + 1e-8  # prevent divide-by-zero
    alpha_s = conf_s / total_conf  # (N,)
    alpha_r = conf_r / total_conf
    alpha_c = conf_c / total_conf

    # Step 4: Weighted fusion
    fused_preference = (
        alpha_s.unsqueeze(1) * spacetime_output +
        alpha_r.unsqueeze(1) * social_output +
        alpha_c.unsqueeze(1) * category_output
    )  # (N, D)

    return fused_preference
from sklearn.metrics.pairwise import cosine_similarity
# t_dim: 时间特征的维度,l_dim: 位置特征的维度,u_dim: 用户特征的维度,embed_dim: 嵌入层的维度,ex: 一个包含最大值和最小值的元组,用于归一化输入数据
class Model(nn.Module):
    def __init__(self, t_dim, l_dim, u_dim, embed_dim, ex, dropout):
        super(Model, self).__init__()
        emb_t = nn.Embedding(t_dim, embed_dim, padding_idx=0)
        emb_l = nn.Embedding(int(l_dim), embed_dim, padding_idx=0)
        emb_u = nn.Embedding(int(u_dim), embed_dim, padding_idx=0)
        emb_su = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_sl = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tu = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tl = nn.Embedding(2, embed_dim, padding_idx=0)
        embed_layers = emb_t, emb_l, emb_u, emb_su, emb_sl, emb_tu, emb_tl

        # 定义用于处理时空信息的时空双层注意力网络层
        self.MultiEmbed = MultiEmbed(ex, embed_dim, embed_layers)
        self.SelfAttn = SelfAttn(embed_dim, embed_dim)
        self.Embed = Embed(ex, embed_dim, l_dim-1, embed_layers)
        self.Attn = Attn(emb_l, l_dim-1)
        #定义用于处理社交信息的CNN层
        self.SocialCNN = SocialCNN(embed_dim, dropout)
        # 定义用于处理类别信息的MLP层
        self.CategoryMLP = CategoryMLP(input_dim=embed_dim, hidden_dim=128, output_dim=embed_dim, dropout=dropout)
        #定义整合多特征的DEEPFM
        # 定义 DeepFM
        self.DeepFM = DeepFM(input_dim=4844, embedding_dim=10, hidden_dims=[500,4852], dropout=0.5)
        # self.DeepFM = DeepFM(input_dim=4852, embedding_dim=6, hidden_dims=[64,32], dropout=0.5)
    # def forward(self, traj, mat1, mat2, vec, traj_len):
    def forward(self, traj, mat1, mat2, vec, traj_len,social_info, category_info):

        # long(N, M, [u, l, t]), float(N, M, M, 2), float(N,L, L), float(N, M), long(N)
        joint, delta = self.MultiEmbed(traj, mat1, traj_len)  # (N, M, emb), (N,  M, M, emb)
        self_attn = self.SelfAttn(joint, delta, traj_len)  # (N, M, emb)
        self_delta = self.Embed(traj[:, :, 1], mat2, vec, traj_len)  # (N, M, L, emb)
        spacetime_output = self.Attn(self_attn, self_delta, traj_len)  # (N, L)
        #
        # 处理社交信息
        social_output = self.SocialCNN(social_info)  # (N, embed_dim)

        # 处理类别信息
        category_output = self.CategoryMLP(category_info)

        output = adaptive_weighted_preference(spacetime_output,social_output,category_output,mat2)


        return output




