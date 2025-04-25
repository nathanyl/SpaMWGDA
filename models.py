import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution, GraphDeConvolution
import torch


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class CGAE(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(CGAE, self).__init__()

        self.z_layer = GraphDeConvolution(nfeat, nhid)
        self.x_hat_layer = GraphDeConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, feat, feat_a, fadj):
        z_ori, z_aug = self.z_layer(feature_ori=feat, feature_aug=feat_a, adjacency=fadj)
        xhat_ori, xhat_aug = self.x_hat_layer(feature_ori=z_ori, feature_aug=z_aug, adjacency=fadj)
        return z_ori, z_aug, xhat_ori, xhat_aug


class CombinedModel(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout):
        super(CombinedModel, self).__init__()
        self.cgae = CGAE(nfeat, nhid1, nhid2, dropout)
        self.gcn = GraphConvolution(nhid2, nhid2)
        self.dropout = dropout

    def forward(self, feat, feat_a, fadj, sadj):
        # CGAE 运行，获得编码后的特征和重构特征
        z_ori, z_aug, xhat_ori, xhat_aug = self.cgae(feat, feat_a, fadj)
        xhat_aug = F.dropout(xhat_aug, self.dropout, training=self.training)
        # GCN 接收重构后的增强特征作为输入，同时处理空间图
        gcn_output = self.gcn(xhat_aug, sadj)
        return z_ori, z_aug, xhat_ori, gcn_output


class AdaptiveAttention(nn.Module):
    def __init__(self, nhid2):
        super(AdaptiveAttention, self).__init__()
        self.project = nn.Sequential(
                nn.Linear(nhid2, nhid2),
                nn.Tanh(),
                nn.Linear(nhid2, 1, bias=False)
            )

    def forward(self, X):
        # X: List of tensors, one for each view
        # Concatenate tensors along the batch dimension and then apply attention
        # concatenated_X = torch.cat(tuple(X), dim=1)  # (batch_size, num_views * feature_dim)
        att_scores = self.project(X).squeeze(2)  # (batch_size, num_views)
        att_weights = F.softmax(att_scores, dim=1)  # (batch_size, num_views)
        return att_weights


class WeightedFusion(nn.Module):
    def __init__(self, num_views, nhid2, use_attention=True):
        super(WeightedFusion, self).__init__()
        self.num_views = num_views
        self.use_attention = use_attention
        if use_attention:
            self.attention = AdaptiveAttention(nhid2)

    def forward(self, X):
        if self.use_attention:
            att_weights = self.attention(X)
        else:
            # If no attention, use uniform weights
            device = X.device
            att_weights = torch.ones(X.size(0), self.num_views, device=device) / self.num_views

        # Apply attention weights
        X = torch.transpose(X, 0, 1)
        weighted_X = torch.stack([att_weights[:, i:i+1] * X[i] for i in range(self.num_views)], dim=0)
        combined_X = torch.sum(weighted_X, dim=0)
        return combined_X


class Attention(nn.Module):
    def __init__(self, in_size):
        super(Attention, self).__init__()
        self.project = nn.Linear(in_size, 1, bias=False)

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta*z).sum(1), beta


class decoder(torch.nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1, track_running_stats=False),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]



class Spatial_MGCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout, num_view=3):
        super(Spatial_MGCN, self).__init__()
        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.FGCN = CombinedModel(nfeat, nhid1, nhid2, dropout)
        self.CGCN = CombinedModel(nfeat, nhid1, nhid2, dropout)
        self.GCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.ZINB = decoder(nfeat, nhid1, nhid2)
        self.dropout = dropout
        self.att = Attention(nhid2)

        self.weightFusion = WeightedFusion(num_view, use_attention=True, nhid2=nhid2)
        self.MLP = nn.Sequential(
            nn.Linear(nhid2, nhid2)
        )

    def forward(self, x, x_a, sadj, sadj_a, fadj):
        emb1_ori = self.SGCN(x, sadj)  # Spatial_GCN
        emb1_aug = self.SGCN(x, sadj_a)  # Spatial_GCN
        com1_ori, com1_aug, com1_xhat, com1 = self.CGCN(x, x_a, fadj, sadj)  # Co_GCN
        com2_ori, com2_aug, com2_xhat, com2 = self.CGCN(x, x_a, fadj, sadj_a)  # Co_GCN
        com3 = self.GCN(x, fadj) # Co_GCN
        emb2_ori, emb2_aug, emb2_xhat, emb2 = self.FGCN(x, x_a, fadj, fadj)  # Feature_GCN

        Xcom = (com1 + com2 + com3) / 3

        emb1 = (emb1_ori + emb1_aug) / 2

        emb = torch.stack([emb1, Xcom, emb2], dim=1)

        combined_x = self.weightFusion(emb)
        emb = self.MLP(combined_x)
        [pi, disp, mean] = self.ZINB(emb)

        return com1, com2, com1_ori, com1_aug, com1_xhat, \
               com2_ori, com2_aug, com2_xhat, \
               emb, emb2_ori, emb2_aug, emb2_xhat, emb2, pi, disp, mean
