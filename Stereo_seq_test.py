from __future__ import division
from __future__ import print_function

import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from rpy2.robjects.packages import importr
from torch.utils.data.datapipes import dataframe
import scanpy as sc
from utils import *
from models import Spatial_MGCN
import os
import argparse
from config import Config
import pandas as pd
import torch.nn.functional as F
importr('mclust')


def load_data(dataset):
    print("load data")
    path = "../generate_data/" + dataset + "/SpaMWGDA.h5ad"
    adata = sc.read_h5ad(path)
    features = torch.FloatTensor(adata.obsm['feat'])
    features_a = torch.FloatTensor(adata.obsm['feat_a'])
    fadj = adata.obsm['fadj']

    sadj = adata.obsm['sadj']
    sadj_a = adata.obsm['sadj_a']
    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)

    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    nsadj_a = normalize_sparse_matrix(sadj_a + sp.eye(sadj_a.shape[0]))
    nsadj_a = sparse_mx_to_torch_sparse_tensor(nsadj_a)
    graph_nei = torch.LongTensor(adata.obsm['graph_nei'])
    graph_neg = torch.LongTensor(adata.obsm['graph_neg'])
    graph_nei_a = torch.LongTensor(adata.obsm['graph_nei_a'])
    graph_neg_a = torch.LongTensor(adata.obsm['graph_neg_a'])
    print("done")
    return adata, features, features_a, nsadj, nsadj_a, nfadj, graph_nei, graph_neg, graph_nei_a, graph_neg_a


def train():
    model.train()
    optimizer.zero_grad()
    com1, com2, com3, com1_ori, com1_aug, com1_xhat, \
    com2_ori, com2_aug, com2_xhat, \
    emb, emb2_ori, emb2_aug, emb2_xhat, emb2, pi, disp, mean = model(features, features_a, sadj, sadj_a, fadj)
    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
    reg_loss1 = regularization_loss(emb, graph_nei, graph_neg)
    reg_loss2 = regularization_loss(emb, graph_nei_a, graph_neg_a)
    reg_loss = reg_loss1 + reg_loss2
    con_loss = consistency_loss(com1, com2, com3)

    loss_feat1 = F.mse_loss(emb2_ori, emb2_aug)
    loss_feat2 = F.mse_loss(emb2_xhat, emb2)

    loss_feat5 = F.mse_loss(com1_ori, com1_aug)
    loss_feat6 = F.mse_loss(com1_xhat, com1)

    loss_feat9 = F.mse_loss(com2_ori, com2_aug)
    loss_feat10 = F.mse_loss(com2_xhat, com2)

    loss_feat = 0.5 * (loss_feat1 + loss_feat2 + loss_feat5 + loss_feat6 + loss_feat9 + loss_feat10)
    total_loss = config.alpha * zinb_loss + config.beta * con_loss + config.gamma * reg_loss + loss_feat
    emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
    mean = pd.DataFrame(mean.cpu().detach().numpy()).fillna(0).values
    total_loss.backward()
    optimizer.step()
    return emb, mean, zinb_loss, reg_loss, con_loss, loss_feat, total_loss


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    plt.rcParams['font.family'] = 'Times New Roman'

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    datasets = ['Stereoseq_MOB']

    for i in range(len(datasets)):
        dataset = datasets[i]
        savepath = './result/' + dataset + '/'
        config_file = './config/' + dataset + '.ini'
        hire_path = '../data/' + dataset + '/crop1.png'

        img = plt.imread(hire_path)
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        print(dataset)

        adata, features, features_a, sadj, sadj_a, fadj, graph_nei, graph_neg, graph_nei_a, graph_neg_a = load_data(dataset)

        config = Config(config_file)
        cuda = not config.no_cuda and torch.cuda.is_available()
        use_seed = not config.no_seed

        config.n = features.shape[0]
        config.class_num = 7


        if cuda:
            features = features.cuda()
            features_a = features_a.cuda()
            sadj = sadj.cuda()
            sadj_a = sadj_a.cuda()
            fadj = fadj.cuda()
            graph_nei = graph_nei.cuda()
            graph_neg = graph_neg.cuda()
            graph_nei_a = graph_nei_a.cuda()
            graph_neg_a = graph_neg_a.cuda()

        config.epochs = config.epochs + 1
        import random

        np.random.seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        os.environ['PYTHONHASHSEED'] = str(config.seed)
        if not config.no_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

        print(dataset, ' ', config.lr, ' ', config.alpha, ' ', config.beta, ' ', config.gamma)

        model = Spatial_MGCN(nfeat=config.fdim,
                             nhid1=config.nhid1,
                             nhid2=config.nhid2,
                             dropout=config.dropout,
                             )
        if cuda:
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        for epoch in range(config.epochs):
            emb, mean, zinb_loss, reg_loss, con_loss, loss_feat, total_loss = train()
            print(dataset, ' epoch: ', epoch, ' zinb_loss = {:.2f}'.format(zinb_loss),
                  ' reg_loss = {:.2f}'.format(reg_loss),
                  ' con_loss = {:.2f}'.format(con_loss),
                  ' loss_feat = {:.2f}'.format(loss_feat),
                  ' total_loss = {:.2f}'.format(total_loss))

        # 在单细胞转录组数据中，obsm 常用于存储降维后的数据，如 PCA、t-SNE 或 UMAP 的结果。
        adata.obsm['emb'] = emb

        # 将 NumPy 数组转换为 Pandas DataFrame
        df = pd.DataFrame(adata.obsm['emb'])
        df.to_csv(savepath + 'SpaMWGDA_emb.csv', header=None, index=None)

        sc.pp.neighbors(adata, use_rep='emb')
        sc.tl.umap(adata)
        sc.tl.louvain(adata, resolution=0.8)
        adata.obs['louvain'].to_csv(savepath + 'SpaMWGDA_idx.csv', header=None, index=None)

        plt.rcParams["figure.figsize"] = (3, 3)
        sc.pl.embedding(adata, basis="spatial", color="louvain", s=6, show=False,
                        title='SpaMWGDA')
        plt.savefig(savepath + 'SpaMWGDA.jpg', bbox_inches='tight', dpi=600)
        plt.axis('off')

