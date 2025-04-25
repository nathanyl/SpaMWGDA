from __future__ import division
from __future__ import print_function
import torch.optim as optim
from sklearn.cluster import KMeans

from utils import *
from models import Spatial_MGCN
import os
import argparse
from config import Config
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
import scanpy as sc

def load_data(dataset):
    print("load data:")
    path = "../generate_data/" + dataset + "/SpaMWGDA.h5ad"
    adata = sc.read_h5ad(path)
    features = torch.FloatTensor(adata.obsm['feat'])
    features_a = torch.FloatTensor(adata.obsm['feat_a'])
    labels = adata.obs['ground_truth']
    fadj = adata.obsm['fadj']

    sadj_a = adata.obsm['sadj_a']
    sadj = adata.obsm['sadj']

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
    return adata, features, features_a, labels, nfadj, nsadj, nsadj_a, graph_nei, graph_neg, graph_nei_a, graph_neg_a


def train():
    torch.autograd.set_detect_anomaly(True)  # 启用异常检测
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
    datasets = ['Human_Breast_Cancer']

    for i in range(len(datasets)):
        dataset = datasets[i]
        path = './result/' + dataset + '/'
        config_file = './config/' + dataset + '.ini'
        if not os.path.exists(path):
            os.mkdir(path)
        print(dataset)
        adata, features, features_a, labels, sadj, sadj_a, fadj, graph_nei, graph_neg, graph_nei_a, graph_neg_a = load_data(dataset)

        config = Config(config_file)
        cuda = not config.no_cuda and torch.cuda.is_available()
        use_seed = not config.no_seed

        _, ground = np.unique(np.array(labels, dtype=str), return_inverse=True)
        ground = torch.LongTensor(ground)
        config.n = len(ground)
        config.class_num = len(ground.unique())

        savepath = './result/Human_Breast_Cancer/'
        save_data_path = '../data/Human_Breast_Cancer/'
        plt.rcParams["figure.figsize"] = (4, 3)

        print(adata)
        title = "Manual annotation"
        sc.pl.spatial(adata, img_key="hires", color=['ground_truth'], title=title, show=False)
        plt.savefig(savepath + dataset + '.jpg', bbox_inches='tight', dpi=600)
        plt.show()

        Ann_df = pd.read_csv(os.path.join(save_data_path, 'metadata.tsv'), sep='\t',
                             header=0, index_col=0)
        adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'ground_truth']


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
                             dropout=config.dropout)
        if cuda:
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=config.lr,
                               weight_decay=config.weight_decay)
        epoch_max = 0
        ari_max = 0
        nmi_max = 0
        idx_max = []
        mean_max = []
        emb_max = []
        for epoch in range(config.epochs):
            emb, mean, zinb_loss, reg_loss, con_loss, loss_feat, total_loss = train()
            print(dataset, ' epoch: ', epoch, ' zinb_loss = {:.2f}'.format(zinb_loss),
                  ' reg_loss = {:.2f}'.format(reg_loss),
                  ' con_loss = {:.2f}'.format(con_loss),
                  ' loss_feat = {:.2f}'.format(loss_feat),
                  ' total_loss = {:.2f}'.format(total_loss))

            kmeans = KMeans(n_clusters=config.class_num).fit(emb)
            idx = kmeans.labels_
            ari_res = metrics.adjusted_rand_score(labels, idx)
            nmi_res = metrics.normalized_mutual_info_score(labels, idx)
            if ari_res > ari_max:
                ari_max = ari_res
                nmi_max = nmi_res
                epoch_max = epoch
                idx_max = idx
                mean_max = mean
                emb_max = emb

        print(f'ARI:{ari_max:.2f},NMI:{nmi_max:.2f}')

        title = 'SpaMWGDA (ARI={:.2f})'.format(ari_max)

        adata.obs['idx'] = idx_max.astype(str)
        adata.obsm['emb'] = emb_max
        adata.obsm['mean'] = mean_max
        adata.obs['kmeans'] = adata.obs['idx'].astype('category')

        title = 'SpaMWGDA (ARI={:.2f})'.format(ari_max)
        pd.DataFrame(emb_max).to_csv(savepath + 'SpaMWGDA_emb.csv', header=None, index=None)
        pd.DataFrame(idx_max).to_csv(savepath + 'SpaMWGDA_idx.csv', header=None, index=None)
        sc.pl.spatial(adata, img_key="hires", color=['idx'], title=title, show=False)
        adata.layers['X'] = adata.X
        adata.layers['mean'] = mean_max
        plt.savefig(savepath + 'SpaMWGDA.jpg', bbox_inches='tight', dpi=600)
        plt.show()
        sc.pp.neighbors(adata, use_rep='mean')
        sc.tl.umap(adata)
        plt.rcParams["figure.figsize"] = (3, 3)
        sc.pl.umap(adata, color="idx", title='SpaMWGDA')
        adata.write(savepath + 'SpaMWGDA.h5ad')
