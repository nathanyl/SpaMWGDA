import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import matplotlib.pyplot as plt
import os
from utils import *
from config import Config


inpath = r'..\data\Stereoseq_MOB'
counts_file = os.path.join(inpath, 'RNA_counts.tsv')
coor_file = os.path.join(inpath, 'position.tsv')

counts = pd.read_csv(counts_file, sep='\t', index_col=0)
coor_df = pd.read_csv(coor_file, sep='\t')


counts.columns = ['Spot_'+str(x) for x in counts.columns] 
coor_df.index = coor_df['label'].map(lambda x: 'Spot_'+str(x))
coor_df = coor_df.loc[:, ['x', 'y']]
coor_df.head()

adata = sc.AnnData(counts.T)
adata.var_names_make_unique()

coor_df = coor_df.loc[adata.obs_names, ['y', 'x']]
adata.obsm["spatial"] = coor_df.to_numpy()
sc.pp.calculate_qc_metrics(adata, inplace=True)

plt.rcParams["figure.figsize"] = (5,4)
sc.pl.embedding(adata, basis="spatial", color="n_genes_by_counts", show=False,save='_stereo_MOB01_UMAP.png')
plt.title("")
plt.axis('off')

used_barcode = pd.read_csv(os.path.join(inpath,'used_barcodes.txt'), sep='\t', header=None)
used_barcode = used_barcode[0]
adata = adata[used_barcode]

plt.rcParams["figure.figsize"] = (5,4)
sc.pl.embedding(adata, basis="spatial", color="n_genes_by_counts", show=False,save='_stereo_MOB02_UMAP.png')
plt.title("")
plt.axis('off')

sc.pp.filter_genes(adata, min_cells=50)
print('After flitering: ', adata.shape)

#Normalization
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# load ST
config_file = './config/' + 'Stereoseq_MOB' + '.ini'
config = Config(config_file)
X = np.nan_to_num(adata.X, nan=0)

feat = adata.X
feat_a = permutation(feat)
adata.obsm['feat'] = feat
adata.obsm['feat_a'] = feat_a
fadj = features_construct_graph(X, k=config.k)

sadj, graph_nei, graph_neg = spatial_construct_graph(adata.obsm['spatial'], config.k)
sadj_a, graph_nei_a, graph_neg_a = spatial_construct_graph1(adata, config.radius)

adata.obsm["fadj"] = fadj

adata.obsm["sadj"] = sadj
adata.obsm["graph_nei"] = graph_nei.numpy()
adata.obsm["graph_neg"] = graph_neg.numpy()

adata.obsm["sadj_a"] = sadj_a
adata.obsm["graph_nei_a"] = graph_nei_a.numpy()
adata.obsm["graph_neg_a"] = graph_neg_a.numpy()
adata.var_names_make_unique()

if not os.path.exists("../generate_data/"):
    os.mkdir("../generate_data/")
savepath = "../generate_data/" + 'Stereoseq_MOB' + "/"
if not os.path.exists(savepath):
    os.mkdir(savepath)
print("saving")
adata.write(savepath + 'SpaMWGDA.h5ad')
print("done")
