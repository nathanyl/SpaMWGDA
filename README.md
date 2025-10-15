# SpaMWGDA
SpaMWGDA11
# Requirements
<ul>
  <li>python=3.10.14</li>
  <li>pytorch=2.4.0</li>
  <li>stlearn=0.4.12</li>
  <li>scanpy=1.10.4</li>
  <li>numpy=1.25.0</li>
  <li>pandas=2.0.3</li>
  <li>torch_geometric=2.5.3</li>
  <li>torch_scatter==2.1.2</li>
  <li>torch_sparse=0.6.13</li>
</ul>

# Installation
Download SpaMWGDA by


    git clone https://github.com/nathanyl/SpaMWGDA

# Run SpaMWGDA model

## 1 Configuration
    python config.py
        Arguments:
          --fidm: int; the number of highly variable genes
          --k: int; the k of knn metrics
          --radius: int; the radius of radius metrics
          --seed: int; the random parameter
          --lr: float; the learning rate
          --weight_decay: float; the weight decay
          --nhid1: int; the dimension of the first hidden layer
          --nhid2: int; the dimension of the second hidden layer
          --dropout: float; the dropout rate
          --epochs: int; the training epochs
          --alpha: float; the factor of zinb_loss
          --beta: float; the factor of con_loss
          --gamma: float; the factorof reg_loss
          --no_cuda: boolean; whether to use cuda
          --no_seed: boolean; whether to use random parameter
## 2 Data Processing
    python DLPFC_generate_data.py
## 3 Run
    python DLPFC_test.py
For Human Breast Cancer dataset and mouse olfactory daatset, you can follow the command to run
    
    
    python HBC.test.py 
    python Stereo_seq_test.py

# Datasets
<div>The human dorsolateral prefrontal cortex(DLPFC) dataset is available for download at http://research.libd.org/spatialLIBD/</div>
<div>The human breast cancer dataset is available for download at http://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Breast_Cancer_Block_A_Section_1</div>
<div>The mouse olfactory dataset obtained from Stereo-seq is available for download at http://github.com.JinmiaoChenLab/SEDR_analyses/</br>
