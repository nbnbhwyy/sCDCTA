import sCDCTA
import scanpy as sc
import numpy as np
import warnings 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import scipy.sparse as sp
warnings.filterwarnings ("ignore")
import torch
import torch.nn.functional as F
print(torch.__version__)
print(torch.cuda.get_device_capability(device=None),  torch.cuda.get_device_name(device=None))
import collections

our_query_adata = sc.read('/path_data/Muraro_CAS.h5ad')
print(our_query_adata)
embedding = pd.read_csv('/path_data/Muraro_embedding.csv')
our_query_adata.uns["embedd"] = embedding.values[:,1:]
sCDCTA.train(our_query_adata, gmt_path=None, label_name='broad_cell_type',epochs=50,project='hGOBP_demo',batch_size = 128)
