import random
import numpy as np
from torch.utils.data import Dataset
import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import os
import math
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import time
import platform
import torch.nn.functional as F
from .sCDCTA_model import *#scTrans_model as create_model
from .sCDCTA_model import scTrans_model as create_model
from sklearn.metrics import f1_score,precision_recall_fscore_support
    
def f1(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

    return (precision_macro, recall_macro, f1_macro), (precision_micro, recall_micro, f1_micro)
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

class MyDataSet(Dataset):
    """ 
    Preproces input matrix and labels.

    """
    def __init__(self, exp,embedding, label):
        self.exp = exp
        self.embedding = embedding
        self.label = label
        self.len = len(label)
    def __getitem__(self,index):
        
        return self.exp[index],self.embedding[index],self.label[index],index
    def __len__(self):
        return self.len
  
def get_gmt(gmt):
    import pathlib
    root = pathlib.Path(__file__).parent
    gmt_files = {
        "human_gobp": [root / "resources/GO_bp.gmt"],
        "human_immune": [root / "resources/immune.gmt"],
        "human_reactome": [root / "resources/reactome.gmt"],
        "human_tf": [root / "resources/TF.gmt"],
        "mouse_gobp": [root / "resources/m_GO_bp.gmt"],
        "mouse_reactome": [root / "resources/m_reactome.gmt"],
        "mouse_tf": [root / "resources/m_TF.gmt"]
    }
    return gmt_files[gmt][0]

def read_gmt(fname, sep='\t', min_g=0, max_g=5000):
    """
    Read GMT file into dictionary of gene_module:genes.\n
    min_g and max_g are optional gene set size filters.

    Args:
        fname (str): Path to gmt file
        sep (str): Separator used to read gmt file.
        min_g (int): Minimum of gene members in gene module.
        max_g (int): Maximum of gene members in gene module.
    Returns:
        OrderedDict: Dictionary of gene_module:genes.
    """
    dict_pathway = OrderedDict()
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            val = line.split(sep)
            if min_g <= len(val[2:]) <= max_g:
                dict_pathway[val[0]] = val[2:]
    return dict_pathway

def create_pathway_mask(feature_list, dict_pathway, add_missing=1, fully_connected=True, to_tensor=False):
    """
    Creates a mask of shape [genes,pathways] where (i,j) = 1 if gene i is in pathway j, 0 else.

    Expects a list of genes and pathway dict.
    Note: dict_pathway should be an Ordered dict so that the ordering can be later interpreted.

    Args:
        feature_list (list): List of genes in single-cell dataset.
        dict_pathway (OrderedDict): Dictionary of gene_module:genes.
        add_missing (int): Number of additional, fully connected nodes.
        fully_connected (bool): Whether to fully connect additional nodes or not.
        to_tensor (False): Whether to convert mask to tensor or not.
    Returns:
        torch.tensor/np.array: Gene module mask.
    """
    assert type(dict_pathway) == OrderedDict
    p_mask = np.zeros((len(feature_list), len(dict_pathway)))
    pathway = list()
    for j, k in enumerate(dict_pathway.keys()):
        pathway.append(k)
        for i in range(p_mask.shape[0]):
            if feature_list[i] in dict_pathway[k]:
                p_mask[i,j] = 1.
    if add_missing:
        n = 1 if type(add_missing)==bool else add_missing
        # Get non connected genes
        if not fully_connected:
            idx_0 = np.where(np.sum(p_mask, axis=1)==0)
            vec = np.zeros((p_mask.shape[0],n))
            vec[idx_0,:] = 1.
        else:
            vec = np.ones((p_mask.shape[0], n))
        p_mask = np.hstack((p_mask, vec))
        for i in range(n):
            x = 'node %d' % i
            pathway.append(x)
    if to_tensor:
        p_mask = torch.Tensor(p_mask)
    return p_mask,np.array(pathway)

def train_one_epoch_fin(model, optimizer, data_loader, device, epoch):
    """
    Train the model and updata weights.
    """
    model.train()
    loss_function = torch.nn.KLDivLoss(reduction='sum')
    accu_loss = torch.zeros(1).to(device) 
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        exp,embedding, label,index = data
        sample_num += exp.shape[0]
        _,pred,_ = model(exp.to(device),embedding.to(device),index)
        pred_classes = torch.max(pred, dim=1)[1]
        label_classes = torch.max(label, dim=1)[1]
        accu_num += torch.eq(pred_classes, label_classes.to(device)).sum()

        loss =  loss_function(F.log_softmax(pred, dim=1).float(), label.float().to(device))

        loss.backward()

        accu_loss += loss.detach()
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step() 
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate_2(model, data_loader, device, epoch):
    model.eval()
    data_loader = tqdm(data_loader)
    pred_all = None
    latent_all = None
    attn_weights_all = None
    for step, data in enumerate(data_loader):
        exp,embedding, label,index = data
        latent,pred,attn_weights = model(exp.to(device),embedding.to(device),index)
        if step == 0:
            pred_all = pred.cpu().numpy()
            latent_all = latent.cpu().numpy()
            attn_weights_all = attn_weights.cpu().numpy()
        else:
            pred_all = np.concatenate(((pred_all),(pred.cpu().numpy())),axis=0)
            latent_all = np.concatenate(((latent_all),(latent.cpu().numpy())),axis=0)
            attn_weights_all = np.concatenate(((attn_weights_all),(attn_weights.cpu().numpy())),axis=0)
    return latent_all, pred_all, attn_weights_all

def fit_model(adata, gmt_path, project = None, pre_weights='', label_name='Celltype',max_g=300,max_gs=300, mask_ratio = 0.015,n_unannotated = 1,batch_size=64, embed_dim=200,depth=3,num_heads=4,lr=0.005, epochs= 10, lrf=0.01):
  #  GLOBAL_SEED = 1
  #  set_seed(GLOBAL_SEED)  
    device = 'cuda:2'
    device = torch.device(device if torch.cuda.is_available() else "cpu") 
    print(device)
    today = time.strftime('%Y%m%d',time.localtime(time.time()))
    #train_weights = os.getcwd()+"/weights%s"%today
    project = project or gmt_path.replace('.gmt','')+'_%s'%today
    project_path = os.getcwd()+'/%s'%project#
    if os.path.exists(project_path) is False:
        os.makedirs(project_path)
    tb_writer = SummaryWriter()
    inverse, genes = set(adata.obs[label_name]), adata.var_names

    gmt_path = None
    if gmt_path is None:
        mask = np.random.binomial(1,mask_ratio,size=(len(genes), max_gs))
        pathway = list()
        for i in range(max_gs):
            x = 'node %d' % i
            pathway.append(x)
        print('Full connection!')
    else:
        if '.gmt' in gmt_path:
            gmt_path = gmt_path
        else:
            gmt_path = get_gmt(gmt_path)
        reactome_dict = read_gmt(gmt_path, min_g=0, max_g=max_g)
        mask,pathway = create_pathway_mask(feature_list=genes,
                                          dict_pathway=reactome_dict,
                                          add_missing=n_unannotated,
                                          fully_connected=True)
        pathway = pathway[np.sum(mask,axis=0)>4]
        mask = mask[:,np.sum(mask,axis=0)>4]
        pathway = pathway[sorted(np.argsort(np.sum(mask,axis=0))[-min(max_gs,mask.shape[1]):])]
        mask = mask[:,sorted(np.argsort(np.sum(mask,axis=0))[-min(max_gs,mask.shape[1]):])]
        print('Mask loaded!')

    np.save(project_path+'/mask.npy',mask)
    pd.DataFrame(pathway).to_csv(project_path+'/pathway_fin.csv') 
    pd.DataFrame(inverse,columns=[label_name]).to_csv(project_path+'/label_dictionary_fin.csv', quoting=None)
    num_classes = len(set(adata.obs["broad_cell_type"]))

    expdata = adata.X
    embedding = adata.uns["embedd"]
    data_norm = adata.uns['Celltype_soft']
    data_norm = F.softmax(torch.from_numpy(data_norm),dim=1)
    y_pre = torch.max(data_norm, dim=1)[1].numpy()
    thr = []
    for index in range(len(adata.uns['dicts_cell_type'])):
        temp = adata.uns['Celltype_soft'][np.array(y_pre)==index][:,index]
        thr.append(np.sort(temp)[int(len(temp)*0.8)])

    ext_index = []
    for index in range(len(adata.uns['Celltype_soft'])): 
        if adata.uns['Celltype_soft'][index][y_pre[index]]>thr[y_pre[index]]:
            ext_index.append(index)
    print(len(ext_index))
    adata_ext = adata[ext_index]
    adata_ext.uns['Celltype_soft'] = adata.uns['Celltype_soft'][ext_index]
    embedding_2 = adata.uns["embedd"][ext_index]

    model = create_model(num_classes=num_classes, num_genes=len(expdata[0]),  mask = mask,embed_dim=embed_dim,depth=depth,num_heads=num_heads,has_logits=False,sample_index = ext_index).to(device) 
    print(pre_weights+'\n')
    if pre_weights != "":
        assert os.path.exists(pre_weights), "pre_weights file: '{}' not exist.".format(pre_weights)
        preweights_dict = torch.load(pre_weights, map_location=device)
        print(model.load_state_dict(preweights_dict, strict=False))

    print('Model builded!')
    pg = [p for p in model.parameters() if p.requires_grad]  


    for epoch in range(epochs):
        if epoch<20:
            optimizer = optim.AdamW(pg, lr=0.0001, weight_decay=1E-5)

            test_dataset = MyDataSet(adata_ext.X,embedding_2, adata_ext.uns['Celltype_soft'])
            data_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    pin_memory=True,drop_last=True) #!!!

            train_loss, train_acc = train_one_epoch_fin(model=model,
                                        optimizer=optimizer,
                                        data_loader=data_loader,
                                        device=device,
                                        epoch=epoch)
            print(train_loss)
        else:
            if epoch==20:
                filepath = "/path_data/Muraro_rela.npz"
                loaded_matrix = sparse.load_npz(filepath).toarray() 
                reconstructed_matrix = sparse.csr_matrix(loaded_matrix)
                model.rela = reconstructed_matrix
            optimizer = optim.AdamW(pg, lr=0.000001, weight_decay=1E-7) 

            if epoch%1 == 0:
             #   list_index =[]
                test_dataset = MyDataSet(expdata,embedding , adata.uns['Celltype_soft'])
                data_loader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=batch_size, 
                                                        shuffle=False,
                                                        pin_memory=True,drop_last=False)             
                latent_all, pre_all, attn_weights_all = evaluate_2(model=model,
                                            data_loader=data_loader,
                                            device=device,
                                            epoch=epoch)    

                data_norm = F.softmax(torch.from_numpy(pre_all),dim=1).numpy()
                y_pre = torch.max(torch.from_numpy(data_norm), dim=1)[1].numpy()


                (precision_macro, recall_macro, f1_macro), (precision_micro, recall_micro, f1_micro)= np.round(f1(np.array(adata.obs[label_name]), np.array(y_pre)), 5) 
                print('--------------------')
                print('F1 score: f1_macro = {}, f1_micro = {}'.format(f1_macro, f1_micro))
                print('precision score: precision_macro = {}, precision_micro = {}'.format(precision_macro, precision_micro))
                print('recall score: recall_macro = {}, recall_micro = {}'.format(recall_macro, recall_micro))
                for value in set(adata.obs['broad_cell_type_2']):
                    y_pree, y_true = y_pre[adata.obs['broad_cell_type_2']==value], adata.obs[label_name][adata.obs['broad_cell_type_2']==value]
                    print(value+'-------------'+str(len(y_true)/len(y_pre)))
                    print('ACC: {}'.format(len(y_pree[y_pree==y_true])/len(y_true)))
            
            test_dataset = MyDataSet(expdata,embedding , data_norm)
            data_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    pin_memory=True,drop_last=False)
            train_loss, train_acc = train_one_epoch_fin(model=model,
                                        optimizer=optimizer,
                                        data_loader=data_loader,
                                        device=device,
                                        epoch=epoch)
            print(train_loss)
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
    print('Training finished!')