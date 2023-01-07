from __future__ import division
from __future__ import print_function

import os
import os.path as osp
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score
from scipy.sparse import csr_matrix
from normalization import fetch_normalization
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


from utils import load_citation, accuracy, clip_gradient, l_1_penalty
from models_PPI import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--kappa', type=float, default=0.98,
                    help='Projection parameter. ||W|| <= kappa/lpf(A)')
parser.add_argument('--dataset', type=str, default="PPI",
                        help='Dataset to use.')
parser.add_argument('--feature', type=str, default="mul",
                    choices=['mul', 'cat', 'adj'],
                    help='feature-type')
parser.add_argument('--normalization', type=str, default='AugNormAdj',
                   choices=['AugNormAdj'],
                   help='Normalization method for the adjacency matrix.')
parser.add_argument('--degree', type=int, default=2,
                    help='degree of the approximation.')
parser.add_argument('--per', type=int, default=-1,
                    help='Number of each nodes so as to balance.')
parser.add_argument('--experiment', type=str, default="base-experiment",
                    help='feature-type')
parser.add_argument('--model', type=str, default='EIGNN_m_nonlinear_X')
parser.add_argument('--fp_layer', type=str, default='EIGNN_m_MLP_v10')
parser.add_argument('--max_iter', type=int, default=300)
parser.add_argument('--threshold', type=float, default=1e-6)
parser.add_argument('--ks', type=str, default='[1,2]', help='a list of S^k, then concat for EIGNN_m_concat')
parser.add_argument('--num_layers', type=int, default=4, help='number of MLP layers')
parser.add_argument('--save_attention', type=int, default=0, help='whether to save the attention')

parser.add_argument('--gamma', type=float, default=0.8)
parser.add_argument('--path', type=str, default='./results_PPI/')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.path):
    os.mkdir(args.path)

ks = eval(args.ks)
ks_str = '-'.join(map(str, ks))

print('='*3, 'PPI dataset contains graphs with all symmetric adjacencies', '='*3)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'PPI')
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Model and optimizer
device = torch.device('cuda' if args.cuda else 'cpu')
model_t1 = time.time()
if args.model == 'MGNNI_m_att_stack':
    net = MGNNI_m_att_stack
    model = net(m=train_dataset.num_features,
                m_y=train_dataset.num_classes,
                nhid=args.hidden,
                ks=eval(args.ks),
                num_layers=args.num_layers,
                dropout=args.dropout,
                gamma=args.gamma,
                threshold=args.threshold,
                max_iter=args.max_iter).to(device)
    result_name = f'{args.model}_{args.dataset}_{args.epochs}_{ks_str}_l{args.num_layers}_{args.hidden}_dp{args.dropout}_{args.lr}_{args.gamma}.txt'
else:
    raise NotImplementedError(f'cannot find the model {args.model}')

file_name = os.path.join(args.path, result_name)
filep = open(file_name, 'w')
filep.write(str(args) + '\n')
#
# model = net(m=train_dataset.num_features,
#             m_y=train_dataset.num_classes,
#             nhid=args.hidden,
#             ks=eval(args.ks),
#             gamma=args.gamma,
#             threshold=args.threshold,
#             max_iter=args.max_iter,
#             fp_layer=args.fp_layer).to(device)

init_time = time.time() - model_t1

loss_op = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=32)

preprocess_time = 0
def train():
    model.train()

    total_loss = 0
    ys, preds = [], []
    for idx, data in enumerate(train_loader):
        # preload_adj_path = os.path.join(base_adj_path, adj_file_name+f'_train_{idx}.npz')
        num_graphs = data.num_graphs
        data.batch = None
        data = data.to(device)
        optimizer.zero_grad()
        if data.edge_attr is None:
            edge_weight = torch.ones((data.edge_index.size(1), ), dtype=data.x.dtype, device=data.edge_index.device)
        adj_sp = csr_matrix((edge_weight.cpu().numpy(), (data.edge_index[0,:].cpu().numpy(), data.edge_index[1,:].cpu().numpy() )), shape=(data.num_nodes, data.num_nodes))
        # symmetric = (abs(adj_sp - adj_sp.T) > 1e-10).nnz == 0
        # print(f'Whether sp_adj is symmetric: {symmetric}')
        adj_normalizer = fetch_normalization("AugNormAdj")
        adj_sp_nz = adj_normalizer(adj_sp)
        adj = torch.sparse.FloatTensor(torch.LongTensor(np.array([adj_sp_nz.row,adj_sp_nz.col])).to(device), torch.Tensor(adj_sp_nz.data).to(device), torch.Size([data.num_nodes, data.num_nodes])) #normalized adj
        # model.set_adj(adj, adj_sp_nz, preload_adj_path)

        loss = loss_op(model(data.x.T, adj), data.y)
        total_loss += loss.item() * num_graphs
        loss.backward()
        optimizer.step()
        # scheduler.step()

    # y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return total_loss / len(train_loader.dataset)

global save_attention_epoch
def test(loader, phase='test', save_attention=False):
    model.eval()

    ys, preds = [], []
    for idx, data in enumerate(loader):
        # preload_adj_path = os.path.join(base_adj_path, adj_file_name+f'_{phase}_{idx}.npz')
        ys.append(data.y)
        if data.edge_attr is None:
            edge_weight = torch.ones((data.edge_index.size(1), ), dtype=data.x.dtype, device=data.edge_index.device)
        adj_sp = csr_matrix((edge_weight.cpu().numpy(), (data.edge_index[0,:].cpu().numpy(), data.edge_index[1,:].cpu().numpy() )), shape=(data.num_nodes, data.num_nodes))
        adj_normalizer = fetch_normalization("AugNormAdj")
        adj_sp_nz = adj_normalizer(adj_sp)
        adj = torch.sparse.FloatTensor(torch.LongTensor(np.array([adj_sp_nz.row,adj_sp_nz.col])).to(device), torch.Tensor(adj_sp_nz.data).to(device), torch.Size([data.num_nodes, data.num_nodes])) #normalized adj
        # model.set_adj(adj, adj_sp_nz, preload_adj_path)
        with torch.no_grad():
            out = model(data.x.T.to(device), adj.to(device))
            if save_attention:
                att_vals = get_attention(data.x.T.to(device), adj.to(device))
                dir_path = './save_att_vals_PPI'
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                np.save(f'./save_att_vals_PPI/att_vals_{save_attention_epoch}.npy', att_vals.cpu().detach().numpy())
                filep.write(f'attention values: {att_vals}')
                print('attention values: ', att_vals)
            # ipdb.set_trace()
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

@torch.no_grad()
def get_attention(X, adj):
    model.eval()
    att_vals = model.get_att_vals(X, adj)
    return att_vals


best_f1 = 0
best_epoch = 0
t1 = time.time()
for epoch in range(1, int(args.epochs) + 1):
    t_train = time.time()
    loss = train()
    t_train_end = time.time()
    val_f1 = test(val_loader, 'val')
    save_attention = False
    if epoch % 200 == 0 and args.save_attention == 1:
        save_attention = True
        save_attention_epoch = epoch
    test_f1 = test(test_loader, 'test', save_attention)
    if test_f1 > best_f1:
        best_f1 = test_f1
        best_epoch = epoch
    print('Epoch: {:02d}, Train_Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}, Train_Time: {:.5f}s'.format(
        epoch, loss, val_f1, test_f1, t_train_end-t_train))
    filep.write('Epoch: {:02d}, Train_Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}, Train_Time: {:.5f}s\n'.format(
        epoch, loss, val_f1, test_f1, t_train_end-t_train))
    if epoch == 1:
        first_epoch_time = time.time() - t1
        print(f'first epoch time: {first_epoch_time}')

training_time = time.time() - t1
print('Best f1 micro is: {} at epoch {}'.format(best_f1, best_epoch))
filep.write('Best f1 micro is: {} at epoch {}\n'.format(best_f1, best_epoch))
print("training + init time elapsed: {:.4f}s".format(training_time + init_time))
filep.write("training + init elapsed: {:.4f}s\n".format(training_time + init_time))
print(f'{args.epochs} epochs, training time per epoch: {training_time/args.epochs}s')
filep.write(f'{args.epochs} epochs, training time per epoch: {training_time/args.epochs}s\n')