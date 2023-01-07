import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import ImplicitGraph, IDM_SGC, IDM_SGC_topk, EIGNN_w_iterative_solvers, EIGNN_new_iter, EIGNN_exact_u, EIGNN_forward_iter, EIGNN_w_iter_adap_gamma, EIGNN_scale_w_iter, EIGNN_m_iter
from torch.nn import Parameter
from utils import get_spectral_rad, SparseDropout
import torch.sparse as sparse
from torch_geometric.nn import GCNConv, GATConv, SGConv, APPNP, GCN2Conv, JumpingKnowledge, MessagePassing
from torch_geometric.utils import remove_self_loops, to_scipy_sparse_matrix
import numpy as np
from utils import *
import time

class IGNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_node, dropout, kappa=0.9, adj_orig=None):
        super(IGNN, self).__init__()

        self.adj = None
        self.adj_rho = None
        self.adj_orig = adj_orig

        #one layer with V
        self.ig1 = ImplicitGraph(nfeat, nhid, num_node, kappa)
        self.dropout = dropout
        self.X_0 = Parameter(torch.zeros(nhid, num_node), requires_grad=False)
        self.V = nn.Linear(nhid, nclass, bias=False)

    def get_Z_star(self, features, adj):
        # return Z_star, normalized(Z_star)
        x = features
        x = self.ig1(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T
        # output = self.EIGNN(X).t()
        output = x
        return output, F.normalize(output, dim=-1)

    def forward(self, features, adj):
        if adj is not self.adj:
            self.adj = adj
            self.adj_rho = get_spectral_rad(adj)
            # ipdb.set_trace()
        ipdb.set_trace()
        x = features
        x = self.ig1(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T
        x = F.normalize(x, dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.V(x)
        return x


class IGNN_finite(nn.Module):
    def __init__(self, m, m_y, nhid, K, dropout):
        super(IGNN_finite, self).__init__()
        self.lin1 = nn.Linear(m, nhid, bias=False)
        self.lin2 = nn.Linear(nhid, m_y, bias=False)
        self.num_layers = K
        self.hid_layer = nn.Linear(nhid, nhid, bias=False)
        self.dropout = dropout
        # self.prop1 = APPNP(K=K, alpha=alpha)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.hid_layer.reset_parameters()

    def forward(self, x, adj):
        # x: (f,n), lin1: (f->h)
        # lin2 (h -> m_y), hid_layer(h -> h)
        x_first = self.lin1(x.T).T
        x = x_first
        # ipdb.set_trace()
        for _ in range(self.num_layers):
            # tmp = torch.spmm(self.hid_layer(x.T).T, adj) + x_first
            tmp = torch.spmm(torch.transpose(adj, 0, 1), self.hid_layer(x.T)).T + x_first
            x = F.relu(tmp)
        x = x.T
        x = F.normalize(x, dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin2(x)
        return x


class IDM_SGC_Linear(nn.Module):
    def __init__(self, adj, sp_adj, m, m_y, num_eigenvec, gamma):
        super(IDM_SGC_Linear, self).__init__()
        # self.B = nn.Parameter(torch.FloatTensor(m_y, m), requires_grad=True)
        self.IDM_SGC = IDM_SGC(adj, sp_adj, m, num_eigenvec, gamma)
        self.B = nn.Linear(m, m_y, bias=False)
        # self.F = nn.Parameter(torch.FloatTensor(m, m), requires_grad=True)
        # S_dense = S.to_dense()
        # self.S = S
        # self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float), requires_grad=False)
        # self.Lambda_S, self.Q_S = torch.symeig(S_dense, eigenvectors=True)
        # self.Lambda_S = self.Lambda_S.view(-1,1)
        #
        # del S_dense
        self.reset_parameters()

    def reset_parameters(self):
        self.B.reset_parameters()
        self.IDM_SGC.reset_parameters()
        # torch.nn.init.xavier_uniform_(self.F)

    def get_Z_star(self, X):
        # return Z_star, normalized(Z_star)
        output = self.IDM_SGC(X).t()
        return output, F.normalize(output, dim=-1)

    def forward(self, X):
        output = self.IDM_SGC(X).t()
        output = F.normalize(output, dim=-1)
        output = F.dropout(output, 0.5, training=self.training)
        output = self.B(output)
        return output

class EIGNN_w_iterative(nn.Module):
    def __init__(self, adj, sp_adj, m, m_y, threshold, max_iter, gamma, chain_len, adaptive_gamma=False):
        super(EIGNN_w_iterative, self).__init__()
        # self.B = nn.Parameter(torch.FloatTensor(m_y, m), requires_grad=True)
        if not adaptive_gamma:
            self.EIGNN = EIGNN_w_iterative_solvers(adj, sp_adj, m, threshold, max_iter, gamma)
        else:
            self.EIGNN = EIGNN_w_iter_adap_gamma(adj, sp_adj, m, threshold, max_iter, gamma, chain_len)
        self.B = nn.Linear(m, m_y, bias=False)
        # self.F = nn.Parameter(torch.FloatTensor(m, m), requires_grad=True)
        # S_dense = S.to_dense()
        # self.S = S
        # self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float), requires_grad=False)
        # self.Lambda_S, self.Q_S = torch.symeig(S_dense, eigenvectors=True)
        # self.Lambda_S = self.Lambda_S.view(-1,1)
        #
        # del S_dense
        self.reset_parameters()

    def get_Z_star(self, X):
        # return Z_star, normalized(Z_star)
        output = self.EIGNN(X).t()
        return output, F.normalize(output, dim=-1)

    def reset_parameters(self):
        self.B.reset_parameters()
        self.EIGNN.reset_parameters()
        # torch.nn.init.xavier_uniform_(self.F)

    def forward(self, X):
        # output, jac_loss = self.EIGNN(X).t()
        output = self.EIGNN(X).t()
        output = F.normalize(output, dim=-1)
        output = F.dropout(output, 0.5, training=self.training)
        output = self.B(output)
        # if self.training:
        #     print(f'gamma: {self.EIGNN.gamma}')
        # return output, jac_loss
        return output


class EIGNN_multi_scale(nn.Module):
    # multiscale, with S^k, the key point is to set decay step different from propagation step
    def __init__(self, adj, sp_adj, m, m_y, k, threshold, max_iter, gamma):
        super(EIGNN_multi_scale, self).__init__()
        # self.B = nn.Parameter(torch.FloatTensor(m_y, m), requires_grad=True)
        self.EIGNN = EIGNN_scale_w_iter(adj, sp_adj, m, k, threshold, max_iter, gamma)
        self.B = nn.Linear(m, m_y, bias=False)
        # self.F = nn.Parameter(torch.FloatTensor(m, m), requires_grad=True)
        # S_dense = S.to_dense()
        # self.S = S
        # self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float), requires_grad=False)
        # self.Lambda_S, self.Q_S = torch.symeig(S_dense, eigenvectors=True)
        # self.Lambda_S = self.Lambda_S.view(-1,1)
        #
        # del S_dense
        self.reset_parameters()

    def reset_parameters(self):
        self.B.reset_parameters()
        self.EIGNN.reset_parameters()
        # torch.nn.init.xavier_uniform_(self.F)

    def forward(self, X):
        output = self.EIGNN(X).t()
        output = F.normalize(output, dim=-1)
        output = F.dropout(output, 0.5, training=self.training)
        output = self.B(output)
        return output


class EIGNN_multi_scale_concat(nn.Module):
    # multiscale, with S^k, the key point is to set decay step different from propagation step
    # concatenate S^1, S^2
    def __init__(self, adj, sp_adj, m, m_y, ks, threshold, max_iter, gamma):
        super(EIGNN_multi_scale_concat, self).__init__()
        # self.B = nn.Parameter(torch.FloatTensor(m_y, m), requires_grad=True)
        self.EIGNNs = torch.nn.ModuleList()
        for k in ks:
            self.EIGNNs.append(EIGNN_scale_w_iter(adj, sp_adj, m, k, threshold, max_iter, gamma))
            # self.EIGNN_1 = EIGNN_scale_w_iter(adj, sp_adj, m, 1, threshold, max_iter, gamma)
            # self.EIGNN_2 = EIGNN_scale_w_iter(adj, sp_adj, m, 2, threshold, max_iter, gamma)
        self.B = nn.Linear(len(ks)*m, m_y, bias=False)
        # self.F = nn.Parameter(torch.FloatTensor(m, m), requires_grad=True)
        # S_dense = S.to_dense()
        # self.S = S
        # self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float), requires_grad=False)
        # self.Lambda_S, self.Q_S = torch.symeig(S_dense, eigenvectors=True)
        # self.Lambda_S = self.Lambda_S.view(-1,1)
        #
        # del S_dense
        self.reset_parameters()

    def reset_parameters(self):
        self.B.reset_parameters()
        for i in range(len(self.EIGNNs)):
            self.EIGNNs[i].reset_parameters()
        # torch.nn.init.xavier_uniform_(self.F)

    def forward(self, X):
        EIGNN_outputs = [model(X).t() for model in self.EIGNNs]
        output = torch.cat(EIGNN_outputs, dim=-1)
        output = F.normalize(output, dim=-1)
        output = F.dropout(output, 0.5, training=self.training)
        output = self.B(output)
        return output


class Attention(nn.Module):
    # https://github.com/jindi-tju/U-GCN/blob/1bb9f95d2c4bfdb1e0a45de461baced95114d688/models.py#L47
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        # return (beta * z).sum(1), beta
        return beta

class EIGNN_multi_scale_att(nn.Module):
    # multiscale, with S^k, the key point is to set decay step different from propagation step
    # concatenate S^1, S^2
    def __init__(self, adj, sp_adj, m, m_y, ks, threshold, max_iter, gamma, dropout=0.5,
                 layer_norm=False, batch_norm=False, learnable_alphas=False, spectral_radius_mode=False):
        super(EIGNN_multi_scale_att, self).__init__()
        # self.B = nn.Parameter(torch.FloatTensor(m_y, m), requires_grad=True)
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.spectral_radius_mode = spectral_radius_mode
        self.EIGNNs = torch.nn.ModuleList()
        self.att = Attention(in_size=m)
        for k in ks:
            self.EIGNNs.append(EIGNN_scale_w_iter(adj, sp_adj, m, k, threshold, max_iter, gamma, layer_norm=layer_norm,
                                                  spectral_radius_mode=self.spectral_radius_mode))
        # self.B = nn.Linear(len(ks)*m, m_y, bias=False)
        # self.bn1 = nn.BatchNorm1d(len(ks)*m)
        # self.learnable_alphas = learnable_alphas
        # self.alphas = nn.Parameter(torch.ones(len(self.EIGNNs)), requires_grad=bool(self.learnable_alphas))
        # self.sm = nn.Softmax(dim=0)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(m)
        self.B = nn.Parameter(1. / np.sqrt(m) * torch.randn(m_y, m), requires_grad=True)
        print(f'model config, Layer_norm:{self.layer_norm}, batch_norm: {self.batch_norm}')
        self.reset_parameters()

    def reset_parameters(self):
        # self.B.reset_parameters()
        for i in range(len(self.EIGNNs)):
            self.EIGNNs[i].reset_parameters()
        # torch.nn.init.xavier_uniform_(self.F)

    def get_Z_star(self, X):
        # return Z_star, normalized(Z_star)
        output = self.EIGNNs[-1](X).t()
        return output, F.normalize(output, dim=-1)


    def forward(self, X):
        # EIGNN_outputs = [model(X).t() for model in self.EIGNNs]
        EIGNN_outputs = []
        for idx, model in enumerate(self.EIGNNs):
            tmp_output = model(X).t()
            EIGNN_outputs.append(tmp_output)
        # output = torch.cat(EIGNN_outputs, dim=-1)
        output = torch.stack(EIGNN_outputs, dim=1) # (n, len(ks), nfeat)
        # output = F.normalize(output, dim=-1)
        att_vals = self.att(output) # (n, len(ks), 1)
        output = (output * att_vals).sum(1)
        # if not self.training:
        # print(f'attention vals[:50]: {att_vals[:50]}')
        if self.batch_norm:
            output = self.bn1(output)
        output = F.dropout(output, self.dropout, training=self.training)
        # output = self.B(output)
        # ipdb.set_trace()
        output = output @ self.B.t()
        return output



class EIGNN_m_MLP_wo_B(nn.Module):
    def __init__(self, m, m_y, nhid, ks, threshold, max_iter, gamma, fp_layer='EIGNN_m_att', dropout=0.5, batch_norm=False):
        super(EIGNN_m_MLP_wo_B, self).__init__()
        # self.B = nn.Parameter(1. / np.sqrt(m) * torch.randn(m, nhid), requires_grad=True)
        # self.IDM_SGC = IDM_SGC(m, num_eigenvec, gamma)
        self.fc1 = nn.Linear(m, nhid)
        self.fc2 = nn.Linear(nhid, m_y)
        self.dropout = dropout
        try:
            self.EIGNN = eval(fp_layer)(m_y, m_y, ks, threshold, max_iter, gamma, dropout=dropout, batch_norm=batch_norm)
        except Exception:
            raise NotImplementedError(f'Cannot find the {fp_layer}')

    def forward(self, X, adj):
        X = F.dropout(X.t(), p=self.dropout, training=self.training)# (n, nfeat)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = self.fc2(X)
        output = self.EIGNN(X.t(), adj)
        return output

class EIGNN_m_att_wo_B(nn.Module):
    # multiscale, with S^k, the key point is to set decay step different from propagation step
    # concatenate S^1, S^2
    def __init__(self, m, m_y, ks, threshold, max_iter, gamma, dropout=0.5,
                 layer_norm=False, batch_norm=False, spectral_radius_mode=False):
        super(EIGNN_m_att_wo_B, self).__init__()
        # self.B = nn.Parameter(torch.FloatTensor(m_y, m), requires_grad=True)
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.spectral_radius_mode = spectral_radius_mode
        self.EIGNNs = torch.nn.ModuleList()
        self.att = Attention(in_size=m)
        for k in ks:
            self.EIGNNs.append(EIGNN_m_iter(m, k, threshold, max_iter, gamma, layer_norm=layer_norm))
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(m)
        self.B = nn.Parameter(1. / np.sqrt(m) * torch.randn(m_y, m), requires_grad=True)
        print(f'model config, Layer_norm:{self.layer_norm}, batch_norm: {self.batch_norm}')
        self.reset_parameters()

    def reset_parameters(self):
        # self.B.reset_parameters()
        for i in range(len(self.EIGNNs)):
            self.EIGNNs[i].reset_parameters()
        # torch.nn.init.xavier_uniform_(self.F)

    def get_Z_star(self, X):
        # return Z_star, normalized(Z_star)
        output = self.EIGNNs[-1](X).t()
        return output, F.normalize(output, dim=-1)

    def forward(self, X, adj):
        # EIGNN_outputs = [model(X).t() for model in self.EIGNNs]
        EIGNN_outputs = []
        for idx, model in enumerate(self.EIGNNs):
            tmp_output = model(X, adj).t()
            EIGNN_outputs.append(tmp_output)
        output = torch.stack(EIGNN_outputs, dim=1) # (n, len(ks), nfeat)
        att_vals = self.att(output) # (n, len(ks), 1)
        output = (output * att_vals).sum(1)
        return output

class EIGNN_m_MLP(nn.Module):
    def __init__(self, m, m_y, nhid, ks, threshold, max_iter, gamma, fp_layer='EIGNN_m_att', dropout=0.5, batch_norm=False):
        super(EIGNN_m_MLP, self).__init__()
        # self.B = nn.Parameter(1. / np.sqrt(m) * torch.randn(m, nhid), requires_grad=True)
        # self.IDM_SGC = IDM_SGC(m, num_eigenvec, gamma)
        self.fc1 = nn.Linear(m, nhid)
        self.fc2 = nn.Linear(nhid, nhid)
        self.dropout = dropout
        try:
            self.EIGNN = eval(fp_layer)(nhid, m_y, ks, threshold, max_iter, gamma, dropout=dropout, batch_norm=batch_norm)
        except Exception:
            raise NotImplementedError(f'Cannot find the {fp_layer}')

    def forward(self, X, adj):
        X = F.dropout(X.t(), p=self.dropout, training=self.training)# (n, nfeat)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = self.fc2(X)
        output = self.EIGNN(X.t(), adj)
        return output

class EIGNN_m_att(nn.Module):
    # multiscale, with S^k, the key point is to set decay step different from propagation step
    # concatenate S^1, S^2
    def __init__(self, m, m_y, ks, threshold, max_iter, gamma, dropout=0.5,
                 layer_norm=False, batch_norm=False, spectral_radius_mode=False):
        super(EIGNN_m_att, self).__init__()
        # self.B = nn.Parameter(torch.FloatTensor(m_y, m), requires_grad=True)
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.spectral_radius_mode = spectral_radius_mode
        self.EIGNNs = torch.nn.ModuleList()
        self.att = Attention(in_size=m)
        for k in ks:
            self.EIGNNs.append(EIGNN_m_iter(m, k, threshold, max_iter, gamma, layer_norm=layer_norm))
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(m)
        self.B = nn.Parameter(1. / np.sqrt(m) * torch.randn(m_y, m), requires_grad=True)
        print(f'model config, Layer_norm:{self.layer_norm}, batch_norm: {self.batch_norm}')
        self.reset_parameters()

    def reset_parameters(self):
        # self.B.reset_parameters()
        for i in range(len(self.EIGNNs)):
            self.EIGNNs[i].reset_parameters()
        # torch.nn.init.xavier_uniform_(self.F)

    def get_Z_star(self, X):
        # return Z_star, normalized(Z_star)
        output = self.EIGNNs[-1](X).t()
        return output, F.normalize(output, dim=-1)

    def forward(self, X, adj):
        # EIGNN_outputs = [model(X).t() for model in self.EIGNNs]
        EIGNN_outputs = []
        for idx, model in enumerate(self.EIGNNs):
            tmp_output = model(X, adj).t()
            EIGNN_outputs.append(tmp_output)
        # output = torch.cat(EIGNN_outputs, dim=-1)
        output = torch.stack(EIGNN_outputs, dim=1) # (n, len(ks), nfeat)
        # output = F.normalize(output, dim=-1)
        att_vals = self.att(output) # (n, len(ks), 1)
        output = (output * att_vals).sum(1)
        if self.batch_norm:
            output = self.bn1(output)
        output = F.dropout(output, self.dropout, training=self.training)
        output = output @ self.B.t()
        return output

class EIGNN_new_iter_model(nn.Module):
    def __init__(self, adj, sp_adj, m, m_y, threshold, max_iter, gamma, solver, solver_mode='backward'):
        print('#'*10, f'using EIGNN_new_iter_model with the {solver} solver, solver_mode: {solver_mode}', '#'*10)
        print('max_iter: ', max_iter)
        super(EIGNN_new_iter_model, self).__init__()
        # self.B = nn.Parameter(torch.FloatTensor(m_y, m), requires_grad=True)
        if solver_mode == 'backward': # use a solver for backward and closed-form solution for forward.
            self.EIGNN = EIGNN_new_iter(adj, sp_adj, m, threshold, max_iter, gamma, solver)
        elif solver_mode == 'forward': # use a solver for forward and closed-form solution for backward.
            self.EIGNN = EIGNN_forward_iter(adj, sp_adj, m, threshold, max_iter, gamma, solver)
        self.B = nn.Linear(m, m_y, bias=True)
        self.max_iter = max_iter
        self.reset_parameters()

    def get_Z_star(self, X):
        # return Z_star, normalized(Z_star)
        output = self.EIGNN(X).t()
        return output, F.normalize(output, dim=-1)

    def reset_parameters(self):
        self.B.reset_parameters()
        self.EIGNN.reset_parameters()
        # torch.nn.init.xavier_uniform_(self.F)

    def forward(self, X):
        output = self.EIGNN(X).t()
        output = F.normalize(output, dim=-1)
        output = F.dropout(output, 0.5, training=self.training)
        output = self.B(output)
        return output

class EIGNN_exact_model(nn.Module):
    def __init__(self, adj, sp_adj, m, m_y, threshold, max_iter, gamma, solver):
        print('#'*10, f'using EIGNN_exact_model (exact solutions for both forward and backward)', '#'*10)
        print('max_iter: ', max_iter)
        super(EIGNN_exact_model, self).__init__()
        # self.B = nn.Parameter(torch.FloatTensor(m_y, m), requires_grad=True)
        self.EIGNN = EIGNN_exact_u(adj, sp_adj, m, threshold, max_iter, gamma, solver)
        self.B = nn.Linear(m, m_y, bias=True)
        self.max_iter = max_iter
        self.reset_parameters()

    def reset_parameters(self):
        self.B.reset_parameters()
        self.EIGNN.reset_parameters()
        # torch.nn.init.xavier_uniform_(self.F)

    def forward(self, X):
        output = self.EIGNN(X).t()
        output = F.normalize(output, dim=-1)
        output = F.dropout(output, 0.5, training=self.training)
        output = self.B(output)
        return output


class IDM_SGC_Linear_topk(nn.Module):
    def __init__(self, adj, sp_adj, m, m_y, num_eigenvec, gamma):
        super(IDM_SGC_Linear_topk, self).__init__()
        # self.B = nn.Parameter(torch.FloatTensor(m_y, m), requires_grad=True)
        self.IDM_SGC = IDM_SGC_topk(adj, sp_adj, m, num_eigenvec, gamma)
        self.B = nn.Linear(m, m_y, bias=False)
        # self.F = nn.Parameter(torch.FloatTensor(m, m), requires_grad=True)
        # S_dense = S.to_dense()
        # self.S = S
        # self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float), requires_grad=False)
        # self.Lambda_S, self.Q_S = torch.symeig(S_dense, eigenvectors=True)
        # self.Lambda_S = self.Lambda_S.view(-1,1)
        #
        # del S_dense
        self.reset_parameters()

    def reset_parameters(self):
        self.B.reset_parameters()
        self.IDM_SGC.reset_parameters()
        # torch.nn.init.xavier_uniform_(self.F)

    def forward(self, X):
        output = self.IDM_SGC(X).t()
        output = F.normalize(output, dim=-1)
        output = F.dropout(output, 0.5, training=self.training)
        output = self.B(output)
        return output


class IDM_SGC_Linear_norm(nn.Module):
    def __init__(self, adj, sp_adj, m, m_y, num_eigenvec, gamma):
        super(IDM_SGC_Linear_norm, self).__init__()
        self.B = nn.Parameter(1. / np.sqrt(m) * torch.randn(m_y, m), requires_grad=True)
        self.bn1 = nn.BatchNorm1d(m)
        self.IDM_SGC = IDM_SGC(adj, sp_adj, m, num_eigenvec, gamma)
        # self.B = nn.Linear(m, m_y, bias=False)
        # self.F = nn.Parameter(torch.FloatTensor(m, m), requires_grad=True)
        # S_dense = S.to_dense()
        # self.S = S
        # self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float), requires_grad=False)
        # self.Lambda_S, self.Q_S = torch.symeig(S_dense, eigenvectors=True)
        # self.Lambda_S = self.Lambda_S.view(-1,1)
        #
        # del S_dense
        # self.reset_parameters()

    def reset_parameters(self):
        # self.B.reset_parameters()
        self.IDM_SGC.reset_parameters()
        # torch.nn.init.xavier_uniform_(self.F)

    def forward(self, X):
        output = self.IDM_SGC(X).t()
        # output = F.normalize(output, dim=-1)
        output = self.bn1(output)
        output = F.dropout(output, 0.5, training=self.training)
        # output = self.B @ output
        output = output @ self.B.t()
        # return output.t()
        return output


class IDM_SGC_MLP(nn.Module):
    def __init__(self, adj, sp_adj, m, m_y, nhid, num_eigenvec, gamma):
        super(IDM_SGC_MLP, self).__init__()
        # self.B = nn.Parameter(torch.FloatTensor(m_y, m), requires_grad=True)
        self.IDM_SGC = IDM_SGC(adj, sp_adj, m, num_eigenvec, gamma)
        self.B = nn.Linear(m, nhid, bias=True)
        self.fc = nn.Linear(nhid, m_y, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.B.reset_parameters()
        self.IDM_SGC.reset_parameters()
        # torch.nn.init.xavier_uniform_(self.F)

    def forward(self, X):
        output = self.IDM_SGC(X).t()
        output = F.normalize(output, dim=-1)
        output = F.dropout(output, 0.5, training=self.training)
        output = self.B(output)
        output = F.relu(output)
        output = self.fc(output)
        return output

epsilon_F = 10**(-12)
def g(F):
    FF = F.t() @ F
    FF_norm = torch.norm(FF, p='fro')
    return (1/(FF_norm+epsilon_F)) * FF


def get_G(Lambda_F, Lambda_S, gamma):
    G = 1.0 - gamma * Lambda_F @ Lambda_S.t()
    G = 1 / G
    return G


class GCN(nn.Module):
    def __init__(self, m, m_y, hidden):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(m, hidden)
        self.gc2 = GCNConv(hidden, m_y)

    def forward(self, x, edge_index):
        out = self.gc1(x, edge_index)
        out = F.relu(out)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.gc2(out, edge_index)
        return out

class GAT(nn.Module):
    def __init__(self, m, m_y, hidden, heads):
        super(GAT, self).__init__()
        self.gat1 = GATConv(m, hidden, heads=heads)
        self.gat2 = GATConv(heads*hidden, m_y, heads=heads)

    def forward(self, x, edge_index):
        out = self.gat1(x, edge_index)
        out = F.elu(out)
        out = F.dropout(out, p=0.8, training=self.training)
        out = self.gat2(out, edge_index)
        return out

class SGC(nn.Module):
    def __init__(self, m, m_y, K):
        super(SGC, self).__init__()
        self.sgc = SGConv(m, m_y, K)
        self.reset_parameters()

    def reset_parameters(self):
        self.sgc.reset_parameters()

    def forward(self, x, edge_index):
        out = self.sgc(x, edge_index)
        return out

class APPNP_Net(nn.Module):
    def __init__(self, m, m_y, nhid, K, alpha):
        super(APPNP_Net, self).__init__()
        self.lin1 = nn.Linear(m, nhid)
        self.lin2 = nn.Linear(nhid, m_y)
        self.prop1 = APPNP(K=K, alpha=alpha)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        out = self.prop1(x, edge_index)
        return out



class GCN_JKNet(torch.nn.Module):
    def __init__(self, m, m_y, hidden, layers=8):
        in_channels = m
        out_channels = m_y

        super(GCN_JKNet, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden))
        for _ in range(layers-1):
            self.convs.append(GCNConv(hidden, hidden))
        # self.conv1 = GCNConv(in_channels, hidden)
        # self.conv2 = GCNConv(hidden, hidden)
        self.lin1 = nn.Linear(layers*hidden, out_channels)
        # self.lin1 = torch.nn.Linear(64, out_channels)
        # self.one_step = APPNP(K=1, alpha=0)
        # self.JK = JumpingKnowledge(mode='lstm',
        #                            channels=64,
        #                            num_layers=4)
        self.JK = JumpingKnowledge(mode='cat')

    def forward(self, x, edge_index):

        final_xs = []
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            final_xs.append(x)

        x = self.JK(final_xs)
        # x1 = F.relu(self.conv1(x, edge_index))
        # x1 = F.dropout(x1, p=0.5, training=self.training)
        #
        # x2 = F.relu(self.conv2(x1, edge_index))
        # x2 = F.dropout(x2, p=0.5, training=self.training)
        # x = self.JK([x1, x2])
        # x = self.one_step(x, edge_index)
        x = self.lin1(x)
        return x

class GCNII_Model(torch.nn.Module):
    def __init__(self, m, m_y, hidden=64, layers=64, alpha=0.5, theta=1.):
        super(GCNII_Model, self).__init__()
        self.lin1 = nn.Linear(m, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(layers):
            self.convs.append(GCN2Conv(channels=hidden,
                                       alpha=alpha, theta=theta, layer=i+1))
        self.lin2 = nn.Linear(hidden, m_y)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        x_0 = x
        for conv in self.convs:
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(conv(x, x_0, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        out = self.lin2(x)
        return out


class H2GCN_Prop(MessagePassing):
    def __init__(self):
        super(H2GCN_Prop, self).__init__()

    def forward(self, h, norm_adj_1hop, norm_adj_2hop):
        h_1 = torch.sparse.mm(norm_adj_1hop, h) # if OOM, consider using torch-sparse
        h_2 = torch.sparse.mm(norm_adj_2hop, h)
        h = torch.cat((h_1, h_2), dim=1)
        return h


class H2GCN(torch.nn.Module):
    def __init__(self, m, m_y, hidden, edge_index, dropout=0.5, act='relu'):
        super(H2GCN, self).__init__()
        self.dropout = dropout
        self.lin1 = nn.Linear(m, hidden, bias=False)
        self.act = torch.nn.ReLU() if act == 'relu' else torch.nn.Identity()
        self.H2GCN_layer = H2GCN_Prop()
        self.num_layers = 1
        self.lin_final = nn.Linear((2**(self.num_layers+1)-1)*hidden, m_y, bias=False)
        # self.lin_final = nn.Linear((self.num_layers+1)*hidden, m_y, bias=False)

        adj = to_scipy_sparse_matrix(remove_self_loops(edge_index)[0])
        adj_2hop = adj.dot(adj)
        adj_2hop = adj_2hop - sp.diags(adj_2hop.diagonal())
        adj = indicator_adj(adj)
        adj_2hop = indicator_adj(adj_2hop)
        norm_adj_1hop = get_normalized_adj(adj)
        self.norm_adj_1hop = sparse_mx_to_torch_sparse_tensor(norm_adj_1hop, 'cuda')
        norm_adj_2hop = get_normalized_adj(adj_2hop)
        self.norm_adj_2hop = sparse_mx_to_torch_sparse_tensor(norm_adj_2hop, 'cuda')

    def forward(self, x, edge_index=None):
        hidden_hs = []
        h = self.act(self.lin1(x))
        hidden_hs.append(h)
        for i in range(self.num_layers):
            h = self.H2GCN_layer(h, self.norm_adj_1hop, self.norm_adj_2hop)
            hidden_hs.append(h)
        h_final = torch.cat(hidden_hs, dim=1)
        # print(f'lin_final.size(): {self.lin_final.weight.size()}, h_final.size(): {h_final.size()}')
        h_final = F.dropout(h_final, p=self.dropout, training=self.training)
        output = self.lin_final(h_final)
        return output