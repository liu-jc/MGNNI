import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from torch.nn import Parameter
from utils import get_spectral_rad, SparseDropout
import torch.sparse as sparse
from torch_geometric.nn import GCNConv, GATConv, SGConv, APPNP, JumpingKnowledge, GCN2Conv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, to_scipy_sparse_matrix
import numpy as np
from utils import *
import scipy.sparse as sp

class FSGNN(nn.Module):
    def __init__(self,nfeat,nlayers,nhidden,nclass,dropout):
        super(FSGNN,self).__init__()
        self.fc2 = nn.Linear(nhidden*nlayers,nclass)
        self.dropout = dropout
        self.act_fn = nn.ReLU()
        self.fc1 = nn.ModuleList([nn.Linear(nfeat,int(nhidden)) for _ in range(nlayers)])
        self.att = nn.Parameter(torch.ones(nlayers), requires_grad=False)
        self.sm = nn.Softmax(dim=0)



    def forward(self,list_mat,layer_norm):

        mask = self.sm(self.att)
        list_out = list()
        for ind, mat in enumerate(list_mat):
            tmp_out = self.fc1[ind](mat)
            if layer_norm == True:
                tmp_out = F.normalize(tmp_out,p=2,dim=1)
            tmp_out = torch.mul(mask[ind],tmp_out)

            list_out.append(tmp_out)

        final_mat = torch.cat(list_out, dim=1)
        out = self.act_fn(final_mat)
        out = F.dropout(out,self.dropout,training=self.training)
        out = self.fc2(out)

        return F.log_softmax(out, dim=1)

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

        x = features
        x = self.ig1(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T
        x = F.normalize(x, dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.V(x)
        return x

class IDM_SGC_Linear(nn.Module):
    def __init__(self, adj, sp_adj, m, m_y, num_eigenvec, gamma, all_eigenvec=True):
        super(IDM_SGC_Linear, self).__init__()
        self.B = nn.Parameter(1. / np.sqrt(m) * torch.randn(m_y, m), requires_grad=True)
        if all_eigenvec:
            self.IDM_SGC = IDM_SGC(adj, sp_adj, m, num_eigenvec, gamma)
        else:
            self.IDM_SGC = IDM_SGC_topk(adj, sp_adj, m, num_eigenvec, gamma)
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
        # return f(X, self.F, self.B)
        # TODO
        # Lambda_F, Q_F = torch.symeig(g(self.F), eigenvectors=True)
        # Lambda_F = Lambda_F.view(-1,1)
        # We can also save FF and FF_norm for backward to reduce cost a bit.
        # G = get_G(Lambda_F, self.Lambda_S, self.gamma)
        # ipdb.set_trace()
        # Z = Q_F @ (G * (Q_F.t() @ X @ self.Q_S)) @ self.Q_S.t()
        # ipdb.set_trace()
        # output = Z.t()
        # output = F.normalize(output, dim=-1)
        # output = F.dropout(output, 0.5, training=self.training)
        # output = self.B(output)
        output = self.IDM_SGC(X)
        output = self.B @ output
        return output.t()


class IDM_SGC_Linear_norm(nn.Module):
    def __init__(self, adj, sp_adj, m, m_y, num_eigenvec, gamma, all_eigenvec=True, dropout=0.5,
                 adj_preload_file=None):
        super(IDM_SGC_Linear_norm, self).__init__()
        self.B = nn.Parameter(1. / np.sqrt(m) * torch.randn(m_y, m), requires_grad=True)
        self.bn1 = nn.BatchNorm1d(m)
        self.dropout = dropout
        print(f'dropout: {dropout}')
        if all_eigenvec:
            self.IDM_SGC = IDM_SGC(adj, sp_adj, m, num_eigenvec, gamma,
                                   adj_preload_file=adj_preload_file)
        else:
            self.IDM_SGC = IDM_SGC_topk(adj, sp_adj, m, num_eigenvec, gamma)
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

    def get_Z_star(self, X):
        # return Z_star, normalized(Z_star)
        output = self.IDM_SGC(X).t()
        return output, F.normalize(output, dim=-1)

    def forward(self, X):
        output = self.IDM_SGC(X).t()
        # output = F.normalize(output, dim=-1)
        output = self.bn1(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        # output = self.B @ output
        output = output @ self.B.t()
        # return output.t()
        return output


class EIGNN_w_iterative(nn.Module):
    def __init__(self, adj, sp_adj, m, m_y, threshold, max_iter, gamma, chain_len, adaptive_gamma=False,
                 spectral_radius_mode=False, compute_jac_loss=False):
        super(EIGNN_w_iterative, self).__init__()
        # self.B = nn.Parameter(torch.FloatTensor(m_y, m), requires_grad=True)
        if not adaptive_gamma:
            self.EIGNN = EIGNN_w_iterative_solvers(adj, sp_adj, m, threshold, max_iter, gamma,
                                                   spectral_radius_mode=spectral_radius_mode, compute_jac_loss=compute_jac_loss)
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

    def reset_parameters(self):
        self.B.reset_parameters()
        self.EIGNN.reset_parameters()
        # torch.nn.init.xavier_uniform_(self.F)

    def forward(self, X):
        output, jac_loss = self.EIGNN(X)
        output = output.t()
        output = F.normalize(output, dim=-1)
        output = F.dropout(output, 0.5, training=self.training)
        output = self.B(output)
        # if self.training:
        #     print(f'gamma: {self.EIGNN.gamma}')
        return output, jac_loss



class EIGNN_finite(nn.Module):
    @staticmethod
    def g(F):
        FF = F.t() @ F
        FF_norm = torch.norm(FF, p='fro')
        return (1 / (FF_norm + epsilon_F)) * FF

    def __init__(self, adj, sp_adj, m, m_y, K, gamma, spectral_radius_mode=False, compute_jac_loss=False):
        super(EIGNN_finite, self).__init__()
        self.B = nn.Parameter(1. / np.sqrt(m) * torch.randn(m_y, m), requires_grad=True)
        # self.fc1 = nn.Linear(m, m)
        self.F = nn.Parameter(torch.FloatTensor(m, m), requires_grad=True)
        self.bn1 = nn.BatchNorm1d(m)
        # TODO:
        # define a finite model
        self.K = K
        self.gamma = gamma
        self.adj = adj.requires_grad_(False)
        self.sp_adj = sp_adj
        self.S_t = torch.transpose(self.adj, 0, 1)
        # self.S_t = self.adj

        # if not adaptive_gamma:
        #     self.EIGNN = EIGNN_w_iterative_solvers(adj, sp_adj, m, threshold, max_iter, gamma,
        #                                            spectral_radius_mode=spectral_radius_mode, compute_jac_loss=compute_jac_loss)
        # else:
        #     self.EIGNN = EIGNN_w_iter_adap_gamma(adj, sp_adj, m, threshold, max_iter, gamma, chain_len)
        # self.B = nn.Linear(m, m_y, bias=False)
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
        torch.nn.init.xavier_uniform_(self.F)
        # self.B.reset_parameters()
        # self.EIGNN.reset_parameters()
        # torch.nn.init.xavier_uniform_(self.F)

    def forward(self, X):
        X_first = X
        for i in range(self.K):
            # S_t = torch.transpose(self.S_k, 0, 1)
            # Z_new = self.gamma * g(self.F) @ torch.sparse.mm(S_t, Z.t()).t() + X
            # ipdb.set_trace()
            X = self.gamma * EIGNN_finite.g(self.F) @ torch.sparse.mm(self.S_t, X.t()).t() + X_first
        output = X.t()
        # output = F.normalize(output, dim=-1)
        output = self.bn1(output)
        output = F.dropout(output, 0.5, training=self.training)
        # output = F.relu(self.fc1(output))
        output = output @ self.B.t()
        # if self.training:
        #     print(f'gamma: {self.EIGNN.gamma}')
        return output


class EIGNN_multi_scale_concat(nn.Module):
    # multiscale, with S^k, the key point is to set decay step different from propagation step
    # concatenate S^1, S^2
    def __init__(self, adj, sp_adj, m, m_y, ks, threshold, max_iter, gamma, dropout=0.5,
                 layer_norm=False, batch_norm=False, learnable_alphas=False, spectral_radius_mode=False):
        super(EIGNN_multi_scale_concat, self).__init__()
        # self.B = nn.Parameter(torch.FloatTensor(m_y, m), requires_grad=True)
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.spectral_radius_mode = spectral_radius_mode
        self.EIGNNs = torch.nn.ModuleList()
        for k in ks:
            self.EIGNNs.append(EIGNN_scale_w_iter(adj, sp_adj, m, k, threshold, max_iter, gamma, layer_norm=layer_norm,
                                                  spectral_radius_mode=self.spectral_radius_mode))
            # self.EIGNN_1 = EIGNN_scale_w_iter(adj, sp_adj, m, 1, threshold, max_iter, gamma)
            # self.EIGNN_2 = EIGNN_scale_w_iter(adj, sp_adj, m, 2, threshold, max_iter, gamma)
        # self.B = nn.Linear(len(ks)*m, m_y, bias=False)
        # self.bn1 = nn.BatchNorm1d(len(ks)*m)
        self.learnable_alphas = learnable_alphas
        self.alphas = nn.Parameter(torch.ones(len(self.EIGNNs)), requires_grad=bool(self.learnable_alphas))
        self.sm = nn.Softmax(dim=0)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(len(ks)*m)
        self.B = nn.Parameter(1. / np.sqrt(m) * torch.randn(m_y, len(ks)*m), requires_grad=True)
        print(f'model config, Layer_norm:{self.layer_norm}, batch_norm: {self.batch_norm}, learnable_alphas: {self.learnable_alphas}')
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
        alphas = self.sm(self.alphas)
        for idx, model in enumerate(self.EIGNNs):
            tmp_output = model(X).t()
            # if self.layer_norm:
            #     tmp_output = F.normalize(tmp_output, p=2, dim=-1)
            if self.learnable_alphas:
                tmp_output = alphas[idx] * tmp_output
                print(f'alphas: {alphas}')
            EIGNN_outputs.append(tmp_output)
        output = torch.cat(EIGNN_outputs, dim=-1)
        # output = F.normalize(output, dim=-1)
        if self.batch_norm:
            output = self.bn1(output)
        output = F.dropout(output, self.dropout, training=self.training)
        # output = self.B(output)
        # ipdb.set_trace()
        output = output @ self.B.t()
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

class MGNNI_m_MLP(nn.Module):
    def __init__(self, m, m_y, nhid, ks, threshold, max_iter, gamma, fp_layer='MGNNI_m_att', dropout=0.5, batch_norm=False):
        super(MGNNI_m_MLP, self).__init__()
        # self.B = nn.Parameter(1. / np.sqrt(m) * torch.randn(m, nhid), requires_grad=True)
        # self.IDM_SGC = IDM_SGC(m, num_eigenvec, gamma)
        self.fc1 = nn.Linear(m, nhid)
        self.fc2 = nn.Linear(nhid, nhid)
        self.dropout = dropout
        try:
            self.MGNNI = eval(fp_layer)(nhid, m_y, ks, threshold, max_iter, gamma, dropout=dropout, batch_norm=batch_norm)
        except Exception:
            raise NotImplementedError(f'Cannot find the {fp_layer}')


    def forward(self, X, adj):
        X = F.dropout(X.t(), p=self.dropout, training=self.training)# (n, nfeat)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = self.fc2(X)
        output = self.MGNNI(X.t(), adj)
        return output

class MGNNI_m_att(nn.Module):
    # multiscale, with S^k, the key point is to set decay step different from propagation step
    # concatenate S^1, S^2
    def __init__(self, m, m_y, ks, threshold, max_iter, gamma, dropout=0.5,
                 layer_norm=False, batch_norm=False):
        super(MGNNI_m_att, self).__init__()
        # self.B = nn.Parameter(torch.FloatTensor(m_y, m), requires_grad=True)
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.MGNNIs = torch.nn.ModuleList()
        self.att = Attention(in_size=m)
        for k in ks:
            self.MGNNIs.append(MGNNI_m_iter(m, k, threshold, max_iter, gamma, layer_norm=layer_norm))
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(m)
        self.B = nn.Parameter(1. / np.sqrt(m) * torch.randn(m_y, m), requires_grad=True)
        print(f'model config, Layer_norm:{self.layer_norm}, batch_norm: {self.batch_norm}')
        self.reset_parameters()

    def reset_parameters(self):
        # self.B.reset_parameters()
        for i in range(len(self.MGNNIs)):
            self.MGNNIs[i].reset_parameters()
        # torch.nn.init.xavier_uniform_(self.F)


    def get_att_vals(self, X, adj):
        EIGNN_outputs = []
        for idx, model in enumerate(self.MGNNIs):
            tmp_output = model(X, adj).t()
            EIGNN_outputs.append(tmp_output)
            # output = torch.cat(EIGNN_outputs, dim=-1)
        output = torch.stack(EIGNN_outputs, dim=1) # (n, len(ks), nfeat)
        # output = F.normalize(output, dim=-1)
        att_vals = self.att(output) # (n, len(ks), 1)
        return att_vals

    def forward(self, X, adj):
        EIGNN_outputs = []
        for idx, model in enumerate(self.MGNNIs):
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
        self.gat2 = GATConv(heads*hidden, m_y, heads=1)

    def forward(self, x, edge_index):
        out = self.gat1(x, edge_index)
        out = F.elu(out)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.gat2(out, edge_index)
        return out

class SGC(nn.Module):
    def __init__(self, m, m_y, K):
        super(SGC, self).__init__()
        self.sgc = SGConv(m, m_y, K)

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