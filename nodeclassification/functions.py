import torch
import numpy as np
import scipy.sparse as sp
from torch.autograd import Function
import torch.nn.functional as F
from utils import sparse_mx_to_torch_sparse_tensor
import ipdb
import pickle

class ImplicitFunction(Function):
    #ImplicitFunction.apply(input, A, U, self.X_0, self.W, self.Omega_1, self.Omega_2)
    @staticmethod
    def forward(ctx, W, X_0, A, B, phi, fd_mitr=300, bw_mitr=300):
        X_0 = B if X_0 is None else X_0
        X, err, status, D, results = ImplicitFunction.inn_pred(W, X_0, A, B, phi, mitr=fd_mitr, compute_dphi=True, name='Forward')
        # print(f'err: {err}')
        ctx.save_for_backward(W, X, A, B, D, X_0, torch.tensor(bw_mitr))
        if status not in "converged":
            print("Iterations not converging!", err, status)
        return X

    @staticmethod
    def backward(ctx, *grad_outputs):

        #import pydevd
        #pydevd.settrace(suspend=False, trace_only_current_thread=True)

        W, X, A, B, D, X_0, bw_mitr = ctx.saved_tensors
        bw_mitr = bw_mitr.cpu().numpy()
        grad_x = grad_outputs[0]

        dphi = lambda X: torch.mul(X, D)
        grad_z, err, status, _, results = ImplicitFunction.inn_pred(W.T, X_0, A, grad_x, dphi, mitr=bw_mitr, trasposed_A=True, name='Backward')
        #grad_z.clamp_(-1,1)
        if status not in "converged":
            print("Iterations not converging! (In backward)", err, status)
            if err > 0.5:
                print('Warning!! Err > 0.5')
                # pickle.dump(results, open(f'./non_convergence_analysis/err{err}_{bw_mitr}.pkl', 'wb'))
            # print(f'abs error trace: {results["abs_err"]}')
            # print(f'X diff trace: {results["X_diff"]}')
            # print(f'X_new trace: {results["X_new"]}')
            # print(f'X trace: {results["X"]}')
        grad_W = grad_z @ torch.spmm(A, X.T)
        grad_B = grad_z

        # Might return gradient for A if needed
        return grad_W, None, torch.zeros_like(A), grad_B, None, None, None

    @staticmethod
    def inn_pred(W, X, A, B, phi, mitr=300, tol=3e-6, trasposed_A=False, compute_dphi=False, name='Forward'):
        # TODO: randomized speed up
        At = A if trasposed_A else torch.transpose(A, 0, 1)
        #X = B if X is None else X
        results = dict()
        results['abs_err'] = []
        results['X_new'] = []
        results['X'] = []
        results['X_diff'] = []
        err = 0
        status = 'max itrs reached'
        for i in range(mitr):
            # WXA
            X_ = W @ X
            support = torch.spmm(At, X_.T).T
            # tmp = support + B
            # X_new = phi(F.normalize(tmp, dim=0))
            X_new = phi(support + B)
            err = torch.norm(X_new - X, np.inf)
            results['abs_err'].append(err.item())
            results['X_diff'].append((X_new - X).cpu().numpy())
            results['X_new'].append(X_new.cpu().numpy())
            results['X'].append(X.cpu().numpy())
            if err < tol:
                status = 'converged'
                break
            X = X_new
        # print(f'{name}, no. iterations: {i}, err: {err}')
        dphi = None
        if compute_dphi:
            with torch.enable_grad():
                support = torch.spmm(At, (W @ X).T).T
                Z = support + B
                Z.requires_grad_(True)
                X_new = phi(Z)
                dphi = torch.autograd.grad(torch.sum(X_new), Z, only_inputs=True)[0]

        return X_new, err, status, dphi, results

epsilon_F = 10**(-12)
def g(F):
    FF = F.t() @ F
    FF_norm = torch.norm(FF, p='fro')
    return (1/(FF_norm+epsilon_F)) * FF


def get_G(Lambda_F, Lambda_S, gamma):
    G = 1.0 - gamma * Lambda_F @ Lambda_S.t()
    G = 1 / G
    return G


class IDMFunction(Function):
    @staticmethod
    def forward(ctx, X, F, S, Q_S, Lambda_S, gamma):
        # TODO
        # print(f'X.requires_grad: {X.requires_grad}')
        Lambda_F, Q_F = torch.symeig(g(F), eigenvectors=True)
        Lambda_F = Lambda_F.view(-1,1)
        # We can also save FF and FF_norm for backward to reduce cost a bit.
        G = get_G(Lambda_F, Lambda_S, gamma)
        # ipdb.set_trace()
        Z = Q_F @ (G * (Q_F.t() @ X @ Q_S)) @ Q_S.t()
        # ipdb.set_trace()
        # Z = Q_F @ (G * (Q_F.t() @ X @ Q_S)) @ torch.inverse(Q_S) # revised version for asymmetric adj
        # ipdb.set_trace()
        # Y_hat = B @ Z
        ctx.save_for_backward(F, S, Q_F, Q_S, Z, G, X, gamma)
        return Z

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        grad_Z = grad_output
        F, S, Q_F, Q_S, Z, G, X, gamma = ctx.saved_tensors
        FF = F.t() @ F
        FF_norm = torch.norm(FF, p='fro')
        # R = G * ((Q_F.t() @ B.t()) @ (grad_Y_hat @ Q_S))
        R = G * (Q_F.t() @ grad_Z @ Q_S)
        # R = G * (Q_F.t() @ grad_Z @ torch.inverse(Q_S).t()) # revised version for asymmetric adj.
        R = Q_F @ R @ Q_S.t() @ torch.sparse.mm(S, Z.t())
        scalar_1 = gamma * (1/(FF_norm+epsilon_F))
        scalar_2 = torch.sum(FF * R)
        scalar_2 = 2 * scalar_2 * (1/(FF_norm**2 + epsilon_F * FF_norm))
        grad_F = (R + R.t()) - scalar_2 * FF
        grad_F = scalar_1 * (F @ grad_F)
        # ipdb.set_trace()
        # grad_X = torch.autograd.grad(Z, X, grad_outputs=grad_Z, only_inputs=True)
        # grad_X = grad_Z @ kronecker(Q_S, Q_F) @ torch.diag(G.view(-1)) @ kronecker(Q_S.t(), Q_F.t())
        # need to chagne to vectorization form, otherwise kronecker product will cause OOM.
        # grad_X = Q_F @ (G * (Q_F.t() @ Q_S)) @ Q_S.t()
        # print(f'grad_X.shape: {grad_X.shape}')
        grad_X = None
        return grad_X, grad_F, None, None, None, None
        # need to change the first var to grad_x? should be U^{-1}?
        # if we add some deep nets before this module (i.e., x is no longer node attributes)

class Forward_Iter(Function):
    @staticmethod
    def forward(ctx, X, F, S, Q_S, Lambda_S, gamma, threshold=3e-6, max_iter=300, solver='standard'):
        # TODO
        # print(f'X.requires_grad: {X.requires_grad}')
        Lambda_F, Q_F = torch.symeig(g(F), eigenvectors=True)
        Lambda_F = Lambda_F.view(-1,1)
        # We can also save FF and FF_norm for backward to reduce cost a bit.
        G = get_G(Lambda_F, Lambda_S, gamma)
        # # ipdb.set_trace()
        # Z = Q_F @ (G * (Q_F.t() @ X @ Q_S)) @ Q_S.t()
        # # ipdb.set_trace()
        # # Y_hat = B @ Z
        # ctx.save_for_backward(F, S, Q_F, Q_S, Z, G, X, gamma)

        # how to handle the grad here?
        # with torch.no_grad():
        #     Z, abs_diff = Forward_Iter.inn_iter(lambda Z: Forward_Iter._inner_func(gamma, S, F, Z, X), z_init=torch.zeros_like(X), threshold=threshold,
        #                                 max_iter=max_iter)
        if solver == 'standard':
            Z, abs_diff = Forward_Iter.inn_iter(lambda Z: Forward_Iter._inner_func(gamma, S, F, Z, X), z_init=torch.zeros_like(X), threshold=threshold,
                                                max_iter=max_iter)
        elif solver == 'new':
            # implement a new solver
            # g(F) @ h @ torch.transpose(S,0,1). But we need to use sparse.mm. So, g(F) @ torch.transpose(torch.sparse.mm(S, X.t()), 0, 1)
            func = lambda h: gamma * g(F) @ torch.transpose(torch.sparse.mm(S, h.t()), 0, 1)
            Z = new_solver(X.clone().detach(), X.clone().detach(), F, S, gamma, max_iter)
            Z_standard, abs_diff = Forward_Iter.inn_iter(lambda h: Forward_Iter._inner_func2(gamma, S, F, h, X), z_init=torch.zeros_like(X), threshold=threshold,
                                                max_iter=max_iter)
            # ipdb.set_trace()
        else:
            raise NotImplementedError(f'Cannot find the solver {solver}')
        # ipdb.set_trace()
        ctx.save_for_backward(F, S, Q_F, Q_S, Z, G, X, gamma)
        return Z

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        grad_Z = grad_output
        # ipdb.set_trace()
        F, S, Q_F, Q_S, Z, G, X, gamma = ctx.saved_tensors
        FF = F.t() @ F
        FF_norm = torch.norm(FF, p='fro')
        # R = G * ((Q_F.t() @ B.t()) @ (grad_Y_hat @ Q_S))
        R = G * (Q_F.t() @ grad_Z @ Q_S)
        R = Q_F @ R @ Q_S.t() @ torch.sparse.mm(S, Z.t())
        scalar_1 = gamma * (1/(FF_norm+epsilon_F))
        scalar_2 = torch.sum(FF * R)
        scalar_2 = 2 * scalar_2 * (1/(FF_norm**2 + epsilon_F * FF_norm))
        grad_F = (R + R.t()) - scalar_2 * FF
        grad_F = scalar_1 * (F @ grad_F)
        grad_X = None
        return grad_X, grad_F, None, None, None, None, None, None, None
        # need to change the first var to grad_x? should be U^{-1}?
        # if we add some deep nets before this module (i.e., x is no longer node attributes)

    @staticmethod
    def _inner_func(gamma, S, F, Z, X):
        S_t = torch.transpose(S, 0, 1)
        Z_new = gamma * g(F) @ torch.sparse.mm(S_t, Z.t()).t() + X
        return Z_new

    @staticmethod
    def _inner_func2(gamma, S, F, Z, X):
        S_t = torch.transpose(S, 0, 1)
        # ipdb.set_trace()
        Z_new = gamma * g(F) @ torch.sparse.mm(S_t, Z.t()).t() + X
        return Z_new

    @staticmethod
    def inn_iter(f, z_init, threshold, max_iter, name='Forward'):
        z_prev, z = z_init, f(z_init)
        nstep = 0
        while nstep < max_iter:
            z_prev, z = z, f(z)
            abs_diff = torch.norm(z_prev - z).item()
            if abs_diff < threshold:
                break
            nstep += 1
        if nstep == max_iter:
            print(f'step {nstep}, not converged, abs_diff: {abs_diff}')
        return z, abs_diff


def new_solver(q_1, h_1, F, S, gamma, max_iter):
    nstep = 0
    # q and h should be initialized as x.
    # ipdb.set_trace()
    while nstep < max_iter:
        S_t = torch.transpose(S, 0, 1)
        h_1 = gamma * g(F) @ torch.sparse.mm(S_t, h_1.t()).t() # transpose or not to transpose?
        q_1 = q_1 + h_1
        # print('current step: ', nstep, 'torch.norm(h).item(): ', torch.norm(h_1).item())
        nstep += 1
    return q_1



def one_step_func(W, X, A, B, phi, transposed_A=False):
    At = A if transposed_A else torch.transpose(A, 0, 1)
    X_ = W @ X
    support = torch.spmm(At, X_.T).T
    X_new = phi(support + B)
    return X_new