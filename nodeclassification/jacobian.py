import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import ipdb


def jac_loss_estimate(f0, z0, vecs=2, create_graph=True):
    """Estimating tr(J^TJ)=tr(JJ^T) via Hutchinson estimator

    Args:
        f0 (torch.Tensor): Output of the function f (whose J is to be analyzed)
        z0 (torch.Tensor): Input to the function f
        vecs (int, optional): Number of random Gaussian vectors to use. Defaults to 2.
        create_graph (bool, optional): Whether to create backward graph (e.g., to train on this loss).
                                       Defaults to True.

    Returns:
        torch.Tensor: A 1x1 torch tensor that encodes the (shape-normalized) jacobian loss
    """
    vecs = vecs
    result = 0
    for i in range(vecs):
        v = torch.randn(*z0.shape).to(z0)
        print(f'v: {v}, v.size: {v.size()}')
        print(f'v.device: {v.device}, z0.device: {z0.device}, f0.device: {f0.device}')
        vJ = torch.autograd.grad(f0, z0, v, create_graph=create_graph)[0]
        result += vJ.norm()**2
    return result / vecs / np.prod(z0.shape)


def power_method(f0, z0, n_iters=200):
    """Estimating the spectral radius of J using power method

    Args:
        f0: size (#feat, #nodes)
        f0 (torch.Tensor): Output of the function f (whose J is to be analyzed)
        z0 (torch.Tensor): Input to the function f
        n_iters (int, optional): Number of power method iterations. Defaults to 200.

    Returns:
        tuple: (largest eigenvector, largest (abs.) eigenvalue)
    """
    # evector = torch.randn_like(z0)
    # bsz = evector.shape[0]
    # for i in range(n_iters):
    #     vTJ = torch.autograd.grad(f0, z0, evector, retain_graph=(i < n_iters-1), create_graph=False)[0]
    #     evalue = (vTJ * evector).reshape(bsz, -1).sum(1, keepdim=True) / (evector * evector).reshape(bsz, -1).sum(1, keepdim=True)
    #     evector = (vTJ.reshape(bsz, -1) / vTJ.reshape(bsz, -1).norm(dim=1, keepdim=True)).reshape_as(z0)
    #     # zero/zero. Then, evector becomes nan. Not sure if it's normal to have zero in vTJ?
    #     # if torch.isnan(evector).sum() > 0:
    #     #     ipdb.set_trace()
    #     # print(f'i: {i}, evalue: {evalue}')
    #     # print(f'vTJ: {vTJ}')
    # return (evector, torch.abs(evalue))
    evector = torch.randn_like(z0)
    bsz = evector.shape[1]
    # ipdb.set_trace()
    for i in range(n_iters):
        vTJ = torch.autograd.grad(f0, z0, evector, retain_graph=(i < n_iters-1), create_graph=False)[0]
        evalue = (vTJ * evector).reshape(-1, bsz).sum(0, keepdim=True) / (evector * evector).reshape(-1, bsz).sum(0, keepdim=True)
        evector = (vTJ.reshape(-1, bsz) / vTJ.reshape(-1, bsz).norm(dim=0, keepdim=True)).reshape_as(z0)
        # zero/zero. Then, evector becomes nan. Not sure if it's normal to have zero in vTJ?
        # if torch.isnan(evector).sum() > 0:
        #     ipdb.set_trace()
        # print(f'i: {i}, evalue: {evalue}')
        # print(f'vTJ: {vTJ}')
    return (evector, torch.abs(evalue))