# MGNNI: Multiscale Graph Neural Networks with Implicit Layers
The implementation of MGNNI: Multiscale Graph Neural Networks with Implicit Layers (NeurIPS 2022).

## Requirements
The script has been tested running under Python 3.6.9, with the following packages installed (along with their dependencies):
* pytorch (tested on 1.6.0)
* torch_geometric (tested on 1.6.3)
* scipy (tested on 1.5.2)
* numpy (tested on 1.19.2)

## Run Experiments 
We provides some examples for running experiments for different tasks on different datasets:
### Node classification 
```
cd nodeclassification
```

For chameleon and squirrel datasets,
```
python train_MGNNI_heterophilic.py --dataset chameleon --lr 0.01 --weight_decay 5e-4 --model MGNNI_m_MLP --fp_layer MGNNI_m_att --batch_norm 1 --ks [1,2] --idx_split 0 --epoch 10000 --patience 500 
```

For Cornell, Texas, Wisconsin datasets,
```
python train_MGNNI_heterophilic.py --dataset cornell --lr 0.5 --weight_decay 5e-6 --model MGNNI_m_att --ks [1,2] --epoch 10000 --patience 500 --idx_split 0
```
`idx_split` should be changed accordingly. There are 10 data splits as used in [Geom-GCN](https://github.com/graphdml-uiuc-jlu/geom-gcn).

For PPI dataset,
```
python train_MGNNI_m_att_PPI.py --model MGNNI_m_att_stack --dropout 0.1 --epoch 5000 --hidden 2048 --ks [1,2]
```

### Graph classification
```
cd graphclassification
```
```
python train_MGNNI_att.py --dataset MUTAG --lr 0.01 --weight_decay 0.0 --num_layers 3 --ks [1,2] --epochs 500 
```



This implementation is developed based on the original implementation of [IGNN](https://github.com/SwiftieH/IGNN/tree/main/nodeclassification) and [EIGNN](https://github.com/liu-jc/EIGNN). We thank them for their useful implementation.

If you find our implementation useful in your research, please consider citing our paper:
```bibtex
@inproceedings{liu2022mgnni,
 author = {Liu, Juncheng and Hooi, Bryan and Kawaguchi, Kenji and Xiao, Xiaokui},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {MGNNI: Multiscale Graph Neural Networks with Implicit Layers},
 year = {2022}
}
```