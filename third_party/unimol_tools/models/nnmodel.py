# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import logging
import copy
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import joblib
from torch.utils.data import Dataset
import numpy as np
from ..utils import logger
from .unimol import UniMolModel
from .loss import GHMC_Loss, FocalLossWithLogits, myCrossEntropyLoss

import torch.multiprocessing as mp
import concurrent.futures

NNMODEL_REGISTER = {
    'unimolv1': UniMolModel,
}

LOSS_RREGISTER = {
    'classification': myCrossEntropyLoss,
    'multiclass': myCrossEntropyLoss,
    'regression': nn.MSELoss(),
    'multilabel_classification': {
        'bce': nn.BCEWithLogitsLoss(),
        'ghm': GHMC_Loss(bins=10, alpha=0.5),
        'focal': FocalLossWithLogits,
    },
    'multilabel_regression': nn.MSELoss(),
    'atom_regression': {
        'l1': nn.L1Loss(),
        'mse': nn.MSELoss(),
        'smoothl1': nn.SmoothL1Loss(),
    }
}
ACTIVATION_FN = {
    # predict prob shape should be (N, K), especially for binary classification, K equals to 1.
    'classification': lambda x: F.softmax(x, dim=-1)[:, 1:],
    # softmax is used for multiclass classification
    'multiclass': lambda x: F.softmax(x, dim=-1),
    'regression': lambda x: x,
    # sigmoid is used for multilabel classification
    'multilabel_classification': lambda x: F.sigmoid(x),
    # no activation function is used for multilabel regression
    'multilabel_regression': lambda x: x,
    'atom_regression': lambda x: x,
}
OUTPUT_DIM = {
    'classification': 2,
    'regression': 1,
}


class NNModel(object):
    def __init__(self, data, trainer, **params):
        self.data = data
        self.num_tasks = self.data['num_tasks']
        self.target_scaler = self.data['target_scaler']
        # unimol_input是conformer中生成的
        self.features = data['unimol_input']
        self.model_name = params.get('model_name', 'unimolv1')
        self.material_structures_type = params.get('material_structures_type', 'small_molecule')
        self.loss_key = params.get('loss_key', None)
        self.trainer = trainer
        self.splitter = self.trainer.splitter
        self.model_params = params.copy()
        self.task = params['task']
        self.max_atoms = params.get('max_atoms', 510)
        self.max_tokens = self.max_atoms+2
        
        if self.task in OUTPUT_DIM:
            self.model_params['output_dim'] = OUTPUT_DIM[self.task]
        elif self.task == 'multiclass':
            self.model_params['output_dim'] = self.data['multiclass_cnt']
        elif self.task == 'atom_regression':
            self.model_params['output_dim'] = self.max_tokens

        else:
            self.model_params['output_dim'] = self.num_tasks
        self.model_params['device'] = self.trainer.device
        self.cv = dict()
        self.metrics = self.trainer.metrics
        if self.task == 'multilabel_classification':
            if self.loss_key is None:
                self.loss_key = 'focal'
            self.loss_func = LOSS_RREGISTER[self.task][self.loss_key]
        elif self.task == 'atom_regression':
            if self.loss_key is None:
                self.loss_key = 'l1'
            self.loss_func = LOSS_RREGISTER[self.task][self.loss_key]
        else:
            self.loss_func = LOSS_RREGISTER[self.task]
        self.activation_fn = ACTIVATION_FN[self.task]
        self.save_path = self.trainer.save_path
        self.trainer.set_seed(self.trainer.seed)
        # self.model = self._init_model(**self.model_params)
        # logger.info("Model structure:\n{}".format(self.model))


    def _init_model(self, model_name, **params):
        if model_name in NNMODEL_REGISTER:
            model = NNMODEL_REGISTER[model_name](**params)
        else:
            raise ValueError('Unknown model: {}'.format(self.model_name))
        return model

    def collect_data(self, X, y, idx):
        assert isinstance(y, np.ndarray), 'y must be numpy array'
        if isinstance(X, np.ndarray):
            return torch.from_numpy(X[idx]).float(), torch.from_numpy(y[idx])
        elif isinstance(X, list):
            return {k: v[idx] for k, v in X.items()}, torch.from_numpy(y[idx])
        else:
            raise ValueError('X must be numpy array or dict')

    def run(self):
        logger.info("start training Uni-Mol:{}".format(self.model_name))
        scalar = self.target_scaler

        # X：每个分子对应字典，datahub中传来的
        X = np.asarray(self.features)
        # print(f"nnmodel.py X: {X.shape}")
        # print(f"nnmodel.py X[0]: {X[0]}")

        y = np.asarray(self.data['target'])

        if 'atom_mask' in self.data:
            mask = np.asarray(self.data['atom_mask'])
        else:
            mask = None
        # print(f"nnmodel.py self.data['target'] {self.data['target'][-1,0:20]} ")

        # print(f"nnmodel.py y {y[-1,0:20]} ")
        # print(f"nnmodel.py mask {mask}")
        
        '''scaffold ['O=C1CC(=O)C(C(=O)O)=C1C(=O)O' 'O=C1CC(=O)C=C1F' 'O=C1NC(=O)C=C1' ...
            'O=S(=O)(O)c(c1)c(S(=O)(=O)O)c(S(=O)(=O)O)c(c12)c(O)n(c2O)N(C3=O)C(=O)c(c34)c(S(=O)(=O)O)c(S(=O)(=O)O)c(c4)S(=O)(=O)O'
            'O=S(=O)(O)c1ccc(S(=O)(=O)O)c(c12)c(O)n(c2O)N(C3=O)C(=O)c(c34)c(S(=O)(=O)O)c(S(=O)(=O)O)c(S(=O)(=O)O)c4S(=O)(=O)O'
            'O=S(=O)(O)c1c(S(=O)(=O)O)c(S(=O)(=O)O)c(S(=O)(=O)O)c(c12)c(O)n(c2O)N(C3=O)C(=O)c(c34)c(S(=O)(=O)O)c(S(=O)(=O)O)cc4S(=O)(=O)O'] '''
        scaffold = np.asarray(self.data['scaffolds'])
        # print(f"nnmodel.py scaffold {scaffold}")
        # 01分类的num_tasks是1，但是output_dim是2
        if self.task == 'classification':
            y_pred = np.zeros_like(
                y.reshape(y.shape[0], self.num_tasks)).astype(float)
        else:
            y_pred = np.zeros((y.shape[0], self.model_params['output_dim']))
        for fold, (tr_idx, te_idx) in enumerate(self.splitter.split(X, y, scaffold)):
            X_train, y_train, mask_train = X[tr_idx], y[tr_idx], mask[tr_idx]
            X_valid, y_valid, mask_valid = X[te_idx], y[te_idx], mask[te_idx]
            traindataset = NNDataset(X_train, y_train, mask_train)
            validdataset = NNDataset(X_valid, y_valid, mask_valid)
            # print(f"traindataset0 {traindataset[0]}")
            if fold > 0:
                # need to initalize model for next fold training
                self.model = self._init_model(**self.model_params)
            # print(f"nnmodel.py fold {fold}")
            fold_y_pred = self.trainer.fit_predict(
                self.model, traindataset, validdataset, self.loss_func, self.activation_fn, self.save_path, fold, self.target_scaler)
            y_pred[te_idx] = fold_y_pred
            # print(f"nnmodel.py {fold}  fold_y_pred {fold_y_pred[-1,0:20]}")

            if 'multiclass_cnt' in self.data:
                label_cnt = self.data['multiclass_cnt']
            else:
                label_cnt = None

            # 单独算一份验证集的metric，在这里对y_valid求逆了，而_y_pred没有变
            if scalar is not None:
                # print(f"nnmodel.py scalar is not None fold {fold} y {y_valid.shape} fold_y_pred {fold_y_pred.shape} mask{mask_valid.shape}")
                # print(f"nnmodel.py scalar is not None fold {fold} y {y_valid[mask_valid][-20:]} fold_y_pred {fold_y_pred[mask_valid][-20:]} mask{mask_valid.shape}")
                # inverse_y_preds = scalar.inverse_transform(copied_array, mask=mask_valid)
                inverse_y_truths = scalar.inverse_transform(y_valid, mask=mask_valid)
                single_metric = self.metrics.cal_metric(
                    inverse_y_truths,
                    fold_y_pred,
                    mask=mask_valid,
                    label_cnt=label_cnt
                )
                # print(f"nnmodel.py scalar is not None fold {fold} y {y_valid.shape} fold_y_pred {fold_y_pred.shape} mask{mask_valid.shape}")
                # print(f"nnmodel.py scalar is not None fold {fold} y {y_valid[mask_valid][-20:]} fold_y_pred {fold_y_pred[mask_valid][-20:]} mask{mask_valid.shape}")

            else:
                # print(f"nnmodel.py scalar is None fold {fold} y {y_valid.shape} fold_y_pred {fold_y_pred.shape} mask{mask_valid.shape}")

                single_metric = self.metrics.cal_metric(y_valid, fold_y_pred, mask=mask_valid, label_cnt=label_cnt)
            logger.info("fold {0}, result {1}".format(fold, single_metric)
            )
            # print(f"nnmodel.py {fold}  fold_y_pred {fold_y_pred[-1,0:20]}")

        self.cv['pred'] = y_pred
        # print(f"nnmodel.py self.cv['pred'][-1,0:20] {self.cv['pred'][-1,0:20]} ")

        # 算五份验证集的总metric，在这里已经对y和self.cv['pred']进行变换了，之后在train.py就不需要了
        if scalar is not None:
            # print(f"nnmodel.py scalar is not None  final y {y.shape} pred {self.cv['pred'].shape} mask{mask.shape}")
            # print(f"nnmodel.py scalar is not None  final y {y[-1,0:20]} pred {self.cv['pred'][-1,0:20]} mask{mask.shape}")


            self.cv['metric'] = self.metrics.cal_metric(
                scalar.inverse_transform(y, mask=mask), 
                self.cv['pred'],
                mask=mask
            )
            # print(f"nnmodel.py scalar is not None  final y {y.shape} pred {self.cv['pred'].shape} mask{mask.shape}")
            # print(f"nnmodel.py scalar is not None  final y {y[-1,0:20]} pred {self.cv['pred'][-1,0:20]} mask{mask.shape}")

        else:
            # print(f"nnmodel.py scalar is None final y {y.shape} pred {self.cv['pred'].shape} mask{mask.shape}")
            self.cv['metric'] = self.metrics.cal_metric(y, self.cv['pred'], mask=mask)   

        self.dump(self.cv['pred'], self.save_path, 'cv.data')
        self.dump(self.cv['metric'], self.save_path, 'metric.result')
        logger.info("Uni-Mol metrics score: \n{}".format(self.cv['metric']))
        # print(f"nnmodel.py self.cv['pred'][-1,0:20] {self.cv['pred'][-1,0:20]} ")

        logger.info("Uni-Mol & Metric result saved!")

    def dump(self, data, dir, name):
        path = os.path.join(dir, name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        joblib.dump(data, path)

    def evaluate(self, trainer=None, preloaded_models=[], n_splits=None):
        n_splits = n_splits or self.splitter.n_splits
        logger.info("start predict NNModel:{}".format(self.model_name))
        testdataset = NNDataset(self.features, np.asarray(self.data['target']), np.asarray(self.data['atom_mask']))
        # y_pred_fold=[]
        # for fold in range(n_splits):
        #     fold_y_pred, _, __ = trainer.predict(preloaded_models[fold], testdataset, self.loss_func, self.activation_fn,
        #                                     self.save_path, fold, self.target_scaler, epoch=1, load_model=False)
        #     if fold == 0:
        #         y_pred = np.zeros_like(fold_y_pred)
        #     y_pred += fold_y_pred
        #     y_pred_fold.append(fold_y_pred)
        y_pred_fold, _, __ = trainer.predict_cv(preloaded_models[:n_splits], testdataset, self.activation_fn, self.target_scaler)
        for fold, fold_y_pred in enumerate(y_pred_fold):
            if fold == 0:
                y_pred = np.zeros_like(fold_y_pred)
            y_pred += fold_y_pred
        y_pred /= n_splits
        self.cv['test_pred'] = y_pred
        self.cv['test_pred_fold'] = y_pred_fold

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def NNDataset(data, label=None, mask=None):
    # print(f"nnmodel.py data {data.shape} label {label.shape} mask {mask.shape}")
    return TorchDataset(data, label, mask)


class TorchDataset(Dataset):
    def __init__(self, data, label=None, mask=None):
        self.data = data
        self.label = label if label is not None else np.zeros((len(data), 1))
        self.mask = mask if mask is not None else np.zeros((len(data), 1))
        # print(f"nnmodel.py self.data {self.data.shape} self.label {self.label.shape} self.mask {self.mask.shape}")
        # print(f"nnmodel.py self.mask {self.mask}")

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.mask[idx]

    def __len__(self):
        return len(self.data)
