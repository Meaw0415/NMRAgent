# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import logging
import copy
import os
import argparse
import json
import numpy as np
import pandas as pd
import joblib
from .data import DataHub
from .models import NNModel
from .tasks import Trainer
from .utils import YamlHandler
from .utils import logger

class MolTrain(object):
    def __init__(self, 
                task='classification',
                initial_weight = 'pretrain_self_supervised',
                material_structures_type ='small_molecule',
                selected_atoms = ['all'],
                epochs=10,
                learning_rate=1e-4,
                batch_size=16,
                grad_accum_steps=1,
                loss_key='l1',
                optim_key='adam',
                early_stopping=5,
                metrics= "none",
                usecls=False,
                split='random',
                save_path='./result',
                amp =True,
                remove_hs=False,
                max_atoms=510,
                ):
        config_path = os.path.join(os.path.dirname(__file__), 'config/default.yaml')
        self.yamlhandler = YamlHandler(config_path)
        config = self.yamlhandler.read_yaml()
        config.task = task
        config.initial_weight = initial_weight
        config.material_structures_type = material_structures_type
        config.selected_atoms = selected_atoms
        config.epochs = epochs
        config.learning_rate = learning_rate
        config.batch_size = batch_size
        config.grad_accum_steps = grad_accum_steps
        config.loss_key = loss_key
        config.optim_key = optim_key
        config.amp = amp
        config.patience = early_stopping
        config.metrics = metrics
        config.usecls = usecls
        config.split = split
        config.remove_hs = remove_hs
        config.max_atoms = max_atoms
        self.selected_atoms = selected_atoms
        self.save_path = save_path
        self.config = config


    def fit(self, data, datatype='smiles'):
        self.datahub = DataHub(data = data, datatype=datatype, is_train=True, save_path=self.save_path, **self.config)
        self.data = self.datahub.data
  

        self.update_and_save_config()
        self.trainer = Trainer(save_path=self.save_path, **self.config)
        self.model = NNModel(self.data, self.trainer, **self.config)
        self.model.run()
        scalar = self.data['target_scaler']
        y_pred = self.model.cv['pred']
        y_true = np.array(self.data['target'])
        if 'atom_mask' in self.data:
            label_mask = self.data['atom_mask']
            index_mask = np.array(self.data['raw_data']['atom_mask'].tolist()).astype(int)
        else:
            atom_mask = None
            index_mask = None

        
        # print(f"train.py scalar is none y_pred {y_pred[mask][-20:]}")

        metrics = self.trainer.metrics
        # 在nnmodel里面已经求逆了
        # if scalar is not None:
        #     print(f"train.py scalar is none y_pred {y_pred[mask][-20:]}")
        #     y_pred = scalar.inverse_transform(y_pred, mask=mask)
        #     y_true = scalar.inverse_transform(y_true, mask=mask)
            # print(f"train.py scalar is none y_true {y_true[-1, 0:20]}")
            # print(f"train.py scalar is none y_pred {y_pred[-1, 0:20]}")

        # 计算分类阈值
        if self.config["task"] in ['classification', 'multilabel_classification']:
            threshold = metrics.calculate_classification_threshold(y_true, y_pred)
            joblib.dump(threshold, os.path.join(self.save_path, 'threshold.dat'))
        
        self.cv_pred = y_pred
        self.cv_true = y_true
        self.cv_label_mask = label_mask
        self.cv_index_mask = index_mask

        return

    def update_and_save_config(self):
        # 从data里面获取任务数
        self.config['num_tasks'] = self.data['num_tasks']
        # 标签列用逗号隔开（可能是多任务），是一个字符串
        self.config['target_cols'] = ','.join(self.data['target_cols'])
        # 多分类任务需要记录类别数
        if self.config['task'] == 'multiclass':
            self.config['multiclass_cnt'] = self.data['multiclass_cnt']

        if self.config['split'] == 'random':
            self.config['split'] = 'random_5fold'
        else:
            self.config['split'] = 'scaffold_5fold'

        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                logger.info('Create output directory: {}'.format(self.save_path))
                os.makedirs(self.save_path)
            else:
                logger.info('Output directory already exists: {}'.format(self.save_path))
                logger.info('Warning: Overwrite output directory: {}'.format(self.save_path))
            out_path = os.path.join(self.save_path, 'config.yaml')
            self.yamlhandler.write_yaml(data = self.config, out_file_path = out_path)
        return