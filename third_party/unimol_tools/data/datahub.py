# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import logging
import copy
import os
import pandas as pd
import numpy as np
import csv
from typing import List, Optional
from collections import defaultdict
from .datareader import MolDataReader
from .datascaler import TargetScaler
from .conformer import ConformerGen
from ..utils import logger

class DataHub(object):
    def __init__(self, data=None, datatype='smiles', is_train=True, save_path=None, **params):
        self.data = data
        self.datatype = datatype
        self.is_train = is_train
        self.save_path = save_path
        self.selected_atoms = params.get('selected_atoms', ['all'])
        self.task = params.get('task', None)
        self.target_cols = params.get('target_cols', None)
        self.multiclass_cnt = params.get('multiclass_cnt', None)
        self.ss_method = params.get('target_normalize', 'none')
        self._init_data(**params)
    
    def _init_data(self, **params):
        self.data = MolDataReader().read_data(self.data, self.datatype, self.is_train, **params)
        self.data['target_scaler'] = TargetScaler(self.ss_method, self.task, self.save_path)

        # if self.datatype == 'smiles':
        #     # self.data = MolDataReader().read_data(self.data, datatype=self.datatype, self.is_train, **params)
        #     self.data['target_scaler'] = TargetScaler(self.ss_method, self.task, self.save_path)
        # elif self.datatype == 'mol':
        #     # self.data = pd.DataFrame(self.data, columns=['mol'])
        #     # dd = {}
        #     # dd['mol'] = self.data['mol'].tolist()
        #     # self.data = dd
        #     self.data['target_scaler'] = None
        # else:
        #     raise TypeError('Unsupported datatype: {}'.format(datatype))
     
        if self.task == 'regression': 
            target = np.array(self.data['target']).reshape(-1,1).astype(np.float32)
            if self.is_train:
                self.data['target_scaler'].fit(target, self.save_path)
            self.data['target'] = self.data['target_scaler'].transform(target)
        elif self.task == 'classification':
            target = np.array(self.data['target']).reshape(-1,1).astype(np.int32)
            self.data['target'] = target
        elif self.task =='multiclass':
            target = np.array(self.data['target']).reshape(-1,1).astype(np.int32)
            self.data['target'] = target
            if not self.is_train:
                self.data['multiclass_cnt'] = self.multiclass_cnt
        elif self.task == 'multilabel_regression':
            # print(f"datahub.py target0{np.array(self.data['target']).shape}")
            target = np.array(self.data['target']).reshape(-1,self.data['num_tasks']).astype(np.float32)
            # print(f"datahub.py target{target.shape}")
            if self.is_train:
                self.data['target_scaler'].fit(target, self.save_path)
            self.data['target'] = self.data['target_scaler'].transform(target)
        elif self.task == 'multilabel_classification':
            target = np.array(self.data['target']).reshape(-1,self.data['num_tasks']).astype(np.int32)
            self.data['target'] = target
        elif self.task == 'atom_regression':
            # print(f"datahub.py target0{np.array(self.data['target']).shape}")
            target = self.data['atom_target']
            atom_mask = self.data['atom_mask']
            # print(f"datahub.py target{target[-1,0:20]}")
            if self.is_train:
                self.data['target_scaler'].fit(target, self.save_path, mask=atom_mask)
            self.data['target'] = self.data['target_scaler'].transform(target, mask=atom_mask)
            # print(f"datahub.py target{self.data['target'][-1,0:20]}")

        elif self.task == 'repr':
            pass
        else:
            raise ValueError('Unknown task: {}'.format(self.task))
        # print("datahub.py 1")
        if self.datatype == 'points3d':
            no_h_list = ConformerGen(**params).transform_raw(self.data['atoms'], self.data['coordinates'])
        elif self.datatype == 'smiles':    
            smiles_list = self.data["smiles"]                  
            no_h_list = ConformerGen(**params).transform(smiles_list, self.datatype)
        elif self.datatype == 'mol':
            mol_list = self.data["mol"]                  
            no_h_list = ConformerGen(**params).transform(mol_list, self.datatype)

        else:
            raise TypeError('Unsupported datatype: {}'.format(datatype))

         
        # print(f"datahub no_h_list{len(no_h_list)}len {no_h_list[0]}")
        # 列表：其中每个分子有一个字典，由conformer.py生成[{'src_tokens', 'src_distance', 'src_coord', 'src_edge_type'},...]
        self.data['unimol_input'] = no_h_list
        