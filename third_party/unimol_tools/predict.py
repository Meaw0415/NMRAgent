# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import logging
import copy
import os
import time
import pandas as pd
import numpy as np
import argparse
import joblib
import torch

from .data import DataHub
from .models import NNModel
from .tasks import Trainer
from .utils import YamlHandler
from .utils import logger


def _debug_enabled():
    return os.environ.get("NMR_RERANK_DEBUG", "0").lower() not in {"", "0", "false", "no"}


def _debug(msg):
    if _debug_enabled():
        print(f"[MolPredict DEBUG] {msg}", flush=True)


class MolPredict(object):
    def __init__(self, load_model=None, n_splits=None):
        if not load_model:
            raise ValueError("load_model is empty")
        self.load_model = load_model
        config_path = os.path.join(load_model, 'config.yaml')
        self.config = YamlHandler(config_path).read_yaml()
        self.config.target_cols = self.config.target_cols.split(',')
        self.task = self.config.task
        self.target_cols = self.config.target_cols
        self.trainer, self.model = None, None
        self.n_splits = n_splits

    def predict(self, data, datatype='smiles', save_path=None, metrics='none'):
        t0 = time.time()
        self.save_path = save_path
        if not metrics or metrics != 'none':
            self.config.metrics = metrics
        _debug(
            f"predict start datatype={datatype} n_splits={self.n_splits} "
            f"cuda_available={torch.cuda.is_available()}"
        )
        ## load test data
        s0 = time.time()
        self.datahub = DataHub(data = data, datatype=datatype, is_train = False, save_path=self.load_model, **self.config)
        self.data = self.datahub.data
        _debug(f"DataHub built in {time.time() - s0:.2f}s")
        s1 = time.time()
        self.trainer = Trainer(save_path=self.load_model, **self.config)
        device = getattr(self.trainer, "device", None)
        _debug(f"Trainer built in {time.time() - s1:.2f}s device={device}")
        s2 = time.time()
        self.model = NNModel(self.data, self.trainer, **self.config)
        _debug(f"NNModel built in {time.time() - s2:.2f}s")
        self.n_splits = self.n_splits or self.trainer.splitter.n_splits
        if not hasattr(self, 'preloaded_models'):
            s3 = time.time()
            self.preloaded_models = [self.model._init_model(**self.config.copy()) for _ in range(self.n_splits)]
            for fold in range(self.n_splits):
                model_path = os.path.join(self.load_model, f'model_{fold}.pth')
                self.preloaded_models[fold].load_state_dict(
                    torch.load(model_path, map_location=self.trainer.device)['model_state_dict'])
                self.preloaded_models = [model.to(self.trainer.device).eval() for model in self.preloaded_models]
            print('Preload models.')
            _debug(f"Preloaded {self.n_splits} models in {time.time() - s3:.2f}s")
        else:
            _debug("Using cached preloaded_models")
        s4 = time.time()
        self.model.evaluate(self.trainer, self.preloaded_models, self.n_splits)
        _debug(f"model.evaluate finished in {time.time() - s4:.2f}s total={time.time() - t0:.2f}s")

        y_pred = self.model.cv['test_pred']
        y_pred_fold = self.model.cv['test_pred_fold']
        y_true = np.array(self.data['target'])

        if 'atom_mask' in self.data:
            label_mask = self.data['atom_mask']
            index_mask = np.array(self.data['raw_data']['atom_mask'].tolist()).astype(int)
        else:
            atom_mask = None
            index_mask = None
        scalar = self.data['target_scaler']
        if scalar is not None:
            # y_pred = scalar.inverse_transform(y_pred, mask)
            y_true = scalar.inverse_transform(y_true, label_mask)


        # df = self.datahub.data['raw_data'].copy()
        # predict_cols = ['predict_' + col for col in self.target_cols]
        # if self.task == 'multiclass' and self.config.multiclass_cnt is not None:
        #     prob_cols = ['prob_' + str(i) for i in range(self.config.multiclass_cnt)]
        #     df[prob_cols] = y_pred
        #     df[predict_cols] = np.argmax(y_pred, axis=1).reshape(-1, 1)
        # elif self.task in ['classification', 'multilabel_classification']:
        #     threshold = joblib.load(open(os.path.join(self.load_model, 'threshold.dat'), "rb"))
        #     prob_cols = ['prob_' + col for col in self.target_cols]
        #     df[prob_cols] = y_pred
        #     df[predict_cols] = (y_pred > threshold).astype(int)
        # else:
        #     prob_cols = predict_cols
        #     df[predict_cols] = y_pred


        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
        if not (y_true == -1.0).all().all():
            metrics = self.trainer.metrics.cal_metric(y_true, y_pred, mask=label_mask)
            logger.info("final predict metrics score: \n{}".format(metrics))
            if self.save_path:
                joblib.dump(metrics, os.path.join(self.save_path, 'test_metric.result'))
        else:
            # df.drop(self.target_cols, axis=1, inplace=True)
            pass
        if self.save_path:
            prefix = data.split('/')[-1].split('.')[0] if isinstance(data, str) else 'test'
            self.save_predict(df, self.save_path, prefix)
            logger.info("pipeline finish!")
        self.cv_pred = y_pred
        self.cv_pred_fold = y_pred_fold
        self.cv_true = y_true
        self.cv_label_mask = label_mask
        self.cv_index_mask = index_mask

        return y_pred
    
    def save_predict(self, data, dir, prefix):
        run_id = 0
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            folders = [x for x in os.listdir(dir)]
            while prefix + f'.predict.{run_id}' + '.csv' in folders:
                run_id += 1
        name = prefix + f'.predict.{run_id}' + '.csv'
        path = os.path.join(dir, name)
        data.to_csv(path)
        logger.info("save predict result to {}".format(path))
