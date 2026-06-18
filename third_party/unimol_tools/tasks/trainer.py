# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function
from ast import Load

import logging
import copy
import os
import pandas as pd
import numpy as np
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch.optim import Adam, AdamW
from torch.nn.utils import clip_grad_norm_
from transformers.optimization import get_linear_schedule_with_warmup
from ..utils import Metrics
from ..utils import logger
from .split import Splitter
from tqdm import tqdm

import time
import sys


class Trainer(object):
    def __init__(self, save_path=None, **params):
        self.save_path = save_path
        self.task = params.get('task', None)
        self.max_atoms = params.get('max_atoms', 510)
        self.max_tokens = self.max_atoms+2
        self.optim_key = params.get('optim_key', 'adam')
        self.grad_accum_steps = params.get('grad_accum_steps', 1)
        if self.task != 'repr':
            self.metrics_str = params['metrics']
            self.metrics = Metrics(self.task, self.metrics_str)
        self._init_trainer(**params)


    def _init_trainer(self, **params):
        ### init common params ###
        self.split_method = params.get('split_method', '5fold_random')
        self.split_seed = params.get('split_seed', 42)
        self.seed = params.get('seed', 42)
        self.set_seed(self.seed)
        self.splitter = Splitter(self.split_method, self.split_seed)
        self.logger_level = int(params.get('logger_level', 1))
        ### init NN trainer params ###
        self.learning_rate = float(params.get('learning_rate', 1e-4))
        # print(f"trainer.py learning_rate{self.learning_rate}")
        # self.selected_atoms = params.get('selected_atoms', ['all'])
        self.batch_size = params.get('batch_size', 32)
        self.max_epochs = params.get('epochs', 50)
        self.warmup_ratio = params.get('warmup_ratio', 0.1)
        self.patience = params.get('patience', 10)
        self.max_norm = params.get('max_norm', 1.0)
        self.cuda = params.get('cuda', True)
        self.amp = params.get('amp', False)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and self.cuda else "cpu")
        # 梯度缩放器：在混合精度训练中使用，将梯度乘以一个缩放因子（通常是小于1的数值），可以减小训练期间的数值不稳定性
        self.scaler = torch.cuda.amp.GradScaler(
        ) if self.device.type == 'cuda' and self.amp == True else None

    def decorate_batch(self, batch, feature_name=None):
        return self.decorate_torch_batch(batch)

    def decorate_graph_batch(self, batch):
        net_input, net_target = {'net_input': batch.to(
            self.device)}, batch.y.to(self.device)
        if self.task in ['classification', 'multiclass', 'multilabel_classification']:
            net_target = net_target.long()
        else:
            net_target = net_target.float()
        return net_input, net_target

    def decorate_torch_batch(self, batch):
        """function used to decorate batch data
        """
        net_input, net_target, net_mask  = batch
        if isinstance(net_input, dict):
            # Some inference batches contain optional fields with value None.
            # Skip device transfer for those entries instead of crashing on `.to(...)`.
            net_input, net_target, net_mask = {
                k: (v.to(self.device) if v is not None else None)
                for k, v in net_input.items()
            }, net_target, net_mask
        else:
            net_input, net_target, net_mask = {
                'net_input': net_input.to(self.device)
            }, net_target, net_mask
        if self.task == 'repr':
            net_target = None
            return net_input, net_target, (net_mask.to(self.device) if net_mask is not None else None)
        elif self.task in ['classification', 'multiclass', 'multilabel_classification']:
            net_target = net_target.long()
        else:
            net_target = net_target.float()
        return (
            net_input,
            net_target.to(self.device) if net_target is not None else None,
            net_mask.to(self.device) if net_mask is not None else None,
        )

    def fit_predict(self, model, train_dataset, valid_dataset, loss_func, activation_fn, dump_dir, fold, target_scaler, feature_name=None):
        model = model.to(self.device)
        train_dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=model.batch_collate_fn,
            drop_last=True,
        )
        # remove last batch, bs=1 can not work on batchnorm1d
        min_val_loss = float("inf")
        max_score = float("-inf")
        wait = 0
        ### init optimizer ###
        num_training_steps = int(len(train_dataloader) * self.max_epochs // self.grad_accum_steps)
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        if self.optim_key == 'adam':
            optimizer = Adam(model.parameters(), lr=self.learning_rate, eps=1e-6)
        elif self.optim_key == 'adamw':
            optimizer = AdamW(model.parameters(), lr=self.learning_rate, eps=1e-6, weight_decay=0.01)
        # 学习率调度器：学习率会在前 num_warmup_steps 步中线性增加，以达到初始的学习率设定。然后，学习率将在接下来的 (num_training_steps - num_warmup_steps) 步中线性减小，直到达到非常小的值或者训练结束。
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        for epoch in range(self.max_epochs):
            model = model.train()
            # Progress Bar 进度条
            start_time = time.time()
            batch_bar = tqdm(total=len(train_dataloader), dynamic_ncols=True,
                             leave=False, position=0, desc='Train', ncols=5)
            trn_loss = []
            for batch_idx, batch in enumerate(train_dataloader):
                # print(f"batch{batch}")
                net_input, net_target, net_mask = self.decorate_batch(
                    batch, feature_name)
                # optimizer.zero_grad()  # Zero gradients
                if self.scaler and self.device.type == 'cuda':
                    # print(f"trainer.py amp this yes")
                    with torch.cuda.amp.autocast():
                        outputs = model(**net_input)
                        batch_max_tokens = outputs.shape[1]
                        
                        net_mask_batch = net_mask[:, :batch_max_tokens].to(self.device)
                        net_target_batch = net_target[:, :batch_max_tokens].masked_select(net_mask_batch).to(self.device)
                        masked_outputs = outputs.masked_select(net_mask_batch).to(self.device)
                        # print(f"  masked_outputs{masked_outputs[:10]}  net_target_batch {net_target_batch[:10]}")
                        loss = loss_func(masked_outputs, net_target_batch) / self.grad_accum_steps

                        # loss = loss_func(outputs, net_target_batch).masked_select(net_mask_batch).mean()

                else:
                    # print(f"trainer.py 000amp")
                    with torch.set_grad_enabled(True):
                        outputs = model(**net_input)
                        batch_max_tokens = outputs.shape[1]

                        net_mask_batch = net_mask[:, :batch_max_tokens].to(self.device)
                        net_target_batch = net_target[:, :batch_max_tokens].masked_select(net_mask_batch).to(self.device)
                        masked_outputs = outputs.masked_select(net_mask_batch).to(self.device)
                        # print(f"  masked_outputs{masked_outputs.shape}  net_target_batch {net_target_batch.shape}")
                        loss = loss_func(masked_outputs, net_target_batch) / self.grad_accum_steps

                        # net_target_batch = net_target[:, :batch_max_tokens].to(self.device)
                        # net_mask_batch = net_mask[:, :batch_max_tokens].to(self.device)
                        # loss = loss_func(outputs, net_target_batch).masked_select(net_mask_batch).mean()
                trn_loss.append(float(loss.data))
                # tqdm lets you add some details so you can monitor training as you train.
                # batch_bar.set_postfix(
                #     Epoch="Epoch {}/{}".format(epoch+1, self.max_epochs),
                #     loss="{:.04f}".format(float(sum(trn_loss) / (i + 1))),
                #     lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
                batch_bar.set_postfix(
                    Epoch="Epoch {}/{}".format(epoch+1, self.max_epochs),
                    loss="{:.06f}".format(float(trn_loss[batch_idx])),
                    lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])))
                if self.scaler and self.device.type == 'cuda':
                    # This is a replacement for loss.backward()
                    # 梯度缩放器在前向传播期间将梯度缩放到较小的范围（通常是 FP16）
                    self.scaler.scale(loss).backward()

                    if (((batch_idx+1) % self.grad_accum_steps == 0) or ((batch_idx+1) == len(train_dataloader))):
                        # unscale the gradients of optimizer's assigned params in-place
                        # 在执行权重更新之前，必须将梯度恢复到原始的数值范围（通常是 FP32）
                        self.scaler.unscale_(optimizer)
                        # Clip the norm of the gradients to max_norm.
                        # 梯度裁剪：防止梯度爆炸问题，梯度的范数超过max_norm时，就会执行梯度裁剪操作
                        clip_grad_norm_(model.parameters(), self.max_norm)                     
                        # This is a replacement for optimizer.step()
                        self.scaler.step(optimizer)
                        scheduler.step()
                        self.scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if (((batch_idx+1) % self.grad_accum_steps == 0) or ((batch_idx+1) == len(train_dataloader))):
                        clip_grad_norm_(model.parameters(), self.max_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                # scheduler.step()
                batch_bar.update()

            batch_bar.close()
            total_trn_loss = np.mean(trn_loss)

            y_preds, val_loss, metric_score = self.predict(
                model, valid_dataset, loss_func, activation_fn, dump_dir, fold, target_scaler, epoch, load_model=False, feature_name=feature_name)
            end_time = time.time()
            total_val_loss = np.mean(val_loss)
            _score = list(metric_score.values())[0]
            _metric = list(metric_score.keys())[0]
            message = 'Epoch [{}/{}] train_loss: {:.6f}, val_loss: {:.6f}, val_{}: {:.6f}, lr: {:.6f}, ' \
                '{:.1f}s'.format(epoch+1, self.max_epochs,
                                 total_trn_loss, total_val_loss,
                                 _metric, _score,
                                 optimizer.param_groups[0]['lr'],
                                 (end_time - start_time))
            logger.info(message)
            is_early_stop, min_val_loss, wait, max_score = self._early_stop_choice(
                wait, total_val_loss, min_val_loss, metric_score, max_score, model, dump_dir, fold, self.patience, epoch)
            if is_early_stop:
                break
        # 所有epoch中最好的那次当成返回值，epoch传入的有点问题
        y_preds, _, _ = self.predict(model, valid_dataset, loss_func, activation_fn,
                                     dump_dir, fold, target_scaler, epoch, load_model=True, feature_name=feature_name)
        return y_preds

    def _early_stop_choice(self, wait, loss, min_loss, metric_score, max_score, model, dump_dir, fold, patience, epoch):
        ### hpyerparameter need to tune if you want to use early stop, currently find use loss is suitable in benchmark test. ###
        if not isinstance(self.metrics_str, str) or self.metrics_str in ['loss', 'none', '']:
            # loss 作为早停 直接用trainer里面的早停函数
            is_early_stop, min_val_loss, wait = self._judge_early_stop_loss(
                wait, loss, min_loss, model, dump_dir, fold, patience, epoch)
        else:
            # 到metric进行判断
            is_early_stop, min_val_loss, wait, max_score = self.metrics._early_stop_choice(
                wait, min_loss, metric_score, max_score, model, dump_dir, fold, patience, epoch)
        return is_early_stop, min_val_loss, wait, max_score

    def _judge_early_stop_loss(self, wait, loss, min_loss, model, dump_dir, fold, patience, epoch):
        is_early_stop = False
        if loss <= min_loss:
            min_loss = loss
            wait = 0
            info = {'model_state_dict': model.state_dict()}
            os.makedirs(dump_dir, exist_ok=True)
            torch.save(info, os.path.join(dump_dir, f'model_{fold}.pth'))
        elif loss >= min_loss:
            wait += 1
            if wait == self.patience:
                logger.warning(f'Early stopping at epoch: {epoch+1}')
                is_early_stop = True
        return is_early_stop, min_loss, wait

    def predict(self, model, dataset, loss_func, activation_fn, dump_dir, fold, target_scaler=None, epoch=1, load_model=False, feature_name=None):
        model = model.to(self.device)
        if load_model == True:
            load_model_path = os.path.join(dump_dir, f'model_{fold}.pth')
            model_dict = torch.load(load_model_path, map_location=self.device)[
                "model_state_dict"]
            model.load_state_dict(model_dict)
            logger.info("load model success!")
            print('load model')
        dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=model.batch_collate_fn,
            num_workers=8,
        )
        model = model.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True,
                         position=0, leave=False, desc='val', ncols=5)
        val_loss = []
        val_len = []
        y_preds = []
        y_truths = []
        y_masks = []
        for batch_idx, batch in enumerate(dataloader):
            net_input, net_target, net_mask= self.decorate_batch(batch, feature_name)
            # Get model outputs
            with torch.no_grad():
                outputs = model(**net_input)
                batch_max_tokens = outputs.shape[1]
                if net_mask is None:
                    net_mask = torch.ones(
                        (outputs.shape[0], self.max_tokens),
                        dtype=torch.bool,
                        device=self.device,
                    )
                if not load_model:

                    net_mask_batch = net_mask[:, :batch_max_tokens]
                    net_target_batch = net_target[:, :batch_max_tokens].masked_select(net_mask_batch).to(self.device)
                    masked_outputs = outputs.masked_select(net_mask_batch.to(self.device))
                    loss = loss_func(masked_outputs, net_target_batch)
                    val_loss.append(float(loss.data))
                    # val_loss.append(0)

            padded_outputs = np.pad(activation_fn(outputs).cpu().numpy(), ((0, 0), (0, self.max_tokens - batch_max_tokens)), mode='constant')
            # print(f"trainer.py outputs {outputs.shape} padded_outputs{padded_outputs.shape} ")
            y_preds.append(padded_outputs)
            # y_preds.append(activation_fn(outputs).cpu().numpy())
            y_truths.append(net_target.detach().cpu().numpy())
            y_masks.append(net_mask.detach().cpu().numpy())
            # if not load_model:
            #     batch_bar.set_postfix(
            #         Epoch="Epoch {}/{}".format(epoch+1, self.max_epochs),
            #         loss="{:.04f}".format(float(np.sum(val_loss) / (i + 1))))
            if not load_model:
                batch_bar.set_postfix(
                    Epoch="Epoch {}/{}".format(epoch+1, self.max_epochs),
                    loss="{:.06f}".format(float(val_loss[batch_idx])))


            batch_bar.update()
        y_preds = np.concatenate(y_preds)
        y_truths = np.concatenate(y_truths)
        y_masks = np.concatenate(y_masks)


        try:
            label_cnt = model.output_dim
        except:
            label_cnt = None

        # 每个epoch计算一次，运行后y_preds和y_truths被求逆了
        if target_scaler is not None:
            # print(f"trainer.py target_scaler is not None epoch{epoch} y_truths{y_truths.shape} y_preds{y_preds.shape}")
            # print(f"trainer.py target_scaler is not None epoch{epoch} y_truths{y_truths[y_masks][-20:]} y_preds{y_preds[y_masks][-20:]}")
       
            inverse_y_preds = target_scaler.inverse_transform(y_preds, mask=y_masks)
            inverse_y_truths = target_scaler.inverse_transform(y_truths, mask=y_masks)
            metric_score = self.metrics.cal_metric(
                inverse_y_truths, inverse_y_preds, mask=y_masks, label_cnt=label_cnt) if not load_model else None
            # print(f"trainer.py target_scaler is not None epoch{epoch} y_truths{y_truths.shape} y_preds{y_preds.shape}")
            # print(f"trainer.py target_scaler is not None epoch{epoch} y_truths{y_truths[y_masks][-20:]} y_preds{y_preds[y_masks][-20:]}")
       
        else:
            # print(f"trainer.py target_scaler is None epoch{epoch} y_truths{y_truths.shape} y_preds{y_preds.shape}")
            metric_score = self.metrics.cal_metric(
                y_truths, y_preds, mask=y_masks, label_cnt=label_cnt) if not load_model else None
        batch_bar.close()
        # y_preds求逆了
        return y_preds, val_loss, metric_score
    
    def predict_cv(self, model_list, dataset, activation_fn, target_scaler=None, feature_name=None):
        # model = model.to(self.device)
        dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=model_list[0].batch_collate_fn,
            num_workers=8,
        )
        # model = model.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True,
                         position=0, leave=False, desc='val', ncols=5)
        y_preds_list = [[] for _ in range(len(model_list))]
        y_masks = []
        for batch_idx, batch in enumerate(dataloader):
            net_input, net_target, net_mask= self.decorate_batch(batch, feature_name)
            # Get model outputs
            with torch.no_grad():
                outputs_list = [model(**net_input) for model in model_list]
            batch_max_tokens = outputs_list[0].shape[1]
            if net_mask is None:
                net_mask = torch.ones(
                    (outputs_list[0].shape[0], self.max_tokens),
                    dtype=torch.bool,
                    device=self.device,
                )

            padded_outputs_list = [
                np.pad(
                    activation_fn(outputs).cpu().numpy(),
                    ((0, 0), (0, self.max_tokens - outputs.shape[1])),
                    mode='constant'
                )
                for outputs in outputs_list
            ]
            for i, padded_outputs in enumerate(padded_outputs_list):
                y_preds_list[i].append(padded_outputs)
            # y_preds.append(padded_outputs)
            y_masks.append(net_mask.detach().cpu().numpy())

            batch_bar.update()
        y_preds_list = [np.concatenate(y_preds) for y_preds in y_preds_list]
        y_masks = np.concatenate(y_masks)

        if target_scaler is not None:
            inverse_y_preds = [target_scaler.inverse_transform(y_preds, mask=y_masks) for y_preds in y_preds_list]
        
        batch_bar.close()
        return y_preds_list, None, None

    def inference(self, model, dataset, feature_name=None, return_repr=True):
        model = model.to(self.device)
        dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=model.batch_collate_fn,
        )
        model = model.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True,
                         position=0, leave=False, desc='inference', ncols=5)
        repr_dict = {"cls_repr": [], "atomic_reprs": []}
        for i, batch in enumerate(dataloader):
            net_input, _, net_mask = self.decorate_batch(batch, feature_name)
            with torch.no_grad():
                outputs = model(return_repr=return_repr, **net_input)
                assert isinstance(outputs, dict)
                for key, value in outputs.items():

                    if key == 'atomic_reprs':
                        batchsize, n, dim = value.shape
                        for i in range(batchsize):
                            current_mask = net_mask[i, :n]
                            current_output = value[i]
                            indices = torch.nonzero(current_mask).squeeze()
                            result = current_output[indices]
                            repr_dict[key].extend(result.cpu().numpy().reshape(-1, dim))
                    else:
                        repr_dict[key].extend([value.cpu().numpy()])
                    # if isinstance(value, list):
                        # value_list = [item.cpu().numpy() for item in value]
                        # repr_dict[key].extend(value_list)
                    # else:
                    #     repr_dict[key].extend([value.cpu().numpy()])
            batch_bar.update()
        repr_dict["cls_repr"] = np.concatenate(repr_dict["cls_repr"]).tolist()
        batch_bar.close()

        return repr_dict

    def set_seed(self, seed):
        """function used to set a random seed
        Arguments:
            seed {int} -- seed number, will set to torch and numpy
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)


def NNDataLoader(feature_name=None, dataset=None, batch_size=None, shuffle=False, collate_fn=None, drop_last=False, **kwargs):

    dataloader = TorchDataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=drop_last,
                                 **kwargs)
    return dataloader
