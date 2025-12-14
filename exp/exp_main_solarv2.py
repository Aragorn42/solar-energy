from data_provider.data_factory_solarv2 import data_provider
from exp.exp_basic_solarv2 import Exp_Basic
from models import Transformer, DLinear, PatchTST, Stat_models, iTransformer, tcn, GBDT
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        if self.args.model == 'GBDT':
            return None  # GBDT 模型在训练/测试时单独处理
        model_dict = {
            'Transformer': Transformer,
            'DLinear': DLinear,
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'LSTM_baseline': Stat_models.LSTM_baseline,
            'tcn': tcn,
        }
        # model = model_dict[self.args.model].Model(self.args).float()
        model_class = model_dict[self.args.model]
        model = model_class(self.args).float() if hasattr(model_class, '__call__') else model_class.Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        if self.args.model == 'GBDT':
            return  # GBDT 模型在此处不进行训练
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        """
        测试阶段：
        1）仍然在标准化空间内计算 mse/mae/rse 等指标，便于快速对比（保持原始行为）；
        2）额外对目标通道做逆标准化，保存真实功率的预测值/真值到 y_pred.npy / y_true.npy；
        3）如果数据集是 Dataset_Custom_Solar（或至少提供 all_timestamps & border1），
           则计算并保存起报时刻 t_origin.npy；否则跳过起报时刻保存，避免报错。
        """
        test_data, test_loader = self._get_data(flag='test')
        if self.args.model == 'GBDT':
            print("Training GBDT model...")
            train_data, train_loader = self._get_data(flag='train')
            vali_data, vali_loader = self._get_data(flag='val')
            gbdt_model = GBDT.Model(self.args)
            gbdt_model.train(train_loader, vali_loader, setting)
            print("Testing GBDT model...")
            gbdt_model.test(test_loader, test_data, setting)
            return
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            )

        preds = []
        trues = []
        inputx = []

        """
        folder_path_vis = './test_results/' + setting + '/'
        if not os.path.exists(folder_path_vis):
            os.makedirs(folder_path_vis)
        """

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())

                """
                # 可选：简单可视化
                if i % 20 == 0:
                    input_arr = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input_arr[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input_arr[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path_vis, str(i) + '.pdf'))
                """

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()

        preds = np.array(preds)   # [num_batches, B, pred_len, C_out]
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])   # [N, pred_len, C_out]
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # ================= 保留原始在线指标（标准化空间） =================
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        # 仍然保存标准化空间的 pred.npy（兼容原有脚本）
        folder_path = './results/solar/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(os.path.join(folder_path, 'pred.npy'), preds)

        # ================== 关键改动 1：逆标准化（仅目标通道） ==================
        scaler = getattr(test_data, 'scaler', None)
        if scaler is not None:
            if self.args.features == 'S':
                mean = scaler.mean_[0]
                scale = scaler.scale_[0]
            else:
                # 对应 Dataset_Custom_Solar 中 df_data 的列顺序：target 被放在最后一列
                mean = scaler.mean_[-1]
                scale = scaler.scale_[-1]

            preds_denorm = preds * scale + mean
            trues_denorm = trues * scale + mean
        else:
            preds_denorm = preds
            trues_denorm = trues

        np.save(os.path.join(folder_path, 'y_pred.npy'), preds_denorm.astype(np.float32))
        np.save(os.path.join(folder_path, 'y_true.npy'), trues_denorm.astype(np.float32))

        # ================== 关键改动 2：起报时刻（带保底机制） ==================
        all_timestamps = getattr(test_data, 'all_timestamps', None)
        border1 = getattr(test_data, 'border1', None)

        if all_timestamps is not None and border1 is not None:
            num_samples = preds_denorm.shape[0]
            local_indices = np.arange(num_samples)
            global_hist_indices = border1 + local_indices + self.args.seq_len - 1

            ts_array = np.array(all_timestamps)
            t_origin = ts_array[global_hist_indices]

            np.save(os.path.join(folder_path, 't_origin.npy'), t_origin)
            print(f"[test] Saved t_origin.npy (datetime) to {folder_path}")
        else:
            # 非 Dataset_Custom_Solar 或未提供时间戳信息时，不保存起报时刻
            print("[test] Dataset has no 'all_timestamps'/'border1'; skip saving t_origin.npy")

        print(f"[test] Saved pred.npy, y_pred.npy, y_true.npy to {folder_path}")
        print("    y_pred shape:", preds_denorm.shape,
              "y_true shape:", trues_denorm.shape)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros(
                    [batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]
                ).float().to(batch_y.device)
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/solar/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(os.path.join(folder_path, 'real_prediction.npy'), preds)

        return