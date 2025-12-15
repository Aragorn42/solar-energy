import numpy as np
import lightgbm as lgb
import os
import torch
import time

class Model:
    def __init__(self, args):
        self.args = args
        self.models = {}  # 存储 {channel_index: model}
        
        # GPU 参数配置
        self.lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'n_estimators': self.args.train_epochs,
            'learning_rate': self.args.learning_rate,
            'num_leaves': 31,
            'verbose': -1,
            'n_jobs': -1
        }
        if hasattr(args, 'use_gpu') and args.use_gpu:
            self.lgb_params.update({
                'device': 'cpu',
                'gpu_platform_id': getattr(args, 'gpu', 0),
                'gpu_device_id': getattr(args, 'gpu_id', 0)
            })

    def _dataloader_to_numpy(self, data_loader):
        """
        核心适配器：将 PyTorch DataLoader 转换为 LightGBM 可用的 Numpy 矩阵
        """
        print(f"Converting DataLoader to Numpy array (Pred Len: {self.args.pred_len})...")
        X_list = []
        y_list = []
        
        # 遍历 DataLoader
        for i, (batch_x, batch_y, _, _) in enumerate(data_loader):
            # batch_x shape: [Batch, Seq_Len, Channels]
            # batch_y shape: [Batch, Label_Len + Pred_Len, Channels]
            
            # 1. 构建特征 X
            # 将 [Batch, Seq, Ch] 展平为 [Batch, Seq * Ch]
            # 这相当于自动构建了所有 Lag 特征
            batch_x_numpy = batch_x.detach().cpu().numpy()
            B, S, C = batch_x_numpy.shape
            X_flat = batch_x_numpy.reshape(B, -1) # Shape: [B, S*C]
            
            # 2. 构建标签 Y
            # batch_y 通常包含 (Label部分 + Pred部分)
            # 我们只需要预测未来第 pred_len 个点，即序列的最后一个点
            batch_y_numpy = batch_y.detach().cpu().numpy()
            
            # 取最后一个时间步作为 Target (对应 t + pred_len)
            target_point = batch_y_numpy[:, -1, :] # Shape: [B, C]
            
            X_list.append(X_flat)
            y_list.append(target_point)
            
        # 拼接所有 Batch
        X_all = np.concatenate(X_list, axis=0) # [Total_Samples, Features]
        y_all = np.concatenate(y_list, axis=0) # [Total_Samples, Channels]
        
        print(f"Data shape - X: {X_all.shape}, y: {y_all.shape}")
        return X_all, y_all

    def train(self, train_loader, val_loader, setting_name):
        """
        接收 train_loader 和 val_loader 进行训练
        """
        # 1. 转换数据
        print("Processing Train Loader...")
        X_train, y_train = self._dataloader_to_numpy(train_loader)
        print("Processing Val Loader...")
        X_val, y_val = self._dataloader_to_numpy(val_loader)
        
        # 获取通道数量 (y_train.shape[1])
        num_channels = y_train.shape[1]
        print(f"Training {num_channels} models (one per channel)...")

        # 2. 逐通道训练
        for ch in range(num_channels):
            print(f"  > Training Channel {ch}...")
            
            # 获取该通道的 Target
            y_train_ch = y_train[:, ch]
            y_val_ch = y_val[:, ch]
            
            # 训练
            model = lgb.LGBMRegressor(**self.lgb_params)
            model.fit(
                X_train, y_train_ch,
                eval_set=[(X_val, y_val_ch)],
                callbacks=[lgb.early_stopping(self.args.patience), lgb.log_evaluation(0)]
            )
            self.models[ch] = model
            
        print("Training completed.")

    def test(self, test_loader, test_set, setting_name):
        """
        接收 test_loader 进行预测，利用 test_set 中的 scaler 进行反归一化
        """
        # 1. 转换数据
        print("Processing Test Loader...")
        X_test, y_test = self._dataloader_to_numpy(test_loader)
        num_channels = y_test.shape[1]
        
        preds_list = []
        
        # 2. 逐通道预测
        for ch in range(num_channels):
            if ch not in self.models:
                raise ValueError(f"Model for channel {ch} not found!")
            
            pred = self.models[ch].predict(X_test)
            preds_list.append(pred)
            
        # 3. 堆叠结果
        # preds_list: List of [N], len=C -> Stack -> [N, C]
        preds_array = np.stack(preds_list, axis=1)
        trues_array = y_test # [N, C]
        
        # 4. 调整维度为 [N, 1, C] (符合你的要求)
        preds_array = preds_array.reshape(preds_array.shape[0], 1, preds_array.shape[1])
        trues_array = trues_array.reshape(trues_array.shape[0], 1, trues_array.shape[1])
        
        print(f"Prediction Shape: {preds_array.shape}")

        # 5. 保存与反归一化
        folder_path = './results/solar/' + setting_name + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 保存归一化的预测值 (用于计算 metrics)
        np.save(os.path.join(folder_path, 'pred.npy'), preds_array)

        # 反归一化 (Inverse Transform)
        # 从 Dataset 对象中获取 Scaler
        if hasattr(test_set, 'inverse_transform'):
            print("Denormalizing using data_set.inverse_transform...")
            
            # inverse_transform 通常接收 (N, Seq, C) 或 (N, C)
            # 我们先把数据挤压回 (N, C) 传给 scaler，然后再变回 (N, 1, C)
            
            # 注意：PyTorch Dataset 的 inverse_transform 通常期望完整的 sequence 或 batch
            # 但这里我们只有一个点。大多数 StandardScaler 只关心最后一维 C
            
            p_tmp = preds_array.squeeze(1) # [N, C]
            t_tmp = trues_array.squeeze(1) # [N, C]
            
            try:
                # 尝试调用 data_set 的方法
                # 注意：有些库实现的 inverse_transform 是针对 tensor 的，有些是针对 numpy 的
                # 为了稳妥，我们手动用 scaler 属性如果存在
                if hasattr(test_set, 'scaler'):
                    scaler = test_set.scaler
                    # 假设是 sklearn scaler
                    p_denorm = scaler.inverse_transform(p_tmp)
                    t_denorm = scaler.inverse_transform(t_tmp)
                else:
                    # 尝试直接调用 data_set 的方法 (通常是 dataset 内部封装的)
                    p_denorm = test_set.inverse_transform(p_tmp)
                    t_denorm = test_set.inverse_transform(t_tmp)
            except Exception as e:
                print(f"Warning: Standard inverse_transform failed ({e}), trying manual calculation...")
                # 备用方案：手动读取 scaler 参数
                scaler = test_set.scaler
                mean = scaler.mean_
                scale = scaler.scale_
                p_denorm = p_tmp * scale + mean
                t_denorm = t_tmp * scale + mean

            # 恢复形状 [N, 1, C]
            preds_denorm = p_denorm.reshape(preds_array.shape)
            trues_denorm = t_denorm.reshape(trues_array.shape)
            
        else:
            preds_denorm = preds_array
            trues_denorm = trues_array

        # 保存真实值
        np.save(os.path.join(folder_path, 'y_pred.npy'), preds_denorm.astype(np.float32))
        np.save(os.path.join(folder_path, 'y_true.npy'), trues_denorm.astype(np.float32))
        
        print(f"Results saved to {folder_path}")