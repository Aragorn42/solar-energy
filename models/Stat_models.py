import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pmdarima as pm
import threading
from sklearn.ensemble import GradientBoostingRegressor

class Naive_repeat(nn.Module):
    def __init__(self, configs):
        super(Naive_repeat, self).__init__()
        self.pred_len = configs.pred_len
        
    def forward(self, x):
        B,L,D = x.shape
        x = x[:,-1,:].reshape(B,1,D).repeat(self.pred_len,axis=1)
        return x # [B, L, D]

class Naive_thread(threading.Thread):
    def __init__(self,func,args=()):
        super(Naive_thread,self).__init__()
        self.func = func
        self.args = args
        self.results = None

    def run(self):
        self.results = self.func(*self.args)
    
    def return_result(self):
        threading.Thread.join(self)
        return self.results

"""
def _arima(seq,pred_len,bt,i):
    model = pm.auto_arima(seq)
    forecasts = model.predict(pred_len) 
    return forecasts,bt,i
"""

def _arima(seq, pred_len, bt, i):
    try:
        # ✅ 1. 确保是 CPU numpy
        if isinstance(seq, torch.Tensor):
            seq = seq.detach().cpu().numpy().flatten()
        elif not isinstance(seq, np.ndarray):
            seq = np.array(seq).flatten()

        # ✅ 2. 避免全 0 或 NaN 序列
        if np.all(np.isnan(seq)) or np.all(seq == 0):
            forecast = np.zeros(pred_len)
        else:
            import pmdarima as pm
            model = pm.auto_arima(seq, suppress_warnings=True, error_action='ignore')
            forecast = model.predict(n_periods=pred_len)

        return forecast, bt, i

    except Exception as e:
        print(f"[ARIMA Thread Error] bt={bt}, i={i}, err={e}")
        # 返回安全值，防止 results 缺失
        return np.zeros(pred_len), bt, i

class Arima(nn.Module):
    """
    Extremely slow, please sample < 0.1
    """
    def __init__(self, configs):
        super(Arima, self).__init__()
        self.pred_len = configs.pred_len
        self.dummy = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        result = np.zeros([x.shape[0],self.pred_len,x.shape[2]])
        threads = []
        for bt,seqs in tqdm(enumerate(x)):
            for i in range(seqs.shape[-1]):
                seq = seqs[:,i]
                one_seq = Naive_thread(func=_arima,args=(seq,self.pred_len,bt,i))
                threads.append(one_seq)
                threads[-1].start()
        for every_thread in tqdm(threads):
            forcast,bt,i = every_thread.return_result()
            result[bt,:,i] = forcast

        # ✅ 确保返回 torch.Tensor（兼容 loss 计算）
        result = torch.tensor(result, dtype=torch.float32, device=x.device)
        return result # [B, L, D]

def _sarima(season,seq,pred_len,bt,i):
    model = pm.auto_arima(seq, seasonal=True, m=season)
    forecasts = model.predict(pred_len) 
    return forecasts,bt,i

class SArima(nn.Module):
    """
    Extremely extremely slow, please sample < 0.01
    """
    def __init__(self, configs):
        super(SArima, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.season = 24
        if 'Ettm' in configs.data_path:
            self.season = 12
        elif 'ILI' in configs.data_path:
            self.season = 1
        if self.season >= self.seq_len:
            self.season = 1

    def forward(self, x):
        result = np.zeros([x.shape[0],self.pred_len,x.shape[2]])
        threads = []
        for bt,seqs in tqdm(enumerate(x)):
            for i in range(seqs.shape[-1]):
                seq = seqs[:,i]
                one_seq = Naive_thread(func=_sarima,args=(self.season,seq,self.pred_len,bt,i))
                threads.append(one_seq)
                threads[-1].start()
        for every_thread in tqdm(threads):
            forcast,bt,i = every_thread.return_result()
            result[bt,:,i] = forcast
        return result # [B, L, D]

def _gbrt(seq,seq_len,pred_len,bt,i):
    model = GradientBoostingRegressor()
    model.fit(np.arange(seq_len).reshape(-1,1),seq.reshape(-1,1))
    forecasts = model.predict(np.arange(seq_len,seq_len+pred_len).reshape(-1,1))  
    return forecasts,bt,i

class GBRT(nn.Module):
    def __init__(self, configs):
        super(GBRT, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
    
    def forward(self, x):
        result = np.zeros([x.shape[0],self.pred_len,x.shape[2]])
        threads = []
        for bt,seqs in tqdm(enumerate(x)):
            for i in range(seqs.shape[-1]):
                seq = seqs[:,i]
                one_seq = Naive_thread(func=_gbrt,args=(seq,self.seq_len,self.pred_len,bt,i))
                threads.append(one_seq)
                threads[-1].start()
        for every_thread in tqdm(threads):
            forcast,bt,i = every_thread.return_result()
            result[bt,:,i] = forcast
        return result # [B, L, D]

class LSTM_baseline(nn.Module):
    def __init__(self, configs):
        super(LSTM_baseline, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.hidden_size = 64
        self.num_layers = 2

        self.lstm = nn.LSTM(
            input_size=configs.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, configs.c_out)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, batch_y=None, *args, **kwargs):
        """
        为了兼容 PatchTST 训练框架的调用方式：
        x_enc       <- batch_x
        x_mark_enc  <- batch_x_mark（这里不用）
        x_dec       <- dec_inp（这里不用）
        x_mark_dec  <- batch_y_mark（这里不用）
        batch_y     <- 真实 y（这里不用）
        """
        x = x_enc  # [B, seq_len, D]

        out, _ = self.lstm(x)          # [B, seq_len, hidden_size]
        out = out[:, -1:, :]           # 取最后一个时间步 [B, 1, hidden_size]
        out = self.fc(out)             # [B, 1, c_out]
        out = out.repeat(1, self.pred_len, 1)  # [B, pred_len, c_out]
        return out