# GEFCom zone1 336→72 Power-only paired benchmark

## Status

`fair_benchmark=true`. Transformer and FusionSFSolar were trained independently by
`run_longExp_solarv2.py` and `Exp_Main`, using validation loss alone for checkpoint selection.
The native raw metrics were computed without clipping. No future weather or target values were
used, and no Chronos-2 embedding code was involved.

## Shared configuration

`dataset/GEFCom/task15.csv`, `custom_solar`, `features=S`, `target=zone1`, `freq=h`,
`seq_len=336`, `label_len=48`, `pred_len=72`, `enc_in=dec_in=c_out=1`, seed 2021,
100 maximum epochs, patience 10, batch size 64, learning rate 0.0005, MSE loss,
`d_model=512`, 8 heads, 3 encoder layers, 2 decoder layers, `d_ff=2048`, dropout 0.05,
`embed=timeF`, one iteration, and all debug batch limits disabled. The independently saved
resolved configurations differ only in `model`.

## Results

| Model | Params | Best epoch | Raw MAE | Raw MSE | Raw RMSE | Physical MAE | Physical RMSE |
| -- | -----: | ---------: | ------: | ------: | -------: | -----------: | ------------: |
| Transformer | 17,874,945 | 3 | 0.250246 | 0.205732 | 0.453577 | 0.063960 | 0.115930 |
| FusionSFSolar | 25,428,993 | 5 | 0.249972 | 0.201739 | 0.449154 | 0.063890 | 0.114799 |

Relative to Transformer, FusionSFSolar changes raw MAE by -0.1095%, raw MSE by -1.9409%,
raw RMSE by -0.9752%, physical MAE by -0.1095%, and physical RMSE by -0.9752% (negative is
lower/better). It uses 42.2605% more trainable parameters. This is a single-seed result.

Transformer stopped at epoch 13 and FusionSFSolar at epoch 15; both stopped by validation
early stopping. Training times were 326.69 s and 491.93 s, with peak allocated GPU memory
6,083,071,488 and 5,772,254,208 bytes respectively.

Both native test outputs have shape `(5568, 72, 1)`. The Dataset exposes 5,625 test windows,
while the native test DataLoader evaluates 5,568 because its established `drop_last=True`
behavior drops the final incomplete batch identically for both models.

## Protocol audit

The two 16 MB manifests contain all 5,625 origin timestamps and all 72 target timestamps for
every window. The comparison passed every check: dataset class, data SHA256, features, target,
frequency, sequence lengths, scaler fit rows, inverse flag, batch size, seed, all split borders,
scaler mean/scale, test-window count, every origin, every target timestamp, and metric function
path/source SHA256.

Machine-readable result: `reports/gefcom_zone1_pred72_power_only_fairness.json`.

## Anomalies

There was no OOM, NaN/Inf, or model collapse; raw prediction standard deviations are 0.979989
for Transformer and 0.968076 for FusionSFSolar. The first FusionSFSolar attempt trained normally
but failed while reloading a CUDA-indexed checkpoint after device visibility remapping. Loading
state dictionaries through CPU was added without changing training semantics, tests passed, and
the complete FusionSFSolar run was repeated from the fixed seed rather than reconstructing a
summary manually.
