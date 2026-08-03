#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results/stage4c/mmsp_24_24_missing_gate/full/seed_2021'; CACHE=ROOT/'results/stage4a/mmsp_24_24_unica_tokens/full/seed_2021'
m=pd.read_csv(CACHE/'test_window_manifest.csv',parse_dates=['context_start','forecast_origin','target_start','target_end']); y=np.load(OUT/'y_true.npy')
names=['complete','satellite_missing_50pct','satellite_missing_100pct']; preds=[np.load(OUT/f'{n}_predictions.npy') for n in names]
m.to_csv(OUT/'window_manifest.csv',index=False); pd.DataFrame({'split':['test'],'source':[str(CACHE/'test_fusion_tokens.npy')],'pooling':['none'],'token_shape':['[24,64]']}).to_csv(OUT/'embedding_manifest.csv',index=False)
site=[]; win=[]
for n,p in zip(names,preds):
    e=np.mean(np.abs(p-y),axis=1)
    for i,v in enumerate(e): win.append({'window_id':m.window_id.iloc[i],'site_id':int(m.site_id.iloc[i]),'scenario':n,'mae':float(v)})
    for s in sorted(m.site_id.unique()):
        mask=m.site_id.to_numpy()==s; err=p[mask]-y[mask]; site.append({'site_id':int(s),'scenario':n,'mae':float(np.mean(np.abs(err))),'rmse':float(np.sqrt(np.mean(err**2))),'nmae':float(np.mean(np.abs(err))),'nrmse':float(np.sqrt(np.mean(err**2)))})
pd.DataFrame(site).to_csv(OUT/'per_site_metrics.csv',index=False); pd.DataFrame(win).to_csv(OUT/'per_window_metrics.csv',index=False)
rows=pd.read_csv(OUT/'scenario_metrics.csv'); base=float(rows.loc[rows.scenario=='complete','mae'].iloc[0]); rows['relative_to_complete_pct']=(rows.mae-base)/base*100; rows.to_csv(OUT/'scenario_metrics.csv',index=False)
(OUT/'metrics.json').write_text(json.dumps({r.scenario:{k:float(r[k]) for k in ('mae','rmse','nmae','nrmse')} for _,r in rows.iterrows()},indent=2)+'\n')
payload={'complete_to_50pct_mae_absolute':float(rows.loc[1,'mae']-base),'complete_to_50pct_mae_relative_pct':float(rows.loc[1,'relative_to_complete_pct']),'complete_to_100pct_mae_absolute':float(rows.loc[2,'mae']-base),'complete_to_100pct_mae_relative_pct':float(rows.loc[2,'relative_to_complete_pct']),'gate_range':float(rows.gate_mean.max()-rows.gate_mean.min()),'gate_changes_material':False,'stage4c_go':False}
(OUT/'comparison.json').write_text(json.dumps(payload,indent=2)+'\n'); (OUT/'comparison.md').write_text(rows.to_string(index=False)+'\n\nGate range across scenarios: '+str(payload['gate_range'])+'. Gate did not materially respond to missingness; Stage 4C is a negative result and requires redesign before reuse.\n')
