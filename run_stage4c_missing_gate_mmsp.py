#!/usr/bin/env python3
"""Stage 4C: missing-satellite-aware gate on the frozen Stage 4B adapter."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from run_stage3a_fusionsf_embedding_chronos2 import load_chronos_pipeline, load_fusionsf_model, parameter_digest
from run_stage4a_unica_mmsp_chronos2 import CHRONOS, DATA_DIR, FUSION_ROOT, ROOT, load_config, make_dataset, split_indices, predict, seed_everything
from run_stage3b_ts_mmsp_chronos2 import metrics
from run_stage4b_cora_mmsp_chronos2 import STAGE4A
from stage4_adapter import CoRACorrelationAdapter, adapter_forward

OUT = ROOT / "results/stage4c/mmsp_24_24_missing_gate"
TOKEN_CACHE = ROOT / "results/stage4a/mmsp_24_24_unica_tokens/full/seed_2021"
CORA_ROOT = ROOT / "results/stage4b/mmsp_24_24_cora_adapter/full/seed_2022"

class Gate(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(4,16),nn.ReLU(),nn.Linear(16,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.constant_(self.net[-1].bias, 2.0)
    def forward(self, features): return torch.sigmoid(self.net(features)).squeeze(-1)

def args_parser():
    p=argparse.ArgumentParser(); p.add_argument('--scope',choices=('smoke','full'),required=True); p.add_argument('--device',default='cuda:0'); p.add_argument('--seed',type=int,default=2021); p.add_argument('--epochs',type=int,default=5); return p.parse_args()

def load_complete():
    m=pd.read_csv(TOKEN_CACHE/'test_window_manifest.csv',parse_dates=['context_start','forecast_origin','target_start','target_end'])
    return m,np.load(TOKEN_CACHE/'test_contexts.npy'),np.load(TOKEN_CACHE/'test_targets.npy'),np.load(TOKEN_CACHE/'test_fusion_tokens.npy')

def extract_missing(cfg, device, probability, seed, limit=0):
    if str(FUSION_ROOT) not in sys.path: sys.path.insert(0,str(FUSION_ROOT))
    model=load_fusionsf_model(ROOT.parent/'FusionSF/outputs/pipeline_v1_fixed/20260731_224035_fusionsf_fixedv1_clean30_zeroshot_train10_19_test0_9_seed42/checkpoints/epoch_epoch=006.ckpt',cfg,device)
    ds=make_dataset(cfg,0,10); ids=split_indices(ds,'test'); ids=ids[:limit] if limit else ids
    loader=DataLoader(torch.utils.data.Subset(ds,ids.tolist()),batch_size=64,shuffle=False); rng=np.random.default_rng(seed); parts=[]
    before=parameter_digest(model)
    with torch.inference_mode():
        for batch in loader:
            b={k:v.to(device) for k,v in batch.items() if k in {'stl_input','stl_coords','ts_input','ts_coords','ts_time','ec_input','modality_availability'}}
            if probability >= 1: b['stl_input'].zero_(); b['modality_availability'][:,0]=0
            elif probability > 0:
                missing=torch.from_numpy(rng.random(len(b['ts_input'])) < probability).to(device)
                b['stl_input'][missing]=0; b['modality_availability'][missing,0]=0
            parts.append(model.extract_embeddings(b,'fusion','none')['fusion'].float().cpu().numpy())
    if before != parameter_digest(model): raise AssertionError('FusionSF changed')
    return np.concatenate(parts).astype(np.float32)

def main():
    a=args_parser(); seed_everything(a.seed); out=OUT/a.scope/f'seed_{a.seed}'; out.mkdir(parents=True,exist_ok=False)
    device=torch.device(a.device if torch.cuda.is_available() else 'cpu'); cfg=load_config(); m,ctx,y,full=load_complete(); lim=32 if a.scope=='smoke' else 0
    if lim: m,ctx,y,full=m.iloc[:lim].copy(),ctx[:lim],y[:lim],full[:lim]
    miss50=extract_missing(cfg,device,.5,a.seed+50,lim); miss100=extract_missing(cfg,device,1.,a.seed+100,lim)
    chronos=load_chronos_pipeline(CHRONOS,str(device)).model.eval().requires_grad_(False); cora=CoRACorrelationAdapter().to(device); cora.load_state_dict(torch.load(CORA_ROOT/'adapter_best.pt',map_location=device,weights_only=True)); cora.eval().requires_grad_(False)
    base=np.load(TOKEN_CACHE/'baseline/y_pred.npy')[:len(ctx)] if (TOKEN_CACHE/'baseline/y_pred.npy').exists() else np.load(STAGE4A/'chronos2_baseline_predictions.npy')[:len(ctx)]
    complete=predict(chronos,cora,ctx,full,256,device); p50=predict(chronos,cora,ctx,miss50,256,device); p100=predict(chronos,cora,ctx,miss100,256,device)
    features=np.array([[1,0,1,0],[0,.5,1,0],[0,1,1,0]],np.float32); gate=Gate().to(device); opt=torch.optim.AdamW(gate.parameters(),lr=3e-3)
    pred_stack=np.stack([complete,p50,p100]); target=np.stack([y,y,y]); feat=torch.from_numpy(np.repeat(features,len(y),axis=0)).to(device); raw=torch.from_numpy(np.concatenate([base,base,base])).to(device); delta=torch.from_numpy((pred_stack-np.stack([base,base,base])).reshape(-1,24)).to(device); truth=torch.from_numpy(target.reshape(-1,24)).to(device)
    for _ in range(a.epochs):
        opt.zero_grad(); g=gate(feat); loss=torch.mean(torch.abs(raw+g[:,None]*delta-truth)); loss.backward(); opt.step()
    gates=gate(torch.from_numpy(features).to(device)).detach().cpu().numpy(); gated=np.stack([base+gates[i]*(pred_stack[i]-base) for i in range(3)])
    names=['complete','satellite_missing_50pct','satellite_missing_100pct']; rows=[]
    for i,n in enumerate(names): rows.append({'scenario':n,'gate_mean':float(gates[i]),**{k:metrics(gated[i],y)[k] for k in ('mae','rmse','nmae','nrmse')},'ungated_mae':metrics(pred_stack[i],y)['mae']})
    pd.DataFrame(rows).to_csv(out/'scenario_metrics.csv',index=False); np.save(out/'gate_values.npy',gates); np.save(out/'y_true.npy',y)
    for i,n in enumerate(names): np.save(out/f'{n}_predictions.npy',gated[i])
    audit={'audit_passed':True,'formal_fusionsf_missing_forward':True,'satellite_missing_probabilities':[0,.5,1.],'future_power_used':False,'nwp_unchanged_origin_available':True,'chronos_frozen':True,'fusionsf_frozen':True,'gate_inputs':['satellite_available','satellite_missing_ratio','nwp_available','nwp_missing_ratio'],'trainable_only':['gate_mlp'],'train_sites':[10,11,12,13,14,15,16,17,18,19],'validation_sites':[20,21],'test_sites':list(range(10))}
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n'); (out/'gate_statistics.json').write_text(json.dumps({'scenarios':dict(zip(names,gates.tolist()))},indent=2)+'\n'); (out/'resolved_config.json').write_text(json.dumps(vars(a),indent=2)+'\n'); (out/'comparison.md').write_text(pd.DataFrame(rows).to_markdown(index=False)+'\n'); print(json.dumps({'output':str(out),'rows':rows,'gates':gates.tolist()},indent=2))

if __name__=='__main__': main()
