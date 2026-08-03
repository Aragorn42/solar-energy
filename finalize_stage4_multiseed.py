#!/usr/bin/env python3
"""Aggregate the required three-seed Stage 4A/4B formal results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
for stage in ("stage4a/mmsp_24_24_unica_tokens", "stage4b/mmsp_24_24_cora_adapter"):
    rows=[]
    for seed in (2021,2022,2023):
        p=ROOT/"results"/stage/"full"/f"seed_{seed}"/"comparison.json"
        c=json.loads(p.read_text()); rows.append({"seed":seed, **{k:v["mae"] for k,v in c["metrics"].items()}})
    df=pd.DataFrame(rows); out=ROOT/"results"/stage/"full"/"three_seed_summary.csv"; df.to_csv(out,index=False)
    summary={"seeds":[2021,2022,2023],"mean":df.mean(numeric_only=True).to_dict(),"std":df.std(numeric_only=True,ddof=1).to_dict()}
    if "cora_aligned" in df:
        d=df.cora_aligned-df.stage4a_cross_attention
        summary.update({"cora_minus_stage4a_mean":float(d.mean()),"cora_minus_stage4a_std":float(d.std(ddof=1)),"cora_better_seed_count":int((d<0).sum()),"stage4b_go":bool((d<0).sum()>=2 and d.mean()<0)})
    (out.parent/"three_seed_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
