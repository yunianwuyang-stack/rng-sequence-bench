#!/usr/bin/env python3
"""Sensitivity audit for missing human responses in Brockbank & Vul (2024) RPS data.

Input: one row per human game-round CSV with columns game_id, round_index,
player_move; optional bot_move and outcome. If the source CSV is long-format with
`is_bot` and both human/bot rows in `player_move`, pass --is-bot-col and the script
will pivot bot moves before building the human sequence mask.

No missing-at-random assumption is made. The script describes missingness by
participant and trial, models its association with observed lagged history where
available, compares complete-case and masked summaries, and relocates each
participant's missing positions at random (preserving their missing count) as a
pattern-sensitivity reference. This is not an imputation procedure.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

K=3
MOVE={"rock":0,"paper":1,"scissors":2,"0":0,"1":1,"2":2}

def encode(x):
    if pd.isna(x): return -1
    return MOVE.get(str(x).strip().lower(), -1)

def make_panel(df, id_col, trial_col, move_col, bot_col=None, is_bot_col=None, bot_flag="1"):
    d=df.copy()
    d[trial_col]=pd.to_numeric(d[trial_col],errors="coerce")
    d=d[d[trial_col].between(1,300)].copy()
    if is_bot_col and is_bot_col in d.columns:
        bot_rows=d[d[is_bot_col].astype(str)==str(bot_flag)][[id_col,trial_col,move_col]].rename(columns={move_col:"bot_move"})
        human=d[d[is_bot_col].astype(str)!=str(bot_flag)].copy()
        if bot_col is None:
            d=human.merge(bot_rows,on=[id_col,trial_col],how="left")
            bot_col="bot_move"
        else: d=human
    d["_move"]=d[move_col].map(encode)
    ids=sorted(d[id_col].dropna().unique(),key=str)
    n=len(ids); choices=np.zeros((n,300),dtype=np.int8); mask=np.zeros((n,300),dtype=bool)
    bot=np.full((n,300),-1,dtype=np.int8)
    for i,gid in enumerate(ids):
        sub=d[d[id_col]==gid]
        for _,row in sub.iterrows():
            t=int(row[trial_col])-1; y=int(row["_move"])
            if y>=0: choices[i,t]=y; mask[i,t]=True
            if bot_col and bot_col in sub.columns:
                b=encode(row[bot_col]);
                if b>=0: bot[i,t]=b
    return ids,choices,mask,bot

def cyclic_rates(y,m):
    valid=m[:,1:] & m[:,:-1]
    if valid.sum()==0: return {"upgrade":None,"downgrade":None,"n_transitions":0}
    delta=(y[:,1:]-y[:,:-1])%3
    return {"upgrade":float(np.mean(delta[valid]==1)),"downgrade":float(np.mean(delta[valid]==2)),"n_transitions":int(valid.sum())}

def tv95(y,m):
    cnt=np.stack([((y==k)&m).sum(axis=1) for k in range(3)],axis=1)
    denom=m.sum(axis=1)
    ok=denom>0
    props=cnt[ok]/denom[ok,None]
    tv=.5*np.abs(props-1/3).sum(axis=1)
    return float(np.quantile(tv,.95)),float(np.median(tv)),int(ok.sum())

def logistic_fit(X,y):
    def obj(b):
        eta=X@b
        loss=np.logaddexp(0,eta).sum()-np.dot(y,eta)+.5e-5*np.dot(b[1:],b[1:])
        grad=X.T@(expit(eta)-y); grad[1:]+=.00001*b[1:]
        return loss,grad
    res=minimize(obj,np.zeros(X.shape[1]),jac=True,method="BFGS")
    return res.x

def randomize_observed_positions(y, m, rng):
    """Relocate valid response slots while preserving each person's observed move order."""
    yr = np.zeros_like(y)
    mr = np.zeros_like(m)
    for i in range(len(y)):
        vals = y[i, m[i]]
        if len(vals):
            pos = np.sort(rng.choice(m.shape[1], size=len(vals), replace=False))
            yr[i, pos] = vals
            mr[i, pos] = True
    return yr, mr


def randomized_position_reference(y, m, seed=8141, reps=500):
    """Randomly relocate each person's observed moves, preserving order and count."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        yr, mr = randomize_observed_positions(y, m, rng)
        rates = cyclic_rates(yr, mr)
        out.append({
            "upgrade": rates["upgrade"],
            "downgrade": rates["downgrade"],
            "valid_counts": mr.sum(axis=1).tolist(),
        })
    return out


def missingness_model(y,m,bot,seed=8141,boot=500):
    n,t=m.shape
    rows=[]
    for i in range(n):
        for j in range(1,t):
            prev= y[i,j-1] if m[i,j-1] else -1
            prevbot=bot[i,j-1]
            outcome=-1
            if prev>=0 and prevbot>=0:
                outcome=0 if prev==prevbot else (1 if (prev-prevbot)%3==1 else 2)
            rows.append((i,j,not m[i,j],prev,outcome))
    a=np.asarray(rows,dtype=float); pid=a[:,0].astype(int); trial=a[:,1]/299; miss=a[:,2]
    prev=a[:,3].astype(int); outcome=a[:,4].astype(int)
    cols=[np.ones(len(a)),trial]
    for k in [0,1,2]: cols.append((prev==k).astype(float))
    cols.extend([(prev<0).astype(float)])
    for k in [0,1,2]: cols.append((outcome==k).astype(float))
    cols.append((outcome<0).astype(float))
    X=np.column_stack(cols)
    b=logistic_fit(X,miss)
    rng=np.random.default_rng(seed); boot_beta=[]
    for _ in range(boot):
        take=rng.choice(n,n,replace=True)
        idx=np.concatenate([np.flatnonzero(pid==i) for i in take])
        if len(np.unique(miss[idx]))<2: continue
        try: boot_beta.append(logistic_fit(X[idx],miss[idx]))
        except Exception: pass
    bb=np.asarray(boot_beta)
    return {"n_person_trial_rows":int(len(a)),"missing_rate":float(miss.mean()),
            "trial_position_odds_ratio_per_100_trials":float(np.exp(b[1]*100/299)),
            "trial_position_or_cluster_bootstrap_ci95":np.exp(np.quantile(bb[:,1]*100/299,[.025,.975])).tolist() if len(bb) else None,
            "lagged_move_and_outcome_coefficients":b[2:].tolist(),
            "lagged_move_and_outcome_coefficient_names":[
                "previous_move_rock","previous_move_paper","previous_move_scissors",
                "previous_move_unavailable","previous_outcome_tie","previous_outcome_win",
                "previous_outcome_loss","previous_outcome_unavailable"
            ],
            "lagged_history_joint_likelihood_ratio_not_computed":True,
            "bootstrap_successful_replicates":int(len(bb)),
            "note":"Lagged move/outcome coefficients are descriptive associations. Missing current responses are never imputed; unavailable lagged outcomes are explicitly coded as unavailable."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("csv",type=Path); p.add_argument("output",type=Path)
    p.add_argument("--id-col",default="game_id"); p.add_argument("--trial-col",default="round_index")
    p.add_argument("--move-col",default="player_move"); p.add_argument("--bot-col",default="bot_move")
    p.add_argument("--is-bot-col",default="is_bot"); p.add_argument("--bot-flag",default="1")
    p.add_argument("--random-mask-reps",type=int,default=500); p.add_argument("--bootstrap-reps",type=int,default=200)
    p.add_argument("--seed",type=int,default=8141)
    a=p.parse_args(); df=pd.read_csv(a.csv)
    ids,y,m,bot=make_panel(df,a.id_col,a.trial_col,a.move_col,a.bot_col if a.bot_col in df.columns else None,
                           a.is_bot_col if a.is_bot_col in df.columns else None,a.bot_flag)
    n=len(ids)
    nmiss=(~m).sum(axis=1); by_trial=(~m).mean(axis=0)
    obs_rates=cyclic_rates(y,m); obs_tv=tv95(y,m)
    complete=m.all(axis=1)
    cc_rates=cyclic_rates(y[complete],m[complete]) if complete.any() else None
    cc_tv=tv95(y[complete],m[complete]) if complete.any() else None
    # Random-position reference preserves each person's observed moves in order
    # and does not impute unknown responses.
    ref=randomized_position_reference(y,m,a.seed,a.random_mask_reps)
    def interval(key):
        x=np.array([z[key] for z in ref if z[key] is not None],float)
        return np.quantile(x,[.025,.5,.975]).tolist() if len(x) else None
    for row in ref:
        row.pop("valid_counts", None)
    result={"source_file":a.csv.name,"participants":n,"trials":300,"missing_total":int((~m).sum()),
      "missing_rate":float((~m).mean()),"participants_with_missing":int((nmiss>0).sum()),
      "missing_per_participant":{"median":float(np.median(nmiss)),"mean":float(nmiss.mean()),"max":int(nmiss.max()),
                                  "quantiles":np.quantile(nmiss,[0,.25,.5,.75,.95,1]).tolist()},
      "missing_by_trial_position":{"first_50":float((~m[:,:50]).mean()),"middle_100":float((~m[:,50:150]).mean()),
                                   "last_150":float((~m[:,150:]).mean()),"per_trial":by_trial.tolist()},
      "observed_masked_metrics":{"tv95":obs_tv[0],"tv_median":obs_tv[1],"n_people":obs_tv[2],**obs_rates},
      "complete_case":{"participants":int(complete.sum()),"tv95":cc_tv[0] if cc_tv else None,
                        "tv_median":cc_tv[1] if cc_tv else None,"transitions":cc_rates},
       "random_position_reference":{"replicates":a.random_mask_reps,
          "cyclic_upgrade_q025_median_q975":interval("upgrade"),"cyclic_downgrade_q025_median_q975":interval("downgrade"),
          "note":"Marginal TV is omitted because random relocation preserves each person's observed category counts exactly; the reference is intended for order-sensitive transition metrics."},
      "lagged_history_missingness_model":missingness_model(y,m,bot,a.seed,boot=a.bootstrap_reps),
      "interpretation":"This is a missingness sensitivity analysis, not a MAR proof. Comparisons are conditional on the released participant cohort and observed response process."}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ["missing_by_trial_position","lagged_history_missingness_model"]},indent=2))
if __name__=="__main__": main()
