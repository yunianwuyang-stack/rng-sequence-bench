"""Fit a pooled Markov model to the public nine-key cohort and audit both starts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from rng_sequence_bench import audit_sequence_contexts, check_forecast_causality, check_prefix_preprocessing

COHORT_HASH = "8dded4707450d96ebac38e7a088f3d58ac16e13fd6956778e85e71ba3b715494"


def prepare(history, cutoff):
    prefix = history[:, :cutoff].astype(int)
    cells = 9*prefix[:, :-1] + prefix[:, 1:]
    count = np.bincount(cells.ravel(), minlength=81).reshape(9,9)
    probability = (count + .5) / (count.sum(axis=1,keepdims=True) + 4.5)
    return {"transition": probability, "smoothing_per_cell": .5}


def simulate_suffix(cdf, uniforms, previous=None):
    n, h = uniforms.shape
    out = np.empty((n,h), dtype=np.int16)
    if previous is None:
        out[:,0] = (9*uniforms[:,0]).astype(int)
    else:
        out[:,0] = (uniforms[:,0,None] >= cdf[previous]).sum(axis=1)
    for t in range(1,h):
        out[:,t] = (uniforms[:,t,None] >= cdf[out[:,t-1]]).sum(axis=1)
    return out


def run(cohort, replicates):
    with cohort.open("rb") as stream:
        digest=hashlib.file_digest(stream,"sha256").hexdigest()
    if digest != COHORT_HASH:
        raise ValueError("The input does not match the documented 830-person release matrix")
    y = np.load(cohort, allow_pickle=False)
    cut = 140
    state = prepare(y,cut)
    train_check = check_prefix_preprocessing(prepare,y,cut)
    def forecast(history):
        return state["transition"][history[:,:-1]]
    future_check = check_forecast_causality(forecast,y,cut,9,mode="all")
    cdf = state["transition"].cumsum(axis=1)
    cdf[:,-1] = 1.
    random = np.random.default_rng(127315)
    def pairs():
        for _ in range(replicates):
            u = random.random((len(y),y.shape[1]-cut))
            yield simulate_suffix(cdf,u), simulate_suffix(cdf,u,y[:,cut-1])
    audit = audit_sequence_contexts(y,probabilities=forecast(y),replicated_pairs=pairs(),
                                    fit_through=cut,seed=91351)
    return dict(cohort_sha256=digest, fitting_trials=[1,140], scoring_trials=[141,200],
                source="Angelike and Musch, OSF xwzup; analytic cohort matched to deposited model data",
                paired_seed=127315, audit_seed=91351, preprocessing=train_check,forecast_check=future_check,audit=audit)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort",type=Path)
    parser.add_argument("report",type=Path)
    parser.add_argument("--replicates",type=int,default=32)
    args=parser.parse_args()
    if args.replicates<2:parser.error("Supply at least two pairs")
    result=run(args.cohort,args.replicates)
    args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result["audit"]["holdout_forecasts"]["named_scores"]))
