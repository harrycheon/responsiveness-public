import os
import sys
import psutil
import rich

sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import argparse
from tqdm.auto import tqdm
from reach.src import ReachableSetDatabase

DB_ACTION_SET_NAME = "complex_nD"

settings = {
    "data_name": "german",
    "action_set_name": "complex_nD",
    "model_type": "logreg",
}

ppid = os.getppid()
process_type = psutil.Process(ppid).name()
if process_type not in ("pycharm"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--action_set_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    args, _ = parser.parse_known_args()
    settings.update(vars(args))

from src.paths import *
from src.ext import fileutils

# load action set and processed data
data = fileutils.load(get_data_file(**settings))
action_set = fileutils.load(get_action_set_file(**settings))

# load database
db = ReachableSetDatabase(action_set=action_set, path=get_reachable_db_file(
    data_name=settings["data_name"], action_set_name=DB_ACTION_SET_NAME
))

# load processed model
model_results = fileutils.load(get_model_file(**settings))
clf = model_results["model"]
scaler = model_results["scaler"]
reformat = lambda x: x.reshape(1, -1) if x.ndim == 1 else x
if scaler is None:
    rescale = lambda x: reformat(x)
else:
    rescale = lambda x: scaler.transform(reformat(x))

# Helper functions
def calc_rij(x):
    """
    calculates rij
    
    :param x: input vector (1-d)
    :param infeasible: boolean flag to indicate whether this point has recourse or not
                       if True, returns 0 for all features (reduces computation)
    """
    rs = db[x]
    act_feats = rs.action_set.actionable_features

    return np.array([[f, *rs.recourse_scores(f, clf.predict, scaler)] for f in act_feats])

# run audit
null_action = np.zeros(data.d)
nan_action = np.repeat(np.nan, data.d)
results = {}
rij_df_rows = []

predictions = clf.predict(rescale(data.U))
for idx, (x, y, fx) in tqdm(list(enumerate(zip(data.U, data.y[data.u_idx], predictions)))):
    # pull reachable set
    R = db[x]
    flipped_idx = np.flatnonzero(clf.predict(rescale(R.X)) != fx)
    feasible = len(flipped_idx) > 0
    recourse_exists = R.complete and feasible

    results[idx] = {
        "y_true": y > 0,
        "orig_prediction": fx > 0,
        "flip_action_idx": flipped_idx if feasible else [],
        "actionable": R.actions.shape[0] > 0,
        "abstains": (R.complete == False) and not feasible,
        "recourse_exists": recourse_exists,
    }

    # rij calculation
    arr = calc_rij(x)
    idx_row = np.hstack(
        (np.array([idx] * arr.shape[0]).reshape(-1, 1), arr)
    )
    rij_df_rows.append(idx_row)

rij_df = pd.DataFrame(
    np.vstack(rij_df_rows), 
    columns=['u_index', 'feature', 'marginal_action_count', 'recourse_count']
    )

rij_df['rij'] = (rij_df['recourse_count'] / rij_df['marginal_action_count']).fillna(0)

# save results
fileutils.save(
    results,
    path=get_audit_results_file(**settings),
    overwrite=True,
    check_save=False,
)

fileutils.save(
    rij_df,
    path=get_rij_file(**settings),
    overwrite=True,
    check_save=False,
)