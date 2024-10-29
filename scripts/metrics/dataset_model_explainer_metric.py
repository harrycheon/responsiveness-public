import sys
import os
import numpy as np
import pandas as pd
from itertools import product

sys.path.append(os.getcwd())
import psutil
import argparse
from src.ext import fileutils
from src.paths import *
from reach.src import ReachableSetDatabase

settings = {
    "data_name": "german",
    "action_set_name": "complex_nD",
    "model_type": "logreg",
    "explainer_type": "rij",
    "method_name": "reach",
    "overwrite": False,
    "masker": None,
}

ppid = os.getppid()
process_type = psutil.Process(ppid).name()
if process_type not in ("Code Helper (Plugin)"):  # add your favorite IDE process here
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--explainer_type", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--overwrite", default=settings["overwrite"], action="store_true")
    # parser.add_argument("--seed", type=int, default=settings["random_seed"])
    args, _ = parser.parse_known_args()
    settings.update(vars(args))

# load dataset
data = fileutils.load(get_data_file(**settings))

action_set = fileutils.load(get_action_set_file(**settings))
db = ReachableSetDatabase(action_set=action_set, path=get_reachable_db_file(**settings))

audit_results = fileutils.load(get_audit_results_file(**settings))
audit_df = pd.DataFrame.from_dict(audit_results, orient="index")

K = 4
DESC_STATS = ['min', '50%', 'max', 'mean']

def main():
    metric = {}

    neg_pred_idx = audit_df[~audit_df['orig_prediction']].index

    rij_df = fileutils.load(get_rij_file(**settings))
    rij_df = rij_df[rij_df['u_index'].isin(neg_pred_idx)]
    rij_df['rij'] = (rij_df['recourse_count'] / rij_df['marginal_action_count']).fillna(0)

    if settings['explainer_type'] == 'rij':
        rij_df['exp_abs'] = rij_df['rij']
        melted = (
            pd.DataFrame(
                index=neg_pred_idx, 
                columns=np.arange(data.d)
                )
            .melt(ignore_index=False)
            .reset_index()
            .rename(columns={'variable': 'feature'})
            .drop(columns='value')
            )
    else:
        exp_obj = fileutils.load(get_explainer_file(**settings))
        df = exp_obj.get_explanations()['values']
        df.rename({feat: idx for idx, feat in enumerate(exp_obj.data.names.X)}, axis=1, inplace=True)
        df = df.iloc[neg_pred_idx]

        melted = df.melt(var_name='feature', value_name='exp', ignore_index=False).reset_index()
        melted[f'exp_abs'] = melted['exp'].abs()

    mrged = melted.merge(
        rij_df, 
        left_on=['index', 'feature'], 
        right_on=['u_index', 'feature'], 
        how='left'
        )
    mrged.fillna(0, inplace=True)
    mrged['exp_rank'] = mrged.groupby('index')[['exp_abs']].rank(ascending=False, method='first')

    top_k = mrged[(mrged['exp_rank'] <= K)]

    metric_names = ['AAN_exp_size', 'AAN_actionable_feature_count', 'AAN_recourse_feature_count']
    dummy = pd.Series(np.zeros(K+1))  # dummy series to fill in missing count values (0 to K)

    # 1. Number of features in AAN
    cnt_df = top_k.groupby('index')['exp_abs'].apply(np.count_nonzero)
    aan_all_idx = np.repeat(cnt_df, data.cnt[cnt_df.index])
    cnt_cnt_1 = aan_all_idx.value_counts()

    top_k = top_k[top_k['exp_abs'] > 0]

    # 2. Number of actionable features among top-K features (would be reported in AAN)
    top_k_act_feats = top_k[top_k['feature'].isin(action_set.actionable_features)]
    act_feat_cnt = top_k_act_feats.groupby('index')['feature'].size()
    act_feat_all_idx = np.repeat(act_feat_cnt, data.cnt[act_feat_cnt.index])
    cnt_cnt_2 = act_feat_all_idx.value_counts()

    # 2-1. Proportion of actionable features among top-K features
    act_feat_prop = act_feat_cnt / cnt_df.loc[act_feat_cnt.index]
    metric['AAN_actionable_feature_prop_mean'] = act_feat_prop.mean()

    # 3. Number of features in AAN that provide recourse (rij > 0)
    cnt_recourse = top_k.groupby('index')[['rij']].apply(lambda x: (x > 0).sum())
    cnt_recourse_all_idx = np.repeat(cnt_recourse['rij'], data.cnt[cnt_recourse.index])
    cnt_cnt_3 = cnt_recourse_all_idx.value_counts()

    # 3-1. Proportion of features in AAN that provide recourse
    recourse_prop = cnt_recourse['rij'] / cnt_df.loc[cnt_recourse.index]
    metric['AAN_recourse_feature_prop_mean'] = recourse_prop.mean()

    # 4. Proportion of Providing Reasons without Recourse (rwor)
    fixed_idx = audit_df[(~audit_df['orig_prediction']) & (~audit_df['recourse_exists'])]
    top_k_size = top_k.groupby('index').size().reset_index().rename(columns={0: 'size'})
    fixed_exp_size = fixed_idx.merge(top_k_size, left_index=True, right_on='index', how='left')['size'].fillna(0)
    rwor = (fixed_exp_size > 0).mean()
    
    metric['reasons_wo_recourse_prop'] = rwor

    # 5. Giving wrong reasons (all reasons do not provide recourse)
    top_k_rec = top_k.groupby('index')['recourse_count'].sum().reset_index()
    all_feat_df = mrged.groupby('index')['recourse_count'].sum().reset_index()
    oneD_rec = all_feat_df[all_feat_df['recourse_count'] > 0]
    wrong = (top_k_rec.loc[top_k_rec['index'].isin(oneD_rec['index']), 'recourse_count'] == 0).mean()

    metric['wrong_reasons_prop'] = wrong

    metric_df = pd.concat([cnt_cnt_1, cnt_cnt_2, cnt_cnt_3, dummy], axis=1).fillna(0).astype(int).sort_index()

    # Calculate mean values for each metric
    metric_df = metric_df.append(
        metric_df.apply(lambda x: (x.index * x).sum() / x.sum(), axis=0).fillna(0),
        ignore_index=True
        )

    for (metric_col_idx, metric_name), cnt_idx in product(enumerate(metric_names), np.arange(K+2)):
        m_name = f'{metric_name}_{cnt_idx}' if cnt_idx <= K else f'{metric_name}_mean'
        metric[m_name] = metric_df.iloc[cnt_idx, metric_col_idx]

    fileutils.save(metric, get_metrics_file(**settings), overwrite=True)

if __name__ == "__main__":
    main()