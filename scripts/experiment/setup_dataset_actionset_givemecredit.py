import os
import sys

# fmt:off
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

import pprint
pp = pprint.PrettyPrinter(depth=2)

from src.paths import *

from reach.src import ActionSet, ReachableSetDatabase
from reach.src.constraints import *
from src.ext import fileutils
from src.ext.data import BinaryClassificationDataset
from src.ext.training import extract_predictor
from reach.src.utils import check_processing_loss, tabulate_actions

settings = {
    "data_name": "givemecredit",
    "action_set_names": ["complex_nD"],
    "check_processing_loss": True,
    "generate_reachable_sets": False,
    "fold_id": "K05N01",
    "random_seed": 2338,
    }


def process_dataset(raw_df):
    """
    `NoSeriousDlqin2yrs`:Person did not experience 90 days past due delinquency or worse
    `Age`: Age of borrower in years
    `NumberOfDependents`: Number of dependents in family excluding themselves (spouse, children etc.)
    #
    `MonthlyIncome`: Monthly income
    #
    `DebtRatio`: Monthly debt payments, alimony, living costs divided by monthy gross income
    `RevolvingUtilizationOfUnsecuredLines`: Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits
    #
    `NumberRealEstateLoansOrLines`: Number of mortgage and real estate loans including home equity lines of credit
    `NumberOfOpenCreditLinesAndLoans`: Number of Open loans (installment like car loan or mortgage) + Lines of credit (e.g. credit cards)
    #
    `NumberOfTime30-59DaysPastDueNotWorse`: Number of times borrower has been 30-59 days past due but no worse in the last 2 years.
    `NumberOfTime60-89DaysPastDueNotWorse`: Number of times borrower has been 60-89 days past due but no worse in the last 2 years.
    `NumberOfTimes90DaysLate`:Number of times borrower has been 90 days or more past due.
    """

    raw_df = pd.DataFrame(raw_df)
    raw_df = raw_df[raw_df.age >= 21]  # note: one person has age == 0

    df = pd.DataFrame()
    df["NotSeriousDlqin2yrs"] = raw_df["NotSeriousDlqin2yrs"]

    # Age
    df["Age_leq_24"] = (raw_df["age"] <= 24)
    df["Age_bt_25_to_30"] = (raw_df["age"] >= 25) & (raw_df["age"] < 30)
    df["Age_bt_30_to_59"] = (raw_df["age"] >= 30) & (raw_df["age"] <= 59)
    df["Age_geq_60"] = raw_df["age"] >= 60

    # Dependents
    df["NumberOfDependents_eq_0"] = raw_df["NumberOfDependents"] == 0
    df["NumberOfDependents_eq_1"] = raw_df["NumberOfDependents"] == 1
    df["NumberOfDependents_geq_2"] = raw_df["NumberOfDependents"] >= 2
    df["NumberOfDependents_geq_5"] = raw_df["NumberOfDependents"] >= 5

    # Debt Ratio
    df["DebtRatio_geq_1"] = raw_df["DebtRatio"] >= 1

    cash_thresholds = (3, 5, 10)  # old versions: (3, 7, 11)
    income = raw_df["MonthlyIncome"] / 1000
    for t in cash_thresholds:
        df[f"MonthlyIncome_geq_{t}K"] = income >= t

    utilization = raw_df["RevolvingUtilizationOfUnsecuredLines"]
    utilization_thresholds = (0.1, 0.2, 0.5, 0.7, 1.0)  # old versions: (3, 7, 11)
    for t in utilization_thresholds:
        df[f"CreditLineUtilization_geq_{t*100}"] = utilization >= t

    df["AnyRealEstateLoans"] = raw_df["NumberRealEstateLoansOrLines"] >= 1
    df["MultipleRealEstateLoans"] = raw_df["NumberRealEstateLoansOrLines"] >= 2
    df["AnyCreditLinesAndLoans"] = raw_df["NumberOfOpenCreditLinesAndLoans"] >= 1
    df["MultipleCreditLinesAndLoans"] = raw_df["NumberOfOpenCreditLinesAndLoans"] >= 2

    df["HistoryOfLatePayment"] = np.any(raw_df[["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse"]].values > 0, axis = 1)
    df["HistoryOfDelinquency"] = np.any(raw_df[["NumberOfTimes90DaysLate"]].values > 0, axis = 1)

    return df

    # Debt Ratio
def simple_1D(data):
    A = ActionSet(data.X_df)
    immutable_features = [
        "Age_leq_24",
        "Age_bt_25_to_30",
        "Age_bt_30_to_59",
        "Age_geq_60",
        "NumberOfDependents_eq_0",
        "NumberOfDependents_eq_1",
        "NumberOfDependents_geq_2",
        "NumberOfDependents_geq_5",
        "HistoryOfLatePayment",
        "HistoryOfDelinquency",
        ]

    A[immutable_features].actionable = False

    A["AnyRealEstateLoans"].lb = 0.0
    A["AnyRealEstateLoans"].ub = 1.0
    A["MultipleRealEstateLoans"].lb = 0.0
    A["MultipleRealEstateLoans"].ub = 1.0
    A["AnyCreditLinesAndLoans"].lb = 0.0
    A["AnyCreditLinesAndLoans"].ub = 1.0
    A["MultipleCreditLinesAndLoans"].lb = 0.0
    A["MultipleCreditLinesAndLoans"].ub = 1.0
    return A

def complex_1D(data):
    A = simple_1D(data)
    A["DebtRatio_geq_1"].step_direction = -1
    A["AnyRealEstateLoans"].step_direction = -1
    A["MultipleRealEstateLoans"].step_direction = -1
    A["AnyCreditLinesAndLoans"].step_direction = -1
    A["MultipleCreditLinesAndLoans"].step_direction = -1
    income_variables = [s for s in A.names if "MonthlyIncome_geq_" in s]
    A[income_variables].step_direction = 1
    return A

def complex_nD(data):

    A = complex_1D(data)

    A.constraints.add(
            constraint=ThermometerEncoding(
                    names=[s for s in A.names if "MonthlyIncome_geq_" in s],
                    step_direction=1,
                    )
            )

    A.constraints.add(
            constraint=ThermometerEncoding(
                    names=[s for s in A.names if "Utilization_geq_" in s],
                    step_direction=-1,
                    )
            )

    A.constraints.add(
            constraint=ThermometerEncoding(
                    names=["AnyRealEstateLoans", "MultipleRealEstateLoans"],
                    step_direction=-1,
                    )
            )

    A.constraints.add(
            constraint=ThermometerEncoding(
                    names=["AnyCreditLinesAndLoans", "MultipleCreditLinesAndLoans"],
                    step_direction=-1,
                    )
            )

    return A


# load raw dataset
loaded = BinaryClassificationDataset.read_csv(
        data_file=data_dir / settings["data_name"] / f"{settings['data_name']}"
        )

# process dataset
data_df = process_dataset(raw_df=loaded.df)

# create processed dataset
data = BinaryClassificationDataset.from_df(data_df)
data.generate_cvindices(
        strata=data.y,
        total_folds_for_cv=[1, 3, 4, 5],
        replicates=1,
        seed=settings["random_seed"],
        )
print(f"n = {data.n} points, of which {np.unique(data.X, axis = 0).shape[0]} are unique")

# train models to check change in feature processing
if settings["check_processing_loss"]:
    df_raw = loaded.df.loc[data_df.index]
    data_raw = BinaryClassificationDataset.from_df(df_raw)
    data_raw.cvindices = data.cvindices
    comp_results = check_processing_loss(
            data,
            data_raw,
            model_type="logreg",
            rebalance=None,
            fold_id="K05N01",
            fold_num_test=5,
            )
    pp.pprint(["TRAIN", comp_results["model"]['train'], "TEST", comp_results["model"]['test']])
    pp.pprint(["RAW_TRAIN", comp_results["model_raw"]['train'], "RAW_TEST", comp_results["model_raw"]['test']])

# create actionset
for name in settings["action_set_names"]:

    if name == "simple_1D":
        action_set = simple_1D(data)
    elif name == "complex_1D":
        action_set = complex_1D(data)
    elif name == "complex_nD":
        action_set = complex_nD(data)

    # more
    try:
        assert action_set.validate(data.X)
    except AssertionError:
        violations_df = action_set.validate(data.X, return_df=True)
        violations = ~violations_df.all(axis=0)
        violated_columns = violations[violations].index.tolist()
        print(violated_columns)
        raise AssertionError()

    # save dataset
    fileutils.save(
            data,
            path=get_data_file(settings["data_name"], action_set_name=name),
            overwrite=True,
            check_save=False,
            )

    # save actionset
    fileutils.save(
            action_set,
            path=get_action_set_file(settings["data_name"], action_set_name=name),
            overwrite=True,
            check_save=True,
            )

    # generate reachable set
    print(tabulate_actions(action_set))
    if settings["generate_reachable_sets"] and name == "complex_nD":
        db = ReachableSetDatabase(
                action_set=action_set,
                path=get_reachable_db_file(
                        data_name=settings["data_name"], action_set_name=name
                        ),
                )
        generation_stats = db.generate(data.X)
        print(generation_stats.n_points.describe())

        predictor = extract_predictor(comp_results['model']['model'], scaler = comp_results['model']['scaler'])

        # run sample audit
        audit_df = db.audit(X = data.X, clf = comp_results['model']['model'], scaler = comp_results['model']['scaler'], target = 1)

        # print audit results
        recourse_df = audit_df.query("yhat == False")
        n_total = len(recourse_df)
        n_responsive = recourse_df["recourse"].sum(axis=0)
        n_fixed = n_total-n_responsive
        p_fixed = n_fixed/n_total
        print(f"predictions without recourse: {p_fixed*100:1.1f}% ({n_fixed}/{n_total})")
        print('reachable point distribution')
        print(recourse_df["n_reachable"].describe())
        print('reachable point distribution')
        pp.pprint(tally(recourse_df['n_feasible']))