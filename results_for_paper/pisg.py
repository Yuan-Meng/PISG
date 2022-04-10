import glob
from functools import reduce
import pandas as pd
import numpy as np
# import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
sns.set(font_scale=2)

def extract_from_qualtrics(n_aliens, conditions, has_extra):
    """extract responses from raw qualtrics data into a dataframe"""

    # loop through conditions to collect dataframes
    dfs = []
    for condition in conditions:
        # 6 aliens (Experiment 1)
        if n_aliens == 6:
            df = extract_6aliens("../data/raw/", condition, has_extra)
            df["condition"] = condition
        # 20 aliens (Experiments 2-4)
        elif n_aliens == 20:
            df = extract_20aliens("../data/raw/", condition, has_extra)
        # wrong number of aliens
        else:
            print("please choose 6 or 20 aliens")
            break
        # append dataframe to list
        dfs.append(df)

    # return a single dataframe
    return pd.concat(dfs, ignore_index=True)

def extract_6aliens(data_dir, condition_name, extended):
    """extract responses from a qualtrics survey with 6 aliens"""

    # exp2 filenames contain keyword "social"
    filenames = glob.glob(data_dir + "*social*")

    # select file from extended version
    if extended:
        filename = [
            filename
            for filename in filenames
            if (condition_name in filename and "extend" in filename)
        ][0]
    # select file from non-extended version
    else:
        filename = [
            filename
            for filename in filenames
            if (condition_name in filename and "extend" not in filename)
        ][0]

    # read file as dataframe
    df_raw = pd.read_csv(filename)

    # response columns
    infers = [f"Q{i}.3_13" for i in range(8, 24)]
    scans = [f"Q{i}.4" for i in range(8, 24)]

    # only extended version has 4 questions re: Alex
    if extended:
        others = ["Q2.1", "Q24.1", "Q25.1_1", "Q26.1_1", "Q26.1_2", "Q27.1_1"]
    else:
        others = ["Q2.1", "Q24.1"]

    # drop first two rows
    df_raw = df_raw.iloc[2:].reset_index(drop=True)

    # select above columns
    df_sel = df_raw[infers + scans + others]

    # rename columns
    infer_dict = dict(zip(infers, [f"infer{i}" for i in range(1, 17)]))
    scan_dict = dict(zip(scans, [f"scan{i}" for i in range(1, 17)]))
    other_dict = {
        "Q2.1": "age",
        "Q24.1": "strategy",
        "Q25.1_1": "knowledge",
        "Q26.1_1": "precision",
        "Q26.1_2": "recall",
        "Q27.1_1": "fairness",
    }
    rename_dict = {**infer_dict, **scan_dict, **other_dict}
    df_sel.rename(columns=rename_dict, inplace=True)

    # replace texts with numeric values
    value_dict = {
        "Strongly agree": 7,
        "Agree": 6,
        "Agree somewhat": 5,
        "Neutral": 4,
        "Disagree somewhat": 3,
        "Disagree": 2,
        "Strongly disagree": 1,
        "I want to scan this visitor.": 1,
        "I want to let this visitor pass.": 0,
    }

    df_sel.replace(value_dict, inplace=True)

    # add id and condition
    if extended:
        df_sel["condition"] = condition_name + "_ext"
    else:
        df_sel["condition"] = condition_name
    df_sel["id"] = "exp1_" + df_sel["condition"] + "_" + df_sel.index.astype(str)

    # drop participants who didn't finish
    return df_sel

def extract_20aliens(data_dir, condition_name, has_reminder):
    """extract responses from a qualtrics survey with 20 aliens"""

    # new data with "no reminder"
    if not has_reminder:
        filenames = glob.glob(data_dir + "*no_reminders*")
    # old data with reminders
    else:
        filenames = [
            filename
            for filename in glob.glob(data_dir + "*both_24*")
            if "no_reminders" not in filename
        ]

    # read file from given condition
    filename = [filename for filename in filenames if condition_name in filename][0]
    df_raw = pd.read_csv(filename)

    # drop first two rows
    df_raw = df_raw.iloc[2:].reset_index(drop=True)

    # select columns to keep
    infers = [f"Q{i}.3_13" for i in range(8, 32)]
    scans = [f"Q{i}.4" for i in range(8, 32)]
    others = ["Q2.1", "Q32.1", "Q33.1_1", "Q34.1_1", "Q34.1_2", "Q35.1_1", "Q36.1"]
    df_sel = df_raw[infers + scans + others]

    # rename columns
    infer_dict = dict(zip(infers, [f"infer{i}" for i in range(1, 25)]))
    scan_dict = dict(zip(scans, [f"scan{i}" for i in range(1, 25)]))
    other_dict = {
        "Q2.1": "age",
        "Q32.1": "strategy",
        "Q33.1_1": "knowledge",
        "Q34.1_1": "precision",
        "Q34.1_2": "recall",
        "Q35.1_1": "fairness",
        "Q36.1": "prolific_id",
    }
    rename_dict = {**infer_dict, **scan_dict, **other_dict}
    df_sel.rename(columns=rename_dict, inplace=True)

    # replace texts with numeric values
    value_dict = {
        "Strongly agree": 7,
        "Agree": 6,
        "Agree somewhat": 5,
        "Neutral": 4,
        "Disagree somewhat": 3,
        "Disagree": 2,
        "Strongly disagree": 1,
        "I want to scan this visitor.": 1,
        "I want to let this visitor pass.": 0,
    }

    df_sel.replace(value_dict, inplace=True)

    # add id and condition
    df_sel["condition"] = condition_name
    if not has_reminder:
        df_sel["id"] = "exp4_" + df_sel["condition"] + "_" + df_sel.index.astype(str)
    elif "20to4" in filename:
        df_sel["id"] = "exp2_" + df_sel["condition"] + "_" + df_sel.index.astype(str)
    else:
        df_sel["id"] = "exp3_" + df_sel["condition"] + "_" + df_sel.index.astype(str)

    return df_sel

def extract_20aliens_check(data_dir):
    """extract responses from a qualtrics survey with 20 aliens"""

    # only one file in "check rate only"
    filename = glob.glob(data_dir + "*check_24aliens*")[0]
    df_raw = pd.read_csv(filename)

    # drop first two rows
    df_raw = df_raw.iloc[2:].reset_index(drop=True)

    # select columns to keep
    infers = [f"Q{i}.3_13" for i in range(8, 32)]
    scans = [f"Q{i}.4" for i in range(8, 32)]
    others = ["Q2.1", "Q32.1", "Q33.1_1", "Q34.1_1", "Q34.1_2", "Q35.1_1", "Q36.1"]
    df_sel = df_raw[infers + scans + others]

    # rename columns
    infer_dict = dict(zip(infers, [f"infer{i}" for i in range(1, 25)]))
    scan_dict = dict(zip(scans, [f"scan{i}" for i in range(1, 25)]))
    other_dict = {
        "Q2.1": "age",
        "Q32.1": "strategy",
        "Q33.1_1": "knowledge",
        "Q34.1_1": "precision",
        "Q34.1_2": "recall",
        "Q35.1_1": "fairness",
        "Q36.1": "prolific_id",
    }
    rename_dict = {**infer_dict, **scan_dict, **other_dict}
    df_sel.rename(columns=rename_dict, inplace=True)

    # replace texts with numeric values
    value_dict = {
        "Strongly agree": 7,
        "Agree": 6,
        "Agree somewhat": 5,
        "Neutral": 4,
        "Disagree somewhat": 3,
        "Disagree": 2,
        "Strongly disagree": 1,
        "I want to scan this visitor.": 1,
        "I want to let this visitor pass.": 0,
    }

    df_sel.replace(value_dict, inplace=True)

    # add id and condition
    df_sel["condition"] = "check"
    df_sel["id"] = "exp2_" + df_sel["condition"] + "_" + df_sel.index.astype(str)

    return df_sel

def reshape_to_long(df, targets, n_trials):
    """extract target features and save in long format"""

    # columns to use as ID variables
    id_cols = ["id", "experiment", "condition"]

    # empty list to save long-format dataframe for each target
    dfs = []

    # iterate through all target features (e.g., guess, scan)
    for target in targets:
        # names of target columns
        target_cols = [f"{target}" + str(i) for i in range(1, n_trials + 1)]
        # collect ID and target columns in wide format
        df_wide = df[id_cols + target_cols]
        # convert to long format
        df_long = pd.melt(
            df_wide,
            id_vars=id_cols,
            value_vars=target_cols,
            var_name=f"type_{target}",
            value_name=target,
        ).reset_index()

        # append to collection
        dfs.append(df_long)

    # join all long-format dataframes on index and ID column
    df_merged = reduce(lambda x, y: pd.merge(x, y, on=["index"] + id_cols), dfs)

    # only return columns we need
    df_merged = df_merged.drop(["index", "type_scan"], axis=1)

    # drop those who didn't pass
    df_merged.dropna(inplace=True)

    # convert numeric variables to integers
    df_merged["infer"] = df_merged["infer"].astype(int)
    df_merged["scan"] = df_merged["scan"].astype(int)

    return df_merged

def fill_6aliens_content(design_dir, df):
    """fill in the content of each trial according to design"""

    # get trial design
    trial_info = pd.read_csv(design_dir + "6aliens_16trials/design_6aliens.csv")[
        ["n_check", "n_hit", "question"]
    ]

    # JOIN with trial_info to get numbers of checks + hits
    df_merged = trial_info.merge(df, left_on="question", right_on="type_infer")

    # only keep selected columns
    return df_merged[
        ["id", "experiment", "condition", "n_check", "n_hit", "infer", "scan"]
    ]

def fill_20aliens_content(design_dir, df):
    """fill in the content of each trial according to design"""

    # load all 20 aliens design
    trial_info = pd.read_csv(design_dir + "20aliens_24trials/design_20aliens.csv")

    # replace with new condition names
    trial_info["condition"].replace(
        {
            "4to20": "20_counter",
            "8to16": "16_counter",
            "12to12": "12_counter",
            "16to8": "8_counter",
            "20to4": "4_counter",
        },
        inplace=True,
    )

    # add groups to the dataframe
    df["group"] = df["type_infer"].apply(
        lambda x: chr(ord("A") + (int(x.replace("infer", "")) - 1))
    )

    # "Outcomes Revealed"
    df_revealed = df.merge(
        trial_info[["group", "n_check", "n_hit", "trial", "block", "condition"]],
        on=["condition", "group"],
    )

    # "Outcomes Unknown" (really hacky...)
    df_unknown = df[df["condition"] == "unknown"]
    df_unknown["condition"].replace({"unknown": "4_counter"}, inplace=True)
    df_unknown = df_unknown.merge(
        trial_info[["group", "n_check", "n_hit", "trial", "block", "condition"]],
        on=["condition", "group"],
    )
    df_unknown["condition"].replace({"4_counter": "unknown"}, inplace=True)

    # put together and return
    return pd.concat([df_revealed, df_unknown], ignore_index=True)

def get_final_df(df_merged, group_size):

    # compute sample check and hit rates
    df_merged["n_total"] = group_size
    df_merged["check_rate"] = df_merged["n_check"] * 1.0 / group_size
    df_merged["hit_rate"] = df_merged["n_hit"] * 1.0 / df_merged["n_check"]
    df_merged["human_hit"] = df_merged["infer"] * 1.0 / group_size

    # select columns needed for modeling and analysis
    sel = [
        "id",
        "experiment",
        "condition",
        "n_check",
        "n_hit",
        "infer",
        "scan",
        "check_rate",
        "hit_rate",
        "human_hit",
    ]

    # select given columns
    df_sel = df_merged[sel]

    # add trial content
    df_sel["content"] = (
        df_sel["n_hit"].map(str) + " out of " + df_sel["n_check"].map(str)
    )

    # return final dataframe
    return df_sel


def infer_hit_rate(choice_data, model_type, n_total, n_check, n_hit):
    """infer hit rate for a single trial in given condition using chosen model"""

    # process model inputs
    n_total = np.array([n_total])  # total encountered
    n_check = np.array([n_check])  # total checked
    n_hit = np.array([n_hit])  # total hit

    # specify model
    with pm.Model() as model:
        # choice model is used in CRO + CHR models
        if model_type in ["CRO", "CHR"]:
            # choice model
            intercept = pm.Normal("intercept", 0, sd=1)
            beta = pm.Normal("beta", 0, sd=1)
            likelihood = pm.invlogit(intercept + beta * choice_data["human_hit"])
            logit = pm.Bernoulli(
                name="logit", p=likelihood, observed=choice_data["scan"]
            )

        # all 3 models need to infer hit rate theta
        theta = pm.Beta("theta", alpha=1, beta=1)

        # only CRO and CHR models make use of check rate mu
        if model_type in ["CRO", "CHR"]:
            mu = pm.Deterministic("mu", 1 / (1 + tt.exp(-(theta * beta + intercept))))
            checked = pm.Binomial("checked", n=n_total, p=mu, observed=n_check)

        # only HRO + CHR models make use of observed hit rates
        if model_type in ["HRO", "CHR"]:
            hit = pm.Binomial("hit", n=n_check, p=theta, observed=n_hit)

    # compute MAP estimates
    with model:
        map_estimates = pm.find_MAP(progressbar=False)

    # return estimates of theta
    return map_estimates["theta"]    

def infer_choice_parameters(df_final):
    """given all choices made by each person, infer parameters in their logistic function"""

    # initialize empty list to collect all estimates
    choice_params = []

    # loop through each unique participant
    for id in df_final["id"].unique():

        # initialize empty dictionary to collect each person's estimates
        choice_param = {}

        # add participant id
        choice_param["id"] = id

        # isolate one person's choice data
        choice_data = df_final[df_final["id"] == id][["human_hit", "scan"]]

        # specify choice model
        with pm.Model() as model:
            intercept = pm.Normal("intercept", 0, sd=1)
            beta = pm.Normal("beta", 0, sd=1)
            likelihood = pm.invlogit(intercept + beta * choice_data["human_hit"])
            logit = pm.Bernoulli(
                name="logit", p=likelihood, observed=choice_data["scan"]
            )

        # obtain map estimates
        with model:
            map_estimates = pm.find_MAP(progressbar=False)

        # add intercept and slope to dictionary
        choice_param["intercept"] = map_estimates["intercept"]
        choice_param["beta"] = map_estimates["beta"]

        # append dictionary to list
        choice_params.append(choice_param)

    # return dictionary     
    return choice_params

def make_predictions(df, design_dir, group_size):

    # get design filename
    if group_size == 6:
        filename = "6aliens_16trials/design_6aliens.csv"
    else:
        filename = "20aliens_24trials/design_20aliens.csv"

    # collect model predictions
    preds = []

    # loop over all experiments
    for experiment in df["experiment"].unique():
        # loop over all conditions in given experiment
        for condition in df[df["experiment"] == experiment]["condition"].unique():
            # load trial design for given condition
            if group_size == 6:
                # only one design in the 6 aliens
                pred = pd.read_csv(design_dir + filename)[["trial", "n_check", "n_hit"]]
            else:
                # all designs in 20 aliens
                designs = pd.read_csv(design_dir + filename)
                designs["condition"].replace(
                    {
                        "4to20": "20_counter",
                        "8to16": "16_counter",
                        "12to12": "12_counter",
                        "16to8": "8_counter",
                        "20to4": "4_counter",
                    },
                    inplace=True,
                )
                # choose design for given condition
                if condition == "unknown":
                    pred = designs[designs["condition"] == "4_counter"][
                        ["trial", "n_check", "n_hit"]
                    ]
                else:
                    pred = designs[designs["condition"] == condition][
                        ["trial", "n_check", "n_hit"]
                    ]

            # elaborate on trial contents
            pred["content"] = (
                pred["n_hit"].map(str) + " out of " + pred["n_check"].map(str)
            )

            # add experiment, condition, and group size
            pred["experiment"], pred["condition"], pred["n_total"] = (
                experiment,
                condition,
                group_size,
            )

            # isolate choice data in given experiment + condition
            choice_data = df[
                (df["experiment"] == experiment) & (df["condition"] == condition)
            ][["human_hit", "scan"]]

            # only CRO can make predictions in "Outcomes Unknown"
            if condition == "unknown":
                pred["CRO"] = pred.apply(
                    lambda x: infer_hit_rate(
                        choice_data, "CRO", x["n_total"], x["n_check"], x["n_hit"]
                    ),
                    axis=1,
                )
                # fill predictions of other models with 0
                pred["HRO"], pred["CHR"] = 0, 0
            # all 3 models can make predictions in "Outcomes Revealed"
            else:
                for model_type in ["CRO", "HRO", "CHR"]:
                    pred[model_type] = pred.apply(
                        lambda x: infer_hit_rate(
                            choice_data,
                            model_type,
                            x["n_total"],
                            x["n_check"],
                            x["n_hit"],
                        ),
                        axis=1,
                    )

            # add dataframe to collection
            preds.append(pred)

    # return dataframe with all predictions
    return pd.concat(preds).reset_index(drop=True)