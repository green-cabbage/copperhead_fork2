import os
import pandas as pd
import dask.dataframe as dd
from dask.distributed import get_worker
import pickle
import glob
import re
import uproot3
# import uproot as uproot3

import random
import string
from datetime import datetime
from hist import Hist
import numpy as np

def generate_unique_string(length):
    # Get current time as a formatted string
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # Generate a random part
    characters = string.ascii_letters + string.digits
    random_part = ''.join(random.choice(characters) for _ in range(length))
    # Combine timestamp and random part
    return f"{timestamp}_{random_part}"


def mkdir(path):
    try:
        os.mkdir(path)
    except Exception:
        pass


def remove(path):
    try:
        os.remove(path)
    except Exception:
        pass


def save_stage1_output_to_parquet(output, out_dir):
    name = None
    for key, task in get_worker().tasks.items():
        if task.state == "executing":
            name = key[-32:]
    if not name:
        return
    for dataset in output.dataset.unique():
        df = output[output.dataset == dataset]
        if df.shape[0] == 0:
            return
        mkdir(f"{out_dir}/{dataset}")
        df.to_parquet(path=f"{out_dir}/{dataset}/{name}.parquet")


def delete_existing_stage1_output(datasets, parameters):
    global_path = parameters.get("global_path", None)
    label = parameters.get("label", None)
    year = parameters.get("year", None)

    if (global_path is None) or (label is None) or (year is None):
        return

    for dataset in datasets:
        path = f"{global_path}/{label}/stage1_output/{year}/{dataset}/"
        paths = glob.glob(f"{path}/*.parquet")
        for file in paths:
            remove(file)


def load_dataframe(client, parameters, inputs=[], dataset=None):
    skip_repartition = False
    ncpus = parameters.get("ncpus", 40)
    # ncpus = 1000#120 # temporary overwrite bc idk what 
    # ncpus = 30
    # ncpus = 500
    
    custom_npartitions_dict = parameters.get("custom_npartitions", {})
    custom_npartitions = 0
    if dataset in custom_npartitions_dict.keys():
        custom_npartitions = custom_npartitions_dict[dataset]

    if (custom_npartitions > 0):
        ncpus = custom_npartitions
    print(f"ncpus: {ncpus}")
    
    if isinstance(inputs, list):
        # Load dataframes
        if client:
            df_future = client.map(load_pandas_from_parquet, inputs)
            df_future = client.gather(df_future)
        else:
            df_future = []
            for inp in inputs:
                df_future.append(load_pandas_from_parquet(inp))
        # Merge dataframes
        try:
            df = dd.concat([d for d in df_future if d.shape[1] > 0])
        except Exception:
            return None
        print(f"df.npartitions: {df.npartitions}")
        if not skip_repartition:
            if df.npartitions > 2 * ncpus:
                df = df.repartition(npartitions=2 * ncpus)
            # if (custom_npartitions_dict > 0) and (df.npartitions > 2 * custom_npartitions):
            #     df = df.repartition(npartitions=2*custom_npartitions)
            # elif df.npartitions > 2 * ncpus:
            #     df = df.repartition(npartitions=2 * ncpus)
        print(f"df.npartitions after repartition: {df.npartitions}")
    elif isinstance(inputs, pd.DataFrame):
        df = dd.from_pandas(inputs, npartitions=ncpus)

    elif isinstance(inputs, dd.DataFrame):
        if custom_npartitions > 0:
            df = inputs.repartition(npartitions=custom_npartitions)
        elif inputs.npartitions > 2 * ncpus:
            df = inputs.repartition(npartitions=ncpus)
        else:
            df = inputs

    else:
        print("Wrong input type:", type(inputs))
        return None

    return df


def load_pandas_from_parquet(path):
    df = dd.from_pandas(pd.DataFrame(), npartitions=1)
    df = dd.read_parquet(path)
    if len(path) > 0:
        try:
            df = dd.read_parquet(path)
        except Exception:
            return df
    return df


def save_stage2_output_hists(hist, var_name, dataset, year, parameters, npart=None):
    global_path = parameters.get("global_path", None)
    label = parameters.get("label", None)

    if (global_path is None) or (label is None):
        return

    out_dir = global_path + "/" + label
    mkdir(out_dir)
    out_dir += "/" + "stage2_histograms"
    mkdir(out_dir)
    out_dir += "/" + var_name
    mkdir(out_dir)
    out_dir += "/" + str(year)
    mkdir(out_dir)

    if npart is None:
        path = f"{out_dir}/{dataset}.pickle"
    else:
        path = f"{out_dir}/{dataset}_{npart}_{generate_unique_string(10)}.pickle"
    with open(path, "wb") as handle:
        pickle.dump(hist, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_stage2_output_df(df, var_name, dataset, year, parameters, npart=None):
    global_path = parameters.get("global_path", None)
    label = parameters.get("label", None)

    if (global_path is None) or (label is None):
        return

    out_dir = global_path + "/" + label
    mkdir(out_dir)
    out_dir += "/" + "stage2_histograms"
    mkdir(out_dir)
    out_dir += "/" + var_name
    mkdir(out_dir)
    out_dir += "/" + str(year)
    mkdir(out_dir)

    if npart is None:
        path = f"{out_dir}/{dataset}.parquet"
    else:
        path = f"{out_dir}/{dataset}_{npart}_{generate_unique_string(10)}.parquet"

    df.to_parquet(path=path)


def delete_existing_stage2_hists(datasets, years, parameters):
    var_names = parameters.get("hist_vars", [])
    global_path = parameters.get("global_path", None)
    label = parameters.get("label", None)

    if (global_path is None) or (label is None):
        return

    for year in years:
        for var_name in var_names:
            for dataset in datasets:
                path = f"{global_path}/{label}/stage2_histograms/{var_name}/{year}/"
                # original start ----------------------------------------------------
                # try:
                #     paths = [f"{path}/{dataset}.pickle"]
                #     for fname in os.listdir(path):
                #         if re.fullmatch(rf"{dataset}_[0-9]+.pickle", fname):
                #             paths.append(f"{path}/{fname}")
                #     for file in paths:
                #         remove(file)
                # except Exception:
                #     pass
                # original end ----------------------------------------------------
                # better soln start --------------------------------------
                # print(f"delete_existing_stage2_hists dataset: {dataset}")
                paths = glob.glob(f"{path}/{dataset}*.pickle") + glob.glob(f"{path}/{dataset}*.parquet")
                # print(f"delete_existing_stage2_hists paths: {paths}")
                for fname in paths:
                    if os.path.exists(fname):
                            remove(fname)
                # raise ValueError
                # better soln end --------------------------------------


def load_stage2_output_hists(argset, parameters):
    year = argset["year"]
    var_name = argset["var_name"]
    dataset = argset["dataset"]
    global_path = parameters.get("global_path", None)
    label = parameters.get("label", None)

    if (global_path is None) or (label is None):
        return

    path = f"{global_path}/{label}/stage2_histograms/{var_name}/{year}/"
    # paths = [f"{path}/{dataset}.pickle"]
    # for fname in os.listdir(path):
    #     if re.fullmatch(rf"{dataset}_[0-9]+.pickle", fname):
    #         paths.append(f"{path}/{fname}")
    paths = glob.glob(f"{path}/{dataset}_*.pickle")
    # print(f"load_stage2_output_hists paths: {paths}")

    hist_df = pd.DataFrame()
    for path in paths:
        try:
            with open(path, "rb") as handle:
                hist = pickle.load(handle)
                new_row = {
                    "year": year,
                    "var_name": var_name,
                    "dataset": dataset,
                    "hist": hist,
                }
                hist_df = pd.concat([hist_df, pd.DataFrame([new_row])])
                hist_df.reset_index(drop=True, inplace=True)
        except Exception:
            pass
    # print(f"load_stage2_output_hists hist_df: {hist_df}")
    return hist_df

def stage2df2hist(df, var_name):
    """
    convert stage2 df output with columns ['value', 'weight', 'region', 'channel', 'variation']
    """
    # print(f"type(df): {type(df)}")
    regions = list(df["region"].unique())
    channels = list(df["channel"].unique())
    variations = list(df["variation"].unique())
    print(f"regions: {regions}")
    print(f"channels: {channels}")
    print(f"variations: {variations}")
    
    bins = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

    # print(f"stage2df2hist bins: {bins}")
    hist = (
        Hist.new.StrCat(regions, name="region")
        .StrCat(channels, name="channel")
        .StrCat(["value", "sumw2"], name="val_sumw2")
    )
    # hist = hist.Var(bins, name=var.name)
    # var_name = "DNN_score"
    print(f"var_name: {var_name}")
    hist = hist.Var(bins, name=var_name)
    hist = hist.StrCat(variations, name="variation")

    # specify container type
    hist = hist.Double()
    for region in regions:
        for channel in channels:
            for variation in variations:
                slicer = (
                        (df.region == region)
                        & (df.channel == channel)
                    )
                data = df.loc[slicer, "value"]
                weight = df.loc[slicer, "weight"]
        
                to_fill = {var_name: data, "region": region, "channel": channel}
                
                to_fill_value = to_fill.copy()
                to_fill_value["val_sumw2"] = "value"
                to_fill_value["variation"] = variation
                hist.fill(**to_fill_value, weight=weight)
                
                to_fill_sumw2 = to_fill.copy()
                to_fill_sumw2["val_sumw2"] = "sumw2"
                to_fill_sumw2["variation"] = variation    
                hist.fill(**to_fill_sumw2, weight=weight * weight)

    print(f"stage2df2hist hist: {hist}")
    return hist
    
    

def load_stage2_output_df2hists(argset, parameters):
    year = argset["year"]
    var_name = argset["var_name"]
    dataset = argset["dataset"]
    global_path = parameters.get("global_path", None)
    label = parameters.get("label", None)

    if (global_path is None) or (label is None):
        return

    # path = f"{global_path}/{label}/stage2_histograms/{var_name}/{year}/"
    # # paths = [f"{path}/{dataset}.pickle"]
    # # for fname in os.listdir(path):
    # #     if re.fullmatch(rf"{dataset}_[0-9]+.pickle", fname):
    # #         paths.append(f"{path}/{fname}")
    # paths = glob.glob(f"{path}/{dataset}_*.parquet")
    # # print(f"load_stage2_output_hists paths: {paths}")

    hist_df = pd.DataFrame()
    try:
        # with open(path, "rb") as handle:
            # hist = pickle.load(handle)
        # df = dd.from_pandas(pd.DataFrame(), npartitions=1)
        # df = dd.read_parquet(path).compute()
        npart, df = argset["df"]
        print(f"load_stage2_output_dfs df: {df}")
        hist = stage2df2hist(df, var_name)
        print(f"load_stage2_output_dfs hist: {hist}")
        
        new_row = {
            "year": year,
            "var_name": var_name,
            "dataset": dataset,
            "hist": hist,
        }
        hist_df = pd.concat([hist_df, pd.DataFrame([new_row])])
        hist_df.reset_index(drop=True, inplace=True)
    except Exception as e:
        print(f"Error in load_stage2_output_df2hists: {e}")
        pass
        
    print(f"load_stage2_output_df2hists hist_df: {hist_df}")
    return hist_df


def save_stage2_output_parquet(df, channel, dataset, year, parameters, npart=None):
    global_path = parameters.get("global_path", None)
    label = parameters.get("label", None)
    if (global_path is None) or (label is None):
        return

    out_dir = global_path + "/" + label
    mkdir(out_dir)
    out_dir += "/" + "stage2_unbinned"
    mkdir(out_dir)
    out_dir += "/" + f"{channel}_{year}"
    mkdir(out_dir)

    if npart is None:
        path = f"{out_dir}/{dataset}.parquet"
    else:
        path = f"{out_dir}/{dataset}_{npart}.parquet"
    df.to_parquet(path=path)


def delete_existing_stage2_parquet(datasets, years, parameters):
    to_delete = parameters.get("tosave_unbinned", {})
    global_path = parameters.get("global_path", None)
    label = parameters.get("label", None)
    if (global_path is None) or (label is None):
        return

    for channel in to_delete.keys():
        for year in years:
            for dataset in datasets:
                path = f"{global_path}/{label}/stage2_unbinned/{channel}_{year}/"
                # original start ----------------------------------------------------

                # paths = [f"{path}/{dataset}.parquet"]
                # if os.path.exists(path):
                #     for fname in os.listdir(path):
                #         # print(f"delete_existing_stage2_parquet fname: {fname}")
                #         if re.fullmatch(rf"{dataset}_[0-9]+.parquet", fname):
                #             paths.append(f"{path}/{fname}")
                #     for file in paths:
                #         if os.path.exists(file):
                #             remove(file)
                # original end ----------------------------------------------------
                # better soln start --------------------------------------
                paths = glob.glob(f"{path}/{dataset}*.parquet")
                for fname in paths:
                    if os.path.exists(fname):
                            remove(fname)
                # better soln end --------------------------------------


def save_template(templates, out_name, parameters):
    out_file = uproot3.recreate(out_name)
    for tmp in templates:
        out_file[tmp._fName] = tmp
    out_file.close()
    return
