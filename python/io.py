import os
import pandas as pd
import dask.dataframe as dd
from dask.distributed import get_worker
import pickle
import glob
import re
import uproot3

import random
import string
from datetime import datetime

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
    # ncpus = parameters.get("ncpus", 40)
    ncpus = 1000#120 # temporary overwrite bc idk what 
    # ncpus = 150
    print(f"ncpus: {ncpus}")
    custom_npartitions_dict = parameters.get("custom_npartitions", {})
    custom_npartitions = 0
    if dataset in custom_npartitions_dict.keys():
        custom_npartitions = custom_npartitions_dict[dataset]

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
        if custom_npartitions > 0:
            df = df.repartition(npartitions=custom_npartitions)
        elif df.npartitions > 2 * ncpus:
            df = df.repartition(npartitions=2 * ncpus)

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
                paths = glob.glob(f"{path}/{dataset}*.pickle")
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
