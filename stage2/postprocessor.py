import dask.dataframe as dd
import pandas as pd
from itertools import chain

from python.workflow import parallelize
from python.io import (
    delete_existing_stage2_hists,
    delete_existing_stage2_parquet,
    save_stage2_output_parquet,
    split_df
)
from stage2.categorizer import (split_into_channels, categorize_by_score, categorize_by_eta, categorize_by_CalibCat, categorize_by_ClosureCat)
from stage2.mva_evaluators import (
    # evaluate_pytorch_dnn,
    # evaluate_pytorch_dnn_pisa,
    evaluate_bdt,
    # evaluate_mva_categorizer,
)
from stage2.histogrammer import make_histograms

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None


def process_partitions(client, parameters, df):
    # for now ignoring some systematics
    ignore_columns = []
    ignore_columns += [c for c in df.columns if "pdf_" in c]

    df = df[[c for c in df.columns if c not in ignore_columns]]

    for key in ["channels", "regions", "categories","syst_variations", "hist_vars", "datasets"]:
        if key in parameters:
            parameters[key] = list(set(parameters[key]))

    years = df.year.unique()
    datasets = df.dataset.unique()
    # delete previously generated outputs to prevent partial overwrite
    delete_existing_stage2_hists(datasets, years, parameters)
    #delete_existing_stage2_parquet(datasets, years, parameters)
    #print(df.year)
    # prepare parameters for parallelization

    #print(years)
    argset = {
        "year": years,
        "dataset": datasets,
    }
    #df=df.compute()
    if isinstance(df, pd.DataFrame):
        argset["df"] = [df]
    elif isinstance(df, dd.DataFrame):
        argset["df"] = [(i, df.partitions[i]) for i in range(df.npartitions)]
    # perform categorization, evaluate mva models, fill histograms
    print("starting parallelize")
    all_dfs = parallelize(on_partition, argset, client, parameters,seq=False)
    hist_info_df_full = all_dfs[0][0]
    df_new = all_dfs[0][1]
    
    #for i in range(5):
        #df_test = df_new[df_new["category"] == f"BDTperyear_2017_cat{i}"]
        #print(df_test)
    # return info for debugging
    #print(len(all_dfs))
    #print("all_dfs[0][1]")
    #print(all_dfs[0][1])
    #print("all_dfs[0][0]")
    #print(all_dfs[0][0])
    #print("all_dfs[1][0]")
    #print(all_dfs[1][0])
    #print("all_dfs[0][2]")
    #print(all_dfs[0][2])
    for i in range(len(all_dfs)-1):
        hist_info_df_full.append(all_dfs[i+1][0])
        #df_new.append(all_dfs[i+1][1])
        #print(all_dfs[i+1][1])
        df_new = pd.concat([df_new, all_dfs[i+1][1]])
    #print(df_new)
    #hist_info_df_full = pd.concat(hist_info_dfs).reset_index(drop=True)

    return hist_info_df_full, df_new


def on_partition(args, parameters):
    
    year = args["year"]
    #print(year)
    #if "2016" in year:
        #year = 2016
    dataset = args["dataset"]
    df = args["df"]

    if "mva_bins" not in parameters:
        parameters["mva_bins"] = {}

    # get partition number, if available
    npart = None
    if isinstance(df, tuple):
        npart = df[0]
        df = df[1]
    
    # convert from Dask DF to Pandas DF
    if isinstance(df, dd.DataFrame):
        df = df.compute()
        
    # preprocess
    #df.loc[df['year'] == "2016postVFP", 'year'] = 2016
    #df.loc[df['year'] == "2016preVFP", 'year'] = 2016
    if parameters["regions"] == ["none"]:
        df.loc[df['region'] != "none", 'region'] = 'none'
    wgts = [c for c in df.columns if "wgt" in c]
    df.loc[:, wgts] = df.loc[:, wgts].fillna(0)
    df.jet1_has_matched_gen_nominal.fillna(False, inplace=True)
    df.jet2_has_matched_gen_nominal.fillna(False, inplace=True)
    df.fillna(-99.0, inplace=True)
    df = df[(df.dataset == dataset) & (df.year == year)]

    # VBF filter
    if "dy_m105_160_amc" in dataset:
        df = df[df.gjj_mass <= 350]
    if "dy_m105_160_vbf_amc" in dataset:
        df = df[df.gjj_mass > 350]

    # if dataset in ["vbf_powheg_dipole", "ggh_amcPS"]:
    #    # improve mass resolution manually
    #    improvement = 0
    #    df["dimuon_mass"] = df["dimuon_mass"] + improvement*(125 - df["dimuon_mass"])

    # < evaluate here MVA scores before categorization, if needed >
    # ...
    # cat_score_name = "mva_categorizer_score"
    # model_name = parameters.get("mva_categorizer", "3layers_64_32_16_all_feat")
    # vbf_mva_cutoff = parameters.get("vbf_mva_cutoff", 0.6819233298301697)
    # df[cat_score_name] = evaluate_mva_categorizer(df, model_name, cat_score_name, parameters)

    # < categorization into channels (ggH, VBF, etc.) >
    # split_into_channels(df, v="nominal", vbf_mva_cutoff=vbf_mva_cutoff)
    split_into_channels(df, v="nominal",ggHsplit=False) #NEEDS TO BE CHANGED WHEN SWITCHING FROM ALL GGH to JET WISE GGH ---> TO BE FIXED
    if parameters["channels"] == ["none"]:
        df.loc[df['channel_nominal'] != "none", 'channel_nominal'] = 'none'
    regions = [r for r in parameters["regions"] if r in df.region.unique()]
    channels = [
        c for c in parameters["channels"] if c in df["channel_nominal"].unique()
    ]

    # split DY by genjet multiplicity

    if "dyblub" in dataset:

        df["two_matched_jets"] = (
            (df.jet1_has_matched_gen_nominal==True) & (df.jet2_has_matched_gen_nominal==True)
        )
        df.loc[
            (df.channel_nominal == "vbf") & (~df.two_matched_jets), "dataset"
        ] = f"{dataset}_01j"
        df.loc[
            (df.channel_nominal == "vbf") & (df.two_matched_jets), "dataset"
        ] = f"{dataset}_2j"
    #print(df)
    # < evaluate here MVA scores after categorization, if needed >
    syst_variations = parameters.get("syst_variations", ["nominal"])
    dnn_models = parameters.get("dnn_models", {})
    bdt_models = parameters.get("bdt_models", {})
    for v in syst_variations:
        # focus on BDT for now start -------------------------------------
        for channel, models in dnn_models.items():
            if channel not in parameters["channels"]:
                continue
            for model in models:
                
                score_name = f"score_{model}_{v}"
                df.loc[
                    df[f"channel_{v}"] == channel, score_name
                ] = evaluate_pytorch_dnn(
                    df[df[f"channel_{v}"] == channel],
                    v,
                    model,
                    parameters,
                    score_name,
                    channel,
                )
                #print(channel)
                #print(score_name)
                #print(df[df[f"channel_{v}"] == channel][f"score_{model}_{v}"])
                """
                df.loc[
                    df[f"channel_{v}"] == channel, score_name
                ] = evaluate_pytorch_dnn_pisa(
                    df[df[f"channel_{v}"] == channel],
                    v,
                    model,
                    parameters,
                    score_name,
                    channel,
                )
                """
        # focus on BDT for now end -------------------------------------
        
        #print(df)
        # evaluate XGBoost BDTs
        for channel, models in bdt_models.items():
            if channel not in parameters["channels"]:
                continue
            for model in models:
                model = f"{model}_{parameters['years'][0]}"
                score_name = f"score_{model}_{v}"
                df.loc[df[f"channel_{v}"] == channel, score_name] = evaluate_bdt(
                    df[df[f"channel_{v}"] == channel], v, model, parameters, score_name
                )
    #print(df)
    # < add secondary categorization / binning here >
    # ...

    # temporary implementation: move from mva score to mva bin number
    for channel, models in chain(dnn_models.items(), bdt_models.items()):
        if channel not in parameters["channels"]:
            continue
        for model_name in models:
            if model_name not in parameters["mva_bins_original"]:
                continue
            score_name = f"score_{model_name}_nominal"
            if score_name in df.columns:
                mva_bins = parameters["mva_bins_original"][model_name][str(year)]
                for i in range(len(mva_bins) - 1):
                    lo = mva_bins[i]
                    hi = mva_bins[i + 1]
                    cut = (df[score_name] > lo) & (df[score_name] <= hi)
                    df.loc[cut, "bin_number"] = i
                df[score_name] = df["bin_number"]
                print(df[score_name])
                print('wrong here')
                parameters["mva_bins"].update(
                    {
                        model_name: {
                            "2016preVFP": list(range(len(mva_bins))),
                            "2016postVFP": list(range(len(mva_bins))),
                            "2017": list(range(len(mva_bins))),
                            "2018": list(range(len(mva_bins))),
                        }
                    }
                )
    #print(df)
    #For ggh: categorise by score based on signal eff
    df["category"] = "All"
    df["categoryPlot"] = "All"
    if parameters["cats_by_score"]:
        categorize_by_score(df, bdt_models, mode = "fixed_ggh", year = parameters["years"][0])
    elif parameters["cats_by_eta"]:
        categorize_by_eta(df)
    elif parameters["cats_by_CalibCat"]:
        categorize_by_CalibCat(df)
    elif parameters["cats_by_ClosureCat"]:
        categorize_by_ClosureCat(df)
    #print(df)
    #for i in range(5):
        #df_test = df[df["category"] == f"BDTperyear_2017_cat{i}"]
        #print(df_test)
    categories = [c for c in df["category"].unique()
       #c for c in parameters["category"] if c in df["category"].unique()
    ]
    df_for_fits = df
    #print(categories)
    #print(df_for_fits)
    #print(df[df[f"channel_{v}"] == channel]["category"])
    # < convert desired columns to histograms >
    # not parallelizing for now - nested parallelism leads to a lock
    hist_info_rows = []
    for var_name in parameters["hist_vars"]:
        print("making histograms")
        hist_info_row = make_histograms(
            df, var_name, year, dataset, regions, channels, categories, npart, parameters
        )
        #print(hist_info_row)
        if hist_info_row is not None:
            hist_info_rows.append(hist_info_row)
        if "dyblub" in dataset:
            for suff in ["01j", "2j"]:
                hist_info_row = make_histograms(
                    df,
                    var_name,
                    year,
                    f"{dataset}_{suff}",
                    regions,
                    channels,
                    npart,
                    parameters,
                )
                if hist_info_row is not None:
                    hist_info_rows.append(hist_info_row)

    if len(hist_info_rows) == 0:
        return pd.DataFrame()
    
    hist_info_df = pd.concat(hist_info_rows).reset_index(drop=True)

    # < save desired columns as unbinned data (e.g. dimuon_mass for fits) >
    do_save_unbinned = parameters.get("save_unbinned", False)
    if do_save_unbinned:
        save_unbinned(df, dataset, year, npart, channels, parameters)

    # < return some info for diagnostics & tests >
    #print("df_for_fits")
    #print(df_for_fits) #Good! Really!

   
    #out_path = f"{parameters['global_path']}/{parameters['label']}/stage2_output/{year}/"
    #save_stage1_output_to_parquet(df_for_fits,out_path)
    return [hist_info_df, df_for_fits]


def save_unbinned(df, dataset, year, npart, channels, parameters):
    to_save = parameters.get("tosave_unbinned", {})
    for channel, var_names in to_save.items():
        if channel not in channels:
            continue
        vnames = []
        for var in var_names:
            if var in df.columns:
                vnames.append(var)
            elif f"{var}_nominal" in df.columns:
                vnames.append(f"{var}_nominal")
        save_stage2_output_parquet(
            df.loc[df["channel_nominal"] == channel, vnames],
            channel,
            dataset,
            year,
            parameters,
            npart,
        )
