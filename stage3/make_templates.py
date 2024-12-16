import numpy as np
import pandas as pd

from python.workflow import parallelize
from python.variable import Variable
from python.io import load_stage2_output_hists, save_template, mkdir, load_stage2_output_df2hists

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from uproot3_methods.classes.TH1 import from_numpy
import glob
import dask.dataframe as dd


decorrelation_scheme = {
    "LHERen": ["DY", "EWK", "ggH", "TT+ST"],
    "LHEFac": ["DY", "EWK", "ggH", "TT+ST"],
    "pdf_2rms": ["DY", "VBF", "ggH"], # ["DY", "qqH_hmm", "ggH_hmm"],
}
shape_only = [
    "wgt_LHERen_up",
    "wgt_LHERen_down",
    "wgt_LHEFac_up",
    "wgt_LHEFac_down",
    "wgt_qgl_up",
    "wgt_qgl_down",
    "wgt_pdf_2rms_up",
    "wgt_pdf_2rms_down",
]


def to_templates(client, parameters, hist_df=None):
    # datasets = list(parameters["datasets"]) # original
    datasets = list(parameters["datasets"]) + ["ewk_lljj_mll105_160_py_dipole", "vbf_powheg_herwig"] # manually add partonShower
    if hist_df is None:
        argset_load = {
            "year": parameters["years"],
            "var_name": parameters["templates_vars"],
            "dataset": datasets,
        }
        # original ----------------------
        hist_rows = parallelize(
            load_stage2_output_hists, argset_load, client, parameters
        )
        # original ---------------------------------
        # new soln ---------------------------------------
        # hist_rows = []
        # for dataset in datasets:
        #     year = argset_load["year"][0]
        #     var_name = argset_load["var_name"][0]
        #     # dataset = argset_load["dataset"]
        #     global_path = parameters.get("global_path", None)
        #     label = parameters.get("label", None)
        #     path = f"{global_path}/{label}/stage2_histograms/{var_name}/{year}/"
        #     paths = glob.glob(f"{path}/{dataset}_*.parquet")
        #     df_l = []
        #     print(f"path: {path}/{dataset}_*.pa")
        #     print(f"paths: {paths}")
        #     for load_path in paths:
        #         df_i = dd.from_pandas(pd.DataFrame(), npartitions=1)
        #         df_i = dd.read_parquet(load_path)
        #         df_l.append(df_i)
        #     print(f"df_l: {df_l}")
        #     if len(df_l) == 0:
        #         continue
        #     df = dd.concat(df_l) 
        #     argset_load["df"] = [(i, df.partitions[i]) for i in range(df.npartitions)]
        #     hist_rows = hist_rows + parallelize(
        #         load_stage2_output_df2hists, argset_load, client, parameters
        #     )
        # new soln -------------------------------------------
        
        hist_df = pd.concat(hist_rows).reset_index(drop=True)
        print(f"hist_df: {hist_df}")
        if hist_df.shape[0] == 0:
            print("No templates to create!")
            return []

    argset = {
        "year": parameters["years"],
        "region": parameters["regions"],
        "channel": parameters["channels"],
        "var_name": [
            v for v in hist_df.var_name.unique() if v in parameters["templates_vars"]
        ],
        "hist_df": [hist_df],
    }
    yield_dfs = parallelize(make_templates, argset, client, parameters, seq=True)
    yield_df = pd.concat(yield_dfs).reset_index(drop=True)
    return yield_df


def make_templates(args, parameters={}):
    year = args["year"]
    print(f"make_template year: {year}")
    print(f'args["hist_df"].year: {args["hist_df"].year}')

    region = args["region"]
    channel = args["channel"]
    var_name = args["var_name"]
    hist_df = args["hist_df"].loc[
        (args["hist_df"].var_name == var_name) & (args["hist_df"].year == year)
    ]
    print(f"hist_df: {hist_df}")
    if "2016" in year:
        year_savepath = year
        year = "2016"
    else:
        year_savepath = year

    if var_name in parameters["variables_lookup"].keys():
        var = parameters["variables_lookup"][var_name]
    else:
        var = Variable(var_name, var_name, 50, 0, 5)

    if hist_df.shape[0] == 0:
        return

    yield_rows = []
    templates = []

    groups = list(set(parameters["grouping"].values()))
    # print(f"groups: {groups}")
    # print(f"hist_df.dataset.unique(): {hist_df.dataset.unique()}")
    
    for group in groups:
        datasets = []
        for d in hist_df.dataset.unique():
            if d not in parameters["grouping"].keys():
                continue
            if parameters["grouping"][d] != group:
                continue
            datasets.append(d)

        if len(datasets) == 0:
            continue

        print(f"datasets: {datasets}")

        # make a list of systematics;
        # avoid situation where different datasets have incompatible systematics
        wgt_variations = []
        for dataset in datasets:
            n_partitions = len(hist_df.loc[hist_df.dataset == dataset, "hist"].values)
            for i in range(n_partitions):
                new_wgt_vars = list(
                    hist_df.loc[hist_df.dataset == dataset, "hist"]
                    .values[i]
                    .axes["variation"]
                )
                # print(f"new_wgt_vars: {new_wgt_vars}")
                if len(wgt_variations) == 0:
                    wgt_variations = new_wgt_vars
                else:
                    wgt_variations = list(set(wgt_variations) & set(new_wgt_vars))

        # print(f"wgt_variations: {wgt_variations}")
        # manually add parton shower variations start -------------------------------
        add_VBF_PartonShower = False
        add_EWK_PartonShower = False
        for wgt_variation in wgt_variations:
            if "qqH_hmm" ==group:
                add_VBF_PartonShower = True
                break
            elif "EWK" ==group:
                add_EWK_PartonShower = True
                break
        if add_VBF_PartonShower:
            wgt_variations += ["qqH_hmm_SignalPartonShowerUp", "qqH_hmm_SignalPartonShowerDown"]
        if add_EWK_PartonShower:
            wgt_variations += ["EWK_EWKPartonShowerUp", "EWK_EWKPartonShowerDown"]

        # print(f"add_VBF_PartonShower: {add_VBF_PartonShower}")
        # print(f"add_EWK_PartonShower: {add_EWK_PartonShower}")
        # manually add parton shower variations end -------------------------------
        # print(f"wgt_variations: {wgt_variations}")
        for variation in wgt_variations:
            # print(f"variation: {variation}")
            # print(f"channel: {channel}")
            
            group_hist = []
            group_sumw2 = []

            slicer_nominal = {
                "region": region,
                "channel": channel,
                "variation": "nominal",
                "val_sumw2": "value",
            }
            slicer_value = {
                "region": region,
                "channel": channel,
                "variation": variation,
                "val_sumw2": "value",
            }
            

            #Parton Shower case start -----------------------------
            if ("PartonShower" in variation):
                slicer_sumw2 = { # slicer_sumw2 needs to be overwritten
                    "region": region,
                    "channel": channel,
                    "variation": "nominal",
                    "val_sumw2": "sumw2",
                }
                if ("qqH_hmm" in variation):
                    baseline_dataset = "vbf_powheg_dipole"
                    variation_dataset = "vbf_powheg_herwig"
                elif ("EWK" in variation):
                    baseline_dataset = "ewk_lljj_mll105_160_ptj0"
                    variation_dataset = "ewk_lljj_mll105_160_py_dipole"
                else:
                    print("no parton shower exists for this sample!")
                    raise ValueError
                # vals_baseline = hist_df.loc[hist_df.dataset == baseline_dataset, "hist"].values 
                print(f'hist_df.loc[hist_df.dataset == baseline_dataset, "hist"]: {hist_df.loc[hist_df.dataset == baseline_dataset, "hist"]}')
                hist_baseline = hist_df.loc[hist_df.dataset == baseline_dataset, "hist"].values.sum()
                
                the_hist_nominal_baseline = hist_baseline[slicer_nominal].project(var.name).values()
                the_sumw2_baseline = hist_baseline[slicer_sumw2].project(var.name).values()
                # print(f"{group} the_hist_nominal_baseline: {the_hist_nominal_baseline}")
                # print(f"{group} the_sumw2_baseline: {the_sumw2_baseline}")

                # vals_variation = hist_df.loc[hist_df.dataset == variation_dataset, "hist"].values 
                hist_variation = hist_df.loc[hist_df.dataset == variation_dataset, "hist"].values.sum()

                the_hist_nominal_variation = hist_variation[slicer_nominal].project(var.name).values()
                the_sumw2_variation = hist_variation[slicer_sumw2].project(var.name).values()
                # print(f"{group} the_hist_nominal_variation: {the_hist_nominal_variation}")
                # print(f"{group} the_sumw2_variation: {the_sumw2_variation}")
                # print(f"{group} the_hist_nominal_variation: {type(the_hist_nominal_variation)}")
                # print(f"{group} the_sumw2_variation: {type(the_sumw2_variation)}")
                

                edges = hist_baseline[slicer_nominal].project(var.name).axes[0].edges
                edges = np.array(edges)
                print(f"edges: {edges}")
                centers = (edges[:-1] + edges[1:]) / 2.0
                name = variation

                if "Up" in variation:
                    group_hist = the_hist_nominal_baseline - (the_hist_nominal_baseline -  the_hist_nominal_variation)
                elif "Down" in variation:
                    group_hist = the_hist_nominal_baseline + (the_hist_nominal_baseline -  the_hist_nominal_variation)
                else:
                    print("unknown variation in parton shower")
                    raise ValueError

                # print(f"group_hist: {group_hist}")
                # group_sumw2 = the_sumw2_variation*0
                group_sumw2 = 2*the_sumw2_baseline + the_sumw2_variation
                

                # print(f"variation name: {name}")
                th1 = from_numpy([group_hist, edges])
                th1._fName = name
                th1._fSumw2 = np.array(np.append([0], group_sumw2)) # -> np.array([0, group_sumw2])
                th1._fTsumw2 = np.array(group_sumw2).sum()
                th1._fTsumwx2 = np.array(group_sumw2 * centers).sum() #-> this is w2*x distibution
                templates.append(th1)

                # variation_fixed = variation.replace("VBF_", "").replace("EWK_", "")
                variation_fixed = variation.replace("qqH_hmm_", "").replace("EWK_", "")

                # print(f"variation_fixed: {variation_fixed}")
                yield_row = {
                        "var_name": var_name,
                        "group": group,
                        "region": region,
                        "channel": channel,
                        "year": year,
                        "variation": variation_fixed,
                        "yield": group_hist.sum(),
                }
                # print(f"yield_rows: {yield_rows}")
                yield_rows.append(yield_row)
                continue # done parton shower, skip the rest of the loop
            # Parton Shower case end -----------------------------
            # do the normal for loop if not PartonShower
            
            slicer_sumw2 = {
                "region": region,
                "channel": channel,
                "variation": variation,
                "val_sumw2": "sumw2",
            }
            for dataset in datasets:
                try:
                    # hist = hist_df.loc[hist_df.dataset == dataset, "hist"].values.sum()

                    # my attempt start -----------------------------------------------------------
                    vals = hist_df.loc[hist_df.dataset == dataset, "hist"].values
                    #---------------------------------------------------------
                    available_axes = ['region', 'channel', 'val_sumw2', 'score_vbf', 'variation'] # debugging
                    # for axes in available_axes:
                    #     print(f"testing axes: {axes}")
                    #     projection = vals[slicer_value].project(var.name)#.values().sum()
                    #     print(f"testing projection: {projection}")
                    # print(f"make_templates vals: {vals}")
                    # sliced_val = vals[slicer_value]
                    # print(f"testing sliced_val: {sliced_val}")
                    # projection = vals[slicer_value].project(var.name).sum()
                    # print(f"testing projection: {projection}")
                    #---------------------------------------------------------
                    # print(f"make_templates len vals: {len(vals)}")
                    # print(f"make_templates type(vals[0]): {type(vals[0])}")
                    # for histogram in list(vals)[:4]:
                    val_l = list(vals)
                    # bad_idxs = [4, 6, 7, 8, 10, 13, 15, 16, 17, 25, 28, 34, 41, 42, 51, 53, 55, 58, 60, 73, 78, 80, 81, 82, 83, 91, 92, 99, 101, 102, 104, 121]
                    bad_idxs = []
                    hist_sum = val_l[0]
                    for hist_idx in range(1, len(val_l)):
                        histogram = val_l[hist_idx]
                        # axes_l = [axis.label for axis in histogram.axes]
                        # print(f"{hist_idx} axes_l: {axes_l}")   
                        if hist_idx in bad_idxs:
                            continue
                        try:
                            hist_sum = hist_sum+histogram
                        except Exception as e:
                            # print(f"Exception {e}")
                            bad_idxs.append(hist_idx)
                            # print(f"bad idx {hist_idx} with error {e}")
                        # print(f"make_templates histogram: {histogram}")
                        # print(f"make_templates histogram.axes: {histogram.axes}")
                        # np_val = histogram.values()
                        # print(f"make_templates histogram.values(): {np_val}")
                        # print(f"make_templates histogram.values().shape: {np_val.shape}")
                        
                    # print(f"make_templates type(vals): {type(vals)}")
                    # print(f"make_templates axes: {vals.axes}")
                    # hist = np.sum(vals.values().flatten())
                    if len(bad_idxs) > 0:
                        print(f"{dataset} bad_idxs: {bad_idxs}")
                    # hist = vals.sum()
                    hist = hist_sum
                    # vals = list(vals)
                    # hist = np.array([val.values() for val in vals]).sum(axis=0)
                    # print(f"make_templates his.shapet: {hist.shape}")
                    # raise ValueError
                    # my attempt end -----------------------------------------------------------
                    
                except Exception as e:
                    print(f"Could not merge histograms for {dataset} due to error {e}")
                    continue

                try: 
                    the_hist = hist[slicer_value].project(var.name).values()
                except Exception as e:
                    print(f"Could not project histograms for {dataset} due to error {e}")
                    continue
                the_hist_nominal = hist[slicer_nominal].project(var.name).values()
                the_sumw2 = hist[slicer_sumw2].project(var.name).values()

                if variation in shape_only:
                    if the_hist.sum() != 0:
                        scale = the_hist_nominal.sum() / the_hist.sum()
                    else:
                        scale = 1.0
                    the_hist = the_hist * scale
                    the_sumw2 = the_sumw2 * scale

                # temporary overwrite for TT+ST group ----------------
                # if group=="TT+ST":
                #     scale=1.081915477
                #     the_hist = the_hist * scale
                #     the_sumw2 = the_sumw2 * scale
                # if group=="Data":
                #     # print("data is present!")
                #     scale=3975.000/3939
                #     the_hist = the_hist * scale
                #     the_sumw2 = the_sumw2 * scale
                # # elif group=="DYJ01":
                # #     scale=1389.6971467649/2525.096684
                # #     the_hist = the_hist * scale
                # #     the_sumw2 = the_sumw2 * scale
                # # elif group=="DYJ2":
                # #     scale=2265.5777773395/1104.819084
                # #     the_hist = the_hist * scale
                # #     the_sumw2 = the_sumw2 * scale
                # elif group=="ggH":
                #     scale=9.240/9.599384
                #     the_hist = the_hist * scale
                #     the_sumw2 = the_sumw2 * scale
                # elif group=="VBF":
                #     scale=11.784/11.936208 
                #     the_hist = the_hist * scale
                #     the_sumw2 = the_sumw2 * scale
                # elif group=="EWK":
                #     scale=125.749/126.96623
                #     the_hist = the_hist * scale
                #     the_sumw2 = the_sumw2 * scale
                # elif group=="VV":
                #     scale=78.102/80.901781
                #     the_hist = the_hist * scale
                #     the_sumw2 = the_sumw2 * scale

                

                # -------------------------------------

                
                if (the_hist.sum() < 0) or (the_sumw2.sum() < 0):
                    continue

                if len(group_hist) == 0:
                    group_hist = the_hist
                    group_sumw2 = the_sumw2
                else:
                    group_hist += the_hist
                    group_sumw2 += the_sumw2

                edges = hist[slicer_value].project(var.name).axes[0].edges
                edges = np.array(edges)
                centers = (edges[:-1] + edges[1:]) / 2.0

            if len(group_hist) == 0:
                continue
            if sum(group_hist) == 0:
                continue

            if group == "Data":
                name = "data_obs"
            else:
                name = group

            if variation == "nominal":
                # variation_core = variation.replace("wgt_", "")
                # variation_core = variation_core.replace("_up", "")
                # variation_core = variation_core.replace("_down", "")
                # print(f"variation_core: {variation_core}")
                
                # else:
                variation_fixed = variation
            else:
                variation_core = variation.replace("wgt_", "")
                variation_core = variation_core.replace("_up", "")
                variation_core = variation_core.replace("_down", "")
                print(f"variation_core: {variation_core}")
                suffix = ""
                if variation_core in decorrelation_scheme.keys():
                    group_LHE = group
                    print(f"group_LHE: {group_LHE}")
                    if group_LHE == "DYJ2" or group_LHE == "DYJ01" :
                        group_LHE = "DY"
                    elif "qqH" in group_LHE :
                        group_LHE = "VBF"
                    elif "ggH" in group_LHE :
                        group_LHE = "ggH"
                    print(f"group_LHE after: {group_LHE}")
                    if group_LHE in decorrelation_scheme[variation_core]:
                        if variation_core == "pdf_2rms" :
                            suffix = "_"+group_LHE+str(year)
                            print(f"pdf_2rms suffix: {suffix}")
                        else:
                            suffix = "_"+group_LHE
                    else:
                        continue
                elif variation_core in ["muID", "muIso", "muTrig"]:
                    suffix = str(year)
                elif variation_core in ["pu", "l1prefiring"]:
                    suffix = "_wgt"+str(year)
                elif variation_core in ["qgl"]:
                    suffix = "_wgt"
                        
                
                    
                # TODO: decorrelate LHE, QGL, PDF uncertainties
                variation_fixed = variation.replace("wgt_", "")               
                variation_fixed = variation_fixed.replace("_up", f"{suffix}Up")
                variation_fixed = variation_fixed.replace("_down", f"{suffix}Down")
                group_name = group
                name = f"{group_name}_{variation_fixed}"
                print(f"name: {name}")

            # print(f"variation name: {name}")
            # print(f"var_name: {var_name}")
            # print(f"variation_fixed: {variation_fixed}")
            th1 = from_numpy([group_hist, edges])
            th1._fName = name
            th1._fSumw2 = np.array(np.append([0], group_sumw2)) # -> np.array([0, group_sumw2])
            th1._fTsumw2 = np.array(group_sumw2).sum()
            th1._fTsumwx2 = np.array(group_sumw2 * centers).sum() #-> this is w2*x distibution
            templates.append(th1)

            yield_rows.append(
                {
                    "var_name": var_name,
                    "group": group,
                    "region": region,
                    "channel": channel,
                    "year": year,
                    "variation": variation_fixed,
                    "yield": group_hist.sum(),
                }
            )

    if parameters["save_templates"]:
        out_dir = parameters["global_path"]
        mkdir(out_dir)
        out_dir += "/" + parameters["label"]
        mkdir(out_dir)
        out_dir += "/" + "stage3_templates"
        mkdir(out_dir)
        out_dir += "/" + var.name
        mkdir(out_dir)

        # out_fn = f"{out_dir}/{channel}_{region}_{year}.root"
        out_fn = f"{out_dir}/{channel}_{region}_{year_savepath}.root"
        
        print(f"out_fn: {out_fn}")
        save_template(templates, out_fn, parameters)

    yield_df = pd.DataFrame(yield_rows)
    return yield_df
