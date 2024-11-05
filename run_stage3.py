import argparse
import dask
from dask.distributed import Client

from config.variables import variables_lookup
from stage3.plotter import plotter
from stage3.make_templates import to_templates
from stage3.make_datacards import build_datacards
import time
__all__ = ["dask"]


parser = argparse.ArgumentParser()
parser.add_argument(
    "-y", "--years", nargs="+", help="Years to process", default=["2018"]
)
parser.add_argument(
    "-sl",
    "--slurm",
    dest="slurm_port",
    default=None,
    action="store",
    help="Slurm cluster port (if not specified, will create a local cluster)",
)
parser.add_argument(
    "-l",
    "--label",
    dest="label",
    default="test",
    action="store",
    help="Unique run label (to create output path)",
)
args = parser.parse_args()

# Dask client settings
use_local_cluster = args.slurm_port is None
node_ip = "128.211.149.133"

if use_local_cluster:
    ncpus_local = 40
    slurm_cluster_ip = ""
    dashboard_address = f"{node_ip}:34875"
else:
    slurm_cluster_ip = f"{node_ip}:{args.slurm_port}"
    dashboard_address = f"{node_ip}:8787"

# global parameters
parameters = {
    # < general settings >
    "slurm_cluster_ip": slurm_cluster_ip,
    "years": args.years,
    "global_path": "/depot/cms/users/yun79/hmm/copperheadV1clean/",
    # "global_path": "/work/users/yun79/copperhead_outputs/copperheadV1clean",
    # "label": "DmitryMaster_JECoff_GeofitFixed_Oct29",
    # "label": "DmitryMaster_JECoff_GeofitFixed_Nov01",
    # "label": "rereco_yun_Nov04",
    "label": args.label,
    "channels": ["vbf"],
    "regions": ["h-peak", "h-sidebands"],
    "syst_variations": ["nominal"],
    #
    # < plotting settings >
    "plot_vars": [],  # "dimuon_mass"],
    "variables_lookup": variables_lookup,
    "save_plots": True,
    "plot_ratio": True,
    "plots_path": "./plots/DmitryMaster_JECoff_GeofitFixed_Nov01/",
    "dnn_models": {
        # "vbf": ["pytorch_test"],
        # "vbf": ["vbf"],
        "vbf": ["pytorch_jun27"],        
    },
    "bdt_models": {},
    #
    # < templates and datacards >
    "save_templates": True,
    "templates_vars": [],  # "dimuon_mass"],
}


parameters["grouping"] = {
    "data_A": "Data",
    "data_B": "Data",
    "data_C": "Data",
    "data_D": "Data",
    "data_E": "Data",
    "data_F": "Data",
    "data_G": "Data",
    "data_H": "Data",
    # # "dy_m105_160_amc": "DY",
    # # "dy_m105_160_vbf_amc": "DY",
    "dy_m105_160_amc_01j": "DYJ01", # bad
    "dy_m105_160_vbf_amc_01j": "DYJ01", # good
    "dy_m105_160_amc_2j": "DYJ2",
    "dy_m105_160_vbf_amc_2j": "DYJ2",
    # "ewk_lljj_mll105_160_py_dipole": "EWK",
    "ewk_lljj_mll105_160_ptj0": "EWK",
    "ttjets_dl": "TT+ST",
    "ttjets_sl": "TT+ST",
    "ttw": "TT+ST",
    "ttz": "TT+ST",
    "st_tw_top": "TT+ST",
    "st_tw_antitop": "TT+ST",
    "ww_2l2nu": "VV",
    "wz_2l2q": "VV",
    "wz_1l1nu2q": "VV",
    "wz_3lnu": "VV",
    "zz": "VV",
    # # "www": "VVV",
    # # "wwz": "VVV",
    # # "wzz": "VVV",
    # # "zzz": "VVV",
    "ggh_amcPS": "ggH",
    "vbf_powheg_dipole": "VBF",
}
# parameters["grouping"] = {"vbf_powheg_dipole": "VBF",}

parameters["plot_groups"] = {
    # "stack": ["DY", "EWK", "TT+ST", "VV", "VVV"],
    "stack": ["DY", "EWK", "TT+ST", "VV"],
    "step": ["VBF", "ggH"],
    "errorbar": ["Data"],
}


if __name__ == "__main__":
    start_time = time.time()
    if use_local_cluster:
        print(
            f"Creating local cluster with {ncpus_local} workers."
            f" Dashboard address: {dashboard_address}"
        )
        # client = Client(
        #     processes=True,
        #     dashboard_address=dashboard_address,
        #     n_workers=ncpus_local,
        #     threads_per_worker=1,
        #     memory_limit="4GB",
        # )
        client =  Client(
            processes=True,
            n_workers=50, # 60
            #dashboard_address=dash_local,
            threads_per_worker=1,
            memory_limit="3GB",
        )
    else:
        print(
            f"Connecting to Slurm cluster at {slurm_cluster_ip}."
            f" Dashboard address: {dashboard_address}"
        )
        client = Client(parameters["slurm_cluster_ip"])
    parameters["ncpus"] = len(client.scheduler_info()["workers"])
    print(f"Connected to cluster! #CPUs = {parameters['ncpus']}")

    # add MVA scores to the list of variables to plot
    dnn_models = list(parameters["dnn_models"].values())
    bdt_models = list(parameters["bdt_models"].values())
    for models in dnn_models + bdt_models:
        for model in models:
            parameters["plot_vars"] += ["score_" + model]
            parameters["templates_vars"] += ["score_" + model]

    parameters["datasets"] = parameters["grouping"].keys()

    # skip plots for now
    # # make plots
    # yields = plotter(client, parameters)
    # print(yields)

    # save templates to ROOT files
    yield_df = to_templates(client, parameters)
    print(f'run stage3 yield_df: {yield_df}')
    # groups = [g for g in yield_df.group.unique() if g != "Data"]
    # print(f'parameters["templates_vars"]: {parameters["templates_vars"]}')
    
    # print(f"yield groups: {groups}")

    datacard_str = parameters["dnn_models"]["vbf"][0]
    print(f"datacard_str: {datacard_str}")
    # make datacards
    build_datacards(f"score_{datacard_str}", yield_df, parameters)
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the elapsed time
    print(f"Execution time: {execution_time:.4f} seconds")