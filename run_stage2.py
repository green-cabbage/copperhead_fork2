import glob
import tqdm
import argparse
import dask
from dask.distributed import Client
import dask.dataframe as dd

from python.io import load_dataframe
from stage2.postprocessor import process_partitions

from config.mva_bins import mva_bins
from config.variables import variables_lookup
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
args = parser.parse_args()

# Dask client settings
use_local_cluster = args.slurm_port is None
node_ip = "128.211.149.133"

if use_local_cluster:
    ncpus_local = 50
    slurm_cluster_ip = ""
    dashboard_address = f"{node_ip}:34875"
else:
    slurm_cluster_ip = f"{node_ip}:{args.slurm_port}"
    dashboard_address = f"{node_ip}:8787"
    ncpus_local = 200

# global parameters
parameters = {
    # < general settings >
    "slurm_cluster_ip": slurm_cluster_ip,
    "global_path": "/depot/cms/users/yun79/hmm/copperheadV1clean/",
    # "global_path": "/work/users/yun79/copperhead_outputs/copperheadV1clean",
    "years": args.years,
    # "label": "DmitryMaster_JECoff_GeofitFixed_Oct29",
    "label": "DmitryMaster_JECoff_GeofitFixed_Nov01",
    "channels": ["vbf"],
    "regions": ["h-peak", "h-sidebands"],
    "syst_variations": ["nominal"],
    # "custom_npartitions": {
    #     "vbf_powheg_dipole": 1,
    # },
    #
    # < settings for histograms >
    "hist_vars": ["dimuon_mass"],
    "variables_lookup": variables_lookup,
    "save_hists": True,
    #
    # < settings for unbinned output>
    "tosave_unbinned": {
        "vbf": ["dimuon_mass", "event", "wgt_nominal", "mu1_pt", "score_pytorch_test"],
        "ggh_0jets": ["dimuon_mass", "wgt_nominal"],
        "ggh_1jet": ["dimuon_mass", "wgt_nominal"],
        "ggh_2orMoreJets": ["dimuon_mass", "wgt_nominal"],
    },
    "save_unbinned": True,
    #
    # < MVA settings >
    # "models_path": "data/trained_models/",
    # "dnn_models": {
    #     "vbf": ["pytorch_test"],
    # },
    # "models_path": "data/trained_models/",
     "models_path" : "/depot/cms/hmm/copperhead/trained_models/",
    "dnn_models": {
        # "vbf": ["vbf"],
         "vbf": ["pytorch_jun27"],
    },
    "bdt_models": {},
    "mva_bins_original": mva_bins,
    "ncpus" : ncpus_local,
}

parameters["datasets"] = [
    # "data_A",
    # "data_B",
    "data_C",
    "data_D",
    # "data_E",
    # "data_F",
    # "data_G",
    # "data_H",
    # "dy_m105_160_amc",
    # "dy_m105_160_vbf_amc",
    # "ewk_lljj_mll105_160_py_dipole",
    # # # "ewk_lljj_mll105_160_ptj0",
    # # "ttjets_dl",
    # # "ttjets_sl",
    # # # # "ttw",
    # # # # "ttz",
    # # "st_tw_top",
    # # "st_tw_antitop",
    # # # "ww_2l2nu",
    # # # "wz_2l2q",
    # # # "wz_1l1nu2q",
    # # # "wz_3lnu",
    # # # "zz",
    # # # "www",
    # # # "wwz",
    # # # "wzz",
    # # # "zzz",
    # "ggh_amcPS",
    # "vbf_powheg_dipole",
]
# using one small dataset for debugging
# parameters["datasets"] = ["ggh_amcPS","vbf_powheg_dipole"]

if __name__ == "__main__":
    start_time = time.time()
    # prepare Dask client
    if use_local_cluster:
        print(
            f"Creating local cluster with {ncpus_local} workers."
            f" Dashboard address: {dashboard_address}"
        )
    #     client = Client(
    #         processes=True,
    #         dashboard_address=dashboard_address,
    #         n_workers=ncpus_local,
    #         threads_per_worker=1,
    #         memory_limit="4GB",
    #     )
        client =  Client(
            processes=True,
            n_workers=ncpus_local, # 60
            #dashboard_address=dash_local,
            threads_per_worker=1,
            memory_limit="3GB",
        )
    else:
    #     print(
    #         f"Connecting to Slurm cluster at {slurm_cluster_ip}."
    #         f" Dashboard address: {dashboard_address}"
    #     )
    #     client = Client(parameters["slurm_cluster_ip"])
    # parameters["ncpus"] = len(client.scheduler_info()["workers"])
        from dask_gateway import Gateway
        gateway = Gateway(
            "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
            proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
        )
        cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
        client = gateway.connect(cluster_info.name).get_client()
        parameters["client"] = client
    # print(f"Connected to cluster! #CPUs = {parameters['ncpus']}")

    # add MVA scores to the list of variables to create histograms from
    dnn_models = list(parameters["dnn_models"].values())
    bdt_models = list(parameters["bdt_models"].values())
    for models in dnn_models + bdt_models:
        for model in models:
            parameters["hist_vars"] += ["score_" + model]

    # prepare lists of paths to parquet files (stage1 output) for each year and dataset
    all_paths = {}
    for year in parameters["years"]:
        all_paths[year] = {}
        for dataset in parameters["datasets"]:
            load_path = f"{parameters['global_path']}/" +f"{parameters['label']}/stage1_output/{year}/"+ f"{dataset}/*.parquet"
            # print(f"load_path {load_path}")
            paths = glob.glob(
                load_path
            )
            all_paths[year][dataset] = paths
    # print(f"all_paths {all_paths}")
    # run postprocessing
    for year in parameters["years"]:
        print(f"Processing {year}")
        for dataset, path in tqdm.tqdm(all_paths[year].items()):
            # print(f"dataset {dataset}")
            # print(f"path {path}")
            if len(path) == 0:
                continue

            # read stage1 outputs
            df = load_dataframe(client, parameters, inputs=[path], dataset=dataset)
            if not isinstance(df, dd.DataFrame):
                continue

            # run processing sequence (categorization, mva, histograms)
            info = process_partitions(client, parameters, df)
            # print(info)

    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
