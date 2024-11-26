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

import logging
logger = logging.getLogger("distributed.utils_perf")
logger.setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings('ignore')



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
    ncpus_local = 15 #30 #2
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
    # "label": "DmitryMaster_JECoff_GeofitFixed_Nov01",
    # "label": "rereco_yun_Nov04",
    "label": args.label,
    "channels": ["vbf"],
    "regions": ["h-peak", "h-sidebands"],
    # "syst_variations": ["nominal"],
    # "syst_variations":['nominal', 'Absolute_up', 'Absolute_down', 'Absolute2018_up', 'Absolute2018_down', 'BBEC1_up', 'BBEC1_down', 'BBEC12018_up', 'BBEC12018_down', 'EC2_up', 'EC2_down', 'EC22018_up', 'EC22018_down', 'HF_up', 'HF_down', 'HF2018_up', 'HF2018_down', 'RelativeBal_up', 'RelativeBal_down', 'RelativeSample2018_up', 'RelativeSample2018_down', 'FlavorQCD_up', 'FlavorQCD_down',],
    "syst_variations": ['nominal', 'Absolute_up', 'Absolute_down', 'Absolute2018_up', 'Absolute2018_down', 'BBEC1_up', 'BBEC1_down', 'BBEC12018_up', 'BBEC12018_down', 'EC2_up', 'EC2_down', 'EC22018_up', 'EC22018_down', 'HF_up', 'HF_down', 'HF2018_up', 'HF2018_down', 'RelativeBal_up', 'RelativeBal_down', 'RelativeSample2018_up', 'RelativeSample2018_down', 'FlavorQCD_up', 'FlavorQCD_down', 'jer1_up', 'jer1_down', 'jer2_up', 'jer2_down', 'jer3_up', 'jer3_down', 'jer4_up', 'jer4_down', 'jer5_up', 'jer5_down', 'jer6_up', 'jer6_down'], # full 2018
    # "syst_variations": ["nominal", "Absolute_up", "RelativeBal_up", "FlavorQCD_up", "RelativeSample2018_up","Absolute_down", "RelativeBal_down", "FlavorQCD_down", "RelativeSample2018_down"],
    # "syst_variations": ["nominal", "Absolute_up", "RelativeBal_up", "FlavorQCD_up", "RelativeSample2016_up","Absolute_down", "RelativeBal_down", "FlavorQCD_down", "RelativeSample2016_down"],
    # "syst_variations": ["nominal", "Absolute2016_up", "Absolute2016_down","Absolute_up",],
    # "syst_variations":['nominal', 'Absolute_up', 'Absolute_down', 'Absolute2016_up', 'Absolute2016_down', 'BBEC1_up', 'BBEC1_down', 'BBEC12016_up', 'BBEC12016_down', 'EC2_up', 'EC2_down', 'EC22016_up', 'EC22016_down', 'HF_up', 'HF_down', 'HF2016_up', 'HF2016_down', 'RelativeBal_up', 'RelativeBal_down', 'RelativeSample2016_up', 'RelativeSample2016_down', 'FlavorQCD_up', 'FlavorQCD_down', 'jer1_up', 'jer1_down', 'jer2_up', 'jer2_down', 'jer3_up', 'jer3_down', 'jer4_up', 'jer4_down', 'jer5_up', 'jer5_down', 'jer6_up', 'jer6_down'], # taken from printing "self.pt_variations" in stage1/processor.py
    # "syst_variations":['nominal', 'Absolute_up', 'Absolute_down'],
     # "syst_variations":['nominal', 'Absolute_up', 'Absolute_down', 'Absolute2016_up', 'Absolute2016_down', 'BBEC1_up', 'BBEC1_down', 'BBEC12016_up', 'BBEC12016_down', 'EC2_up', 'EC2_down', 'EC22016_up', 'EC22016_down', 'HF_up', 'HF_down', 'HF2016_up', 'HF2016_down', 'RelativeBal_up', 'RelativeBal_down', 'RelativeSample2016_up', 'RelativeSample2016_down', 'FlavorQCD_up', 'FlavorQCD_down',], # taken from printing "self.pt_variations" in stage1/processor.py
    # "syst_variations":['nominal', 'Absolute_up', 'Absolute_down', 'Absolute2017_up', 'Absolute2017_down', 'BBEC1_up', 'BBEC1_down', 'BBEC12017_up', 'BBEC12017_down', 'EC2_up', 'EC2_down', 'EC22017_up', 'EC22017_down', 'HF_up', 'HF_down', 'HF2017_up', 'HF2017_down', 'RelativeBal_up', 'RelativeBal_down', 'RelativeSample2017_up', 'RelativeSample2017_down', 'FlavorQCD_up', 'FlavorQCD_down', ], # taken from printing "self.pt_variations" in stage1/processor.py
    
    "custom_npartitions": {
        # "vbf_powheg_dipole": 1,
        "dy_m105_160_amc" : 519,
        "dy_m105_160_vbf_amc" : 295,
    },
    #
    # < settings for histograms >
    # "hist_vars": ["dimuon_mass"],
    "hist_vars": [],
    "variables_lookup": variables_lookup,
    "save_hists": True,
    #
    # < settings for unbinned output>
    "tosave_unbinned": {
        "vbf": ["dimuon_mass", "event", "wgt_nominal", "mu1_pt", "score_pytorch_jun27", "region", "year"],
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
    # "data_C",
    # "data_D",
    # "data_E",
    # "data_F",
    # "data_G",
    # "data_H",
    # "dy_m105_160_amc",
    # "dy_m105_160_vbf_amc",
    "ewk_lljj_mll105_160_py_dipole",
    "ewk_lljj_mll105_160_ptj0",
    # "ttjets_dl",
    # "ttjets_sl",
    # "ttw",
    # "ttz",
    # "st_tw_top",
    # "st_tw_antitop",
    # "ww_2l2nu",
    # "wz_2l2q",
    # "wz_1l1nu2q",
    # "wz_3lnu",
    # "zz",
    # # # "www",
    # # # "wwz",
    # # # "wzz",
    # # # "zzz",
    "ggh_amcPS",
    "ggh_powhegPS",
    "vbf_powheg_dipole",
    # "vbf_powhegPS",
    "vbf_powheg_herwig",
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
            n_workers=ncpus_local, # 15 ncpus_local
            #dashboard_address=dash_local,
            threads_per_worker=2,#1
            memory_limit="20GB",#12
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
            print(f"load_path {load_path}")
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

            print("processing partitions!")
            # run processing sequence (categorization, mva, histograms)
            info = process_partitions(client, parameters, df)
            # print(info)

    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
