import glob
import tqdm
import argparse
import pandas as pd
from stage3.fitter import run_fits



parser = argparse.ArgumentParser()
parser.add_argument(
    "-y", "--year", nargs="+", help="Years to process", default=["2018"]
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




# global parameters
parameters = {
    # < general settings >
    "global_path": "/depot/cms/hmm/vscheure",
    "years": args.year,
    "label": args.label,
    #"channels": ["ggh_0jets","ggh_1jet","ggh_2orMoreJets","vbf"],
    #"channels": ["ggh"],
    "channels": ["ggh"],
        "category": ["cat1","cat2","cat3","cat4","cat5"],
    #"category": ["All"],
    "mva_channels": ["ggh"],
    "cats_by_score": True,
    #"cats_by_score": False,
    
    "signals": ["ggh_powheg"],
    "data": [ "data_x",
            ],
    #"regions": ["h-sidebands","h-peak"],
    "regions": ["h-peak"],
    
    "syst_variations": ["nominal"],

    
}

parameters["datasets"] = [
    #"data_A",
    #"data_B",
    #"data_C",
    #"data_D",
    #"data_E",
    #"data_F",
    #"data_G",
    #"data_H",
    "data_x",
    "ggh_powheg",

]


if __name__ == "__main__":


    
    # prepare lists of paths to parquet files (stage1 output) for each year and dataset
    #client = None
    all_paths = {}
    for year in parameters["years"]:
        all_paths[year] = {}
        for dataset in parameters["datasets"]:
            paths = glob.glob(
                f"{parameters['global_path']}/"
                f"{parameters['label']}/stage2_output/{year}/"
                f"{dataset}/*.csv"
            )
            #print(f"{parameters['global_path']}/"
               #f"{parameters['label']}/stage1_output/{year}/"
                #f"{dataset}/")
            all_paths[year][dataset] = paths
            print(all_paths)
    # run postprocessing
    for year in parameters["years"]:
        print(f"Processing {year}")
        
        for dataset, path in tqdm.tqdm(all_paths[year].items()):
            print(path)
            if len(path) == 0:
                continue

            # read stage2 outputs
            for pat in path:
                df = pd.read_csv(pat)

                run_fits(parameters, df)
