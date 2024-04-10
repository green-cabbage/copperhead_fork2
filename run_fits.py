import glob
import tqdm
import argparse
import pandas as pd
from stage3.fitter import run_fits
import pdb



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
        #"category": ["cat1","cat2","cat3","cat4","cat5"],
    #"category": ["All"],
    "mva_channels": ["ggh"],
    "cats_by_score": True,
    #"cats_by_score": False,
    
    "signals": ["ggh_powheg"],
    "data": ["data_A",
             "data_B",
             "data_C",
            "data_D",
            ],
    #"regions": ["h-sidebands","h-peak"],
    "regions": ["z-peak"],
    "is_Z": True,
    
    "syst_variations": ["nominal"],

    
}

parameters["datasets"] = [
    "data_A",
    #"data_B",
    #"data_C",
    #"data_D",
    #"data_E",
    #"data_F",
    #"data_G",
    #"data_H",
    #"data_x",
    #"ggh_powheg",

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
                f"{dataset}"
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
                do_calib_fits=False
                do_closure_fits=True
                if not do_closure_fits:
                    df = pd.read_csv(f"{pat}/{dataset}.csv")
                    df_all = pd.read_csv(f"{pat}/{dataset}_nocats.csv")
                if args.year[0] == "combined":
                    columns_to_check = ["score_BDTperyear_2016postVFP_nominal", "score_BDTperyear_2016preVFP_nominal", "score_BDTperyear_2017_nominal", "score_BDTperyear_2018_nominal"]
                #else:
                    #columns_to_check = [f"score_BDTperyear_{args.year[0]}_nominal"]
                    #columns_to_check = [f"score_BDTperyear_{args.year[0]}_nominal"]

                # Drop rows where all specified columns have NaN values
                #df_all_filtered = df_all.dropna(subset=columns_to_check, how='all')
                #print(df.keys)
                

                if do_calib_fits:
                    df_all = df
                    BB = ((abs(df["mu1_eta"])<=0.9) & (abs(df["mu2_eta"])<=0.9))
                    BO = ((abs(df["mu1_eta"])<=0.9) & ((abs(df["mu2_eta"])>0.9) & (abs(df["mu2_eta"]) <=1.8)))
                    BE = ((abs(df["mu1_eta"])<=0.9) & ((abs(df["mu2_eta"])>1.8) & (abs(df["mu2_eta"]) <=2.4)))
                    OB = (((abs(df["mu1_eta"])>0.9) & (abs(df["mu1_eta"]) <=1.8)) & (abs(df["mu2_eta"])<=0.9))
                    OO = (((abs(df["mu1_eta"])>0.9) & (abs(df["mu1_eta"]) <=1.8)) & ((abs(df["mu2_eta"])>0.9) & (abs(df["mu2_eta"]) <=1.8)))
                    OE = (((abs(df["mu1_eta"])>0.9) & (abs(df["mu1_eta"]) <=1.8)) & ((abs(df["mu2_eta"])>1.8) & (abs(df["mu2_eta"]) <=2.4)))
                    EB = (((abs(df["mu1_eta"])>1.8) & (abs(df["mu1_eta"]) <=2.4)) & (abs(df["mu2_eta"])<=0.9))
                    EO = (((abs(df["mu1_eta"])>1.8) & (abs(df["mu1_eta"]) <=2.4)) & ((abs(df["mu2_eta"])>0.9) & (abs(df["mu2_eta"]) <=1.8)))
                    EE = (((abs(df["mu1_eta"])>1.8) & (abs(df["mu1_eta"]) <=2.4)) & ((abs(df["mu2_eta"])>1.8) & (abs(df["mu2_eta"]) <=2.4)))
                    selections = [((df["mu1_pt"]>30)&(df["mu1_pt"]<=45)&(BB | OB | EB)),
                                  ((df["mu1_pt"]>30)&(df["mu1_pt"]<=45)&(BO | OO | EO)),
                                  ((df["mu1_pt"]>30)&(df["mu1_pt"]<=45)&(BE | OE | EE)),
                                  ((df["mu1_pt"]>45)&(df["mu1_pt"]<=52)&BB),
                                  ((df["mu1_pt"]>45)&(df["mu1_pt"]<=52)&BO),
                                  ((df["mu1_pt"]>45)&(df["mu1_pt"]<=52)&BE),
                                  ((df["mu1_pt"]>45)&(df["mu1_pt"]<=52)&OB),
                                  ((df["mu1_pt"]>45)&(df["mu1_pt"]<=52)&OO),
                                  ((df["mu1_pt"]>45)&(df["mu1_pt"]<=52)&OE),
                                  ((df["mu1_pt"]>45)&(df["mu1_pt"]<=52)&EB),
                                  ((df["mu1_pt"]>45)&(df["mu1_pt"]<=52)&EO),
                                  ((df["mu1_pt"]>45)&(df["mu1_pt"]<=52)&EE),
                                  ((df["mu1_pt"]>52)&(df["mu1_pt"]<=62)&BB),
                                  ((df["mu1_pt"]>52)&(df["mu1_pt"]<=62)&BO),
                                  ((df["mu1_pt"]>52)&(df["mu1_pt"]<=62)&BE),
                                  ((df["mu1_pt"]>52)&(df["mu1_pt"]<=62)&OB),
                                  ((df["mu1_pt"]>52)&(df["mu1_pt"]<=62)&OO),
                                  ((df["mu1_pt"]>52)&(df["mu1_pt"]<=62)&OE),
                                  ((df["mu1_pt"]>52)&(df["mu1_pt"]<=62)&EB),
                                  ((df["mu1_pt"]>52)&(df["mu1_pt"]<=62)&EO),
                                  ((df["mu1_pt"]>52)&(df["mu1_pt"]<=62)&EE),
                                  ((df["mu1_pt"]>62)&(df["mu1_pt"]<=200)&BB),
                                  ((df["mu1_pt"]>62)&(df["mu1_pt"]<=200)&BO),
                                  ((df["mu1_pt"]>62)&(df["mu1_pt"]<=200)&BE),
                                  ((df["mu1_pt"]>62)&(df["mu1_pt"]<=200)&OB),
                                  ((df["mu1_pt"]>62)&(df["mu1_pt"]<=200)&OO),
                                  ((df["mu1_pt"]>62)&(df["mu1_pt"]<=200)&OE),
                                  ((df["mu1_pt"]>62)&(df["mu1_pt"]<=200)&EB),
                                  ((df["mu1_pt"]>62)&(df["mu1_pt"]<=200)&EO),
                                  ((df["mu1_pt"]>62)&(df["mu1_pt"]<=200)&EE),]
                    
                    for i in range(len(selections)):
                        if  i==11:
                            selection = selections[i]
                            df_i = df[(selection==True)]
                            print(df_i)
                            tag = f"{parameters['label']}_calib_cat{i}"
                            run_fits(parameters, df_i,df_i,tag)

                elif do_closure_fits:
                    for i in [1]:
                        df = pd.read_csv(f"{pat}/{dataset}_Closure_cat{i}.csv")
                        
                        df["category"] = "All"
                        print(df)
                        tag = f"{parameters['label']}_closure_cat{i}"
                        run_fits(parameters, df,df,tag)

                else:
                    tag = ""
                    run_fits(parameters, df,df_all,tag)