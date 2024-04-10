import glob
import tqdm
import argparse
import dask
from dask.distributed import Client
import dask.dataframe as dd

from python.io import load_dataframe,mkdir,save_stage2_output_to_csv

from stage2.postprocessor import process_partitions

from config.mva_bins import mva_bins
from config.variables import variables_lookup
import pdb

__all__ = ["dask"]


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

# Dask client settings
use_local_cluster = args.slurm_port is None
node_ip = "128.211.149.133"
node_ip = "128.211.149.140"

if use_local_cluster:
    ncpus_local = 50
    slurm_cluster_ip = ""
    dashboard_address = f"{node_ip}:34875"
else:
    slurm_cluster_ip = f"{node_ip}:{args.slurm_port}"
    dashboard_address = f"{node_ip}:8787"

# global parameters
parameters = {
    # < general settings >
    "slurm_cluster_ip": slurm_cluster_ip,
    "global_path": "/depot/cms/hmm/vscheure",
    #"global_path": "/work/users/vscheure",
    "years": args.year,
    "label": args.label,
    #"channels": ["ggh_0jets","ggh_1jet","ggh_2orMoreJets"],
    #"channels": ["none"],
    "channels": [ "ggh"],
        "category": ["cat1","cat2","cat3","cat4","cat5"],
    #"category": ["All"],
    #"mva_channels": ["ggh"],
    #"cats_by_score": True,
    "cats_by_score": False,
    "cats_by_eta": False,
    "cats_by_CalibCat": False,
    "cats_by_ClosureCat": False,
    
    "signals": ["ggh_powheg"],
    "data": [ "data_x",
            ],
    "regions": ["h-sidebands","h-peak"],
    #"regions": ["h-peak"],
    #"regions": ["z-peak"],
    "syst_variations": ["nominal"],
    # "custom_npartitions": {
    #     "vbf_powheg_dipole": 1,
    # },
    #
    # < settings for histograms >
    "hist_vars":  ["dimuon_mass","dimuon_pt"],
    #"hist_vars":  ["dimuon_mass","dimuon_pt","dimuon_ebe_mass_res", "mu1_eta",'mu2_eta',"mu1_phi",'mu2_phi',"mu1_pt","dimuon_ebe_mass_res_raw","zpt_weight","dimuon_cos_theta_cs","zeppenfeld","jj_dEta","dimuon_phi_cs","jj_mass","dimuon_dR","njets","njets","mmj_min_dEta","mmj2_dPhi","jet1_eta", "jet1_phi",'jet1_qgl',"jet2_eta", "jet2_phi",'jet2_qgl', 'mu2_iso','jet2_pt','jet1_pt','mu2_pt'],    #['dimuon_cos_theta_cs', 'dimuon_dEta', 'dimuon_dPhi', 'dimuon_dR',  'dimuon_phi', 'dimuon_phi_cs', 'dimuon_pt', 'dimuon_pt_log', 'jet1_eta', 'jet1_phi', 'jet1_pt', 'jet1_qgl', 'jet2_eta', 'jet2_phi', 'jet2_pt', 'jet2_qgl', 'jj_dEta', 'jj_dPhi', 'jj_eta', 'jj_mass', 'jj_mass_log', 'jj_phi', 'jj_pt', 'll_zstar_log', 'mmj1_dEta', 'mmj1_dPhi', 'mmj2_dEta', 'mmj2_dPhi', 'mmj_min_dEta', 'mmj_min_dPhi', 'mmjj_eta', 'mmjj_mass', 'mmjj_phi', 'mmjj_pt', 'mu1_eta', 'mu1_iso', 'mu1_phi', 'mu1_pt_over_mass', 'mu2_eta', 'mu2_iso', 'mu2_phi', 'mu2_pt_over_mass', 'zeppenfeld'],
    

    "variables_lookup": variables_lookup,
    "save_hists": True,
    #
    # < settings for unbinned output>
    "tosave_unbinned": {
        #"vbf": ["dimuon_mass", "event", "wgt_nominal", "mu1_pt", "score_pytorch_test"],
        #"none": ["dimuon_mass", "event", "wgt_nominal", "mu1_pt", "score_pytorch_test"],
        #"ggh_0jets": ["dimuon_mass", "wgt_nominal"],
        #"ggh_1jet": ["dimuon_mass", "wgt_nominal"],
        #"ggh_2orMoreJets": ["dimuon_mass", "wgt_nominal"],
    },
    "save_unbinned": False,
    #
    # < MVA settings >
    "models_path": "/depot/cms/hmm/vscheure/data/trained_models/",
    "dnn_models": {
        #"vbf": ["ValerieDNNtest2","ValerieDNNtest3"],
        #"ggh": ["ggHtest2"]
        #"vbf": ["ValerieDNNtest3"]
        # "vbf": ["pytorch_test"],
        # "vbf": ["pytorch_jun27"],
        #"vbf": ["pytorch_jun27"],
        #"vbf": ["pytorch_jul12"],  # jun27 is best
        # "vbf": ["pytorch_aug7"],
         #"vbf": [
             #"ValerieDNNtest2",
             #"pytorch_jun27",
        #    #"pytorch_sep4",
        #    #"pytorch_sep2_vbf_vs_dy",
        #    #"pytorch_sep2_vbf_vs_ewk",
        #    #"pytorch_sep2_vbf_vs_dy+ewk",
        #    #"pytorch_sep2_ggh_vs_dy",
        #    #"pytorch_sep2_ggh_vs_ewk",
        #    #"pytorch_sep2_ggh_vs_dy+ewk",
        #    #"pytorch_sep2_vbf+ggh_vs_dy",
        #    #"pytorch_sep2_vbf+ggh_vs_ewk",
        #    #"pytorch_sep2_vbf+ggh_vs_dy+ewk",
         #],
        # "vbf": ["pytorch_may24_pisa"],
   },
    # "mva_categorizer": "3layers_64_32_16_all_feat",
    # "vbf_mva_cutoff": 0.5,
    "bdt_models": {
         "ggh": ["BDTperyear"],
    },
    "mva_bins_original": mva_bins,
}

parameters["datasets"] = [
    "data_A",
    "data_B",
    "data_C",
    "data_D",
    #"data_E",
    #"data_F",
    #"data_G",
    #"data_H",
    #"data_x",
    #"dy_M-50",
    #"dy_M-50_nocut",
    "dy_M-100To200",
    #"dy_1j",
    #"dy_2j",
    #"dy_m105_160_amc",
    #"dy_m105_160_vbf_amc",
    #"ewk_lljj_mll105_160_py_dipole",
    #"ewk_lljj_mll50_mjj120",
    #"ttjets_dl",
    #"ttjets_sl",
    #"ttw",
    #"ttz",
    #"st_tw_top",
    #"st_tw_antitop",
    #"ww_2l2nu",
    #"wz_2l2q",
    #"wz_1l1nu2q",
    #"wz_3lnu",
    #"zz",
    #"www",
    #"wwz",
    #wzz",
    #"zzz",
    "ggh_powheg",
    #"vbf_powheg",
]
# using one small dataset for debugging
#parameters["datasets"] = ["ggh_localTest"]

if __name__ == "__main__":
    # prepare Dask client
    if use_local_cluster:
        print(
            f"Creating local cluster with {ncpus_local} workers."
            f" Dashboard address: {dashboard_address}"
        )
        client = Client(
            processes=True,
            #dashboard_address=dashboard_address,
            n_workers=ncpus_local,
            threads_per_worker=1,
            memory_limit="120GB",
        )
    else:
        print(
            f"Connecting to Slurm cluster at {slurm_cluster_ip}."
            f" Dashboard address: {dashboard_address}"
        )
        client = Client(parameters["slurm_cluster_ip"])
    parameters["ncpus"] = len(client.scheduler_info()["workers"])
    print(f"Connected to cluster! #CPUs = {parameters['ncpus']}")

    # add MVA scores to the list of variables to create histograms from
    dnn_models = list(parameters["dnn_models"].values())
    bdt_models =list(parameters["bdt_models"].values())
    #bdt_models = list(parameters["bdt_models"].values())
    for models in dnn_models + bdt_models:
       for model in models:
            parameters["hist_vars"] += ["score_" + model + "_" + str(args.year[0])]
            print( parameters["hist_vars"])
    
    # prepare lists of paths to parquet files (stage1 output) for each year and dataset
    #client = None
    all_paths = {}
    for year in parameters["years"]:
        out_dir = parameters["global_path"]
        mkdir(out_dir)
        out_dir += "/" + parameters["label"]
        mkdir(out_dir)
        out_dir += "/" + "stage2_output"
        mkdir(out_dir)
        out_dir += "/" + str(year)
        mkdir(out_dir)
        all_paths[year] = {}
        for dataset in parameters["datasets"]:
            paths = glob.glob(
                f"{parameters['global_path']}/"
                f"{parameters['label']}/stage1_output/{year}/"
                f"{dataset}/*.parquet"
            )
            #print(f"{parameters['global_path']}/"
               #f"{parameters['label']}/stage1_output/{year}/"
                #f"{dataset}/")
            all_paths[year][dataset] = paths
            #print(all_paths)
    # run postprocessing
    for year in parameters["years"]:
        print(f"Processing {year}")
        
        for dataset, path in tqdm.tqdm(all_paths[year].items()):
            #print(path)
            if len(path) == 0:
                continue

            # read stage1 outputs
            df = load_dataframe(client, parameters, inputs=[path], dataset=dataset)
            #for i in range(0, len(df.columns), 10):
               # print(df.compute().columns[i:i+10])
            #pdb.set_trace()
            if dataset == "data_x":
                df = df.compute()
                df.loc[df.dataset=="data_A", "dataset"] = "data_x"
                df.loc[df.dataset=="data_B", "dataset"] = "data_x"
                df.loc[df.dataset=="data_C", "dataset"] = "data_x"
                df.loc[df.dataset=="data_D", "dataset"] = "data_x"
                df.loc[df.dataset=="data_E", "dataset"] = "data_x"
                df.loc[df.dataset=="data_F", "dataset"] = "data_x"
                df.loc[df.dataset=="data_G", "dataset"] = "data_x"
                df.loc[df.dataset=="data_H", "dataset"] = "data_x"
            print("have df, starting to compute")
            #print(df.compute())
            #if not isinstance(df, dd.DataFrame):
                #print("Dataframe not in correct format")
                #continue
            # run processing sequence (categorization, mva, histograms)
            info, df = process_partitions(client, parameters, df)
            print("processing done starting svaing csvs")
            do_calib = False
            do_closure = False
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
            if do_calib:

                for i in range(len(selections)):
                    columns_to_store = ["dataset","channel_nominal","category","dimuon_mass", "dimuon_ebe_mass_res","dimuon_ebe_mass_res_raw", "mu1_eta", "mu2_eta", "mu1_pt","wgt_nominal"]
                    df_store = df[columns_to_store]
                    cat_name = f"Calibration_cat{i}"
        
                    print(df_store.loc[selections[i]])
                
                    save_stage2_output_to_csv(df_store.loc[selections[i]],out_dir, f"{dataset}_{cat_name}")
                    #print(f"csvfile saved to {out_dir}/{dataset}_{cat_name}.csv")
                save_stage2_output_to_csv(df_store,out_dir, f"{dataset}")
            if do_closure:
                factor = [1.2201396317213455,1.2733852070151181,1.1809747179894707,1.2155635302806347,1.1752846132357713,
1.1577112474160125,1.2313568280842153,1.1596572743326237,1.0938879774699048,1.0946705676434387,
1.0612410287460479,
1.0226838219692747,
1.2459689572295658,
1.206549257362834,
1.0985030556145394,
1.2344789461784635,
1.2029933710784952,
1.151232006040953,
1.1703937430948665,
1.1188527603236753,
1.1075288965271624,
1.2142943072368348,
1.197141320666724,1.1590251747997593,
1.1519290432860205,1.2330753443778018,
1.1792095208933573,1.1820926891105363,1.089318044400659,1.0679024050680632]
                   
                columns_to_store = ["dataset","channel_nominal","category","dimuon_mass", "dimuon_ebe_mass_res_raw", "wgt_nominal"]
                df_store = df[columns_to_store]
                df_store["dimuon_ebe_mass_res_calib"] = 0
                for i in range(len(selections)):
                    df_store.loc[selections[i], "dimuon_ebe_mass_res_calib"] = df_store.loc[selections[i] , "dimuon_ebe_mass_res_raw"]*factor[i]
                    print(df_store.loc[selections[i]]["dimuon_ebe_mass_res_calib"])
                    selectionsClosure = [((df_store["dimuon_ebe_mass_res_calib"]>0.0)&(df_store["dimuon_ebe_mass_res_calib"]<=0.7)),
                                  ((df_store["dimuon_ebe_mass_res_calib"]>0.7)&(df_store["dimuon_ebe_mass_res_calib"]<=0.8)),
                                  ((df_store["dimuon_ebe_mass_res_calib"]>0.8)&(df_store["dimuon_ebe_mass_res_calib"]<=0.9)),
                                  ((df_store["dimuon_ebe_mass_res_calib"]>0.9)&(df_store["dimuon_ebe_mass_res_calib"]<=1.0)),
                                  ((df_store["dimuon_ebe_mass_res_calib"]>1.0)&(df_store["dimuon_ebe_mass_res_calib"]<=1.1)),
                                  ((df_store["dimuon_ebe_mass_res_calib"]>1.1)&(df_store["dimuon_ebe_mass_res_calib"]<=1.2)),
                                  ((df_store["dimuon_ebe_mass_res_calib"]>1.3)&(df_store["dimuon_ebe_mass_res_calib"]<=1.4)),
                                  ((df_store["dimuon_ebe_mass_res_calib"]>1.4)&(df_store["dimuon_ebe_mass_res_calib"]<=1.5)),
                                  ((df_store["dimuon_ebe_mass_res_calib"]>1.5)&(df_store["dimuon_ebe_mass_res_calib"]<=1.7)),
                                  ((df_store["dimuon_ebe_mass_res_calib"]>1.7)&(df_store["dimuon_ebe_mass_res_calib"]<=2.1)),
                                  ((df_store["dimuon_ebe_mass_res_calib"]>2.1)&(df_store["dimuon_ebe_mass_res_calib"]<=2.5)),
                                  ((df_store["dimuon_ebe_mass_res_calib"]>2.5)&(df_store["dimuon_ebe_mass_res_calib"]<=3.5)),]

                    
                for i in range(len(selectionsClosure)):
                    #if i==0:
                    cat_name = f"Closure_cat{i}"
        
                    print(df_store.loc[selectionsClosure[i]])
                
                    save_stage2_output_to_csv(df_store.loc[selectionsClosure[i]],out_dir, f"{dataset}_{cat_name}")
            #run_fits(client, parameters, df)
