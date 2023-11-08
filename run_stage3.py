import argparse
import dask
from dask.distributed import Client

from config.variables import variables_lookup
from stage3.plotter import plotter
from stage3.make_templates import to_templates
from stage3.make_datacards import build_datacards

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
parser.add_argument(
    "-d",
    "--dir",
    dest="plotsdir",
    default="testplots",
    action="store",
    help="Name of directory for saving plots",
)
args = parser.parse_args()

# Dask client settings
use_local_cluster = args.slurm_port is None
node_ip = "128.211.149.133"
node_ip = "128.211.149.140"

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
    "years": args.year,
    "global_path": "/depot/cms/hmm/vscheure/",
    "label": args.label,
     #"channels": ["vbf"],
    "channels": ["ggh"],
      #"channels": ["ggh_0jets"],
     #"channels": ["none"],
    #"channels": ["ggh_0jets","ggh_1jet","ggh_2orMoreJets","vbf"],
    #"regions": ["h-sidebands", "h-peak" ],
    "regions": ["h-peak"],
    #"regions": ["h-sidebands"],
    #"regions": ["none"],
    "syst_variations": ["nominal"],
    #
    # < plotting settings >
    "plot_vars":  ["dimuon_mass","dimuon_pt","dimuon_ebe_mass_res","dimuon_cos_theta_cs","zeppenfeld","jj_dEta","dimuon_phi_cs","jj_mass","dimuon_dR","njets","mu1_pt","mu1_eta","jet1_pt","njets","mmj_min_dEta","mmj2_dPhi","jet1_eta", "jet1_phi",],
    #"plot_vars": ["njets"],
    "variables_lookup": variables_lookup,
    "save_plots": True,
    "plot_ratio": True,
    "plots_path": f"{args.plotsdir}/",
   "dnn_models": {
       #"vbf": ["ValerieDNNtest2","ValerieDNNtest3"],
    "ggh": ["ggHtest2"],
       
       
        #"none": ["ValerieDNNtest2"],
        # "vbf": ["pytorch_test"],
        #"vbf": ["pytorch_jun27"],
       #"vbf": ["pytorch_jun27"],
       # "vbf": ["pytorch_jul12"],  # jun27 is best
        # "vbf": ["pytorch_aug7"],
        # "vbf": [
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
        # ],
        # "vbf": ["pytorch_may24_pisa"],
    },
    #"bdt_models": {},
    #
    # < templates and datacards >
    "save_templates": False,
    "templates_vars": [  "score_ValerieDNNtest3"],
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
    #"dy_M-50": "DY",
    #"dy_M-50_nocut": "DY_nocut",
    #"dy_M-100To200": "DY",
    #"dy_M-50_01j": "DY_01J",
    #"dy_M-50_2j": "DY_2J",
    "dy_M-100To200_01j": "DY_01J",
    "dy_M-100To200_2j": "DY_2J",
    #"dy_1j": "DY",
    #"dy_2j": "DY",
    #"dy_m105_160_amc": "DY",
    #"dy_m105_160_vbf_amc": "DY",
    #"dy_m105_160_amc_01j": "DY_01J",
    #"dy_m105_160_vbf_amc_01j": "DY_01J",
    #"dy_m105_160_amc_2j": "DY_01J",
    #"dy_m105_160_vbf_amc_2j": "DY_2J",
    # "ewk_lljj_mll105_160_py_dipole": "EWK",
    "ewk_lljj_mll50_mjj120": "EWK",
    "ttjets_dl": "TT+ST",
    "ttjets_sl": "TT+ST",
    #"ttw": "TT+ST",
    #"ttz": "TT+ST",
    "st_tw_top": "TT+ST",
    "st_tw_antitop": "TT+ST",
    "ww_2l2nu": "VV",
    "wz_2l2q": "VV",
    #"wz_1l1nu2q": "VV",
    "wz_3lnu": "VV",
    "zz": "VV",
    #"www": "VVV",
    #"wwz": "VVV",
    #"wzz": "VVV",
    #"zzz": "VVV",
    #"ggh_amcPS": "ggH",
    "ggh_powheg": "ggH",
    "vbf_powheg": "VBF",
    #"vbf_powheg_dipole_01j": "VBF_01J",
    # "vbf_powheg_dipole_0j": "VBF_0J",
    # "vbf_powheg_dipole_1j": "VBF_1J",
   # "vbf_powheg_dipole_2j": "VBF_2J",
}
# parameters["grouping"] = {"vbf_powheg_dipole": "VBF",}

parameters["plot_groups"] = {
    #"stack": ["DY","DY-M100","DY_01jets","DY_2jets", "EWK", "TT+ST", "VV", "VVV"],
    "stack": ["DY_01J","DY_2J", "EWK", "TT+ST", "VV", "VVV"],
    #"stack": ["DY", "EWK", "TT+ST", "VV", "VVV"],
    #"stack": ["DY-M50","DY-M100"],
    #"stack": ["DY_nocut"],
    "step": ["VBF", "ggH"],
    #"step": ["DY_nocut"],
    "errorbar": ["Data"],
    #"errorbar": [],
}


if __name__ == "__main__":
    if use_local_cluster:
        print(
            f"Creating local cluster with {ncpus_local} workers."
            f" Dashboard address: {dashboard_address}"
        )
        client = Client(
            processes=True,
            #sdashboard_address=dashboard_address,
            n_workers=ncpus_local,
            threads_per_worker=1,
            memory_limit="4GB",
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
    #dnn_models = []
    dnn_models = list(parameters["dnn_models"].values())
    bdt_models = []
    #bdt_models = list(parameters["bdt_models"].values())
    for models in dnn_models + bdt_models:
       for model in models:
            parameters["plot_vars"] += ["score_" + model]
            parameters["templates_vars"] += ["score_" + model]

    parameters["datasets"] = parameters["grouping"].keys()

    # make plots
    #print(parameters)
    yields = plotter(client, parameters)
    #print(yields)

    # save templates to ROOT files
    yield_df = to_templates(client, parameters)
    #print(yield_df)

    # make datacards
    build_datacards("score_ValerieDNNtest3", yield_df, parameters)
    
