import os
import sys

[sys.path.append(i) for i in [".", ".."]]
import time

import dask
from dask.distributed import Client

from python.io import load_dataframe
from config.variables import variables_lookup
from stage2.postprocessor import process_partitions
from test_tools import almost_equal
from config.mva_bins import mva_bins

__all__ = ["dask"]


parameters = {
    "ncpus": 1,
    "years": [2017],
    "datasets": ["dy"],
    #"channels": ["ggh_0jets","ggh_1jets","ggh_2orMoreJets","vbf"],
    "channels": ["vbf"],
    "regions": ["h-peak","h-sidebands"],
    "hist_vars": ["dimuon_mass"],
    "tosave_unbinned": {
        "vbf": ["dimuon_mass", "event", "wgt_nominal", "mu1_pt", "score_pytorch_test"],
        "ggh_0jets": ["dimuon_mass", "wgt_nominal"],
        "ggh_1jet": ["dimuon_mass", "wgt_nominal"],
        "ggh_2orMoreJets": ["dimuon_mass", "wgt_nominal"],
    },
    "save_unbinned": True,
    "variables_lookup": variables_lookup,
        # < MVA settings >
    "models_path": "/depot/cms/hmm/copperhead/trained_models",
    "dnn_models": {
        "vbf": ["pytorch_sep1_vbf+ggh_vs_dy"],
    },
    "bdt_models": {},
    "mva_bins_original": mva_bins,
}


if __name__ == "__main__":
    tick = time.time()

    client = Client(
        processes=True, n_workers=1, threads_per_worker=1, memory_limit="4GB"
    )

    file_name = "dy_stage1.parquet"
    path = f"{os.getcwd()}/tests/samples/{file_name}"

    df = load_dataframe(client, parameters, inputs=[path])
    out_hist = process_partitions(client, parameters, df)

    elapsed = round(time.time() - tick, 3)
    print(f"Finished everything in {elapsed} s.")
    print(out_hist)
   #assert almost_equal(
        #out_hist.loc[out_hist.variation == "nominal", "yield"].values[0],
        #21.0,
    #)
