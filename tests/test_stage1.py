import os
import sys

[sys.path.append(i) for i in [".", ".."]]
import time

from coffea.processor import DaskExecutor, Runner
from coffea.nanoevents import NanoAODSchema

from stage1.SimpleProcessor import SimpleDimuonProcessor
from stage1.preprocessor import SamplesInfo
from test_tools import almost_equal
import dask
from dask.distributed import Client
from python.io import (
    mkdir,
    save_stage1_output_to_csv,
    delete_existing_stage1_output,
)
from functools import partial

__all__ = ["Client"]


if __name__ == "__main__":
    tick = time.time()

    client = dask.distributed.Client(
        processes=True, n_workers=1, threads_per_worker=1, memory_limit="2.9GB"
    )
    print("Client created")
    #file_path = "root://cmsxrootd.fnal.gov//store/data/Run2016C/SingleMuon/NANOAOD/HIPM_UL2016_MiniAODv2_NanoAODv9-v2/40000/0D1698EF-F93D-D84F-8529-4706B02CCB04.root"
    #file_path = "root://cmsxrootd.fnal.gov//store/mc/RunIISummer20UL16NanoAODAPVv9/DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1/2810000/D0C32DFC-4E62-3148-9801-467E2C205E94.root"
    file_path = "root://cmsxrootd.fnal.gov//store/mc/RunIISummer20UL16NanoAODAPVv9/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1//270000//BCF88722-7D2F-EF4B-A771-20FD4FEBD37A.root"
    #file_path = f"{os.getcwd()}/tests/samples/ewk_lljj_mll105_160_ptj0_NANOV10_2018.root"
    dataset = {"test": file_path}

    samp_info = SamplesInfo(xrootd=False)
    samp_info.paths = dataset
    samp_info.year = "2016preVFP"
    samp_info.load("test", use_dask=False)
    samp_info.lumi_weights["test"] = 1.0
    print(samp_info.fileset)
    out_dir="/home/vscheure/"
    executor_args = {"client": client, "use_dataframes": True, "retries": 0}
    processor_args = {
        "samp_info": samp_info,
        "do_btag_syst": False,
        "regions": ["h-peak"],
        "apply_to_output": partial(save_stage1_output_to_csv, out_dir=out_dir),
    }

    executor = DaskExecutor(**executor_args)
    run = Runner(executor=executor, schema=NanoAODSchema, chunksize=1000, maxchunks=10)
    output = run(
        samp_info.fileset,
        "Events",
        processor_instance=SimpleDimuonProcessor(**processor_args),
        
    )

    df = output.compute()
    print(df)
    print(df["lumi_weights"])

    elapsed = round(time.time() - tick, 3)
    print(f"Finished everything in {elapsed} s.")

    #dimuon_mass = df.loc[df.event == 2254006, "dimuon_mass"].values[0]
    #jj_mass = df.loc[df.event == 2254006, "jj_mass_nominal"].values[0]

    #assert df.shape == (391, 122)
    #assert almost_equal(dimuon_mass, 117.1209375)
    #assert almost_equal(jj_mass, 194.5646039)
