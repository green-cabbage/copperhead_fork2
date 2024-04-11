def for_all_years(value):
    out = {k: value for k in ["2016preVFP","2016postVFP", "2017", "2018","2022EE"]}
    return out


parameters = {}

parameters.update(
    {
        "muon_pt_cut": for_all_years(20.0),
        "muon_eta_cut": for_all_years(2.4),
        "muon_iso_cut": for_all_years(0.25),  # medium iso
        "muon_id": for_all_years("mediumId"),
        # "muon_flags": for_all_years(["isGlobal", "isTracker"]),
        "muon_flags": for_all_years([]),
        "muon_leading_pt": {"2016preVFP": 26.0,"2016postVFP": 26.0, "2017": 29.0, "2018": 26.0, "2022EE": 26.0,},
        "muon_trigmatch_iso": for_all_years(0.15),  # tight iso
        "muon_trigmatch_dr": for_all_years(0.1),
        "muon_trigmatch_id": for_all_years("tightId"),
        "electron_pt_cut": for_all_years(20.0),
        "electron_eta_cut": for_all_years(2.5),
        "electron_id_run3": for_all_years("mvaIso_WP90"), #Run3 ready!
        "electron_id_UL": for_all_years("mvaFall17V2Iso_WP90"),
        "jet_pt_cut": for_all_years(25.0),
        "jet_eta_cut": for_all_years(4.7),
        "jet_id": {"2016preVFP": "loose","2016postVFP": "loose", "2017": "tight", "2018": "tight","2022EE": "tight",},
        "jet_puid": {"2016preVFP": "loose","2016postVFP": "loose", "2017": "loose", "2018": "loose", "2022EE": "loose",},
        "min_dr_mu_jet": for_all_years(0.4),
        "btag_loose_wp": {"2016preVFP": 0.2027,"2016postVFP": 0.1918 ,"2017": 0.1355, "2018": 0.0490,"2022EE":0.1200,}, ## fix 2022
        "btag_medium_wp": {"2016preVFP": 0.6001,"2016postVFP": 0.4847, "2017": 0.4506, "2018": 0.4168, "2022EE":0.4168,}, ## fix 2022
        "softjet_dr2": for_all_years(0.16),
    }
)

parameters["lumimask"] = {
    "2016preVFP": "data/lumimasks/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt",
    "2016postVFP": "data/lumimasks/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt",
    "2017": "data/lumimasks/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt",
    "2018": "data/lumimasks/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt",
    "2022EE": "data/lumimasks/Cert_Collisions2022_355100_362760_Golden.txt",
}

parameters["hlt"] = {
    "2016preVFP": ["IsoMu24", "IsoTkMu24"],
    "2016postVFP": ["IsoMu24", "IsoTkMu24"],
    "2017": ["IsoMu27"],
    "2018": ["IsoMu24"],
    "2022EE": ["IsoMu24"],
}

parameters["roccor_file"] = {
    "2016preVFP": "data/roch_corr/RoccoR2016aUL.txt",
    "2016postVFP": "data/roch_corr/RoccoR2016bUL.txt",
    "2017": "data/roch_corr/RoccoR2017UL.txt",
    "2018": "data/roch_corr/RoccoR2018UL.txt",
    "2022EE": "data/roch_corr/RoccoR2018UL.txt",
}

parameters["nnlops_file"] = for_all_years("data/NNLOPS_reweight.root")

#parameters["btag_sf_csv"] = { #preUL
#    "2016preVFP": "data/btag/DeepCSV_2016LegacySF_V1.csv",
#    "2016postVFP": "data/btag/DeepCSV_2016LegacySF_V1.csv",
#    "2017": "data/btag/DeepCSV_94XSF_V5_B_F.csv",
#    "2018": "data/btag/DeepCSV_102XSF_V1.csv",
#}
parameters["btag_sf_json"] = {
    #"2016preVFP": "data/btag/DeepCSV_106XUL16preVFPSF_v1.csv",
    #"2016postVFP": "data/btag/DeepCSV_106XUL16postVFPSF_v2.csv",
    "2016preVFP": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2016preVFP_UL/btagging.json.gz",
    "2016postVFP": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2016postVFP_UL/btagging.json.gz",
    "2017": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2017_UL/btagging.json.gz",
    "2018": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2018_UL/btagging.json.gz",
    "2022EE": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2022_Summer22EE/btagging.json.gz",
}
parameters["btag_sf_csv"] = {

    "2016preVFP": "data/btag/DeepCSV_2016LegacySF_V1.csv",
    "2016postVFP": "data/btag/DeepCSV_2016LegacySF_V1.csv",
    "2017": "data/btag/DeepCSV_106XUL17SF.csv",
    "2018": "data/btag/DeepCSV_106XUL18SF.csv",
    "2022EE": "data/btag/DeepCSV_106XUL18SF.csv",
}

parameters["pu_file_data"] = {
    "2016preVFP": "data/pileup/puData2016_UL_withVar.root",
    "2016postVFP": "data/pileup/puData2016_UL_withVar.root",
    "2017": "data/pileup/puData2017_UL_withVar.root",
    "2018": "data/pileup/puData2018_UL_withVar.root",
    "2022EE": "data/pileup/puData2018_UL_withVar.root",
}

parameters["pu_file_mc"] = {
    "2016preVFP": "data/pileup/pileup_profile_Summer16.root",
    "2016postVFP": "data/pileup/pileup_profile_Summer16.root",
    "2017": "data/pileup/mcPileup2017.root",
    "2018": "data/pileup/mcPileup2018.root",
    "2022EE": "data/pileup/mcPileup2018.root",
}

parameters["muSFFileList"] = {
    "2016preVFP": [
        {
            "id": (
                "data/muon_sf/year2016/MuonSF_Run2016_UL_HIPM_ID.root",
                "NUM_MediumID_DEN_TrackerMuons_abseta_pt",
            ),
            "iso": (
                "data/muon_sf/year2016/MuonSF_Run2016_UL_HIPM_ISO.root",
                "NUM_TightRelIso_DEN_MediumID_abseta_pt",
            ),
            "trig": (
                "data/muon_sf/mu2016/EfficienciesStudies_2016_trigger_EfficienciesAndSF_RunBtoF.root",
                "IsoMu24_OR_IsoTkMu24_PtEtaBins/efficienciesDATA/abseta_pt_DATA",
                "IsoMu24_OR_IsoTkMu24_PtEtaBins/efficienciesMC/abseta_pt_MC",
            ),
            "scale": 1.0,
        },
    ],
       "2016postVFP": [
        {
            "id": (
                "data/muon_sf/year2016/MuonSF_Run2016_UL_ID.root",
                "NUM_MediumID_DEN_TrackerMuons_abseta_pt",
            ),
            "iso": (
                "data/muon_sf/year2016/MuonSF_Run2016_UL_ISO.root",
                "NUM_TightRelIso_DEN_MediumID_abseta_pt",
            ),
            "trig": (
                "data/muon_sf/mu2016/EfficienciesStudies_2016_trigger_EfficienciesAndSF_RunGtoH.root",
                "IsoMu24_OR_IsoTkMu24_PtEtaBins/efficienciesDATA/abseta_pt_DATA",
                "IsoMu24_OR_IsoTkMu24_PtEtaBins/efficienciesMC/abseta_pt_MC",
            ),
            "scale": 1.0,
        },
    ],
    "2017": [
        {
            "id": (
                "data/muon_sf/year2017/MuonSF_Run2017_UL_ID.root",
                "NUM_MediumID_DEN_TrackerMuons_abseta_pt",
            ),
            "iso": (
                "data/muon_sf/year2017/MuonSF_Run2017_UL_ISO.root",
                "NUM_TightRelIso_DEN_MediumID_abseta_pt",
            ),
            "trig": (
                "data/muon_sf/mu2017/EfficienciesAndSF_RunBtoF_Nov17Nov2017.root",
                "IsoMu27_PtEtaBins/efficienciesDATA/abseta_pt_DATA",
                "IsoMu27_PtEtaBins/efficienciesMC/abseta_pt_MC",
            ),
            "scale": 1.0,
        }
    ],
    "2018": [
        {
            "id": (
                "data/muon_sf/year2018/MuonSF_Run2018_UL_ID.root",
                "NUM_MediumID_DEN_TrackerMuons_abseta_pt",
            ),
            "iso": (
                "data/muon_sf/year2018/MuonSF_Run2018_UL_ISO.root",
                "NUM_TightRelIso_DEN_MediumID_abseta_pt",
            ),
            "trig": (
                "data/muon_sf/mu2018/EfficienciesStudies_2018_trigger_EfficienciesAndSF_2018Data_BeforeMuonHLTUpdate.root",
                "IsoMu24_PtEtaBins/efficienciesDATA/abseta_pt_DATA",
                "IsoMu24_PtEtaBins/efficienciesMC/abseta_pt_MC",
            ),
            "scale": 8.95 / 59.74,
        },
        {
            "id": (
                "data/muon_sf/year2018/MuonSF_Run2018_UL_ID.root",
                "NUM_MediumID_DEN_TrackerMuons_abseta_pt",
            ),
            "iso": (
                "data/muon_sf/year2018/MuonSF_Run2018_UL_ISO.root",
                "NUM_TightRelIso_DEN_MediumID_abseta_pt",
            ),
            "trig": (
                "data/muon_sf/mu2018/EfficienciesStudies_2018_trigger_EfficienciesAndSF_2018Data_AfterMuonHLTUpdate.root",
                "IsoMu24_PtEtaBins/efficienciesDATA/abseta_pt_DATA",
                "IsoMu24_PtEtaBins/efficienciesMC/abseta_pt_MC",
            ),
            "scale": 50.79 / 59.74,
        },
    ],
        "2022EE": [
        {
            "id": (
                "data/muon_sf/year2017/MuonSF_Run2017_UL_ID.root",
                "NUM_MediumID_DEN_TrackerMuons_abseta_pt",
            ),
            "iso": (
                "data/muon_sf/year2017/MuonSF_Run2017_UL_ISO.root",
                "NUM_TightRelIso_DEN_MediumID_abseta_pt",
            ),
            "trig": (
                "data/muon_sf/mu2017/EfficienciesAndSF_RunBtoF_Nov17Nov2017.root",
                "IsoMu27_PtEtaBins/efficienciesDATA/abseta_pt_DATA",
                "IsoMu27_PtEtaBins/efficienciesMC/abseta_pt_MC",
            ),
            "scale": 1.0,
        }
        ],
}

parameters["zpt_weights_file"] = for_all_years("data/reweight_zpt_2018_nJetBinned_new.histo.root")
parameters["puid_sf_file"] = for_all_years("data/PUID_106XTraining_ULRun2_EffSFandUncties_v1.root")
parameters["res_calib_path"] = for_all_years("data/res_calib/")

parameters["sths_names"] = for_all_years(
    [
        "Yield",
        "PTH200",
        "Mjj60",
        "Mjj120",
        "Mjj350",
        "Mjj700",
        "Mjj1000",
        "Mjj1500",
        "PTH25",
        "JET01",
    ]
)

parameters["btag_systs"] = for_all_years(
    [
        "jes",
        "lf",
        "hfstats1",
        "hfstats2",
        "cferr1",
        "cferr2",
        "hf",
        "lfstats1",
        "lfstats2",
    ]
)

parameters.update(
    {
        "event_flags": for_all_years(
            [
                "BadPFMuonFilter",
                "EcalDeadCellTriggerPrimitiveFilter",
                "HBHENoiseFilter",
                "HBHENoiseIsoFilter",
                "globalSuperTightHalo2016Filter",
                "goodVertices",
                "BadChargedCandidateFilter",
            ]
        ),
        "do_l1prefiring_wgts": {"2016preVFP": True,"2016postVFP": True, "2017": True, "2018": False,"2022EE": False},
    }
)

parameters["n_pdf_variations"] = {"2016preVFP": 100, "2016postVFP": 100, "2017": 33, "2018": 33,"2022EE":33}

parameters["dnn_max"] = {"2016preVFP": 1.75, "2016postVFP": 1.75, "2017": 2.0, "2018": 2.35,"2022EE": 2.35}