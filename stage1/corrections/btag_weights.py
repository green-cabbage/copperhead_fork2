import pandas as pd
import correctionlib
from python.misc import onedimeval
from functools import partial

def btag_weights_json(processor, systs, jets, weights, bjet_sel_mask, btag_file):

    btag = pd.DataFrame(index=bjet_sel_mask.index)
    #print(f"len btag1 {len(btag)}")
    #print(f"len jets1 {len(jets)}")
    jets = jets[abs(jets.eta) < 2.4]
    jets["btag_wgt"] = 1.0
    jets.loc[jets.pt > 1000.0, "pt"] = 1000.0
    
    
    btag_json=[btag_file["deepJet_shape"]]
    jets.loc[abs(jets["eta"]) < 2.4, "btag_wgt"] = onedimeval(partial(btag_json[0].evaluate,
        "central"),
        jets.hadronFlavour.values,
        abs(jets.eta.values),
        jets.pt.values,
        jets.btagDeepFlavB.values,
    )
    btag["wgt"] = jets["btag_wgt"].prod(level=0)
    btag["wgt"] = btag["wgt"].fillna(1.0)
    btag.loc[btag.wgt < 0.01, "wgt"] = 1.0
    print(f"jets.btag_wgt: {jets.btag_wgt}")
    #print(f"len btag2 {len(btag)}")
    #print(f"len jets2 {len(jets)}")
    flavors = {
        0: ["jes", "lf", "lfstats1", "lfstats2"],
        1: ["jes", "lf", "lfstats1", "lfstats2"],
        2: ["jes", "lf", "lfstats1", "lfstats2"],
        3: ["jes", "lf", "lfstats1", "lfstats2"],
        4: ["cferr1", "cferr2"],
        5: ["jes", "hf", "hfstats1", "hfstats2"],
        21: ["jes", "lf", "lfstats1", "lfstats2"],
    }
    btag_syst = {}
    for sys in systs:

        jets[f"btag_{sys}_up"] = 1.0
        jets[f"btag_{sys}_down"] = 1.0
        btag[f"{sys}_up"] = 1.0
        btag[f"{sys}_down"] = 1.0

        for f, f_syst in flavors.items():
            if sys in f_syst:
                btag_mask = (abs(jets.hadronFlavour)) == f #& (abs(jets.eta) < 2.4))
                jets.loc[btag_mask, f"btag_{sys}_up"] = onedimeval(partial(btag_json[0].evaluate,
                    f"up_{sys}"),
                    jets.hadronFlavour[btag_mask].values,
                    abs(jets.eta)[btag_mask].values,
                    jets.pt[btag_mask].values,
                    jets.btagDeepFlavB[btag_mask].values,
                    
                )
                jets.loc[btag_mask, f"btag_{sys}_down"] = onedimeval(partial(btag_json[0].evaluate,
                    f"down_{sys}"),
                    jets.hadronFlavour[btag_mask].values,
                    abs(jets.eta)[btag_mask].values,
                    jets.pt[btag_mask].values,
                    jets.btagDeepFlavB[btag_mask].values,
                    
                )

        btag[f"{sys}_up"] = jets[f"btag_{sys}_up"].prod(level=0)
        btag[f"{sys}_down"] = jets[f"btag_{sys}_down"].prod(level=0)
        btag[f"{sys}_up"] =btag[f"{sys}_up"].fillna(1.0)
        btag[f"{sys}_down"] =btag[f"{sys}_down"].fillna(1.0)
        
        btag_syst[sys] = {"up": btag[f"{sys}_up"], "down": btag[f"{sys}_down"]}

    sum_before = weights.df["nominal"][bjet_sel_mask].sum()
    sum_after = (
        weights.df["nominal"][bjet_sel_mask]
        .multiply(btag.wgt[bjet_sel_mask], axis=0)
        .sum()
    )
    # print(f"btag_wgt b4 normalization: {btag.wgt.to_numpy()[bjet_sel_mask]}")
    btag.wgt = btag.wgt * sum_before / sum_after
    #print(f"len btag.wgt {len(btag.wgt)}")
    #print(f"len jets3 {len(jets)}")
    return btag.wgt, btag_syst

def btag_weights_csv(processor, lookup, systs, jets, weights, bjet_sel_mask):

    btag = pd.DataFrame(index=bjet_sel_mask.index)
    jets = jets[abs(jets.eta) < 2.4]
    jets.loc[jets.pt > 1000.0, "pt"] = 1000.0

    jets["btag_wgt"] = lookup.eval(
        "central",
        jets.hadronFlavour.values,
        abs(jets.eta.values),
        jets.pt.values,
        jets.btagDeepFlavB.values,
        True,
    )
    btag["wgt"] = jets["btag_wgt"].prod(level=0)
    btag["wgt"] = btag["wgt"].fillna(1.0)
    btag.loc[btag.wgt < 0.01, "wgt"] = 1.0

    flavors = {
        0: ["jes", "hf", "lfstats1", "lfstats2"],
        1: ["jes", "hf", "lfstats1", "lfstats2"],
        2: ["jes", "hf", "lfstats1", "lfstats2"],
        3: ["jes", "hf", "lfstats1", "lfstats2"],
        4: ["cferr1", "cferr2"],
        5: ["jes", "lf", "hfstats1", "hfstats2"],
        21: ["jes", "hf", "lfstats1", "lfstats2"],
    }

    btag_syst = {}
    for sys in systs:
        jets[f"btag_{sys}_up"] = 1.0
        jets[f"btag_{sys}_down"] = 1.0
        btag[f"{sys}_up"] = 1.0
        btag[f"{sys}_down"] = 1.0

        for f, f_syst in flavors.items():
            if sys in f_syst:
                btag_mask = abs(jets.hadronFlavour) == f
                jets.loc[btag_mask, f"btag_{sys}_up"] = lookup.eval(
                    f"up_{sys}",
                    jets.hadronFlavour[btag_mask].values,
                    abs(jets.eta)[btag_mask].values,
                    jets.pt[btag_mask].values,
                    jets.btagDeepFlavB[btag_mask].values,
                    True,
                )
                jets.loc[btag_mask, f"btag_{sys}_down"] = lookup.eval(
                    f"down_{sys}",
                    jets.hadronFlavour[btag_mask].values,
                    abs(jets.eta)[btag_mask].values,
                    jets.pt[btag_mask].values,
                    jets.btagDeepFlavB[btag_mask].values,
                    True,
                )

        btag[f"{sys}_up"] = jets[f"btag_{sys}_up"].prod(level=0)
        btag[f"{sys}_down"] = jets[f"btag_{sys}_down"].prod(level=0)
        btag_syst[sys] = {"up": btag[f"{sys}_up"], "down": btag[f"{sys}_down"]}

    sum_before = weights.df["nominal"][bjet_sel_mask].sum()
    sum_after = (
        weights.df["nominal"][bjet_sel_mask]
        .multiply(btag.wgt[bjet_sel_mask], axis=0)
        .sum()
    )
    btag.wgt = btag.wgt * sum_before / sum_after

    return btag.wgt, btag_syst