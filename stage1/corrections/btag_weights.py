import pandas as pd
import numpy as np

def btag_weights_old(lookup, systs, jets, weights, bjet_sel_mask, numevents):
    btag_wgt = np.ones(numevents, dtype=float)
    jets_ = jets[abs(jets.eta)<2.4]
    jet_pt_ = awkward.JaggedArray.fromcounts(jets_[jets_.counts>0].counts, np.minimum(jets_.pt.flatten(), 1000.))

    btag_wgt[(jets_.counts>0)] = lookup('central', jets_[jets_.counts>0].hadronFlavour,\
                                              abs(jets_[jets_.counts>0].eta), jet_pt_,\
                                              jets_[jets_.counts>0].btagDeepB, True).prod()
    btag_wgt[btag_wgt<0.01] = 1.

    btag_syst = {}
    flavors = {
        0: ["jes","hf","lfstats1","lfstats2"],
        1: ["jes","hf","lfstats1","lfstats2"],
        2: ["jes","hf","lfstats1","lfstats2"],
        3: ["jes","hf","lfstats1","lfstats2"],
        4: ["cferr1","cferr2"],
        5: ["jes","lf","hfstats1","hfstats2"],
        21: ["jes","hf","lfstats1","lfstats2"],
    }
    
    for sys in systs:
        njets = len(jets_.flatten())
        btag_syst[sys] = [np.ones(njets, dtype=float),np.ones(njets, dtype=float)]
        for f, f_syst in flavors.items():
            if sys in f_syst:
                btag_mask = abs(jets_.hadronFlavour)==f
                btag_syst[sys][0][btag_mask.flatten()] = lookup('up_'+sys, jets_.hadronFlavour[btag_mask],\
                                                      abs(jets_.eta)[btag_mask], jets_.pt[btag_mask],\
                                                      jets_.btagDeepB[btag_mask], True).flatten()
                btag_syst[sys][1][btag_mask.flatten()] = lookup('down_'+sys, jets_.hadronFlavour[btag_mask],\
                                                      abs(jets_.eta)[btag_mask], jets_.pt[btag_mask],\
                                                      jets_.btagDeepB[btag_mask], True).flatten()
        btag_syst[sys][0] = awkward.JaggedArray.fromcounts(jets_.counts, btag_syst[sys][0]).prod()
        btag_syst[sys][1] = awkward.JaggedArray.fromcounts(jets_.counts, btag_syst[sys][1]).prod()
    
    sum_before = weights.df['nominal'][bjet_sel_mask].sum()
    sum_after = weights.df['nominal'][bjet_sel_mask].multiply(btag_wgt[bjet_sel_mask], axis=0).sum()
    btag_wgt = btag_wgt*sum_before/sum_after
    return btag_wgt, btag_syst


def btag_weights(processor, lookup, systs, jets, weights, bjet_sel_mask):

    btag = pd.DataFrame(index=bjet_sel_mask.index)
    # print(f"jet b4 abs eta cut: {jets.head(20).eta}")
    jets = jets[abs(jets.eta) < 2.4]
    # print(f"jet after abs eta cut: {jets.head(20).eta}")
    jets.loc[jets.pt > 1000.0, "pt"] = 1000.0
    # print(f"btag_weights jets: {jets}")
    jets["btag_wgt"] = lookup.eval(
        "central",
        jets.hadronFlavour.values,
        abs(jets.eta.values),
        jets.pt.values,
        jets.btagDeepB.values,
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
                    jets.btagDeepB[btag_mask].values,
                    True,
                )
                jets.loc[btag_mask, f"btag_{sys}_down"] = lookup.eval(
                    f"down_{sys}",
                    jets.hadronFlavour[btag_mask].values,
                    abs(jets.eta)[btag_mask].values,
                    jets.pt[btag_mask].values,
                    jets.btagDeepB[btag_mask].values,
                    True,
                )
        if "cferr1" == sys:
            cferr1_up = jets[f"btag_{sys}_up"]
            # print(f"cferr1_up: {cferr1_up}")
            # print(f"cferr1_up prod: {cferr1_up.prod(level=0)}")
            is_nan = np.isnan(cferr1_up.prod(level=0))
            print(f"cferr1_up is_nan: {np.any(is_nan)}")
            # print(f"cferr1_up : {cferr1_up.prod(level=0)}")
            print(f"cferr1_up : {cferr1_up}")

        # btag_sys = btag[f"{sys}_up"]
        # print(f'btag_sys b4 {btag_sys}')
        btag[f"{sys}_up"] = jets[f"btag_{sys}_up"].prod(level=0)
        btag[f"{sys}_down"] = jets[f"btag_{sys}_down"].prod(level=0)
        
        # btag_sys = btag[f"{sys}_up"]
        # print(f'btag_sys after {btag_sys}')
        
        # fill nan for indices that don't exist in the prod()
        btag[f"{sys}_up"].fillna(1.0, inplace=True)
        btag[f"{sys}_down"].fillna(1.0, inplace=True)
        btag_syst[sys] = {"up": btag[f"{sys}_up"], "down": btag[f"{sys}_down"]}

    sum_before = weights.df["nominal"][bjet_sel_mask].sum()
    sum_after = (
        weights.df["nominal"][bjet_sel_mask]
        .multiply(btag.wgt[bjet_sel_mask], axis=0)
        .sum()
    )
    btag.wgt = btag.wgt * sum_before / sum_after
    # print(f"btag.wgt: {btag.wgt}")
    return btag.wgt, btag_syst
