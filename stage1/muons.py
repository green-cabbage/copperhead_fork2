import numpy as np

from python.math_tools import p4_sum, delta_r, cs_variables, cs_variables_pisa


def fill_muons(processor, output, mu1, mu2, is_mc, is_v9):
    if is_v9:
        mu1_variable_names = ["mu1_pt", "mu1_pt_over_mass", "mu1_eta", "mu1_phi", "mu1_iso","mu1_pt_raw"]
        mu2_variable_names = ["mu2_pt", "mu2_pt_over_mass", "mu2_eta", "mu2_phi", "mu2_iso","mu2_pt_raw"]
    else:
        mu1_variable_names = ["mu1_pt", "mu1_pt_over_mass", "mu1_eta", "mu1_phi", "mu1_iso","mu1_bsConstrainedPt","mu1_bsConstrainedPtErr","mu1_bsConstrainedChi2","mu1_pt_raw"]
        mu2_variable_names = ["mu2_pt", "mu2_pt_over_mass", "mu2_eta", "mu2_phi", "mu2_iso","mu2_bsConstrainedPt","mu2_bsConstrainedPtErr","mu2_bsConstrainedChi2","mu2_pt_raw"]
    dimuon_variable_names = [
        "dimuon_mass",
        "dimuon_ebe_mass_res",
        "dimuon_ebe_mass_res_raw",
        "dimuon_ebe_mass_res_rel",
        "dimuon_pt",
        "dimuon_pt_log",
        "dimuon_eta",
        "dimuon_phi",
        "dimuon_dEta",
        "dimuon_dPhi",
        "dimuon_dR",
        "dimuon_rap",
        "dimuon_cos_theta_cs",
        "dimuon_phi_cs",
    ]
    v_names = mu1_variable_names + mu2_variable_names + dimuon_variable_names

    # Initialize columns for muon variables
    for n in v_names:
        output[n] = 0.0

    # Fill single muon variables
    if is_v9:
        for v in ["pt", "ptErr", "eta", "phi","pt_raw"]:
            output[f"mu1_{v}"] = mu1[v]
            output[f"mu2_{v}"] = mu2[v]
    else:
        for v in ["pt", "ptErr", "eta", "phi","bsConstrainedPt","bsConstrainedPtErr","bsConstrainedChi2","pt_raw"]:
            output[f"mu1_{v}"] = mu1[v]
            output[f"mu2_{v}"] = mu2[v]
    output["mu1_iso"] = mu1.pfRelIso04_all
    output["mu2_iso"] = mu2.pfRelIso04_all


    # Fill dimuon variables
    mm = p4_sum(mu1, mu2)
    for v in ["pt", "eta", "phi", "mass", "rap"]:
        name = f"dimuon_{v}"
        output[name] = mm[v]
        output[name] = output[name].fillna(-999.0)
        
    output["mu1_pt_over_mass"] = output.mu1_pt / output.dimuon_mass
    output["mu2_pt_over_mass"] = output.mu2_pt / output.dimuon_mass

    output["dimuon_pt_log"] = np.log(output.dimuon_pt)

    mm_deta, mm_dphi, mm_dr = delta_r(mu1.eta, mu2.eta, mu1.phi, mu2.phi)

    output["dimuon_dEta"] = mm_deta
    output["dimuon_dPhi"] = mm_dphi
    output["dimuon_dR"] = mm_dr

    output["dimuon_ebe_mass_res"] = mass_resolution(
        is_mc, processor.evaluator, output, processor.year
    )
    output["dimuon_ebe_mass_res_raw"] = mass_resolution_raw(output)
    
    output["dimuon_ebe_mass_res_rel"] = output.dimuon_ebe_mass_res / output.dimuon_mass

    output["dimuon_pisa_mass_res_rel"] = mass_resolution_pisa(
        processor.evaluator, output
    )
    output["dimuon_pisa_mass_res"] = (
        output.dimuon_pisa_mass_res_rel * output.dimuon_mass
    )

    output["dimuon_cos_theta_cs"], output["dimuon_phi_cs"] = cs_variables(mu1, mu2)
    (
        output["dimuon_cos_theta_cs_pisa"],
        output["dimuon_phi_cs_pisa"],
    ) = cs_variables_pisa(mu1, mu2)



def fill_muons_Simple(processor, output, mu1, mu2, is_mc):
    mu1_variable_names = ["mu1_pt", "mu1_pt_over_mass", "mu1_eta", "mu1_phi", "mu1_iso"]
    mu2_variable_names = ["mu2_pt", "mu2_pt_over_mass", "mu2_eta", "mu2_phi", "mu2_iso"]
    dimuon_variable_names = [
        "dimuon_mass",
        "dimuon_pt",
        "dimuon_pt_log",
        "dimuon_eta",
        "dimuon_phi",
        "dimuon_dEta",
        "dimuon_dPhi",
        "dimuon_dR",
        "dimuon_rap",
    ]
    v_names = mu1_variable_names + mu2_variable_names + dimuon_variable_names

    # Initialize columns for muon variables
    for n in v_names:
        output[n] = 0.0

    # Fill single muon variables
    for v in ["pt", "eta", "phi"]:
        output[f"mu1_{v}"] = mu1[v]
        output[f"mu2_{v}"] = mu2[v]

    #output["mu1_iso"] = mu1.pfRelIso04_all
    #output["mu2_iso"] = mu2.pfRelIso04_all
    #output["mu1_pt_over_mass"] = output.mu1_pt / output.dimuon_mass
    #output["mu2_pt_over_mass"] = output.mu2_pt / output.dimuon_mass

    # Fill dimuon variables
    mm = p4_sum(mu1, mu2)
    for v in ["pt", "eta", "phi", "mass"]:
        name = f"dimuon_{v}"
        output[name] = mm[v]
        output[name] = output[name].fillna(-999.0)

    output["dimuon_pt_log"] = np.log(output.dimuon_pt)

    mm_deta, mm_dphi, mm_dr = delta_r(mu1.eta, mu2.eta, mu1.phi, mu2.phi)

    output["dimuon_dEta"] = mm_deta
    output["dimuon_dPhi"] = mm_dphi
    output["dimuon_dR"] = mm_dr


def mass_resolution_raw(df):
    # Returns absolute non-calibrated mass resolution!
    dpt1 = (df.mu1_ptErr * df.dimuon_mass) / (2 * df.mu1_pt)
    dpt2 = (df.mu2_ptErr * df.dimuon_mass) / (2 * df.mu2_pt)

    return np.sqrt(dpt1 * dpt1 + dpt2 * dpt2)


def mass_resolution(is_mc, evaluator, df, year):
    # Returns absolute mass resolution!
    dpt1 = (df.mu1_ptErr * df.dimuon_mass) / (2 * df.mu1_pt)
    dpt2 = (df.mu2_ptErr * df.dimuon_mass) / (2 * df.mu2_pt)
    if "2016" in year:
        yearstr = "2016"
    elif "2022" in year:
        yearstr = "2018"
    else:
        yearstr=year #Work around before there are seperate new files for pre and postVFP
    if is_mc:
        label = f"res_calib_MC_{yearstr}"
    else:
        label = f"res_calib_Data_{yearstr}"
    calibration = np.array(
        evaluator[label](
            df.mu1_pt.values, abs(df.mu1_eta.values), abs(df.mu2_eta.values)
        )
    )

    return np.sqrt(dpt1 * dpt1 + dpt2 * dpt2) * calibration


def mass_resolution_pisa(evaluator, df):
    # Returns relative mass resolution!
    mu1_ptErr = evaluator["PtErrParametrization"](
        np.log(df.mu1_pt).values, np.abs(df.mu1_eta).values
    )
    mu2_ptErr = evaluator["PtErrParametrization"](
        np.log(df.mu2_pt).values, np.abs(df.mu2_eta).values
    )
    return np.sqrt(0.5 * (mu1_ptErr * mu1_ptErr + mu2_ptErr * mu2_ptErr))
