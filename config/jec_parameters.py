def for_all_years(value):
    out = {k: value for k in ["2016preVFP","2016postVFP", "2017", "2018","2022EE",]}
    return out


def get_variations(sources):
    result = []
    for v in sources:
        result.append(v + "_up")
        result.append(v + "_down")
    return result


jec_parameters = {}

# jec_unc_to_consider = {
#     "2016preVFP": [
#         "Absolute",
#         "Absolute2016",
#         "BBEC1",
#         "BBEC12016",
#         "EC2",
#         "EC22016",
#         "HF",
#         "HF2016",
#         "RelativeBal",
#         "RelativeSample2016",
#         "FlavorQCD",
#     ],
#     "2016postVFP": [
#         "Absolute",
#         "Absolute2016",
#         "BBEC1",
#         "BBEC12016",
#         "EC2",
#         "EC22016",
#         "HF",
#         "HF2016",
#         "RelativeBal",
#         "RelativeSample2016",
#         "FlavorQCD",
#     ],
#     "2017": [
#         "Absolute",
#         "Absolute2017",
#         "BBEC1",
#         "BBEC12017",
#         "EC2",
#         "EC22017",
#         "HF",
#         "HF2017",
#         "RelativeBal",
#         "RelativeSample2017",
#         "FlavorQCD",
#     ],
#     "2018": [
#         "Absolute",
#         "Absolute2018",
#         "BBEC1",
#         "BBEC12018",
#         "EC2",
#         "EC22018",
#         "HF",
#         "HF2018",
#         "RelativeBal",
#         "RelativeSample2018",
#         "FlavorQCD",
#     ],
#     "2022EE": [
#         "Absolute",
#         "Absolute2018",
#         "BBEC1",
#         "BBEC12018",
#         "EC2",
#         "EC22018",
#         "HF",
#         "HF2018",
#         "RelativeBal",
#         "RelativeSample2018",
#         "FlavorQCD",
#     ],
# }

# updating to match the terms in https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/master/examples/jercExample.py?ref_type=heads#L233-252
jec_unc_to_consider = {
    "2016preVFP": [
        "Absolute",
        "Absolute_2016",
        "BBEC1",
        "BBEC1_2016",
        "EC2",
        "EC2_2016",
        "HF",
        "HF_2016",
        "RelativeBal",
        "RelativeSample_2016",
        "FlavorQCD",
    ],
    "2016postVFP": [
        "Absolute",
        "Absolute_2016",
        "BBEC1",
        "BBEC1_2016",
        "EC2",
        "EC2_2016",
        "HF",
        "HF_2016",
        "RelativeBal",
        "RelativeSample_2016",
        "FlavorQCD",
    ],
    "2017": [
        "Absolute",
        "Absolute_2017",
        "BBEC1",
        "BBEC1_2017",
        "EC2",
        "EC2_2017",
        "HF",
        "HF_2017",
        "RelativeBal",
        "RelativeSample_2017",
        "FlavorQCD",
    ],
    "2018": [
        "Absolute",
        "Absolute_2018",
        "BBEC1",
        "BBEC1_2018",
        "EC2",
        "EC2_2018",
        "HF",
        "HF_2018",
        "RelativeBal",
        "RelativeSample_2018",
        "FlavorQCD",
    ],
}

jec_parameters["jec_variations"] = {
    year: get_variations(jec_unc_to_consider[year]) for year in ["2016preVFP","2016postVFP", "2017", "2018",]
}

jec_parameters["jec_unc_to_consider"] = {
    year: jec_unc_to_consider[year] for year in ["2016preVFP","2016postVFP", "2017", "2018",]
}

jec_parameters["runs"] = {
    "2016preVFP": ["B", "C", "D", "E", "F"],
    "2016postVFP": ["F", "G", "H"],
    "2017": ["B", "C", "D", "E", "F"],
    "2018": ["A", "B", "C", "D"],
}

# jec_parameters["jec_levels_mc"] = for_all_years(
#     ["L1FastJet", "L2Relative", "L3Absolute"]
# )
# jec_parameters["jec_levels_data"] = for_all_years(
#     ["L1FastJet", "L2Relative", "L3Absolute", "L2L3Residual"]
# )

jec_parameters["jec_levels_mc"] = for_all_years(
    ["L2Relative", "L3Absolute"]
)
jec_parameters["jec_levels_data"] = for_all_years(
    ["L2Relative", "L3Absolute", "L2L3Residual"]
)


jec_parameters["jec_tags"] = {
    "2016preVFP": "Summer19UL16APV_V7_MC",
    "2016postVFP": "Summer19UL16_V7_MC",
    "2017": "Summer19UL17_V5_MC",
    "2018": "Summer19UL18_V5_MC",
}

jec_parameters["jer_tags"] = {
    "2016preVFP": "Summer20UL16APV_JRV3_MC",
    "2016postVFP": "Summer20UL16_JRV3_MC",
    "2017": "Summer19UL17_JRV2_MC",
    "2018": "Summer19UL18_JRV2_MC",
}

jec_parameters["jec_data_tags"] = {
    "2016preVFP": {
        "Summer19UL16APV_RunBCD_V7_DATA": ["B", "C", "D"],
        "Summer19UL16APV_RunEF_V7_DATA": ["E", "F"],
    },
    "2016postVFP": {
        "Summer19UL16_RunFGH_V7_DATA": ["F","G","H"],
    },
    "2017": {
        "Summer19UL17_RunB_V5_DATA": ["B"],
        "Summer19UL17_RunC_V5_DATA": ["C"],
        "Summer19UL17_RunD_V5_DATA": ["D"],
        "Summer19UL17_RunE_V5_DATA": ["E"],
        "Summer19UL17_RunF_V5_DATA": ["F"],
    },
    "2018": {
        "Summer19UL18_RunA_V5_DATA": ["A"],
        "Summer19UL18_RunB_V5_DATA": ["B"],
        "Summer19UL18_RunC_V5_DATA": ["C"],
        "Summer19UL18_RunD_V5_DATA": ["D"],
    },
}

jer_variations = ["jer1", "jer2", "jer3", "jer4", "jer5", "jer6"]
jec_parameters["jer_variations"] = {
    year: get_variations(jer_variations) for year in ["2016preVFP","2016postVFP", "2017", "2018",]
}