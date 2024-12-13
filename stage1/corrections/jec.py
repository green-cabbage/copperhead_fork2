from coffea.jetmet_tools import CorrectedJetsFactory, JECStack
from coffea.lookup_tools import extractor
from config.jec_parameters import jec_parameters
import copy
import os
import correctionlib as core
import awkward as ak

def get_corr_inputs(input_dict, corr_obj):
    """
    Helper function for getting values of input variables
    given a dictionary and a correction object.
    """
    input_values = [input_dict[inp.name] for inp in corr_obj.inputs]
    return input_values

def getDataJecTag(jec_pars, dataset):
    """
    helper function that returns the correct JEC tag for the specific run
    """
    jec_data_tag_dict = jec_pars["jec_data_tags"]
    print(f"jec_data_tag_dict: {jec_data_tag_dict}")
    for jec_tag, runs in jec_data_tag_dict.items():
        for run in runs:
            print(f"run: {run}")
            if run in dataset:
                return jec_tag

    # if nothing gets returned, we have an issue
    print("ERROR: No JEC data TAG was GIVEN")
    raise ValueError

def apply_jec(
    df,
    jets,
    dataset,
    is_mc,
    year,
    do_jec,
    do_jecunc,
    do_jerunc,
    jec_pars,
):
    input_dict = {
        # jet transverse momentum
        "JetPt": ak.flatten(jets.pt_raw),
        # jet pseudorapidity
        "JetEta": ak.flatten(jets.eta),
        # jet azimuthal angle
        "JetPhi": ak.flatten(jets.phi),
        # jet area
        "JetA": ak.flatten(jets.area),
        # median energy density (pileup)
        "Rho": ak.flatten(jets.rho),
        # # systematic variation (only for JER SF)
        # "systematic": "nom",
        # # pT of matched gen-level jet (only for JER smearing)
        # "GenPt": 80.0,  # or -1 if no match
        # # unique event ID used for deterministic
        # # pseudorandom number generation (only for JER smearing)
        # "EventID": 12345,
    }
    # print(f"type(jets.pt_raw): {type(jets.pt_raw)}")
    # print(f"input_dict: {input_dict}")
    algo = "AK4PFchs"
    if is_mc:
        # jec_levels = ["L1FastJet", "L2Relative", "L3Absolute"] # hard code for now
        jec_levels = jec_pars["jec_levels_mc"]
        jec =  jec_parameters["jec_tags"][year]
    else: # data
        # jec_levels = ["L1FastJet", "L2Relative", "L3Absolute", "L2L3Residual"]
        jec_levels = jec_pars["jec_levels_data"]
        jec = getDataJecTag(jec_pars, dataset)

    # print(f"jec: {jec}")
    # print(f"jec_levels: {jec_levels}")
    fname = f"/work/users/yun79/dmitry/another_fork/copperhead_fork2/data/POG/JME/{year}_UL/jet_jerc.json.gz" # Hard code for now
    cset = core.CorrectionSet.from_file(os.path.join(fname))
    print("cset done")
    sf_total = ak.ones_like(ak.flatten(jets.eta))
    for lvl in jec_levels:
        key = "{}_{}_{}".format(jec, lvl, algo)
        print("JSON access to keys: '{}'".format(key))
        sf = cset[key]
        inputs = get_corr_inputs(input_dict, sf)
        sf_val = sf.evaluate(*inputs)
        # print(f"{lvl} sf_val: {sf_val}")
        sf_total = sf_total * sf_val
    # print(f"sf_total: {sf_total}")
    # unflatten the correction
    counts = ak.num(jets.eta, axis=1)
    sf_total = ak.unflatten(
        sf_total,
        counts=counts,
    )
    print(f"sf_total unflattened: {sf_total}")

    # now apply the corrections to pt and mass
    print(f"jets.pt b4: {jets.pt}")
    print(f"jets.mass b4: {jets.mass}")
    if do_jec:
        jets["pt"] = jets.pt_raw*sf_total
        jets["mass"] = jets.mass_raw*sf_total
    jets["pt_jec"] = jets["pt"]
    jets["mass_jec"] = jets["mass"]
    print(f"jets.pt after: {jets.pt}")
    print(f"jets.mass after: {jets.mass}")
    if is_mc:
        jec_vars = jec_pars["jec_unc_to_consider"]
        # print(f"jec_vars: {jec_vars}")
        if do_jecunc:
            # print(f"jec_vars: {jec_vars}")
            for unc in jec_vars:
                # key = "{}_{}_{}".format(jec, unc, algo)
                key = "{}_Regrouped_{}_{}".format(jec, unc, algo)
                print("JSON UNC access to key: '{}'".format(key))
                sf = cset[key]
                
                inputs = get_corr_inputs(input_dict, sf)
                jecunc_sf = sf.evaluate(*inputs)
                jecunc_sf = ak.unflatten(
                    jecunc_sf,
                    counts=counts,
                )
                # print(f"{unc} jecunc_sf: {jecunc_sf}")
                direction = "up"
                scale = (1 + jecunc_sf)
                # print(f"{unc} up scale: {scale}")
                # print(f"pt flag: JES_{unc}_{direction}_pt")
                jets[f"JES_{unc}_{direction}_pt"] = jets.pt * scale
                jets[f"JES_{unc}_{direction}_mass"] = jets.mass * scale
                direction = "down"
                scale = (1 - jecunc_sf)
                # print(f"{unc} down scale: {scale}")
                jets[f"JES_{unc}_{direction}_pt"] = jets.pt * scale
                jets[f"JES_{unc}_{direction}_mass"] = jets.mass * scale

        if do_jerunc: 
            #
            # accessing the JER scale factor
            #
            
            jer = jec_pars["jer_tags"]           
            key = "{}_{}_{}".format(jer, "ScaleFactor", algo)
            print(" JER scale factor access to key: '{}'".format(key))
            sf = cset[key]

            # update the input with jec pt plus other inputs for JER smearing
            input_dict["JetPt"] = ak.flatten(jets.pt)
            input_dict["systematic"] = "nom"
            input_dict["GenPt"] = ak.flatten(jets.pt_gen)
            random_array = jets.pt + jets.eta + jets.phi + jets.mass # just add bunch of kinematics. if you truly want a random number, you could use what nick made 
            input_dict["EventID"] = ak.flatten(random_array)

            sf_input_names = [inp.name for inp in sf.inputs]
            print("Inputs: " + ", ".join(sf_input_names))
            inputs = get_corr_inputs(input_dict, sf)
            jersf_nom_value = sf.evaluate(*inputs)
            # print(f"jersf_nom_value: {jersf_nom_value}")
            
            # obtain jersf down
            input_dict["systematic"] = "down"
            inputs = get_corr_inputs(input_dict, sf)
            jersf_down_value = sf.evaluate(*inputs)
            # print(f"jersf_down_value: {jersf_down_value}")

            #
            # accessing the JER (pT resolution)
            #
            # NOTE: "systematic" only impacts jer SF calculation, so we don't need to change input_dict["systematic"]
            key = "{}_{}_{}".format(jer, "PtResolution", algo)
            print("JSON access to key: '{}'".format(key))
            sf = cset[key]
            
            sf_input_names = [inp.name for inp in sf.inputs]
            print("Inputs: " + ", ".join(sf_input_names))
            inputs = get_corr_inputs(input_dict, sf)
            jer_value = sf.evaluate(*inputs)
            # print(f"jer_value: {jer_value}")

            #
            # performing JER smearing
            # (needs JER/JERSF from previous step)
            #

            fname_jersmear = f"/work/users/yun79/dmitry/another_fork/copperhead_fork2/data/POG/JME/jer_smear.json.gz" # Hard code for now
            # print("\nLoading JSON file: {}".format(fname_jersmear))
            cset_jersmear = core.CorrectionSet.from_file(os.path.join(fname_jersmear))
            
            key_jersmear = "JERSmear"
            # print("JSON access to key: '{}'".format(key_jersmear))
            sf_jersmear = cset_jersmear[key_jersmear]
            
            # add previously obtained JER/JERSF values to inputs
            input_dict["JER"] = jer_value
            input_dict["JERSF"] = jersf_nom_value
            
            sf_input_names = [inp.name for inp in sf_jersmear.inputs]
            # print("Inputs: " + ", ".join(sf_input_names))
            
            inputs = get_corr_inputs(input_dict, sf_jersmear)
            jersmear_factor_nom = sf_jersmear.evaluate(*inputs)

            input_dict["JERSF"] = jersf_down_value
            inputs = get_corr_inputs(input_dict, sf_jersmear)
            jersmear_factor_down = sf_jersmear.evaluate(*inputs)
            

            jersmear_factor_nom = ak.unflatten(
                jersmear_factor_nom,
                counts=counts,
            )
            jersmear_factor_down = ak.unflatten(
                jersmear_factor_down,
                counts=counts,
            )
            jersf_nom_value = ak.unflatten(
                jersf_nom_value,
                counts=counts,
            )
            jersf_down_value = ak.unflatten(
                jersf_down_value,
                counts=counts,
            )

            # print(f"jersmear_factor_nom: {jersmear_factor_nom}")
            # print(f"jersmear_factor_down: {jersmear_factor_down}")

            # now apply the up and down jer variations
            unc = "JER" # NOTE: from Dmitry's July2 2020 presentation, on jet pT is changed on JER uncertainty, but JER smear in general also impacts mass
            direction = "up" # JER up variation: JEC and JER-corrected pT
            scale = jersmear_factor_nom
            # print(f"{unc} up scale: {scale}")
            
            jets[f"{unc}_{direction}_pt"] = jets.pt * scale
            # jets[f"{unc}_{direction}_mass"] = jets.mass * scale
            

            # print(f"jersf_down_value/jersf_nom_value: {jersf_down_value/jersf_nom_value}")
            # print(f"jets.pt_jec - jets.pt_gen: {jets.pt_jec - jets.pt_gen}")
            direction = "down" # JER down variation is more direct correection, formula from Dmitry's presentation on Jul 2nd 2020
            down_pt = jets.pt_gen + (jets.pt_jec - jets.pt_gen) * (jersf_down_value/jersf_nom_value)
            # print(f"{unc} down scale: {scale}")
            jets[f"{unc}_{direction}_pt"] = down_pt
            # jets[f"{unc}_{direction}_mass"] = jets.mass * scale
            
            # print(f"jets.pt b4 smearing: {jets.pt}")
            # print(f"jets.pt_jec b4 smearing: {jets.pt_jec}")
            # print(f" {unc}_up_pt after smearing: {jets[f'{unc}_up_pt']}")
            # print(f" {unc}_down_pt after smearing: {jets[f'{unc}_down_pt']}")
            

    return jets

def apply_jec_rereco(
    df,
    jets,
    dataset,
    is_mc,
    year,
    do_jec,
    do_jecunc,
    do_jerunc,
    jec_factories,
    jec_factories_data,
):
    # print(f"jets.pt b4 apply_jec: {jets.pt }")
    # print(f"jets.mass b4 apply_jec: {jets.mass }")
    # print(f"jets.eta b4 apply_jec: {jets.eta }")
    # print(f"jets.phi b4 apply_jec: {jets.phi }")
    # print(f"do_jec: {do_jec}")    
    revert_jet_kinematics = (not do_jec) and (do_jecunc or do_jerunc)
    print(f"revert_jet_kinematics: {revert_jet_kinematics}")    
    if revert_jet_kinematics: # save current jet pt and masses to overwite the jec that happens regardless of do_jec ==True when either  do_jecunc and do_jerunc True
        pt_orig = copy.deepcopy(jets.pt) # NOTE: if jets.pt_orig and jets.mass_orig get overwritten duirng junc/jer factory build() method, so save on a seperate variable
        mass_orig = copy.deepcopy(jets.mass)
        
    
    cache = df.caches[0]

    # Correct jets (w/o uncertainties)
    if do_jec:
        if is_mc:
            factory = jec_factories["jec"]
        else:
            for run in jec_parameters["runs"][year]:
                if run in dataset:
                    factory = jec_factories_data[run]
        jets = factory.build(jets, lazy_cache=cache)

    # TODO: only consider nuisances that are defined in run parameters
    # Compute JEC uncertainties
    if is_mc and do_jecunc:
        jets = jec_factories["junc"].build(jets, lazy_cache=cache)

    # print(f"jets.pt after junc: {jets.pt }")
    # print(f"jets.mass after junc: {jets.mass }")
    # print(f"jets.eta after junc: {jets.eta }")
    # print(f"jets.phi after junc: {jets.phi }")

    
    # Compute JER uncertainties
    if is_mc and do_jerunc:
        jets = jec_factories["jer"].build(jets, lazy_cache=cache)

    # reverting the jec that happens regardless of do_jec ==True when do_jecunc is True
    if revert_jet_kinematics: 
        jets["pt"] = pt_orig
        jets["mass"] = mass_orig
    
    # TODO: JER nuisances

    # print(f"jets.pt after apply_jec: {jets.pt }")
    # print(f"jets.mass after apply_jec: {jets.mass }")
    # print(f"jets.eta after apply_jec: {jets.eta }")
    # print(f"jets.phi after apply_jec: {jets.phi }")
    return jets


def jec_names_and_sources(jec_pars, year):
    names = {}
    suffix = {
        "jec_names": [f"_{level}_AK4PFchs" for level in jec_pars["jec_levels_mc"]],
        "jec_names_data": [
            f"_{level}_AK4PFchs" for level in jec_pars["jec_levels_data"]
        ],
        "junc_names": ["_Uncertainty_AK4PFchs"],
        "junc_names_data": ["_Uncertainty_AK4PFchs"],
        "junc_sources": ["_UncertaintySources_AK4PFchs"],
        "junc_sources_data": ["_UncertaintySources_AK4PFchs"],
        "jer_names": ["_PtResolution_AK4PFchs"],
        "jersf_names": ["_SF_AK4PFchs"],
    }

    for key, suff in suffix.items():
        if "data" in key:
            names[key] = {}
            for run in jec_pars["runs"]:
                for tag, iruns in jec_pars["jec_data_tags"].items():
                    if run in iruns:
                        names[key].update({run: [f"{tag}{s}" for s in suff]})
        else:
            tag = jec_pars["jer_tags"] if "jer" in key else jec_pars["jec_tags"]
            names[key] = [f"{tag}{s}" for s in suff]

    return names


def jec_weight_sets(jec_pars, year):
    weight_sets = {}
    names = jec_names_and_sources(jec_pars, year)

    extensions = {
        "jec_names": "jec",
        "jer_names": "jr",
        "jersf_names": "jersf",
        "junc_names": "junc",
        "junc_sources": "junc",
    }

    weight_sets["jec_weight_sets"] = []
    weight_sets["jec_weight_sets_data"] = []

    for opt, ext in extensions.items():
        # MC
        weight_sets["jec_weight_sets"].extend(
            [f"* * data/jec/{name}.{ext}.txt" for name in names[opt]]
        )
        # Data
        if "jer" in opt:
            continue
        data = []
        for run, items in names[f"{opt}_data"].items():
            data.extend(items)
        data = list(set(data))
        weight_sets["jec_weight_sets_data"].extend(
            [f"* * data/jec/{name}.{ext}.txt" for name in data]
        )

    return weight_sets


def get_name_map(stack):
    name_map = stack.blank_name_map
    name_map["JetPt"] = "pt"
    name_map["JetMass"] = "mass"
    name_map["JetEta"] = "eta"
    name_map["JetA"] = "area"
    name_map["ptGenJet"] = "pt_gen"
    name_map["ptRaw"] = "pt_raw"
    name_map["massRaw"] = "mass_raw"
    name_map["Rho"] = "rho"
    return name_map


def jec_factories(year):
    jec_pars = {k: v[year] for k, v in jec_parameters.items()}

    # print(f"jec_pars: {jec_pars}")

    weight_sets = jec_weight_sets(jec_pars, year)
    names = jec_names_and_sources(jec_pars, year)
    # print(f"weight_sets: {weight_sets}")
    # print(f"names: {names}")

    jec_factories = {}
    jec_factories_data = {}

    # Prepare evaluators for JEC, JER and their systematics
    jetext = extractor()
    jetext.add_weight_sets(weight_sets["jec_weight_sets"])
    jetext.add_weight_sets(weight_sets["jec_weight_sets_data"])
    jetext.finalize()
    jet_evaluator = jetext.make_evaluator()

    stacks_def = {
        "jec_stack": ["jec_names"],
        "jer_stack": ["jer_names", "jersf_names"],
        "junc_stack": ["junc_names"],
    }

    stacks = {}
    for key, vals in stacks_def.items():
        stacks[key] = []
        for v in vals:
            stacks[key].extend(names[v])

    jec_input_options = {}
    for opt in ["jec", "junc", "jer"]:
        jec_input_options[opt] = {
            name: jet_evaluator[name] for name in stacks[f"{opt}_stack"]
        }

    for src in names["junc_sources"]:
        for key in jet_evaluator.keys():
            if src in key:
                jec_input_options["junc"][key] = jet_evaluator[key]

    # Create separate factories for JEC, JER, JEC variations
    for opt in ["jec", "junc", "jer"]:
        stack = JECStack(jec_input_options[opt])
        jec_factories[opt] = CorrectedJetsFactory(get_name_map(stack), stack)

    # Create a separate factory for each data run
    for run in jec_pars["runs"]:
        jec_inputs_data = {}
        for opt in ["jec", "junc"]:
            jec_inputs_data.update(
                {name: jet_evaluator[name] for name in names[f"{opt}_names_data"][run]}
            )
        for src in names["junc_sources_data"][run]:
            for key in jet_evaluator.keys():
                if src in key:
                    jec_inputs_data[key] = jet_evaluator[key]

        jec_stack_data = JECStack(jec_inputs_data)
        jec_factories_data[run] = CorrectedJetsFactory(
            get_name_map(jec_stack_data), jec_stack_data
        )

    return jec_factories, jec_factories_data


#        if is_mc and self.do_jerunc:
#            jetarrays = {c: df.Jet[c].flatten() for c in
#                         df.Jet.columns if 'matched' not in c}
#            pt_gen_jet = df.Jet['matched_genjet'].pt.flatten(axis=0)
#            # pt_gen_jet = df.Jet.matched_genjet.pt.flatten(axis=0)
#            pt_gen_jet = np.zeros(len(df.Jet.flatten()))
#            pt_gen_jet[df.Jet.matched_genjet.pt.flatten(axis=0).counts >
#                       0] = df.Jet.matched_genjet.pt.flatten().flatten()
#            pt_gen_jet[df.Jet.matched_genjet.pt.flatten(
#                axis=0).counts <= 0] = 0
#            jetarrays['ptGenJet'] = pt_gen_jet
#            jets = JaggedCandidateArray.candidatesfromcounts(
#                df.Jet.counts, **jetarrays)
#            jet_pt_jec = df.Jet.pt
#            self.Jet_transformer_JER.transform(
#                jets, forceStochastic=False)
#            jet_pt_jec_jer = jets.pt
#            jet_pt_gen = jets.ptGenJet
#            jer_sf = ((jet_pt_jec_jer - jet_pt_gen) /
#                      (jet_pt_jec - jet_pt_gen +
#                       (jet_pt_jec == jet_pt_gen) *
#                       (jet_pt_jec_jer - jet_pt_jec)))
#            jer_down_sf = ((jets.pt_jer_down - jet_pt_gen) /
#                           (jet_pt_jec - jet_pt_gen +
#                           (jet_pt_jec == jet_pt_gen) * 10.))
#            jet_pt_jer_down = jet_pt_gen +\
#                (jet_pt_jec - jet_pt_gen) *\
#                (jer_down_sf / jer_sf)
#            jer_categories = {
#                'jer1': (abs(jets.eta) < 1.93),
#                'jer2': (abs(jets.eta) > 1.93) & (abs(jets.eta) < 2.5),
#                'jer3': ((abs(jets.eta) > 2.5) &
#                         (abs(jets.eta) < 3.139) &
#                         (jets.pt < 50)),
#                'jer4': ((abs(jets.eta) > 2.5) &
#                         (abs(jets.eta) < 3.139) &
#                         (jets.pt > 50)),
#                'jer5': (abs(jets.eta) > 3.139) & (jets.pt < 50),
#                'jer6': (abs(jets.eta) > 3.139) & (jets.pt > 50),
#            }
#            for jer_unc_name, jer_cut in jer_categories.items():
#                jer_cut = jer_cut & (jets.ptGenJet > 0)
#                up_ = (f"{jer_unc_name}_up" not in self.pt_variations)
#                dn_ = (f"{jer_unc_name}_down" not in
#                       self.pt_variations)
#                if up_ and dn_:
#                    continue
#                pt_name_up = f"pt_{jer_unc_name}_up"
#                pt_name_down = f"pt_{jer_unc_name}_down"
#                df.Jet[pt_name_up] = jet_pt_jec
#                df.Jet[pt_name_down] = jet_pt_jec
#                df.Jet[pt_name_up][jer_cut] = jet_pt_jec_jer[jer_cut]
#                df.Jet[pt_name_down][jer_cut] =\
#                    jet_pt_jer_down[jer_cut]
#
#                if (f"{jer_unc_name}_up" in self.pt_variations):
#                    jet_variation_names += [f"{jer_unc_name}_up"]
#                if (f"{jer_unc_name}_down" in self.pt_variations):
#                    jet_variation_names += [f"{jer_unc_name}_down"]
#            if self.timer:
#                self.timer.add_checkpoint("Computed JER nuisances")
