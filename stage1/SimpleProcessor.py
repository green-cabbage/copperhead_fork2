import awkward as ak
import numpy as np

# np.set_printoptions(threshold=sys.maxsize)
import pandas as pd

import coffea.processor as processor
from coffea.lookup_tools import extractor



from python.math_tools import p4_sum
from stage1.weights import Weights

# from stage1.corrections.puid_weights import puid_weights

from stage1.muons import fill_muons_Simple


from config.parameters import parameters

from config.variables_few import variables
from config.branches import branches




class SimpleDimuonProcessor(processor.ProcessorABC):
    def __init__(self, **kwargs):
       
        self.apply_to_output = kwargs.get("apply_to_output", None)

        # try to load metadata
        self.samp_info = kwargs.get("samp_info", None)
        if self.samp_info is None:
            print("Samples info missing!")
            return
        self.year = self.samp_info.year
        self.lumi_weights = self.samp_info.lumi_weights
        self.lumi = self.samp_info.lumi

        # load parameters (cuts, paths to external files, etc.)
        self.parameters = {k: v[self.year] for k, v in parameters.items()}




        self.regions = kwargs.get("regions", ["h-peak", "h-sidebands","z-peak"])
        # variables to save
        self.vars_to_save = set([v.name for v in variables])


    def process(self, df):

        # Dataset name (see definitions in config/datasets.py)
        dataset = df.metadata["dataset"]
        is_mc = "data" not in dataset
        numevents = len(df)
        
        
        # ------------------------------------------------------------#
        # Apply LHE cuts for DY sample stitching
        # ------------------------------------------------------------#
        df["exclude_LHE"] = False
        df["LHEMass"] = 0 
        #if dataset == "dy_M-50":# or dataset =="test":
        # Get LHE Particles and calculate LHE mass for all DY events
        LHE_columns = ["pt",
           "eta",
           "phi",
           "mass",
           "pdgId",]
        LHEInfo = df.LHEPart[LHE_columns]
        LHEParts = ak.to_pandas(LHEInfo)
        LHEParts["ele"] = (abs(LHEParts.pdgId) == 11)
        LHEParts["muons"] = (abs(LHEParts.pdgId) == 13) 
        LHEParts["tau"] = (abs(LHEParts.pdgId )== 15)

        LHEmuons=LHEParts[LHEParts["muons"]]
        LHEeles=LHEParts[LHEParts["ele"]]
        LHEtaus=LHEParts[LHEParts["tau"]]
        LHELeptons = pd.concat([LHEmuons,LHEeles,LHEtaus])
        LHElep1 = LHELeptons.loc[LHELeptons.pdgId.groupby("entry").idxmax()]
        LHElep2 = LHELeptons.loc[LHELeptons.pdgId.groupby("entry").idxmin()]

        LHElep1.index = LHElep1.index.droplevel("subentry")
        LHElep2.index = LHElep2.index.droplevel("subentry")

        LHEMass = p4_sum(LHElep1,LHElep2).mass
        LHELeptons["LHEMass"] = LHEMass

        LHEMass = LHEMass.to_frame(name="LHEMass")
        LHEMassVal =  LHEMass["LHEMass"]

        LHEMass["exclude_LHE"] = False
        if dataset == "dy_M-50" or dataset == "test":
            LHEMass.loc[((LHEMassVal > 100) & (LHEMassVal < 200)), "exclude_LHE"] = True

        print(LHEMassVal)
        df["exclude_LHE"] = LHEMass["exclude_LHE"]
        df["LHEMass"] = LHEMassVal
        print(df["LHEMass"])
            
            

        

        
        # ------------------------------------------------------------#
        # Apply HLT, lumimask, genweights, PU weights
        # and L1 prefiring weights
        # ------------------------------------------------------------#

        # All variables that we want to save
        # will be collected into the 'output' dataframe
        output = pd.DataFrame({"run": df.run, "event": df.event})
        output.index.name = "entry"

        output["LHEMass"] = df["LHEMass"]
        print( output["LHEMass"])
        # Separate dataframe to keep track on weights
        # and their systematic variations
        weights = Weights(output)

        if is_mc:
            # For MC: Apply gen.weights, pileup weights, lumi weights,
            # L1 prefiring weights
            mask = np.ones(numevents, dtype=bool)
            genweight = df.genWeight
            weights.add_weight("genwgt", genweight)
            output[genwgt] = genweight
            #if dataset == "DY_M-50":
                #weights.add_weight("lumi", df["new_lumi_weights"])
            #else:
            weights.add_weight("lumi", self.lumi_weights[dataset])




        else:
            # For Data: apply Lumi mask
            lumi_info = LumiMask(self.parameters["lumimask"])
            mask = lumi_info(df.run, df.luminosityBlock)

        # Apply HLT to both Data and MC
        hlt_columns = [c for c in self.parameters["hlt"] if c in df.HLT.fields]
        hlt = ak.to_pandas(df.HLT[hlt_columns])
        if len(hlt_columns) == 0:
            hlt = False
        else:
            hlt = hlt[hlt_columns].sum(axis=1)






        # for ...
        if True:  # indent reserved for loop over muon pT variations
           
            muon_columns = [
                "pt",
                "eta",
                "phi",
                "mass",
                "charge",

            ] + [self.parameters["muon_id"]]
            muons = ak.to_pandas(df.Muon[muon_columns])


            muons["selection"] = (
                (muons.pt > self.parameters["muon_pt_cut"])
                & (abs(muons.eta) < self.parameters["muon_eta_cut"])
            )

            # Count muons
            nmuons = (
                muons[muons.selection]
                .reset_index()
                .groupby("entry")["subentry"]
                .nunique()
            )

            # Find opposite-sign muons
            mm_charge = muons.loc[muons.selection, "charge"].groupby("entry").prod()

            
            LHE_Cut = ak.to_pandas(df.exclude_LHE)
            LHE_Cut = LHE_Cut.product(axis=1)

            # Define baseline event selection
            output["two_muons"] = nmuons == 2
            output["event_selection"] = (
                mask
                & (LHE_Cut ==False)
                & (hlt > 0)
                & (nmuons == 2)
                & (mm_charge == -1)
            )
            #print(output["event_selection"])
            # --------------------------------------------------------#
            # Select two leading-pT muons
            # --------------------------------------------------------#

            # Find pT-leading and subleading muons
            # This is slow for large chunk size.
            # Consider reimplementing using sort_values().groupby().nth()
            # or sort_values().drop_duplicates()
            # or using Numba
            # https://stackoverflow.com/questions/50381064/select-the-max-row-per-group-pandas-performance-issue
            muons = muons[muons.selection & (nmuons == 2)]
            mu1 = muons.loc[muons.pt.groupby("entry").idxmax()]
            mu2 = muons.loc[muons.pt.groupby("entry").idxmin()]
            mu1.index = mu1.index.droplevel("subentry")
            mu2.index = mu2.index.droplevel("subentry")

            # --------------------------------------------------------#
            # Select events with muons passing leading pT cut
            # and trigger matching (trig match not done in final vrsn)
            # --------------------------------------------------------#

            # Events where there is at least one muon passing
            # leading muon pT cut
            pass_leading_pt = mu1.pt > self.parameters["muon_leading_pt"]

            # update event selection with leading muon pT cut
            output["pass_leading_pt"] = pass_leading_pt
            output["event_selection"] = output.event_selection & output.pass_leading_pt
            print("just before fill muons")
            print(output)
            # --------------------------------------------------------#
            # Fill dimuon and muon variables
            # --------------------------------------------------------#

            fill_muons_Simple(self, output, mu1, mu2, is_mc)

        # ------------------------------------------------------------#
        # Fill outputs
        # ------------------------------------------------------------#
        mass = output.dimuon_mass
        print("just after fill muons")
        print(output)
        #output["region"] = None
        #output.loc[((mass > 76) & (mass < 106)), "region"] = "z-peak"
        #output.loc[
            #((mass > 110) & (mass < 115.03)) | ((mass > 135.03) & (mass < 150)),
            #"region",
        #] = "h-sidebands"
        #output.loc[((mass > 115.03) & (mass < 135.03)), "region"] = "h-peak"
        output["dataset"] = dataset
        output["year"] = self.year
        print(self.vars_to_save)
        columns_to_save = [
            c
            for c in output.columns
            if (c[0] in self.vars_to_save)
            or (c[0] in ["dataset", "year"])
        ]
        #output = output.loc[output.event_selection, columns_to_save]
        #output = output.reindex(sorted(output.columns), axis=1)
        #output.columns = ["_".join(col).strip("_") for col in output.columns.values]
        #output = output[output.region.isin(self.regions)]
        output = output[output.event_selection]
        print(output)
        print(output.keys)
        to_return = None
        if self.apply_to_output is None:
            to_return = output
        else:
            self.apply_to_output(output)
            to_return = self.accumulator.identity()

        return to_return
    def prepare_lookups(self):





        # Muon scale factors
        self.musf_lookup = musf_lookup(self.parameters)



        # --- Evaluator
        self.extractor = extractor()


        self.extractor.finalize()
        self.evaluator = self.extractor.make_evaluator()

      

        return
    @property
    def accumulator(self):
        return processor.defaultdict_accumulator(int)

    @property
    def columns(self):
        return branches

    def postprocess(self, accumulator):
        return accumulator
