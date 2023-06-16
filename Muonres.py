import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from python.math_tools import p4_sum, delta_r
from stage1.corrections.geofit import apply_geofit
from stage1.corrections.rochester import apply_roccor
from stage1.corrections.fsr_recovery import fsr_recovery
from scipy.optimize import curve_fit
import sys
import ROOT
ROOT.gSystem.Load("stage3/lib/RooDoubleCB/RooDoubleCB_cxx")
from stage3.fit_models import doubleCB
from stage3.fit_models import BWxDCB
from coffea.lookup_tools import txt_converters, rochester_lookup
from stage3.fit_models import Voigtian


rochester_data = txt_converters.convert_rochester_file(
        "data/roch_corr/RoccoR2018UL.txt", loaduncs=True
        )
roccor_lookup = rochester_lookup.rochester_lookup(rochester_data)


def fitDCB(Mass):
    x = ROOT.RooRealVar("x", "x", 105, 150)
    data = ROOT.RooDataSet.from_pandas({"x": Mass}, [x])
    DCB=doubleCB(x,"test")
    result= DCB[0].fitTo(data)
    return data,x,DCB
    
def fitVoigtian(Mass):
    x = ROOT.RooRealVar("x", "x", 65, 125)
    data = ROOT.RooDataSet.from_pandas({"x": Mass}, [x])
    Voigt=Voigtian(x,"test")
    result= Voigt[0].fitTo(data)
    return data,x,Voigt
    
def fitBWDCB(Mass):
    x = ROOT.RooRealVar("x", "x", 65, 125)
    x.setBins(100)
    data = ROOT.RooDataSet.from_pandas({"x": Mass}, [x])
    datahist =  ROOT.RooDataHist("dh","binned version of data",ROOT.RooArgSet(x),data)
    BWDCB=BWxDCB(x,"test")
    l = ROOT.RooLinkedList()
    l.Add(ROOT.RooFit.Range(70,110))
    result= BWDCB[0].fitTo(datahist,l)
    return data,x,BWDCB
def gauss(x, A,mu,sigma):
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))    
def mass_resolution(df):
    # Returns absolute mass resolution!
    dpt1 = (df.mu1_ptErr * df.dimuon_mass) / (2 * df.mu1_pt)
    dpt2 = (df.mu2_ptErr * df.dimuon_mass) / (2 * df.mu2_pt)
    return np.sqrt(dpt1 * dpt1 + dpt2 * dpt2)
def getGenMass(obj1,obj2):
    result = pd.DataFrame(
        index=obj1.index.union(obj2.index),
        columns=["px", "py", "pz", "e", "pt", "eta", "phi", "mass", "rap"],
    ).fillna(0.0)
    for obj in [obj1, obj2]:
        px_ = obj.genpt * np.cos(obj.genphi)
        py_ = obj.genpt * np.sin(obj.genphi)
        pz_ = obj.genpt * np.sinh(obj.geneta)
        e_ = np.sqrt(px_**2 + py_**2 + pz_**2 + obj.genmass**2)
        result.px += px_
        result.py += py_
        result.pz += pz_
        result.e += e_
    result.pt = np.sqrt(result.px**2 + result.py**2)
    result.eta = np.arcsinh(result.pz / result.pt)
    result.phi = np.arctan2(result.py, result.px)
    result.mass = np.sqrt(
        result.e**2 - result.px**2 - result.py**2 - result.pz**2
    )
    return result.mass
   
def fill_muons(output, mu1, mu2):
    mu1_variable_names = ["mu1_pt", "mu1_pt_over_mass", "mu1_eta", "mu1_phi", "mu1_iso"]
    mu2_variable_names = ["mu2_pt", "mu2_pt_over_mass", "mu2_eta", "mu2_phi", "mu2_iso"]
    dimuon_variable_names = [
        "dimuon_mass",
        "dimuon_ebe_mass_res",
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
    for v in ["pt", "ptErr", "eta", "phi"]:
        output[f"mu1_{v}"] = mu1[v]
        output[f"mu2_{v}"] = mu2[v]

    output["mu1_iso"] = mu1.pfRelIso04_all
    output["mu2_iso"] = mu2.pfRelIso04_all
    output["mu1_pt_over_mass"] = output.mu1_pt / output.dimuon_mass
    output["mu2_pt_over_mass"] = output.mu2_pt / output.dimuon_mass

    # Fill dimuon variables
    mm = p4_sum(mu1, mu2)
    for v in ["pt", "eta", "phi", "mass", "rap"]:
        name = f"dimuon_{v}"
        output[name] = mm[v]
        output[name] = output[name].fillna(-999.0)

    output["dimuon_pt_log"] = np.log(output.dimuon_pt)

    mm_deta, mm_dphi, mm_dr = delta_r(mu1.eta, mu2.eta, mu1.phi, mu2.phi)

    output["dimuon_dEta"] = mm_deta
    output["dimuon_dPhi"] = mm_dphi
    output["dimuon_dR"] = mm_dr

    output["dimuon_mass_res"] = mass_resolution(output)
 
if sys.argv[1]=="data":
    fname = "root://cmsxrootd.fnal.gov///store/data/Run2018A/SingleMuon/NANOAOD/UL2018_MiniAODv2_NanoAODv9-v2/2550000/00EBBD1F-032C-9B49-A998-7645C9966432.root"
   
if sys.argv[1] =="Z":
    fname = "root://cmsxrootd.fnal.gov///store/mc/RunIISummer20UL18NanoAODv9/DY1JetsToLL_M-50_MatchEWPDG20_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/3F74AD49-80AE-9B4B-9B30-CE5E644724E4.root"
if (sys.argv[1] =="Higgs") or   (sys.argv[1] =="cats"):
    fname = "root://cmsxrootd.fnal.gov///store/mc/RunIISummer20UL18NanoAODv9/GluGluHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/2810000/C4DAB63C-E2A1-A541-93A8-3F46315E362C.root"
#fname = "tests/samples/ewk_lljj_mll105_160_ptj0_NANOV10_2018.root"

events = NanoEventsFactory.from_root(
    fname,
    schemaclass=NanoAODSchema.v6,
    #metadata={"dataset": "DY"},
    #entry_stop=200000,
).events()


#has_fsr = fsr_recovery(events)
#events["Muon", "pt"] = events.Muon.pt_fsr
#events["Muon", "eta"] = events.Muon.eta_fsr
#events["Muon", "phi"] = events.Muon.phi_fsr
#events["Muon", "pfRelIso04_all"] = events.Muon.iso_fsr

#numevents= len(events)
#mask = np.ones(numevents, dtype=bool)
#apply_geofit(events, "2018", mask)
#events["Muon", "pt"] = events.Muon.pt_gf

apply_roccor(events, roccor_lookup, 1)
events["Muon", "pt"] = events.Muon.pt_roch

#events.Muon["genPt"] = events.Muon.matched_gen.pt
GenMu_columns = ["pt",
               #"eta",
                #"phi",
                #"charge",
                #"ptErr",
                #"mass",
                #"pfRelIso04_all",
                #"mediumId"
                #"genPartIdx"
                
            ]
muon_columns = [
                "pt",
                "eta",
                "phi",
                "charge",
                "ptErr",
                "mass",
                "pfRelIso04_all",
                "mediumId",
                #"genPartIdx"
                
            ]
muons = ak.to_pandas(events.Muon[muon_columns])
if sys.argv[1] != "data":
    GenMuonsPt = ak.to_pandas(events.Muon.matched_gen.pt)
    GenMuonsPt.rename(columns={"values":"genpt"}, inplace =True)
    GenMuonsEta = ak.to_pandas(events.Muon.matched_gen.eta)
    GenMuonsEta.rename(columns={"values":"geneta"}, inplace =True)
    GenMuonsPhi = ak.to_pandas(events.Muon.matched_gen.phi)
    GenMuonsPhi.rename(columns={"values":"genphi"}, inplace =True)
    GenMuonsMass = ak.to_pandas(events.Muon.matched_gen.mass)
    GenMuonsMass.rename(columns={"values":"genmass"}, inplace =True)

    GenMuons = GenMuonsPt.join(GenMuonsEta)
    GenMuons = GenMuons.join(GenMuonsPhi)
    GenMuons = GenMuons.join(GenMuonsMass)
    #GenParticles["mu"] = (((GenParticles.pdgId == 13) | (GenParticles.pdgId == -13)) & (GenParticles.statusFlags ==12))
    #GenParticles = GenParticles[GenParticles.mu]
    #print(GenMuons)

    muons = muons.join(GenMuons)
muons["selection"] = (
                (muons.pt > 20)
                & (abs(muons.eta < 2.4))
                & (muons.pfRelIso04_all < 0.25)
                & muons["mediumId"]
            )
# Count muons
nmuons = (
           muons[muons.selection]
                .reset_index()
                .groupby("entry")["subentry"]
                .nunique()
            )

#print(nmuons)

# Find opposite-sign muons
mm_charge = muons.loc[muons.selection, "charge"].groupby("entry").prod()
output = pd.DataFrame()
output["event_selection"] = (
          (nmuons == 2)
         & (mm_charge == -1)
            )        
muons = muons[muons.selection & (nmuons == 2)]
mu1 = muons.loc[muons.pt.groupby("entry").idxmax()]
mu2 = muons.loc[muons.pt.groupby("entry").idxmin()]
#sgenmu1 = 
mu1.index = mu1.index.droplevel("subentry")
mu2.index = mu2.index.droplevel("subentry")

fill_muons(output,mu1,mu2)
output= output[output.event_selection==True]
print(output)
EtaCats = [[0.0,0.9,0.0,0.9],[0.0,0.9,0.9,1.8],[0.0,0.9,1.8,2.4], [0.9,1.8,0.0,0.9], [0.9,1.8,0.9,1.8], [0.9,1.8,1.8,2.4],[1.8,2.4,0.0,0.9],[1.8,2.4,0.9,1.8],[1.8,2.4,1.8,2.4]]
nbins=80
if(sys.argv[2]=="cats"):
    for Category in EtaCats:
        if (Category[0] == 0.0) & (Category[2] == 0.0):
            name = "Barrel-Barrel"
        if (Category[0] == 0.0) & (Category[2] == 0.9):
            name = "Barrel-Overlap"
        if (Category[0] == 0.0) & (Category[2] == 1.8):
            name = "Barrel-Endcap"
        if (Category[0] == 0.9) & (Category[2] == 0.0):
            name = "Overlap-Barrel"
        if (Category[0] == 0.9) & (Category[2] == 0.9):
            name = "Overlap-Overlap"
        if (Category[0] == 0.9) & (Category[2] == 1.8):
            name = "Overlap-Endcap"
        if (Category[0] == 1.8) & (Category[2] == 0.0):
            name = "Endcap-Barrel"
        if (Category[0] == 1.8) & (Category[2] == 0.9):
            name = "Endcap-Overlap"
        if (Category[0] == 1.8) & (Category[2] == 1.8):
            name = "Endcap-Endcap"
        output["Muoncat"] = ((abs(output.mu1_eta)>Category[0]) &(abs(output.mu2_eta)>Category[2]) & (abs(output.mu1_eta)<Category[1])& (abs(output.mu2_eta)<Category[3]))
        mu1["Muoncat"] = ((abs(mu1.eta)>Category[0]) & (abs(mu1.eta)<Category[1]))
        mu1_cat = mu1[mu1.Muoncat==True]
        mu2["Muoncat"] = ((abs(mu2.eta)>Category[2]) & (abs(mu2.eta)<Category[3]))
        mu2_cat = mu2[mu2.Muoncat==True]
        output_cat = output[output.Muoncat==True]
        #etahist = plt.hist(output_cat.mu1_eta)
        #plt.show()
        Mass = output_cat["dimuon_mass"]
        if sys.argv[1]!="data":
            GenMass = getGenMass(mu1_cat,mu2_cat)
            Residuals = (Mass-GenMass)/GenMass
            #Residuals_cleaned =[]
                #plt.show()
                #sys.exit()
            #for value in Residuals:
            #   if (abs(value)-np.mean(Residuals)) < 1 & (np.isnan(value)==False) :
            #       Residuals_cleaned.append(value)
            #Residuals_cleaned= np.array(Residuals_cleaned)

            Residuals_cleaned = Residuals
            plt.figure(name)
            #print(Residuals_cleaned.max())
            hist, bin_edges, patches = plt.hist(Residuals_cleaned, nbins, range = [Residuals_cleaned.min(),Residuals_cleaned.max()])
            p0 = [0.1, 0., 0.1]
            coeff, var_matrix = curve_fit(gauss,np.linspace(Residuals_cleaned.min(),Residuals_cleaned.max(),nbins), hist,p0)


        # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
        
        #print(hist.max())
        
        #print(np.linspace(Residuals_cleaned.min(),Residuals_cleaned.max(),80))
        # Get the fitted curve
        #print(coeff) 
            Gaussian_fit = gauss(np.linspace(Residuals_cleaned.min(),Residuals_cleaned.max(),nbins), *coeff)
            plt.plot(np.linspace(Residuals_cleaned.min(),Residuals_cleaned.max(),nbins),Gaussian_fit)
    
            plt.title("Mass residuals without Geofit corrections "+ name)
            plt.xlabel("Dimuon mass residuals")
            plt.ylabel("a.u.")
            plt.figtext(0.15, 0.8, "Resolution from fit:"+str(abs(coeff[2]).round(4)),fontsize=12)
            #plt.show()
            plt.savefig("MassResiduals" +name+".png")
        MassRes = output_cat["dimuon_mass_res"]#/output_cat["dimuon_mass"]
        MassResRel = output_cat["dimuon_mass_res"]/output_cat["dimuon_mass"]
        plt.figure(2)
        plt.hist(MassResRel,150,histtype="step",label=name,range=[0.0,0.05],density=True)
        plt.title("mass resolution without Geofit corrections "+ name)
        plt.xlabel("Dimuon mass resolution")
        plt.ylabel("a.u.")
        plt.legend()
        plt.savefig("FancymassRes.png")
        plt.figure(name + "2")
        plt.hist(MassResRel,40,label=name)
        plt.xlabel("Relative dimuon mass resolution")
        plt.ylabel("a.u.")
        m = np.median(MassResRel)
        plt.figtext(0.15, 0.8, "Resolution from median: "+str(m.round(4)),fontsize=12)
        plt.savefig("MassResRel"+name+".png")
        
        #plt.clf()
    
        del output_cat
        del mu1_cat
        del mu2_cat
        Mass_all = output["dimuon_mass"]
        if sys.argv[1]=="Higgs":
            data,x,fitfunc = fitDCB(Mass)
        if sys.argv[1]=="Z" or sys.argv[1]=="data":
            data,x,fitfunc = fitBWDCB(Mass)
        c = ROOT.TCanvas("blub","blub", 800,600)
        c.cd()
        xframe2 = x.frame(Title="DCB fit to "+str(sys.argv[1])+" peak (NanoAOD with Geofit)")
        data.plotOn(xframe2, DrawOption="F",FillColor="kRed")
        fitfunc[0].plotOn(xframe2)


        xframe2.Draw()
        fit_vals=fitfunc[1]
        #pt.Draw()
        #c.Update()
        textnumber = fit_vals[0].getVal()
        textnumber = round(textnumber,2)
        text= "Peak Mass from fit: "+ str(textnumber)
        if sys.argv[1]=="Higgs":
            textnumber2 = fit_vals[1].getVal()
            textnumber2 = round(textnumber2,2)
            text2=  "Resolution from fit: "+ str(textnumber2)
        if sys.argv[1]=="Z" or sys.argv[1]=="data":
            textnumber2 = fit_vals[2].getVal()
            textnumber2 = round(textnumber2,2)
            text2= "Resolution from fit: "+ str(textnumber2)
        latex= ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextAlign(11)
        latex.SetTextFont(42)
        latex.SetTextSize(0.04)
        latex.DrawLatex(0.48, 0.81,text )
        latex.DrawLatex(0.48, 0.75,text2 )
        c.SaveAs("Resolutionfrom"+name+str(sys.argv[1])+"Peak_roccor.png")
        print("-----------------------------------")
        print(fit_vals[0].getVal())
        print(fit_vals[1].getVal())
        #print("Higgs mass resolution from fit: "+fit_vals)
        print("-----------------------------------")
        





      
            
            
            
