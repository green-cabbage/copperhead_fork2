import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from python.math_tools import p4_sum, delta_r
from stage1.corrections.geofit import apply_geofit
from scipy.optimize import curve_fit
import sys
import ROOT
from stage3.fit_models import doubleCB

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
    
    
fname = "root://eos.cms.rcac.purdue.edu//store/mc/RunIISummer20UL17NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1/130000/A9F4CE6E-5AA7-7044-91B4-45F8B4E2B570.root"
#fname = "tests/samples/ewk_lljj_mll105_160_ptj0_NANOV10_2018.root"

events = NanoEventsFactory.from_root(
    fname,
    schemaclass=NanoAODSchema.v6,
    metadata={"dataset": "DY"},
    entry_stop=10000,
).events()

events.Muon["genPt"] = events.Muon.matched_gen.pt
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
                "genPartIdx"
                
            ]
muons = ak.to_pandas(events.Muon[muon_columns])
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
    Mass_all = output["dimuon_mass"]
    GenMass = getGenMass(mu1_cat,mu2_cat)
    Residuals = (Mass-GenMass)/GenMass
    #Residuals_cleaned =[]
    if sys.argv[1]=="zpeak":
        hist,bin_edges,patches =plt.hist(Mass_all,80,range=[55,140])
        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
        x = ROOT.RooRealVar("x", "x", 55, 130)
        xframe = x.frame()
        data = ROOT.RooDataSet.from_numpy({"x": Mass_all}, [x])
        doubleCB.fitTo(data)
        xframe2 = x.frame()
        data.plotOn(xframe2)
        doubleCB.plotOn(xframe2)
        xframe2.Draw()
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
    
    def gauss(x, A,mu,sigma):
        
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))

    # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
    p0 = [0.1, 0., 0.1]
    #print(hist.max())
    coeff, var_matrix = curve_fit(gauss,np.linspace(Residuals_cleaned.min(),Residuals_cleaned.max(),nbins), hist,p0)
    #print(np.linspace(Residuals_cleaned.min(),Residuals_cleaned.max(),80))
    # Get the fitted curve
    print(coeff) 
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
    plt.title("mass resolution without Geofit corrections")
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
    m2 = np.std(Residuals)
    print(m)
    print(m2)
    #plt.clf()
    
    del output_cat
    del mu1_cat
    del mu2_cat






      
            
            
            
