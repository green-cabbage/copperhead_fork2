import ROOT as rt
import pandas as pd
from python.workflow_noDask import non_parallelize
from stage3.fit_plots import plot
from stage3.fit_models import chebyshev, doubleCB, doubleCB_forZ, SumTwoExpPdf, bwZ, bwGamma, bwZredux, bernstein,BWxDCB,Voigtian_Erf,Erf
import pdb
rt.RooMsgService.instance().setGlobalKillBelow(rt.RooFit.ERROR)
#t.gSystem.Load ("../CMSSW_12_4_15/lib/el8_amd64_gcc10/libHiggsAnalysisCombinedLimit.so")
simple = False
def mkdir(path):
    try:
        os.mkdir(path)
    except Exception:
        pass
def run_fits(parameters, df,df_all,tag):
    signal_ds = parameters.get("signals", [])
    data_ds = parameters.get("data", [])
    year = parameters.get("years")[0]
    is_Z = parameters.get("is_Z")
    all_datasets = df.dataset.unique()
    signals = [ds for ds in all_datasets if ds in signal_ds]
    backgrounds = [ds for ds in all_datasets if ds in data_ds]
    fit_setups = []
    if is_Z == True:
        fit_setup = {"label": f"Zfit_{tag}", "mode": "Z", "year": year, "df":  df_all[df_all.dataset.isin(backgrounds)]}
        fit_setups.append(fit_setup)
        argset = {
            "fit_setup": fit_setups,
            "channel": parameters["mva_channels"],
            "category": ["All"]
        }
        fit_ret = non_parallelize(fitter, argset, parameters)
        return fit_ret
    elif len(backgrounds) > 0:
        fit_setup_multi = {
           "label": "background_all",
           "mode": "bkg_all",
            "year": year,
           "df": df_all[df_all.dataset.isin(backgrounds)],
           "blinded": False,
        }
        for ds in signals:
            fit_setup = {"label": ds, "mode": "sig", "year": year, "df": df[df.dataset == ds]}
            fit_setups.append(fit_setup)
        fit_setups.append(fit_setup_multi)
        argset = {
            "fit_setup": fit_setups,
            "channel": parameters["mva_channels"],
            "category": ["All"]
        }
        print(df_all[df_all.dataset.isin(backgrounds)])
        print("-----------------------------------")
        print("starting non-cat bkg fit")
        print("-----------------------------------")
        #db.set_trace()
        #print(argset)
        #print (parameters)
        fit_ret = non_parallelize(fitter, argset, parameters)
    fit_setups = []
    if len(backgrounds) > 0 & (is_Z ==False):
        if simple ==False:
            fit_setup = {
                "label": "background_cats",
                "mode": "bkg_cats",
                "year": year,
                "df": df[df.dataset.isin(backgrounds)],
                "blinded": False,
            }
        else:
            fit_setup = {
                "label": "background_cats_simple",
                "mode": "bkg_cats_simple",
                "year": year,
                "df": df[df.dataset.isin(backgrounds)],
                "blinded": False,
            }
        fit_setups.append(fit_setup)
    if is_Z ==False:
        for ds in signals:
            fit_setup = {"label": ds, "mode": "sig", "year": year, "df": df[df.dataset == ds]}
            fit_setups.append(fit_setup)
       

    argset = {
        "fit_setup": fit_setups,
        "channel": parameters["mva_channels"],
        "category": df["category"].dropna().unique(),
    }
    print(df["category"])
    fit_ret = non_parallelize(fitter, argset, parameters)
    df_fits = pd.DataFrame(columns=["label", "channel", "category", "chi2"])
    for fr in fit_ret:
        df_fits = pd.concat([df_fits, pd.DataFrame.from_dict(fr)])
    # choose fit function with lowest chi2/dof
    df_fits.loc[df_fits.chi2 <= 0, "chi2"] = 999.0
    df_fits.to_pickle("all_chi2.pkl")
    idx = df_fits.groupby(["label", "channel", "category"])["chi2"].idxmin()
    df_fits = (
        df_fits.loc[idx]
        .reset_index()
        .set_index(["label", "channel"])
        .sort_index()
        .drop_duplicates()
    )
    #print(df_fits)
    df_fits.to_pickle("best_chi2.pkl")
    return fit_ret


def fitter(args, parameters={}):
    fit_setup = args["fit_setup"]
    df = fit_setup["df"]
    label = fit_setup["label"]
    year = fit_setup["year"]
    mode = fit_setup["mode"]
    print(mode)
    blinded = fit_setup.get("blinded", False)
    save = parameters.get("save_fits", False)
    save_path = parameters.get("save_fits_path", "fits/")
    channel = args["channel"]
    if (mode != 'bkg_all') or (mode != 'Z') :
        category = args["category"]
    else:
        category = 'All'
    
    if mode == "Z":
        save_path = save_path + f"/calib_fits/BWxDCB/"
    else:
        save_path = save_path + f"/fits_{channel}_{category}/"
    mkdir(save_path)
    #print(df)
    #with channel selection
    #df = df[(df.channel_nominal == channel) & (df.category == category)]
    #without chennel selection
    df = df[(df.category == category)]
    #print(channel)
    #print(category)
    #print("in fitter")
    #print(df)
    norm = df.wgt_nominal.sum()
    print(f"Channel: {channel}, Category: {category}, {norm}")
    #norm = 1.
    
    the_fitter = Fitter(
        fitranges={"low": 110,"low_signal": 118,"low_Z": 80, "high": 150,"high_signal": 132,"high_Z": 100, "SR_left": 120, "SR_right": 130},
        #fitranges_signal={"low": 115, "high": 135},
        fitmodels={
            "bwz": bwZ,
            "bwz_redux": bwZredux,
            "bwgamma": bwGamma,
            "bernstein": bernstein,
            "SumTwoExpPdf": SumTwoExpPdf,
            "dcb": doubleCB,
            "BWxDCB": BWxDCB,
            "Voigtian": Voigtian_Erf,
            "chebyshev": chebyshev,
        },
        requires_order=["chebyshev", "bernstein"],
        channel=channel,
        mode=mode,
        label=label,
        category = category,
        filename_ext="",
        
    )
    if mode == "bkg_all":
         print("doing non-categorized bkg fits")
         chi2 = the_fitter.simple_fit(
            dataset=df,
            label=label,
            category="All",
            blinded=blinded,
            model_names=["SumTwoExpPdf", "bwz_redux",],
            #model_names=["bwz_redux",],
            model_names_multi=[],
            #orders = {"chebyshev": [2,3],},
            fix_parameters=False,
            store_multipdf=False,
            doProdPDF=False,
            title="Background",
            save=True,
            save_path=save_path,
            norm=norm,
         )
    if mode == "bkg_cats":
         print("doing categorized bkg fits")
         chi2 = the_fitter.simple_fit(
            dataset=df,
            label=label,
            category=category,
            blinded=blinded,
            model_names_multi=["SumTwoExpPdf", "bwz_redux",],
            model_names=["bernstein"],#"bernstein"],
            orders = {"bernstein": [2],
                      #"bernstein": [4]
                     },
            fix_parameters=False,
            store_multipdf=True,
            doProdPDF=True,
            title="Background",
            save=True,
            save_path=save_path,
            norm=norm,
         )
    if mode == "bkg_cats_simple":
         print("doing categorized bkg fits")
         chi2 = the_fitter.simple_fit(
            dataset=df,
            label=label,
            category=category,
            blinded=blinded,
            model_names_multi=[],
            model_names=["bwz_redux",],
            #orders = {"bernstein": [2],
                      #"bernstein": [4]
                     #},
            fix_parameters=False,
            store_multipdf=False,
            doProdPDF=False,
            title="Background",
            save=True,
            save_path=save_path,
            norm=norm,
         )
        # generate and fit pseudo-data
        #the_fitter.fit_pseudodata(
            #label="pseudodata_" + label,
            #category=category,
            #blinded=blinded,
            #model_names=["bwz", "bwz_redux", "bwgamma"],
            #fix_parameters=False,
            #title="Pseudo-data",
            #save=save,
            #save_path=save_path,
            #norm=norm,
        #)

    if mode == "sig":
        print("doing signal fits")
        chi2 = the_fitter.simple_fit(
            dataset=df,
            label=label,
            category=category,  # temporary
            blinded=False,
            model_names=["dcb"],
            model_names_multi=[],
            fix_parameters=True,
            store_multipdf=False,
            doProdPDF=False,
            title="Signal",
            save=True,
            save_path=save_path,
            norm=norm,
        )
    if mode == "Z":
        print("doing Z fits")
        chi2 = the_fitter.simple_fit(
            dataset=df,
            label=label,
            category=category,  # temporary
            blinded=False,
            model_names=["BWxDCB"],
            #model_names=["Voigtian_Erf"],
            model_names_multi=[],
            fix_parameters=True,
            store_multipdf=False,
            doProdPDF=False,
            title="Z",
            save=True,
            save_path=save_path,
            norm=norm,
        )
    ret = {"label": label, "channel": channel, "category": category, "chi2": chi2}
    return ret


class Fitter(object):
    def __init__(self, **kwargs):
        self.fitranges = kwargs.get(
            "fitranges", {"low": 110,"low_signal": 120,"low_Z": 80, "high": 150,"high_signal": 130,"high_Z": 100, "SR_left": 120, "SR_right": 130}
        )
        self.fitmodels = kwargs.get("fitmodels", {})
        self.requires_order = kwargs.get("requires_order", [])
        self.channel = kwargs.get("channel", "ggh")
        self.filename_ext = kwargs.get("filename_ext", "")

        self.data_registry = {}
        self.model_registry = []
        self.mode = kwargs.get("mode", "sig")
        self.label = kwargs.get("label", "dummylabel")
        self.category = kwargs.get("category", "All")
        binned=False
        self.workspace = self.create_workspace(self.mode,binned, self.category)

    def simple_fit(
        self,
        dataset=None,
        label="test",
        category="cat0",
        blinded=False,
        binned=False,
        model_names=[],
        model_names_multi=[],
        orders={},
        fix_parameters=False,
        store_multipdf=True,
        doProdPDF=False,
        title="",
        save=True,
        save_path="./",
        norm=0,
    ):
        if dataset is None:
            raise Exception("Error: dataset not provided!")
        if len(model_names) == 0:
            raise Exception("Error: empty list of fit models!")

        ds_name = f"data_{label}"
        self.add_data(dataset, norm, ds_name=ds_name, blinded=blinded, binned=binned)
        print(dataset["dimuon_mass"])
        ndata = len(dataset["dimuon_mass"].values)
        print(ndata)
        for model_name in model_names:
            if (model_name in self.requires_order) and (model_name in orders.keys()):
                for order in orders[model_name]:
                    self.add_model(model_name, category=category, order=order)
            else:
                self.add_model(model_name, category=category)

        # self.workspace.Print()
        chi2 = self.fit(
            ds_name,
            ndata,
            model_names,
            model_names_multi = model_names_multi,
            orders=orders,
            doMultiPDF=store_multipdf,
            doProdPDF=doProdPDF,
            blinded=blinded,
            fix_parameters=fix_parameters,
            category=category,
            label=label,
            title=title,
            save=save,
            save_path=save_path,
            norm=norm,
        )

        if True:
            mkdir(save_path)
            if doProdPDF:
                for model_name in model_names:
                    for order in orders[model_name]:
                        self.save_workspace(
                            f"{save_path}/workspace_{self.channel}_{category}_{label}_{model_name}{order}{self.filename_ext}"
                )
            else:
                self.save_workspace(
                    f"{save_path}/workspace_{self.channel}_{category}_{label}{self.filename_ext}"
                )
        return chi2

    def create_workspace(self, mode, binned=False, category = "All"):
        w = rt.RooWorkspace("w", "w")
        print(self.label)
        if mode == "Z":
            
            if ("calib_cat3" in self.label) or ("calib_cat7" in self.label) or  ("calib_cat21" in self.label):
                mh_ggh = rt.RooRealVar(
                    "mh_ggh", "mh_ggh", 83, 99
                )
                self.fitranges["low_Z"] = 83
                self.fitranges["high_Z"] = 99

            elif ("calib_cat15" in self.label):
                mh_ggh = rt.RooRealVar(
                    "mh_ggh", "mh_ggh", 82, 100
                )
                self.fitranges["low_Z"] = 82
                self.fitranges["high_Z"] = 100
                
            elif ("calib_cat19" in self.label):
                mh_ggh = rt.RooRealVar(
                    "mh_ggh", "mh_ggh", 80.5, 101.1
                )
                self.fitranges["low_Z"] = 80.5
                self.fitranges["high_Z"] = 101.1
            elif ("calib_cat24" in self.label):
                mh_ggh = rt.RooRealVar(
                    "mh_ggh", "mh_ggh", 82.5, 100.5
                )
                self.fitranges["low_Z"] = 82.5
                self.fitranges["high_Z"] = 100.5

                
            elif ("calib_cat11" in self.label):
                mh_ggh = rt.RooRealVar(
                    "mh_ggh", "mh_ggh", 79, 101.8
                )
                self.fitranges["low_Z"] = 79
                self.fitranges["high_Z"] = 101.8

            elif ("calib_cat18" in self.label) or ("calib_cat20" in self.label)or ("calib_cat25" in self.label)or ("calib_cat28" in self.label) or ("calib_cat27" in self.label):
                mh_ggh = rt.RooRealVar(
                    "mh_ggh", "mh_ggh", 81, 101
                )
                self.fitranges["low_Z"] = 81
                self.fitranges["high_Z"] = 101
            

                
            elif ("calib_cat17" in self.label)  or ("calib_cat29" in self.label):
                mh_ggh = rt.RooRealVar(
                    "mh_ggh", "mh_ggh",80, 102
                )
                self.fitranges["low_Z"] = 80
                self.fitranges["high_Z"] = 102
            elif ("closure_cat0" in self.label):
                mh_ggh = rt.RooRealVar(
                    "mh_ggh", "mh_ggh", 84.5, 94.6
                )
                self.fitranges["low_Z"] = 84.5
                self.fitranges["high_Z"] = 94.6
            elif self.label=="Zfit_no_e_cut_UL_closure_cat1":
                mh_ggh = rt.RooRealVar(
                    "mh_ggh", "mh_ggh", 83, 97.5
                )
                self.fitranges["low_Z"] = 83
                self.fitranges["high_Z"] = 97.5
            elif ("closure_cat9" in self.label):
                mh_ggh = rt.RooRealVar(
                    "mh_ggh", "mh_ggh", 80, 101
                )
                self.fitranges["low_Z"] = 80
                self.fitranges["high_Z"] = 101
            elif ("closure_cat10" in self.label):
                mh_ggh = rt.RooRealVar(
                    "mh_ggh", "mh_ggh", 76, 106
                )
                self.fitranges["low_Z"] = 76
                self.fitranges["high_Z"] = 106
            elif ("closure_cat11" in self.label):
                mh_ggh = rt.RooRealVar(
                    "mh_ggh", "mh_ggh", 80, 104
                )
                self.fitranges["low_Z"] = 80
                self.fitranges["high_Z"] = 104
            else:
                mh_ggh = rt.RooRealVar(
                "mh_ggh", "mh_ggh", self.fitranges["low_Z"], self.fitranges["high_Z"]
            )
        else:
            mh_ggh = rt.RooRealVar(
                "mh_ggh", "mh_ggh", self.fitranges["low"], self.fitranges["high"]
            )
        mh_ggh.setRange(
            "sideband_left", self.fitranges["low"], self.fitranges["SR_left"]
        )
        mh_ggh.setRange(
            "sideband_right", self.fitranges["SR_right"], self.fitranges["high"]
        )
        if mode == "Z":
            mh_ggh.setRange("window", self.fitranges["low_Z"], self.fitranges["high_Z"])
        else:
            mh_ggh.setRange("window", self.fitranges["low"], self.fitranges["high"])
        mh_ggh.SetTitle("m_{#mu#mu}")
        mh_ggh.setUnit("GeV")
        if binned:
            mh_ggh.setBins(30)
        w.Import(mh_ggh)
        #getattr(w, "import")(mh_ggh)
        # w.Print()
        return w

    def save_workspace(self, out_name):
        outfile = rt.TFile(f"{out_name}.root", "recreate")
        self.workspace.Write()
        outfile.Close()

    def add_data(self, data, norm, ds_name="ds", blinded=False, binned=False):
        if ds_name in self.data_registry.keys():
            raise Exception(
                f"Error: Dataset with name {ds_name} already exists in workspace!"
            )
        norm_var = rt.RooRealVar(f"norm{ds_name}", f"norm{ds_name}", norm)
        try:
            #self.workspace.Import(norm_var)
            getattr(self.workspace, "import")(norm_var)
        except Exception:
            print(f"{norm_var} already exists in workspace, skipping...")
        if isinstance(data, pd.DataFrame):
            data = self.fill_dataset(
                data["dimuon_mass"].values,norm_var, self.workspace.obj("mh_ggh"), ds_name=ds_name, binned =binned
            )
        elif isinstance(data, pd.Series):
            data = self.fill_dataset(
                data.values, norm_var,self.workspace.obj("mh_ggh"), ds_name=ds_name,binned =binned
            )
        elif not (
            isinstance(data, rt.TH1F)
            or isinstance(data, rt.RooDataSet)
            or isinstance(data, rt.RooDataHist)
        ):
            raise Exception(f"Error: trying to add data of wrong type: {type(data)}")

        if blinded:
            if ds_name == "data_signal":
                data = data.reduce(rt.RooFit.CutRange("low,sideband_left"))
                data = data.reduce(rt.RooFit.CutRange("sideband_right,high"))
            else:
                data = data.reduce(rt.RooFit.CutRange("sideband_left,sideband_right"))
        
        self.data_registry[ds_name] = type(data)
        # self.workspace.Import(data, ds_name)
        getattr(self.workspace, "import")(data)


    def fit_pseudodata(
        self,
        label="test",
        category="cat0",
        blinded=False,
        model_names_multi=[],
        model_names=[],
        orders={},
        fix_parameters=False,
        title="",
        save=True,
        save_path="./",
        norm=0,
    ):
        tag = f"_{self.channel}_{category}"
        chi2 = {}
        model_names_all = []
        for model_name in model_names:
            if (model_name in self.requires_order) and (model_name in orders.keys()):
                for order in orders[model_name]:
                    model_names_all.append({"name": model_name, "order": order})
            else:
                model_names_all.append({"name": model_name, "order": 0})

        for model_names_order in model_names_all:
            model_name = model_names_order["name"]
            order = model_names_order["order"]
            if model_name in self.requires_order:
                model_key = f"{model_name}{str(order)}" + tag
            else:
                model_key = model_name + tag
            # print(model_key)
            # self.workspace.pdf(model_key).Print()
            data = self.workspace.pdf(model_key).generate(
                rt.RooArgSet(self.workspace.obj("mh_ggh")), norm
            )
            ds_name = f"data_{model_key}_pseudo"
            self.add_data(data, norm, ds_name=ds_name)
            chi2[model_key] = self.fit(
                ds_name,
                norm,
                [model_name],
                orders={model_name: order},
                doMultiPDF=False, 
                doProdPDF=False,
                blinded=blinded,
                fix_parameters=fix_parameters,
                category=category,
                label=label,
                title=title,
                save=save,
                save_path=save_path,
                norm=norm,
            )[model_key]
        if save:
            mkdir(save_path)
            self.save_workspace(
                f"{save_path}/workspace_{self.channel}_{category}_{label}{self.filename_ext}"
            )
        return chi2

    def fill_dataset(self,  data,norm_var, x, ds_name="ds", binned=False):
        cols = rt.RooArgSet(x)
        ds = rt.RooDataSet(ds_name, ds_name, cols) 
        for datum in data:
            if (datum < x.getMax()) and (datum > x.getMin()):
                x.setVal(datum)
                ds.add(cols)
        if binned:
            x.setBins(30)
            cols_binned = rt.RooArgSet(x)
            ds = rt.RooDataHist(ds_name, ds_name, cols_binned ,ds)
        return ds

    def generate_data(self, model_name, category, xSec, lumi):
        tag = f"_{self.channel}_{category}"
        model_key = model_name + tag
        if model_key not in self.model_registry:
            self.add_model(model_name, category=category)
        return self.workspace.pdf(model_key).generate(
            rt.RooArgSet(self.workspace.obj("mh_ggh")), xSec * lumi
        )

    def add_model(self, model_name, order=None, category="cat0", prefix=""):
        if model_name not in self.fitmodels.keys():
            raise Exception(f"Error: model {model_name} does not exist!")
        tag = f"_{self.channel}_{category}"
        if model_name == "BWxDCB":
            DCB_forZ, params1 = doubleCB_forZ(self.workspace.obj("mh_ggh"), tag)
            BW_forZ, params2 = bwZ(self.workspace.obj("mh_ggh"), tag)
            self.workspace.Import(DCB_forZ)
            self.workspace.Import(BW_forZ)
            self.workspace.obj("mh_ggh").setBins(200,"cache")
            self.workspace.obj("mh_ggh").setMin("cache",50.5) ;
            self.workspace.obj("mh_ggh").setMax("cache",130.5) ;
            model = rt.RooFFTConvPdf(f"{model_name}{tag}",f"{model_name}{tag}",self.workspace.obj("mh_ggh"), BW_forZ, DCB_forZ) 
            print(model)
        elif order is None:
            model, params = self.fitmodels[model_name](self.workspace.obj("mh_ggh"), tag)
            #print(model)
        else:
            if model_name in self.requires_order:
                model, params = self.fitmodels[model_name](
                    self.workspace.obj("mh_ggh"), tag, order
                )
            else:
                raise Exception(
                    f"Warning: model {model_name} does not require to specify order!"
                )

        model_key = model_name + tag
        if model_key not in self.model_registry:
            self.model_registry.append(model_key)
        self.workspace.Import(model,rt.RooFit.RenameConflictNodes("_pdf"))
        #getattr(self.workspace, "import")(model)

    def fit(
        self,
        ds_name,
        ndata,
        model_names,
        model_names_multi=[],
        orders={},
        doMultiPDF=False,
        doProdPDF=False,
        blinded=False,
        fix_parameters=False,
        save=True,
        save_path="./",
        category="cat0",
        label="",
        title="",
        norm=0,
    ):
        if ds_name not in self.data_registry.keys():
            raise Exception(f"Error: Dataset {ds_name} not in workspace!")

        pdfs = {}
        chi2 = {}
        tag = f"_{self.channel}_{category}"
        model_names_all = []
        if doProdPDF == False and fix_parameters == False and ("All" in category):
            print('doing fewz')
            FEWZ_file = rt.TFile("fits/NNLO_Bourilkov_2017.root", "READ")
        
      
            FEWZ_histo = FEWZ_file.Get("full_36fb")
            
            bernstein_add, params_fewzBern = bernstein(
                   self.workspace.obj("mh_ggh"), "fewz", 3
               )
            bernstein_only1, params_fewzBernonly1 = bernstein(
                   self.workspace.obj("mh_ggh"), "fewz", 6
               )
            bernstein_only2, params_fewzBernonly2 = bernstein(
                   self.workspace.obj("mh_ggh"), "fewz", 7
              )
            bernstein_only3, params_fewzBernonly3 = bernstein(
                self.workspace.obj("mh_ggh"), "fewz", 8
               )
            bernstein_only4, params_fewzBernonly4 = bernstein(
                   self.workspace.obj("mh_ggh"), "fewz", 9
               )
            bernstein_only5, params_fewzBernonly5 = bernstein(
                   self.workspace.obj("mh_ggh"), "fewz", 10
               )
            FEWZ_data = rt.RooDataHist("fewzdata","fewzdata",self.workspace.obj("mh_ggh"),FEWZ_histo)
            n_points = FEWZ_histo.GetNbinsX()
            x_vals = []
            y_vals = []
            z_vals = []
            n=0
            for i in range(n_points):
                if i<0 or i >=42:
                    continue
                n=n+1
                if (FEWZ_histo.GetBinCenter(i)) < 110.0:
                    x_vals.append(110.0)
                    y_vals.append(FEWZ_histo.GetBinContent(i+1)*1.05)  
                    continue
                if (FEWZ_histo.GetBinCenter(i)) >150:
                    x_vals.append(150.0)
                    y_vals.append(FEWZ_histo.GetBinContent(i)*0.95)  
                    continue
                x_vals.append(FEWZ_histo.GetBinCenter(i))
                
                y_vals.append(FEWZ_histo.GetBinContent(i))  
                #z_vals.append(i)
            #y_vals = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
            print(x_vals)
            print(y_vals)
            import ctypes
            x_vals_c_list = (ctypes.c_double * len(x_vals))(*x_vals)
            y_vals_c_list = (ctypes.c_double * len(y_vals))(*y_vals)
            #z_vals_c_list = (ctypes.c_double * len(z_vals))(*z_vals)
            #Create the RooSpline1D
            #import pdb
            ##pdb.setTrace()
            print('doing spline')
            spline = rt.RooSpline1D("spline_FEWZ_model", "Interpolated Spline FEWZ_model", self.workspace.obj("mh_ggh"), n, x_vals_c_list,y_vals_c_list, "CSPLINE")
            #self.workspace.Import(spline)
            c = rt.TCanvas("dgf","dgfdgfdgf",800,600)
            
            

            Fewzlist = rt.RooArgList()
            Fewzlist.add(spline)
            Fewzlist.add(bernstein_add)
            #pdb.setTrace()
            FEWZxBern_func = rt.RooProduct("FEWZxBern_func", "Spline times Bernstein Result", Fewzlist)
            FEWZxBern = rt.RooGenericPdf("FEWZxBern", "Spline * Bernstein PDF", "@0", rt.RooArgList(FEWZxBern_func))
            #FEWZxBern_func.plotOn(xframe2, DrawOption="L")
            print(FEWZxBern)            
            fewzpdfs =[]
            FEWZ_pdf_new = FEWZxBern.Clone(f"fewz{tag}")
            FEWZ_pdf_new2 = bernstein_only1.Clone(f"fewz2{tag}")
            
            FEWZ_pdf_new3 = bernstein_only2.Clone(f"fewz3{tag}")
            FEWZ_pdf_new4 = bernstein_only3.Clone(f"fewz4{tag}")
            FEWZ_pdf_new5 = bernstein_only4.Clone(f"fewz5{tag}")
            FEWZ_pdf_new6 = bernstein_only5.Clone(f"fewz6{tag}")
            self.workspace.Import(FEWZ_pdf_new)
            self.workspace.Import(FEWZ_pdf_new2)
            fewzpdfs.append(FEWZ_pdf_new2)
            self.workspace.Import(FEWZ_pdf_new3)
            fewzpdfs.append(FEWZ_pdf_new3)
            self.workspace.Import(FEWZ_pdf_new4)
            fewzpdfs.append(FEWZ_pdf_new4)
            self.workspace.Import(FEWZ_pdf_new5)
            fewzpdfs.append(FEWZ_pdf_new5)
            self.workspace.Import(FEWZ_pdf_new6)
            fewzpdfs.append(FEWZ_pdf_new6)
            for i in range(len(fewzpdfs)):
                fewzpdfs[i].fitTo(
                    FEWZ_data,
                    rt.RooFit.Save(),
                    rt.RooFit.PrintLevel(0),
                    #rt.RooFit.SumW2Error(0),
                    rt.RooFit.AsymptoticError(1),
                    #rt.RooFit.Minimizer("Minuit2","migrad"),
                    rt.RooFit.Verbose(rt.kFALSE),
                    )
                fewzpdfs[i].getParameters(rt.RooArgSet()).setAttribAll("Constant")
            #pdb.set_trace()
            normfewz = FEWZ_data.sumEntries()
            fewznorm_var = rt.RooRealVar(f"fewz2{tag}_norm", f"fewz2{tag}_norm", normfewz)
            #fewznorm_var = rt.RooRealVar(f"fewz{tag}_norm", f"fewz{tag}_norm", normfewz)
            print(f"Norm_{category} = {normfewz}")
            try:
                self.workspace.Import(fewznorm_var)
                #self.workspace.Import(fewznorm_var2)
            except Exception:
                print(f"{norm_var} already exists in workspace, skipping...")
            xframe2 = self.workspace.obj("mh_ggh").frame(Title="Polynomial order 7 (8 dof)")
            offset=0.5
            leg0 = rt.TLegend(0.15 + offset, 0.6, 0.5 + offset, 0.82)
            leg0.SetFillStyle(0)
            #leg0.SetLineColor(0)
            leg0.SetTextSize(0.03)

            FEWZ_data.plotOn(xframe2, rt.RooFit.Name("fewzdata"))
            #FEWZ_pdf_new.plotOn(xframe2,rt.RooFit.LineColor(rt.kRed),rt.RooFit.Name(FEWZ_pdf_new.GetName()),)
            #leg0.AddEntry(FEWZ_pdf_new, "#splitline{" + FEWZ_pdf_new.GetName() + "}{FEWZ_pdf_new}", "l")
            FEWZ_pdf_new2.plotOn(xframe2,rt.RooFit.LineColor(rt.kBlue),rt.RooFit.Name(FEWZ_pdf_new2.GetName()),rt.RooFit.RefreshNorm())
            FEWZ_pdf_new3.plotOn(xframe2,rt.RooFit.LineColor(rt.kRed),rt.RooFit.Name(FEWZ_pdf_new3.GetName()),rt.RooFit.RefreshNorm())
            FEWZ_pdf_new4.plotOn(xframe2,rt.RooFit.LineColor(rt.kGreen),rt.RooFit.Name(FEWZ_pdf_new4.GetName()),rt.RooFit.RefreshNorm())
            FEWZ_pdf_new5.plotOn(xframe2,rt.RooFit.LineColor(rt.kYellow),rt.RooFit.Name(FEWZ_pdf_new5.GetName()),rt.RooFit.RefreshNorm())
            FEWZ_pdf_new6.plotOn(xframe2,rt.RooFit.LineColor(rt.kOrange),rt.RooFit.Name(FEWZ_pdf_new6.GetName()),rt.RooFit.RefreshNorm())
            spline.plotOn(xframe2, rt.RooFit.LineColor(rt.kYellow),rt.RooFit.Name(spline.GetName()))
            leg0.AddEntry(FEWZ_pdf_new2, "Bernstein polynomial order 7", "l")
            #leg0.AddEntry(spline, "#splitline{" + spline.GetName() + "}{spline}", "l")
            xframe2.Draw()
            Chi2_1 = xframe2.chiSquare(f"fewz2{tag}", "fewzdata")
            
            print (f"Chi2 fewz spline: {Chi2_1}")#
            #Chi2_2 = {}
            
            #for i in range(len(fewzpdfs)):
                #Chi2_2 = xframe2.chiSquare(f"fewz{i+2}{tag}", "fewzdata")
                #print (f"Chi2 fewz bernstein{i+3}: {Chi2_2}")
           
            #print(Chi2_2)
            leg0.Draw("Same")
            c.SaveAs("order7.png")
        
            #model_names.append("fewz")
            model_names.append("fewz2")
            print("fewz")
            print(FEWZ_pdf_new)
            print("fewz done")
        for model_name in model_names:
            if (model_name in self.requires_order) and (model_name in orders.keys()):
                for order in orders[model_name]:
                    model_names_all.append(f"{model_name}{str(order)}")
            else:
                model_names_all.append(model_name)
                #print(model_name)
        if doProdPDF == True:
            WS_all_file = rt.TFile(f"fits/fits_{self.channel}_All/workspace_{self.channel}_All_background_all.root", "READ") 
            WS_all = WS_all_file.Get("w")
            data_all = WS_all.data("data_background_all")
            #print(data_all)
            ##print(self.workspace.obj(ds_name))
            hist_data_all = rt.TH1F("hist1", "Histogram for all data", 80, 110, 150)
            hist_data_cat = rt.TH1F("hist2", "Histogram for cat data", 80, 110, 150)

            data_all.fillHistogram(hist_data_all, rt.RooArgList(WS_all.obj("mh_ggh")))
            self.workspace.obj(ds_name).fillHistogram(hist_data_cat, rt.RooArgList(self.workspace.obj("mh_ggh")))
            
            # Calculate the bin-by-bin ratio
            ratio_hist = hist_data_cat.Clone("ratioHist")
            ratio_hist.Divide(hist_data_all)

            ratio_hist_for_fit = rt.RooDataHist("ratio_hist_for_fit", "Data ratio_hist_for_fit", rt.RooArgList(self.workspace.obj("mh_ggh")), ratio_hist)
            
            multipdfs={}
            ProdPDFs = []
            prod_model_names_all = []
            #model_names_multi.append("fewz")
            model_names_multi.append("fewz2")
            #ds_name = "ratio_hist_for_fit"
            self.workspace.Import(ratio_hist_for_fit)
            #for model_name in model_names_multi:
                #multipdfs[model_name] = WS_all.pdf(f"{model_name}_ggh_All")
                #print(multipdfs[model_name])
            

            

                #for model_name_poly in model_names_all:
                    #print(model_name_poly)
                    #prodpdflist=rt.RooArgList()
                    #model_key = model_name_poly + tag
                    #
                    #self.add_model("chebyshev",order=1,category=category+model_name)
                    
                    #pdf = self.workspace.pdf(model_key+model_name)
                    #prodpdflist.add(pdf)
                    #self.workspace.Import(pdf_for_fit,rt.RooFit.RenameConflictNodes(f"_for{model_name}"))
                    #print(pdf)
                    #prodpdflist.add(multipdfs[model_name])
                    #ProdPDF = rt.RooProdPdf(f"ProdPDF{model_name}{model_name_poly}{tag}", f"ProdPDF {model_name}{model_name_poly}{tag}", prodpdflist)
                    #prod_model_names_all.append(ProdPDF)
                    #print("ProdPDF created")
                    #ProdPDFs.append(f"ProdPDF{model_name}{model_name_poly}")
                    #print("ProdPDF appended")
                    #self.workspace.Import(ProdPDF,rt.RooFit.RenameConflictNodes(f"_for{model_name}"))
            #model_names_all = ProdPDFs
        print(model_names_all)
        for model_name in model_names_all:
            model_key = model_name + tag
            pdfs[model_key] = self.workspace.pdf(model_key)
            print(model_key)
            #print(pdfs[model_key])
            if doProdPDF == True:
                pdfs[model_key].fitTo(
                    self.workspace.obj("ratio_hist_for_fit"),
                    rt.RooFit.Save(),
                    rt.RooFit.PrintLevel(0),
                    rt.RooFit.AsymptoticError(1),
                    rt.RooFit.Verbose(rt.kFALSE),
                )
            else:
                pdfs[model_key].fitTo(
                    self.workspace.obj(ds_name),
                    rt.RooFit.Save(),
                    rt.RooFit.PrintLevel(-1),
                    #rt.RooFit.AsymptoticError(1),
                    rt.RooFit.PrefitDataFraction(0.01),
                    #rt.RooFit.Minimizer("Minuit2","minimize"),
                    rt.RooFit.Verbose(rt.kFALSE),
                )
            print(pdfs[model_key])
            
            #if fix_parameters:
                #pdfs[model_key].getParameters(rt.RooArgSet()).setAttribAll("Constant")
                #a1 = self.workspace.var(f"alpha1{tag}")
                #a2 = self.workspace.var(f"alpha2{tag}")
                #n1 = self.workspace.var(f"n1{tag}")
                #n2 = self.workspace.var(f"n2{tag}")
                #mean = self.workspace.var(f"mean{tag}")
                #sigma = self.workspace.var(f"sigma{tag}")
                #mean.Print()
                #sigma.Print()
                #a1.Print()
                #a2.Print()
                #n1.Print()
                #n2.Print()
            chi2[model_key] = self.get_chi2(model_key, ds_name, ndata)
            print(f"Chi2 = {chi2[model_key]}")
            pdfs_to_plot = {}
            polypdfs = {}
            if doProdPDF == True:
                for model_name in model_names:
                    for order in orders[model_name]:
                        for model_name_multi in model_names_multi:
                            print(model_name_multi)
                            multipdfs[model_name_multi] = WS_all.pdf(f"{model_name_multi}_ggh_All")
                            #print(multipdfs[model_name_multi])
            

            

                
                            print(model_name)
                        
                            prodpdflist=rt.RooArgList()
                            model_key = model_name +str(order) + tag
                    
                            #self.add_model("chebyshev",order=1,category=category+model_name)
                    
                            polypdf = pdfs[model_key]
                            print(polypdf)
                            polypdfs[model_key] = polypdf
                            prodpdflist.add(multipdfs[model_name_multi])
                            prodpdflist.add(polypdf)
                            #self.workspace.Import(pdf_for_fit,rt.RooFit.RenameConflictNodes(f"_for{model_name}"))
                            #print(pdf)
                            
                            ProdPDF = rt.RooProdPdf(f"ProdPDF{model_name_multi}{model_name}{str(order)}{tag}", f"ProdPDF {model_name_multi}{tag}", prodpdflist)
                            self.workspace.Import(ProdPDF, rt.RooFit.RenameConflictNodes(f"_for{model_name_multi}"))
                        #prod_model_names_all.append(ProdPDF)
                            print(ProdPDF)
                            print("ProdPDF created")
                            ProdPDFs.append(f"ProdPDF{model_name_multi}{model_name}{str(order)}{tag}")
                            pdfs[f"ProdPDF{model_name_multi}{model_name}{str(order)}{tag}"] = ProdPDF
                            pdfs_to_plot[f"ProdPDF{model_name_multi}{model_name}{str(order)}{tag}"] = ProdPDF
                            print("ProdPDF appended")
                            norm_var = rt.RooRealVar(f"ProdPDF{model_name_multi}{model_name}{str(order)}{tag}_norm", f"ProdPDF{model_name_multi}{model_name}{str(order)}{tag}_norm", norm)
                            try:
                                self.workspace.Import(norm_var)
                                getattr(self.workspace, "import")(norm_var)
                            except Exception:
                                print(f"{norm_var} already exists in workspace, skipping...")
      
                            chi2[f"ProdPDF{model_name_multi}{model_name}{str(order)}{tag}"] = self.get_chi2(f"ProdPDF{model_name_multi}{model_name}{str(order)}{tag}", ds_name, ndata)
                            print(chi2[f"ProdPDF{model_name_multi}{model_name}{str(order)}{tag}"])
                        model_names_all = ProdPDFs
                        print(model_names_all)
  


                            
                    #for order in orders[model_name]:
                       #with open(f'chi2{tag}{model_name}{order}.txt', 'w') as f:
                            #print(chi2, file=f)
            if doProdPDF == False:           
                norm_var = rt.RooRealVar(f"{model_key}_norm", f"{model_key}_norm", norm)
                print(f"Norm_{category} = {norm}")
                try:
                    self.workspace.Import(norm_var)
                    getattr(self.workspace, "import")(norm_var)
                except Exception:
                    print(f"{norm_var} already exists in workspace, skipping...")


        if save:
            mkdir(save_path)
            if doProdPDF == False:
                #pdfs["fewz"] = FEWZ_pdf_new
                
                plot(self, ds_name, pdfs, blinded, category, label, title, save_path)
            if doProdPDF:
                plot(self, "ratio_hist_for_fit", polypdfs, blinded, category, "ShapeMod", "ShapeModifierFunction", save_path)
                plot(self, ds_name, pdfs_to_plot, blinded, category, label, title, save_path)
        if doMultiPDF==True:
            cat = rt.RooCategory(
                f"pdf_index",
                "index of the active pdf",
            )
            pdflist = rt.RooArgList()
            for model_name in model_names_all:
                if doProdPDF:
                    tag = ""
                model_key = model_name + tag
                #print(f"adding {model_name} to Multipdf")
                pdflist.add( pdfs_to_plot[model_key])
                norm_var = rt.RooRealVar(f"{model_name}__for_multi_norm", f"ProdPDF{model_name}__for_multi_norm", norm)
                try:
                    self.workspace.Import(norm_var)
                    getattr(self.workspace, "import")(norm_var)
                except Exception:
                    print(f"{norm_var} already exists in workspace, skipping...")
            #print(pdflist)
            multipdf = rt.RooMultiPdf(
                f"multipdf_{self.channel}_{category}", "multipdf", cat, pdflist
            )
            norm_var = rt.RooRealVar(f"multipdf_{self.channel}_{category}_norm",f"multipdf_{self.channel}_{category}_norm", norm)
            try:
                self.workspace.Import(norm_var)
                getattr(self.workspace, "import")(norm_var)
            except Exception:
                print(f"{norm_var} already exists in workspace, skipping...")

            #multipdf.fitTo(
            #    self.workspace.obj(ds_name),
            #    rt.RooFit.Save(),
            #    rt.RooFit.PrintLevel(-1),
            #    rt.RooFit.Verbose(rt.kFALSE),
            #)
            #multipdfs= {}
            #multipdfs["multipdf"] = multipdf
            #plot(self, ds_name, multipdfs, blinded, category, f"{label}Multipdf", title, save_path)
            #self.add_model("multipdf", category=category)
            getattr(self.workspace, "import")(cat)
            self.workspace.Import(multipdf, rt.RooFit.RenameConflictNodes("_for_multi"))


        #print(pdfs[model_key])
        print(f'Chi2 = {chi2}')
        
        return chi2

    def get_chi2(self, model_key, ds_name, ndata):
        normalization = rt.RooRealVar(
            "normaliazation", "normalization", ndata, 0.5 * ndata, 2 * ndata
        )
        model = rt.RooExtendPdf(
            "ext", "ext", self.workspace.pdf(model_key), normalization
        )
        #print(f'In GetChi2 {model_key}')
        xframe = self.workspace.obj("mh_ggh").frame()
        ds = self.workspace.data(ds_name)
        ds.plotOn(xframe, rt.RooFit.Name(ds_name))
        model.plotOn(xframe, rt.RooFit.Name(model_key))
        nparam = model.getParameters(ds).getSize()
        #print(nparam)
        chi2 = xframe.chiSquare(model_key, ds_name, nparam)
        if chi2 <= 0:
            chi2 == 999
        return chi2
