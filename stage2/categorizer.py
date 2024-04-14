def split_into_channels(df, v="", nochannels=False, ggHsplit = True):
    df.loc[df[f"njets_{v}"] ==-99, f"njets_{v}"] = 0.0
    columns_print = [f"njets_{v}"]
    with open("dfsplitintoc.txt", "w") as f:
        print(df[columns_print], file=f)
    print(df)
    df.loc[:, f"channel_{v}"] = "none"
    if nochannels == False:
        df.loc[
        (df[f"nBtagLoose_{v}"] >= 2) | (df[f"nBtagMedium_{v}"] >= 1), f"channel_{v}"
        ] = "ttHorVH"

        df.loc[
            (df[f"channel_{v}"] == "none")
            & (df[f"jj_mass_{v}"] > 400)
            & (df[f"jj_dEta_{v}"] > 2.5)
            & (df[f"jet1_pt_{v}"] > 35),
            f"channel_{v}",
        ] = "vbf"
        if ggHsplit ==True:
            df.loc[
                (df[f"channel_{v}"] == "none") & (df[f"njets_{v}"] < 1), f"channel_{v}"
            ] = "ggh_0jets"
            df.loc[
                (df[f"channel_{v}"] == "none") & (df[f"njets_{v}"] == 1.0), f"channel_{v}"
            ] = "ggh_1jet"
            df.loc[
                (df[f"channel_{v}"] == "none") & (df[f"njets_{v}"] > 1), f"channel_{v}"
            ] = "ggh_2orMoreJets"
        else:
            #print("-------------")
            #print("not splitting ggH")
            #print("-------------")
            df.loc[
                (df[f"channel_{v}"] == "none"), f"channel_{v}"
            ] = "ggh"

def categorize_by_score(df, scores, mode="uniform",year="2018", **kwargs):
    nbins = kwargs.pop("nbins", 4)
    for channel, score_names in scores.items():
        for score_name in score_names:
            score_name = f"{score_name}_{year}"
            score = df.loc[df.channel_nominal == channel, f"score_{score_name}_nominal"]
            score = df[f"score_{score_name}_nominal"]
            if mode == "uniform":
                for i in range(nbins):
                    cat_name = f"{score_name}_cat{i}"
                    cut_lo = score.quantile(i / nbins)
                    cut_hi = score.quantile((i + 1) / nbins)
                    cut = (df.channel == channel) & (score > cut_lo) & (score < cut_hi)
                    df.loc[cut, "category"] = cat_name
            if mode == "fixed_ggh":
                #df["category"] = "cat0"
                #score_ggh = df[df.loc[(df.channel_nominal == channel) & (df.dataset == "ggh_powheg"), f"score_{score_name}_nominal"]]
                signal_eff_bins = {}
                signal_eff_bins["2018"] =  [0.0,
                                            0.31099143624305725, 
                                            0.4707504212856293, 
                                            0.5497796535491943, 
                                            0.6779066324234009,
                                            1]#ggHnew
                
                                            #[0.0, 
                                           #0.578787624835968, 
                                           #0.7073342800140381, 
                                           #0.7585148215293884, 
                                           #0.8222802877426147,
                                           #1]#BDTv12
                
                                           #[0.0, 
                                           #0.34052810072898865, 
                                           #0.6200955510139465, 
                                           #0.7057817578315735, 
                                           #0.8094207644462585, 
                                           #1]#BDTperyear

                     
                signal_eff_bins["2017"] = [0.0, 
                                           0.33528196811676025, 
                                           0.5249117016792297, 
                                           0.6191316843032837, 
                                           0.7578054666519165, 
                                           1]
                signal_eff_bins["2016postVFP"] = [0.0, 
                                                  0.46456924080848694, 
                                                  0.611092209815979, 
                                                  0.6676164269447327, 
                                                  0.7403154969215393, 
                                                  1]
                signal_eff_bins["2016preVFP"] = [0.0, 
                                                 0.4651411175727844, 
                                                 0.6060909628868103, 
                                                 0.6604294180870056, 
                                                 0.7282804846763611, 
                                                 1]
                #signal_eff_bins = [0,1]
                
                #print("Score all:")
                #print(score.dropna())
                #print("Score signal")
                #print(score_ggh.dropna())
                print(f"doing categorisation for {year}")
                print(f"doing categorisation for score_{score_name}_nominal")
                print(signal_eff_bins[year])
                for i in range(len(signal_eff_bins[year])-1):
                    cut_lo = signal_eff_bins[year][i]
                    cut_hi = signal_eff_bins[year][i + 1]
                    #print(f"Cut hi = {cut_hi}")
                    cat_name = f"{score_name}_cat{i}"
                    cut = (df.channel_nominal == channel) & (score > cut_lo) & (score <= cut_hi)
                    df.loc[cut, "category"] = cat_name
def categorize_by_eta(df , **kwargs):
    nbins = kwargs.pop("nbins", 4)

    print(f"doing categorisation for score_eta")
    etacats = [0.0,0.8,1.4,2.4]
    for i in range(len(etacats)-1):
        cut_lo = etacats[i]
        cut_hi = etacats[i + 1]
        cat_name = f"EtaCats_cat{i}"
        cut = (df.channel_nominal == "ggh") & (df.mu1_eta > cut_lo) & (df.mu1_eta <= cut_hi) & (df.mu2_eta > cut_lo) & (df.mu2_eta <= cut_hi)
        
        df.loc[cut, "category"] = cat_name


def categorize_by_CalibCat(df , **kwargs):
    nbins = kwargs.pop("nbins", 4)

    print(f"doing categorisation for calibration")
    BB = ((abs(df["mu1_eta"])<=0.9) & (abs(df["mu2_eta"])<=0.9))
    BO = ((abs(df["mu1_eta"])<=0.9) & ((abs(df["mu2_eta"])>0.9) & (abs(df["mu2_eta"]) <=1.8)))
    BE = ((abs(df["mu1_eta"])<=0.9) & ((abs(df["mu2_eta"])>1.8) & (abs(df["mu2_eta"]) <=2.4)))
    OB = (((abs(df["mu1_eta"])>0.9) & (abs(df["mu1_eta"]) <=1.8)) & (abs(df["mu2_eta"])<=0.9))
    OO = (((abs(df["mu1_eta"])>0.9) & (abs(df["mu1_eta"]) <=1.8)) & ((abs(df["mu2_eta"])>0.9) & (abs(df["mu2_eta"]) <=1.8)))
    OE = (((abs(df["mu1_eta"])>0.9) & (abs(df["mu1_eta"]) <=1.8)) & ((abs(df["mu2_eta"])>1.8) & (abs(df["mu2_eta"]) <=2.4)))
    EB = (((abs(df["mu1_eta"])>1.8) & (abs(df["mu1_eta"]) <=2.4)) & (abs(df["mu2_eta"])<=0.9))
    EO = (((abs(df["mu1_eta"])>1.8) & (abs(df["mu1_eta"]) <=2.4)) & ((abs(df["mu2_eta"])>0.9) & (abs(df["mu2_eta"]) <=1.8)))
    EE = (((abs(df["mu1_eta"])>1.8) & (abs(df["mu1_eta"]) <=2.4)) & ((abs(df["mu2_eta"])>1.8) & (abs(df["mu2_eta"]) <=2.4)))
    selections = [((df["mu1_pt"]>30)&(df["mu1_pt"]<=45)&(BB | OB | EB)),
                                  ((df["mu1_pt"]>30)&(df["mu1_pt"]<=45)&(BO | OO | EO)),
                                  ((df["mu1_pt"]>30)&(df["mu1_pt"]<=45)&(BE | OE | EE)),
                                  ((df["mu1_pt"]>45)&(df["mu1_pt"]<=52)&BB),
                                  ((df["mu1_pt"]>45)&(df["mu1_pt"]<=52)&BO),
                                  ((df["mu1_pt"]>45)&(df["mu1_pt"]<=52)&BE),
                                  ((df["mu1_pt"]>45)&(df["mu1_pt"]<=52)&OB),
                                  ((df["mu1_pt"]>45)&(df["mu1_pt"]<=52)&OO),
                                  ((df["mu1_pt"]>45)&(df["mu1_pt"]<=52)&OE),
                                  ((df["mu1_pt"]>45)&(df["mu1_pt"]<=52)&EB),
                                  ((df["mu1_pt"]>45)&(df["mu1_pt"]<=52)&EO),
                                  ((df["mu1_pt"]>45)&(df["mu1_pt"]<=52)&EE),
                                  ((df["mu1_pt"]>52)&(df["mu1_pt"]<=62)&BB),
                                  ((df["mu1_pt"]>52)&(df["mu1_pt"]<=62)&BO),
                                  ((df["mu1_pt"]>52)&(df["mu1_pt"]<=62)&BE),
                                  ((df["mu1_pt"]>52)&(df["mu1_pt"]<=62)&OB),
                                  ((df["mu1_pt"]>52)&(df["mu1_pt"]<=62)&OO),
                                  ((df["mu1_pt"]>52)&(df["mu1_pt"]<=62)&OE),
                                  ((df["mu1_pt"]>52)&(df["mu1_pt"]<=62)&EB),
                                  ((df["mu1_pt"]>52)&(df["mu1_pt"]<=62)&EO),
                                  ((df["mu1_pt"]>52)&(df["mu1_pt"]<=62)&EE),
                                  ((df["mu1_pt"]>62)&(df["mu1_pt"]<=200)&BB),
                                  ((df["mu1_pt"]>62)&(df["mu1_pt"]<=200)&BO),
                                  ((df["mu1_pt"]>62)&(df["mu1_pt"]<=200)&BE),
                                  ((df["mu1_pt"]>62)&(df["mu1_pt"]<=200)&OB),
                                  ((df["mu1_pt"]>62)&(df["mu1_pt"]<=200)&OO),
                                  ((df["mu1_pt"]>62)&(df["mu1_pt"]<=200)&OE),
                                  ((df["mu1_pt"]>62)&(df["mu1_pt"]<=200)&EB),
                                  ((df["mu1_pt"]>62)&(df["mu1_pt"]<=200)&EO),
                                  ((df["mu1_pt"]>62)&(df["mu1_pt"]<=200)&EE),]
    for i in range(len(selections)):

        cat_name = f"Calibration_cat{i}"
        df.loc[selections[i], "category"] = cat_name
        #print(df.loc[selections[i]])

def categorize_by_ClosureCat(df , **kwargs):
    nbins = kwargs.pop("nbins", 4)

    print(f"doing categorisation for closure test")
    selections = [((df["dimuon_ebe_mass_res"]>0.6)&(df["dimuon_ebe_mass_res"]<=0.7)),
                                  ((df["dimuon_ebe_mass_res"]>0.7)&(df["dimuon_ebe_mass_res"]<=0.8)),
                                  ((df["dimuon_ebe_mass_res"]>0.8)&(df["dimuon_ebe_mass_res"]<=0.9)),
                                  ((df["dimuon_ebe_mass_res"]>0.9)&(df["dimuon_ebe_mass_res"]<=1.0)),
                                  ((df["dimuon_ebe_mass_res"]>1.0)&(df["dimuon_ebe_mass_res"]<=1.1)),
                                  ((df["dimuon_ebe_mass_res"]>1.1)&(df["dimuon_ebe_mass_res"]<=1.2)),
                                  ((df["dimuon_ebe_mass_res"]>1.3)&(df["dimuon_ebe_mass_res"]<=1.4)),
                                  ((df["dimuon_ebe_mass_res"]>1.4)&(df["dimuon_ebe_mass_res"]<=1.5)),
                                  ((df["dimuon_ebe_mass_res"]>1.5)&(df["dimuon_ebe_mass_res"]<=1.7)),
                                  ((df["dimuon_ebe_mass_res"]>1.7)&(df["dimuon_ebe_mass_res"]<=2.0)),
                                  ((df["dimuon_ebe_mass_res"]>2.0)&(df["dimuon_ebe_mass_res"]<=2.5)),
                                  ((df["dimuon_ebe_mass_res"]>2.5)&(df["dimuon_ebe_mass_res"]<=3.5)),]
    for i in range(len(selections)):

        cat_name = f"Closure_cat{i}"
        df.loc[selections[i], "category"] = cat_name
        #print(df.loc[selections[i]])
                
                
def categorize_dnn_output(df, score_name, channel, region, year, yearstr):
    # Run 2 (VBF yields)
    target_yields = {
        "2016": [
            0.35455259,
            0.50239086,
            0.51152889,
            0.52135985,
            0.5282209,
            0.54285134,
            0.54923751,
            0.56504687,
            0.57204477,
            0.58273066,
            0.5862248,
            0.59568793,
            0.60871905,
        ],
                "2016postVFP": [
            0.35455259,
            0.50239086,
            0.51152889,
            0.52135985,
            0.5282209,
            0.54285134,
            0.54923751,
            0.56504687,
            0.57204477,
            0.58273066,
            0.5862248,
            0.59568793,
            0.60871905,
        ],
        "2017": [
            0.44194544,
            0.55885064,
            0.57796123,
            0.58007343,
            0.58141001,
            0.58144682,
            0.57858609,
            0.59887533,
            0.59511901,
            0.59644831,
            0.59825915,
            0.59283309,
            0.57329743,
        ],
        "2018": [
            0.22036263,
            1.31808978,
            1.25396849,
            1.183724,
            1.12620194,
            1.06041376,
            0.99941623,
            0.93224412,
            0.87074753,
            0.80599462,
            0.73469265,
            0.668018,
            0.60991945,
        ],
    }

    bins = [df[score_name].max()]
    #print(bins)
    #print(channel)
    print(region)
    #print(year)
    
    slicer = (
        (df.channel_nominal == channel) & ((df.region == "h-peak") or (df.region == "h-sidebands")) & (df.year == yearstr)
    )
    df_sorted = (
        df.loc[slicer, :]
        .sort_values(by=score_name, ascending=False)
        .reset_index(drop=True)
    )
    #print(df.loc[slicer, :][score_name])
    df_sorted["wgt_cumsum"] = df_sorted.wgt_nominal.cumsum()
    #print(df_sorted)
    tot_yield = 0
    last_yield = 0
    #print(target_yields)
    for yi in reversed(target_yields[year]):
        tot_yield += yi
        #print(yi)
        for i in range(df_sorted.shape[0] - 1):
            value = df_sorted.loc[i, "wgt_cumsum"]
            value1 = df_sorted.loc[i + 1, "wgt_cumsum"]
            #print(value)
            #print(value1)
            if (value < tot_yield) & (value1 > tot_yield):
                if abs(last_yield - tot_yield) < 1e-06:
                    continue
                if abs(tot_yield - sum(target_yields[year])) < 1e-06:
                    continue
                bins.append(df_sorted.loc[i, score_name])
                last_yield = tot_yield
    bins.append(0.0)
    bins = sorted(bins)
    #print(bins)
def categorize_dnn_output_ggh(df, score_name, channel, region, year, yearstr):
    # Run 2 (ggh yields)
    target_yields = [0.05,0.2,0.35,0.7,1]
    target_yields = [0.3,0.65,0.8,0.95,1]
    #target_yields = [1.0,0.95,0.8,0.65,0.3,0]

    bins = [df[score_name].max()]
    #print(bins)
    #print(channel)
    #print(region)
    #print(year)
    
    slicer = (
        (df.channel_nominal == channel) & (df.year == yearstr)
    )
    df_sorted = (
        df.loc[slicer, :]
        .sort_values(by=score_name, ascending=True)
        .reset_index(drop=True)
    )
    #print(df.loc[slicer, :][score_name])
    df_sorted["wgt_cumsum_normed"] = df_sorted.wgt_nominal.cumsum()/df_sorted.wgt_nominal.sum()
    #print(df_sorted)
    tot_yield = 0
    last_yield = 0
    #print(target_yields)
    for yi in (target_yields):
        for i in range(df_sorted.shape[0] - 1):
            value = df_sorted.loc[i, "wgt_cumsum_normed"]
            value1 = df_sorted.loc[i + 1, "wgt_cumsum_normed"]
            #print(value)
            #print(value1)
            if (value < yi) & (value1 >= yi):
                #if abs(last_yield - tot_yield) < 1e-06:
                    #continue
                #if abs(tot_yield - sum(target_yields) < 1e-06:
                    #continue
                bins.append(df_sorted.loc[i, score_name])
                #last_yield = tot_yield
    bins.append(0.0)
    bins = sorted(bins)
    print(bins)
    for i in range(len(bins)-1):
        cut_lo = bins[i]
        cut_hi = bins[i + 1]
        #print(f"Cut hi = {cut_hi}")
        cat_name = f"{score_name}_cat{i}"
        cut = (df_sorted.channel_nominal == channel) & (df_sorted[score_name] > cut_lo) & (df_sorted[score_name] <= cut_hi)
        df_sorted.loc[cut, "category"] = cat_name
    #print(df_sorted["category"])
    #for i in range(5):
        #df_test = df_sorted[df_sorted["category"] == f"{score_name}_cat{i}"]
        #print(df_test[score_name])
        #print(df_test["wgt_cumsum_normed"])