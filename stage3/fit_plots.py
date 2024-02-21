import ROOT as rt

colors = [
    rt.kRed,
    rt.kGreen,
    rt.kYellow,
    rt.kBlue,
    rt.kViolet,
    rt.kGray,
    rt.kOrange,
    rt.kPink,
    rt.kMagenta,
    rt.kAzure,
    rt.kCyan,
    rt.kTeal,
    rt.kSpring,
    rt.kRed + 1,
    rt.kGreen + 1,
    rt.kBlue + 1,
    rt.kYellow + 1,
    rt.kViolet + 1,
    rt.kGray + 1,
    rt.kOrange + 1,
    rt.kPink + 1,
    rt.kMagenta + 1,
]


def plotter(ws, objNames, isBlinded, channel, category, OutputFilename, title):
    c = rt.TCanvas("c_cat" + category, "c_cat" + category, 800, 800)
    xframe = ws.obj("mh_ggh").frame(rt.RooFit.Title(title + " in cat" + category))
    count = 0
    for name in objNames:
        if "model" in name:
            print(name)
            ws.pdf(name).plotOn(
                xframe,
                rt.RooFit.Range("window"),
                rt.RooFit.NormRange("window"),
                rt.RooFit.LineColor(colors[count]),
                rt.RooFit.Name(name),
            )
            count += 1
            # ws.pdf(name).plotOn(xframe,rt.RooFit.Range("window"))
        elif "ds_fake" in name:
            ws.obj(name).plotOn(xframe, rt.RooFit.Binning(80))
        elif "ds" in name:
            ws.data(name).plotOn(xframe, rt.RooFit.Binning(80))
    xframe.Draw()
    c.Update()
    c.SaveAs(OutputFilename + "_cat" + category + ".root")
    c.SaveAs(OutputFilename + "_cat" + category + ".pdf")
    c.SaveAs(OutputFilename + "_cat" + category + ".png")
    c.SaveAs(OutputFilename + "_cat" + category + ".C")


def plot(fitter, ds_name, models, blinded, category, label, title, save_path):
    ws = fitter.workspace
    c = rt.TCanvas("c_cat" + category, "c_cat" + category, 800, 800)
    offset = 0.5
    upper_pad = rt.TPad("upper_pad", "upper_pad", 0, 0.25, 1, 1)
    lower_pad = rt.TPad("lower_pad", "lower_pad", 0, 0, 1, 0.35)
    upper_pad.SetBottomMargin(0.14)
    lower_pad.SetTopMargin(0.00001)
    lower_pad.SetBottomMargin(0.25)
    upper_pad.Draw()
    lower_pad.Draw()
    upper_pad.cd()
    #print('Hallo')
    mh_ggh = ws.obj("mh_ggh")
    #print('Hallo')
    xframe = mh_ggh.frame(rt.RooFit.Title(title + " Fit in cat" + category))
    # dataset.plotOn(xframe,rt.RooFit.CutRange("sideband_left"))
    # dataset.plotOn(xframe,rt.RooFit.CutRange("sideband_right"))

    ws.obj(ds_name).plotOn(xframe, rt.RooFit.Binning(80))
    #print('Hallo')
    leg0 = rt.TLegend(0.15 + offset, 0.6, 0.5 + offset, 0.82)
    leg0.SetFillStyle(0)
    #leg0.SetLineColor(0)
    leg0.SetTextSize(0.03)
    #leg0.AddEntry(h_data,"Data","lep")
    if blinded:
        count = 0
        for model_key, model in models.items():
            model.plotOn(
                xframe,
                rt.RooFit.Range("window"),
                rt.RooFit.NormRange("sideband_left,sideband_right"),
                rt.RooFit.LineColor(colors[count]),
                rt.RooFit.Name(model.GetName()),
            )
            leg0.AddEntry(model, "#splitline{" + model.GetName() + "}{model}", "l")
            count += 1
    else:
        count = 0
        #print('Hallo before plot')
        for model_key, model in models.items():
            model.plotOn(
                xframe,
                rt.RooFit.Range("window"),
                # rt.RooFit.NormRange("window"),
                rt.RooFit.LineColor(colors[count]),
                rt.RooFit.Name(model.GetName()),
            )
            #leg0.AddEntry(model, "#line{" + model.GetName() + "}{model}", "l")
            leg0.AddEntry(model, model.GetName() + "model", "l")
            count += 1
    # upper_pad = rt.TPad("up_cat"+category,"up_cat"+category,0.0,0.2,1.0,1.0,21)
    # lower_pad = rt.TPad("lp_cat"+category,"lp_cat"+category,0.0,0.0,1.0,0.2,22)
    xframe.SetMinimum(0.0001)
    xframe.Draw()
    #print('Hallo before ggh')
    if ("ggh_powheg" in label) or ("Z" in label)  :
        print("Fitting ggH signal")
        # Add TLatex to plot
        for model_key, model in models.items():
            
            h_pdf = model.createHistogram("h_pdf", mh_ggh, rt.RooFit.Binning(80))
        #print(h_pdf.GetMaximum())
        effSigma = getEffSigma(h_pdf)
        effSigma_low, effSigma_high = (
            h_pdf.GetMean() - effSigma,
            h_pdf.GetMean() + effSigma,
        )
        h_effSigma = h_pdf.Clone()
        h_effSigma.GetXaxis().SetRangeUser(effSigma_low, effSigma_high)
        h_data = mh_ggh.createHistogram("h_data", rt.RooFit.Binning(80))
        lat0 = rt.TLatex()
        lat0.SetTextFont(42)
        lat0.SetTextAlign(11)
        lat0.SetNDC()
        lat0.SetTextSize(0.045)
        lat0.DrawLatex(0.15, 0.92, "#bf{CMS} #it{Simulation}")
        lat0.DrawLatex(0.77, 0.92, "13 TeV")
        lat0.DrawLatex(0.16 + 0.02, 0.83, "H#rightarrow#mu#mu")
        leg0 = rt.TLegend(0.15 + offset, 0.6, 0.5 + offset, 0.82)
        leg0.SetFillStyle(0)
        leg0.SetLineColor(0)
        leg0.SetTextSize(0.03)
        leg0.AddEntry(h_data, "Simulation", "lep")
        #leg0.AddEntry(h_pdf, "#splitline{Double Crystal-Ball}{model}", "l")
        leg0.AddEntry(h_pdf, "DCB model", "l")
        leg0.Draw("Same")

        leg1 = rt.TLegend(0.17 + offset, 0.45, 0.4 + offset, 0.61)
        leg1.SetFillStyle(0)
        leg1.SetLineColor(0)
        leg1.SetTextSize(0.03)
        for model_key, model in models.items():
            print (model_key)
            #key = model_key.split("dcb_")[1]
            if "All" in model_key:
                key = "ggh_All"
            else:
                key = model_key.split("dcb_")[1]
            print(key)
            sigma = ws.var(f"sigma_{key}").getVal()
            sigmaErr = ws.var(f"sigma_{key}").getError()
            
        leg1.AddEntry(
            h_pdf, f"Sigma = {sigma}+-{sigmaErr} GeV", "l"
        )
        # leg1.AddEntry(h_pdf_splitByYear['2017'],"2017: #scale[0.8]{#sigma_{eff} = %1.2f GeV}"%getEffSigma(h_pdf_splitByYear['2017']),"l")
        # leg1.AddEntry(h_pdf_splitByYear['2018'],"2018: #scale[0.8]{#sigma_{eff} = %1.2f GeV}"%getEffSigma(h_pdf_splitByYear['2018']),"l")
        # leg1.Draw("Same")

        leg2 = rt.TLegend(0.07 + offset, 0.45, 0.5 + offset, 0.45)
        leg2.SetFillStyle(0)
        leg2.SetLineColor(0)
        leg2.SetTextSize(0.03)
        leg2.AddEntry(
            #h_effSigma,
            #"#sigma_{eff} = %1.2f GeV" % (0.5 * (effSigma_high - effSigma_low)),
            #"fl",
             h_pdf, f"#sigma = {sigma:.3f}+-{sigmaErr:.3f} GeV", "p"
        )
        leg2.Draw("Same")
        h_effSigma.SetLineColor(15)
        h_effSigma.SetFillStyle(1001)
        h_effSigma.SetFillColor(15)
        h_effSigma.Draw("Same Hist F")
        vline_effSigma_low = rt.TLine(
            effSigma_low,
            0,
            effSigma_low,
            h_pdf.GetBinContent(h_pdf.FindBin(effSigma_low)),
        )
        vline_effSigma_high = rt.TLine(
            effSigma_high,
            0,
            effSigma_high,
            h_pdf.GetBinContent(h_pdf.FindBin(effSigma_high)),
        )
        vline_effSigma_low.SetLineColor(15)
        vline_effSigma_high.SetLineColor(15)
        vline_effSigma_low.SetLineWidth(2)
        vline_effSigma_high.SetLineWidth(2)
        vline_effSigma_low.Draw("Same")
        vline_effSigma_high.Draw("Same")
        fwhm_low = h_pdf.GetBinCenter(h_pdf.FindFirstBinAbove(0.5 * h_pdf.GetMaximum()))
        fwhm_high = h_pdf.GetBinCenter(h_pdf.FindLastBinAbove(0.5 * h_pdf.GetMaximum()))
        fwhmArrow = rt.TArrow(
            fwhm_low,
            0.5 * h_pdf.GetMaximum(),
            fwhm_high,
            0.5 * h_pdf.GetMaximum(),
            0.02,
            "<>",
        )
        fwhmArrow.SetLineWidth(2)
        fwhmArrow.Draw("Same <>")
        fwhmText = rt.TLatex()
        fwhmText.SetTextFont(42)
        fwhmText.SetTextAlign(11)
        fwhmText.SetNDC()
        fwhmText.SetTextSize(0.03)
        fwhmText.DrawLatex(
            0.20 + offset, 0.38, "FWHM = %1.2f GeV" % (fwhm_high - fwhm_low)
        )
    else:
        

        lat0 = rt.TLatex()
        lat0.SetTextFont(42)
        lat0.SetTextAlign(11)
        lat0.SetNDC()
        lat0.SetTextSize(0.045)
        lat0.DrawLatex(0.15, 0.92, "#bf{CMS} #it{Run2}")
        lat0.DrawLatex(0.77, 0.92, "13 TeV")
        lat0.DrawLatex(0.16 + 0.02, 0.83, "H#rightarrow#mu#mu")
        #leg0.AddEntry(h_data, "Data", "lep")
        #leg0.AddEntry(h_pdf, model_key, "l")
        leg0.Draw("Same")

    hpull = xframe.pullHist()
    lower_pad.cd()
    #print('Hallo before frame')
    xframe2 = mh_ggh.frame()
    xframe2.SetTitle("")
    xframe2.addPlotable(hpull, "P")
    xframe2.GetYaxis().SetTitle("Pull")
    xframe2.GetYaxis().SetRangeUser(-5, 8)
    if blinded:
        xframe2.GetYaxis().SetRangeUser(-4, 4)
    xframe2.GetYaxis().SetTitleOffset(0.3)
    xframe2.GetYaxis().SetTitleSize(0.08)
    xframe2.GetYaxis().SetLabelSize(0.08)
    xframe2.GetXaxis().SetLabelSize(0.08)
    xframe2.GetXaxis().SetTitle("m_{#mu#mu} (GeV)")
    xframe2.GetXaxis().SetTitleSize(0.08)
    line = rt.TLine(0.1, 0.5, 0.9, 0.5)
    line.SetNDC(rt.kTRUE)
    line.SetLineWidth(2)
    line.Draw()
    xframe2.Draw()
    c.Modified()
    c.Update()
    #print('Hallo')
    out_name = (
        f"{save_path}/fit_{label}_{fitter.channel}_{category}{fitter.filename_ext}"
    )
    # extensions = [".root", ".pdf", ".png", ".C"]
    extensions = [".png"]
    for e in extensions:
        c.SaveAs(f"{out_name}{e}")


def getEffSigma(_h):
    nbins, binw, xmin = (
        _h.GetXaxis().GetNbins(),
        _h.GetXaxis().GetBinWidth(1),
        _h.GetXaxis().GetXmin(),
    )
    mu, rms, total = _h.GetMean(), _h.GetRMS(), _h.Integral()
    # Scan round window of mean: window RMS/binWidth (cannot be bigger than 0.1*number of bins)
    nWindow = int(rms / binw) if (rms / binw) < 0.1 * nbins else int(0.1 * nbins)
    # Determine minimum width of distribution which holds 0.693 of total
    rlim = 0.683 * total
    wmin = 9999999
    # iscanmin = -999
    for iscan in range(-1 * nWindow, nWindow + 1):
        # Find bin idx in scan: iscan from mean
        i_centre = int((mu - xmin) / binw + 1 + iscan)
        x_centre = (i_centre - 0.5) * binw + xmin  # * 0.5 for bin centre
        x_up, x_down = x_centre, x_centre
        i_up, i_down = i_centre, i_centre
        # Define counter for yield in bins: stop when counter > rlim
        y = _h.GetBinContent(i_centre)  # Central bin height
        r = y
        reachedLimit = False
        for j in range(1, nbins):
            if reachedLimit:
                continue
            # Up:
            if (i_up < nbins) & (not reachedLimit):
                i_up += 1
                x_up += binw
                y = _h.GetBinContent(i_up)  # Current bin height
                r += y
                if r > rlim:
                    reachedLimit = True
            else:
                print(
                    " --> Reach nBins in effSigma calc: %s. Returning 0 for effSigma"
                    % _h.GetName()
                )
                return 0
            # Down:
            if not reachedLimit:
                if i_down > 0:
                    i_down -= 1
                    x_down -= binw
                    y = _h.GetBinContent(i_down)  # Current bin height
                    r += y
                    if r > rlim:
                        reachedLimit = True
                else:
                    print(
                        " --> Reach 0 in effSigma calc: %s. Returning 0 for effSigma"
                        % _h.GetName()
                    )
                    return 0
        # Calculate fractional width in bin takes above limt (assume linear)
        if y == 0.0:
            dx = 0.0
        else:
            dx = (r - rlim) * (binw / y)
        # Total width: half of peak
        w = (x_up - x_down + binw - dx) * 0.5
        if w < wmin:
            wmin = w
            # iscanmin = iscan
        # Return effSigma
        return wmin
