import ROOT

ROOT.gStyle.SetOptStat(0)
# Open the ROOT files and get the TTrees
file1 = ROOT.TFile.Open("/depot/cms/hmm/vscheure/SimpleTest/DYoutput.root")
file2 = ROOT.TFile.Open("/depot/cms/hmm/vscheure/SimpleTest/DYoutput100.root")
file3 = ROOT.TFile.Open("/depot/cms/hmm/vscheure/SimpleTest/DYoutput_nocut.root")

tree1 = file1.Get("my_ttree")
tree2 = file2.Get("my_ttree")
tree3 = file3.Get("my_ttree")

# Create a canvas for plotting
c = ROOT.TCanvas("Canvas", '', 1000, 800)
c.SetLogy()
c.SetTicks()
pad1 = ROOT.TPad("pad1", "pad1", 0., 0.3, 1., 1.0)
pad1.SetLeftMargin(0.1)
pad1.SetRightMargin(0.03)
pad1.SetTopMargin(0.1)
pad1.SetBottomMargin(0.02)
pad1.SetLogy()
pad1.SetTickx()
pad1.SetTicky()
pad1.Draw()
pad1.cd()


# Create a histogram stack to hold the distributions
stack = ROOT.THStack("stack", "Dimuon Mass Distribution")

# Create histograms for the first two TTrees and fill them with data
histogram1 = ROOT.TH1F("histogram1", "Dimuon Mass Distribution", 50, 45, 200)
#weightDY =1
weightDY = 16800*6200/71839442
for event in tree1:
    histogram1.Fill(event.dimuon_mass)
histogram1.Scale(weightDY)
testerr = histogram1.Clone()
histogram1.SetFillColor(ROOT.kBlue-1)
stack.Add(histogram1)

histogram2 = ROOT.TH1F("histogram2", "Dimuon Mass Distribution", 50, 45, 200)
#weight100=1
weight100 = 16800*254.8/219887619
for event in tree2:
    histogram2.Fill(event.dimuon_mass)
histogram2.Scale(weight100)
#errhist2 = histogram2.Clone()
histogram2.SetFillColor(ROOT.kOrange+2)
stack.Add(histogram2)
testerr.Add(histogram2)



# Create a histogram for the third TTree as a line
histogram3 = ROOT.TH1F("histogram3", "Dimuon Mass Distribution", 50, 45, 200)
#weightDYn =1
weightDYn = 16800*6450/71839442
for event in tree3:
    histogram3.Fill(event.dimuon_mass)
histogram3.Scale(weightDYn)
histogram3.SetLineColor(ROOT.kGreen)
histogram3.SetLineWidth(4)

# Create TGraphAsymmErrors for statistical uncertainty for each histogram
testerr.SetFillStyle(3244)
testerr.SetFillColor(1)

#stat_uncertainty1 = ROOT.TGraphAsymmErrors(histograms)



# Set the marker style and color for the bands

# Customize the appearance of the stacked plot
#histogram3.GetXaxis().SetTitle("Dimuon Mass")
#histogram3.SetTitle("Dimuon Mass Distribution")
textsize = 24./(pad1.GetWh()*pad1.GetAbsHNDC())
histogram3.SetTitle("Dimuon mass")
histogram3.GetXaxis().SetTitleSize(0)
histogram3.GetXaxis().SetLabelSize(0)
histogram3.GetYaxis().SetTitle("Events")
histogram3.GetYaxis().SetTitleFont(42)
histogram3.GetYaxis().SetTitleSize(textsize)
histogram3.GetYaxis().SetTitleOffset(1.)
#main_pad.cd()
# Draw the stacked plot
histogram3.Draw("HIST")
stack.Draw("HIST SAME")


# Draw the statistical uncertainty bands on top of the stacked plot

#histogram3.SetFillStyle(3244)
#histogram3.SetFillColor(ROOT.kGreen)
#histogram3.Draw("E2 SAME")
testerr.Draw("E2 SAME")
# Add a legend
legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
legend.AddEntry(histogram1, "DY-M-50", "f")
legend.AddEntry(histogram2, "DY-M100To200", "f")
legend.AddEntry(histogram3, "DY-M-50 no cut", "l")
legend.Draw()
c.cd()

# Create the ratio pad
pad2 = ROOT.TPad("pad2", "pad2", 0, 0., 1, 0.3)
pad2.SetLeftMargin(0.1)
pad2.SetRightMargin(0.03)
pad2.SetTopMargin(0.04)
pad2.SetBottomMargin(0.35)
pad2.SetGridy()
pad2.SetTickx()
pad2.SetTicky()
pad2.Draw()
pad2.cd()


# Clone the histograms for the ratio plot
ratio_histogram1 = histogram1.Clone()
ratio_testerr = testerr.Clone()
ratio_histogram3 = histogram3.Clone()

# Calculate the ratio histograms
# Calculate the ratio of the statistical uncertainty to the main histogram
for bin in range(1, testerr.GetNbinsX() + 1):
    bin_content = testerr.GetBinContent(bin)
    if bin_content != 0:
        ratio_testerr.SetBinContent(bin,testerr.GetBinError(bin) / bin_content)
        
for bin in range(1, ratio_histogram3.GetNbinsX() + 1):
    bin_content = histogram3.GetBinContent(bin)
    if bin_content != 0:
        ratio_histogram3.SetBinContent(bin,histogram3.GetBinError(bin) / bin_content)
        #ratio_histogram3.SetBinContent(bin, histogram3.GetBinError(bin) / bin_content)
# Customize the appearance of the ratio histograms
ratio_testerr.SetFillStyle(3144)
ratio_histogram3.SetLineWidth(0)

ratio_histogram3.SetFillStyle(3145)
ratio_histogram3.SetFillColor(ROOT.kGreen+2)

textsize = 27./(pad2.GetWh()*pad2.GetAbsHNDC())
ratio_histogram3.SetTitleSize(0)
ratio_histogram3.SetTitle("  ")
ratio_histogram3.GetXaxis().SetTitle("Dimuon mass")
ratio_histogram3.GetXaxis().SetTitleOffset(1.)
ratio_histogram3.GetXaxis().SetTitleSize(textsize)
ratio_histogram3.GetXaxis().SetLabelSize(textsize)
ratio_histogram3.GetXaxis().SetTitleFont(42)
ratio_histogram3.GetXaxis().SetRangeUser(0., 0.031)
ratio_histogram3.GetYaxis().SetTitle("Stat. unc.")
ratio_histogram3.GetYaxis().CenterTitle(True)
ratio_histogram3.GetYaxis().SetNdivisions(5)
ratio_histogram3.GetYaxis().SetLabelSize(textsize)
ratio_histogram3.GetYaxis().SetTitleFont(42)
ratio_histogram3.GetYaxis().SetTitleSize(textsize)
ratio_histogram3.GetYaxis().SetTitleOffset(1.)



# Draw the ratio pad
ratio_histogram3.Draw("HIST")
ratio_testerr.Draw("HIST SAME")
#ratio_histogram2.Draw("SAME HIST")


# Show the canvas
c.Update()
c.Draw()



# Show the canvas



c.SaveAs("DYPlot.png")
x = input()

