#!/bin/bash
s=("2017" "2016postVFP" "2016preVFP"); 
for year in ${s[@]}; 
do
combineCards.py datacard_vbf_SR_"$year".txt datacard_vbf_SB_"$year".txt   > combined_"$year".txt
text2workspace.py combined_"$year".txt -m 125 
combineTool.py -M Impacts -d combined_"$year".root -m 125 --doInitialFit --robustFit 1
combineTool.py -M Impacts -d combined_"$year".root -m 125 --robustFit 1 --doFits
combineTool.py -M Impacts -d combined_"$year".root -m 125 -o impacts_"$year".json
plotImpacts.py -i impacts_"$year".json -o impacts_"$year"
done