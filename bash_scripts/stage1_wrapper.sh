#!/bin/bash
s=("2016postVFP" "2016preVFP"); 
label="ul_yun_Dec15_JecOff_JesJerUncOn_2016LumiFix"
for year in ${s[@]}; 
do
nohup python run_stage1.py -y "$year" --label "$label" -sl dummy
done