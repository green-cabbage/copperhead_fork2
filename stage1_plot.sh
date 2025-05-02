#!/bin/bash
# Stop execution on any error
set -e

#
data_l="A B C D E F G H"
bkg_l="DY TT ST VV EWK OTHER"
sig_l="ggH VBF"

status="Private_Work"

# label="April19_NanoV12_JEROff"
# label="jetHornStudy_29Apr2025_JecOnJerOff"
label="jetHornStudy_29Apr2025_JecOnJerOn"
# label="jetHornStudy_29Apr2025_JecOnJerOn_tightJetPuId"
# label="jetHornStudy_29Apr2025_JecOnJerStrat1"
# label="jetHornStudy_29Apr2025_JecOnJerStrat2"

year="2018"

lumi="59.97"
vars2plot="jet"
# region="z-peak"
region="signal"

# load_path = f"/depot/cms/users/yun79/hmm/copperheadV1clean/{label}/stage1_output/{year}/f1_0/"
# load_path="/depot/cms/users/shar1172/hmm/copperheadV1clean/${label}/stage1_output/${year}/f1_0/"
load_path="/depot/cms/users/yun79/hmm/copperheadV1clean/${label}/stage1_output/${year}/*/"

# python validation_plotter_unified.py -y $year --load_path $load_path -var $vars2plot --data $data_l --background $bkg_l --signal $sig_l --lumi $lumi --status $status -cat nocat -reg $region --label $label --linear_scale
python validation_plotter_unified.py -y $year --load_path $load_path -var $vars2plot --data $data_l --background $bkg_l --signal $sig_l --lumi $lumi --status $status -cat ggh -reg $region --label $label
# python validation_plotter_unified.py -y $year --load_path $load_path -var $vars2plot --data $data_l --background $bkg_l --signal $sig_l --lumi $lumi --status $status -cat vbf -reg $region --label $label

# ! python validation_plotter_unified.py -y {year} --load_path {load_path}  -var {' '.join(vars2plot)} --data {' '.join(data_l)} --background {' '.join(bkg_l)} --signal {' '.join(sig_l)} --lumi {lumi} --status {status} -cat vbf -reg {region} --label {label}  
