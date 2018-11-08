#!/usr/bin/env python
import os, subprocess, sys
from array import array
from ROOT import *
from math import sqrt, sin, cos, tan, exp
import numpy as np
workingDir = os.getcwd()

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# cd /home/acaan/VHbbNtuples_8_0_x/CMSSW_7_4_7/src/ ; cmsenv ; cd -
# python do_limits_rebining.py --channel "0l_2tau" --variables "oldVarA"  --BINtype "regular"
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--channel ", type="string", dest="channel", help="The ones whose variables implemented now are:\n   - 1l_2tau\n   - 2lss_1tau\n It will create a local folder and store the report*/xml", default="2lss_1tau")
parser.add_option("--variables", type="string", dest="variables", help="  Set of variables to use -- it shall be put by hand in the code", default="none")
parser.add_option("--BDTtype", type="string", dest="BDTtype", help="Variable set", default="1B")
parser.add_option("--BINtype", type="string", dest="BINtype", help="regular / ranged / quantiles", default="regular")
(options, args) = parser.parse_args()

user="acaan"
year="2016"
channel=options.channel

if channel == "2lss_1tau" :
    label='2lss_1tau_2018Feb28_VHbb_TLepMTau_shape'
    bdtTypes=["tt","ttV","SUM_T","SUM_M","1B_T","1B_M"]
if channel == "1l_2tau" :
    label= "1l_2tau_2018Mar02_VHbb_TLepTTau_shape"
    bdtTypes=["ttbar","ttV","SUM_T","SUM_VT","1B_T","1B_VT"]
if channel == "2l_2tau" :
    label= "2l_2tau_2018Feb20_VHbb_TLepMTau"
    bdtTypes= ["tt","ttV","SUM_M","SUM_T","SUM_VT","1B_M","1B_T","1B_VT"]
if channel == "3l_1tau" :
    label= "3l_1tau_2018Mar12_VHbb_TLepMTau_shape"
    bdtTypes= ["tt","ttV","SUM_M","SUM_T","SUM_VT","1B_M","1B_T","1B_VT"]
if channel == "2017" :
    label="datacards_ICHEP"
    bdtTypes= ["plainKin_SUM_VT"]
    channelsTypes= [ "2l_2tau"]
if channel == "0l_2tau" :
    year="2017"
    #label='0l_2tau_datacards_2018Oct07_withBoost'
    label="0l_2tau_datacards_2018Oct25_withBoostSubjetCat" #'0l_2tau_datacards_2018Oct07_withBoost_looseTau'
    bdtTypes=[
    #"mvaOutput_0l_2tau_HTT_sum",
    #"mva_Updated", #"mva_oldVar", "mTauTauVis", "mTauTau",
    #"mva_Boosted_AK12", "mva_Boosted_AK12_basic","mva_Boosted_AK12_noISO",
    #"mva_Boosted_AK8", #"mva_Boosted_AK8_basic",
    "mva_Boosted_AK8_noISO"
    ]
    channelsTypes= [ "0l_2tau" ]
if channel == "3l_0tau" :
    year="2017"
    label='3l_0tau_datacards_2018Oct07_withBoost_multilepCat'
    #label='0l_2tau_datacards_2018Oct07_withBoost_looseTau'
    bdtTypes=[
    "mva_AK12", "mva_Boosted_AK12_basic", "mva_Boosted_AK12_noISO", "mva_Boosted_AK12",
    "mva_Boosted_AK8_noISO", "mva_Boosted_AK8",
    "mva_oldVar", "mva_Updated"
    ] # "mvaOutput_0l_2tau_HTT_sum",
    channelsTypes= [ "3l_0tau" ]
if channel == "2lss_0tau" :
    year="2017"
    label = "2lss_0tau_datacards_2018Oct25_withBoostCat" #"2lss_0tau_datacards_2018Oct07_withBoost_multilepCat_2"
    bdtTypes=[
     "mva_oldVar", "mva_Updated",
    #"mva_Boosted_AK12", "mva_Boosted_AK12_noISO", # "mva_Boosted_AK12_basic",
    "mva_Boosted_AK8", #"mva_Boosted_AK8_basic",
    "mva_Boosted_AK8_noISO",
    ]
    channelsTypes= [ "2lss_0tau" ]
if channel == "2los_1tau" :
    year="2017"
    label='2los_1tau_BDTtraining_Ltau_100bin_10Oct_2018'
    bdtTypes=[
    "mvaOutput_2los_1tau_evtLevelSUM_TTH_19Var",
    ]
    channelsTypes= [ "2los_1tau" ]
if channel == "hh_bb2l" :
    year="2017"
    label='hh_bb2l'
    bdtTypes=[
    "hh_bb2lOS_MVAOutput_400",
    ]


sources=[]
bdtTypesToDo=[]
bdtTypesToDoFile=[]
channelToDo=[]
local=workingDir+"/"+options.channel+"_"+label+"/"+options.variables+"/"

print ("to run this script your CMSSW_base should be the one that CombineHavester installed")

def run_cmd(command):
  print "executing command = '%s'" % command
  p = subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
  stdout, stderr = p.communicate()
  print stderr
  return stdout

nbinRegular=np.arange(1, 20) #list(divisorGenerator(originalBinning))
nbinQuant= np.arange(10,30)
#
#nbinRegular=np.arange(8, 11) #list(divisorGenerator(originalBinning))
#nbinQuant= np.arange(10,11)

"""
python do_limits_rebining.py --channel "0l_2tau"  --BINtype "quantiles" --variables "teste" &
python do_limits_rebining.py --channel "0l_2tau"  --BINtype "regular" --variables "teste" &

python do_limits_rebining.py --channel "3l_0tau"  --BINtype "quantiles" --variables "teste" &
python do_limits_rebining.py --channel "3l_0tau"  --BINtype "regular" --variables "teste" &

python makePostFitPlots_FromCombine.py --channel "ttH_2los_1tau" --input /home/acaan/CMSSW_9_4_0_pre1/src/tth-bdt-training-test/treatDatacards/2los_1tau_BDTtraining_Ltau_100bin_10Oct_2018/ttH_teste_2los_1tau_mvaOutput_2los_1tau_evtLevelSUM_TTH_19Var_nbin_14_shapes.root --odir /home/acaan/CMSSW_9_4_0_pre1/src/tth-bdt-training-test/treatDatacards/2los_1tau_BDTtraining_Ltau_100bin_10Oct_2018/ --minY 0.1 --maxY 50000 --notFlips  --fromHavester --useLogPlot --nameOut teste_2los_1tau_mvaOutput_2los_1tau_evtLevelSUM_TTH_19Var_nbin_14

"""

counter=0
if channel == "2lss_1tau" :
    source=local+"/prepareDatacards_"+channel+"_sumOS_"
    if options.variables=="MEM" :
        #source= source+"memOutput_LR"+".root"
        my_file = source+'memOutput_LR.root'
        if os.path.exists(my_file) :
            sources = sources + [source+'memOutput_LR']
            bdtTypesToDo = bdtTypesToDo +['1D']
            print ("rebinning ",sources[counter])
        else : print ("does not exist ",source+"memOutput_LR.root")
    else :
        for bdtType in bdtTypes :
            my_file = source+'mvaOutput_2lss_'+options.variables+'_'+bdtType+'.root'
            if os.path.exists(my_file) :
                sources = sources + [source+'mvaOutput_2lss_'+options.variables+'_'+bdtType]
                bdtTypesToDo = bdtTypesToDo +[bdtType]
                print (sources[counter],"rebinning ")
                counter=counter+1
            else : print (source+"mvaOutput_2lss_"+options.variables+"_"+bdtType+".root","does not exist ")
    bdtTypesToDoFile=bdtTypesToDo
if channel == "1l_2tau" or channel == "2l_2tau" or channel == "3l_1tau" :
    source=local+"/prepareDatacards_"+channel+"_"
    if options.variables=="oldTrain" :
        oldVar=["1l_2tau_ttbar_Old", "1l_2tau_ttbar_OldVar","ttbar_OldVar"] #["1l_2tau_ttbar"] #"1l_2tau_ttbar_Old", "1l_2tau_ttbar_OldVar", "ttbar_OldVar"]
        for ii,nn in enumerate(oldVar) :
            my_file = source+"mvaOutput_"+nn+'.root'
            print my_file
            if os.path.exists(my_file) :
                sources = sources + [source+"mvaOutput_"+nn]
                bdtTypesToDo = bdtTypesToDo +['1D']
                bdtTypesToDoFile=bdtTypesToDoFile+[oldVar[ii]]
                print ("rebinning ",sources[counter])
            else : print ("does not exist ",source+"mvaOutput_"+nn+".root")
    elif "HTT" in options.variables :
        for bdtType in bdtTypes :
            #if channel == "2l_2tau" :
            fileName=options.variables+"_"+bdtType
            #elif channel == "1l_2tau" : fileName=bdtType+"_"+options.variables
            my_file =  source+"mvaOutput_"+fileName+".root"
            if os.path.exists(my_file) :
                sources = sources + [source+"mvaOutput_"+fileName]
                bdtTypesToDo = bdtTypesToDo +[bdtType]
                bdtTypesToDoFile=bdtTypesToDoFile+[fileName]
                print (sources[counter],"rebinning ")
                counter=counter+1
            else : print (source+fileName+".root","does not exist ")
    elif "mTauTauVis" in options.variables :
        my_file = source+options.variables+".root"
        if os.path.exists(my_file) :
            proc=subprocess.Popen(["cp "+my_file+" " +local],shell=True,stdout=subprocess.PIPE)
            out = proc.stdout.read()
            sources = sources + [source+options.variables]
            bdtTypesToDo = bdtTypesToDo +[options.variables]
            bdtTypesToDoFile=bdtTypesToDoFile+[options.variables]
            print (sources[counter],"rebinning ")
            counter=counter+1
        else : print (source+options.variables+".root","does not exist ")
    else : print ("options",channel,options.variables,"are not compatible")
if channel=="2017" :
    for ii, bdtType in enumerate(bdtTypes) :
        source=local+"/prepareDatacards_"+channelsTypes[ii]+"_mvaOutput_"+bdtTypes[ii]+"_noRebin"
        my_file = source+".root"
        if os.path.exists(my_file) :
            sources = sources + [source]
            bdtTypesToDo = bdtTypesToDo +[channelsTypes[ii]+"_mvaOutput_"+bdtTypes[ii]]
            bdtTypesToDoFile=bdtTypesToDoFile+[channelsTypes[ii]+"_mvaOutput_"+bdtTypes[ii]]
            channelToDo=channelToDo+[channelsTypes[ii]]
            print (sources[counter],"rebinning ")
            counter=counter+1
        else : print (source+options.variables+".root","does not exist ")
if channel == "0l_2tau" :
    local="/home/acaan/CMSSW_9_4_0_pre1/src/tth-bdt-training-test/treatDatacards/"+label+"/"
    for ii, bdtType in enumerate(bdtTypes) :
        source=local+"/prepareDatacards_0l_2tau_"+bdtTypes[ii]
        my_file = source+".root"
        if os.path.exists(my_file) :
            sources = sources + [source]
            bdtTypesToDo = bdtTypesToDo +["0l_2tau_"+bdtTypes[ii]]
            bdtTypesToDoFile=bdtTypesToDoFile+["0l_2tau_"+bdtTypes[ii]]
            channelToDo=channelToDo+["0l_2tau"]
            print (sources[counter],"rebinning ")
            counter=counter+1
        else : print (source+" "+my_file, "does not exist ")
if channel == "3l_0tau" :
    local="/home/acaan/CMSSW_9_4_0_pre1/src/tth-bdt-training-test/treatDatacards/"+label+"/"
    for ii, bdtType in enumerate(bdtTypes) :
        source=local+"/prepareDatacards_3l_"+bdtTypes[ii]
        my_file = source+".root"
        if os.path.exists(my_file) :
            sources = sources + [source]
            bdtTypesToDo = bdtTypesToDo +["3l_0tau_"+bdtTypes[ii]]
            bdtTypesToDoFile=bdtTypesToDoFile+["3l_0tau_"+bdtTypes[ii]]
            channelToDo=channelToDo+["3l"]
            print (sources[counter],"rebinning ")
            counter=counter+1
        else : print (source+" "+my_file, "does not exist ")
if channel == "2lss_0tau" :
    local="/home/acaan/CMSSW_9_4_0_pre1/src/tth-bdt-training-test/treatDatacards/"+label+"/"
    for ii, bdtType in enumerate(bdtTypes) :
        source=local+"/prepareDatacards_2lss_"+bdtTypes[ii]
        my_file = source+".root"
        if os.path.exists(my_file) :
            sources = sources + [source]
            bdtTypesToDo = bdtTypesToDo +["2lss_0tau_"+bdtTypes[ii]]
            bdtTypesToDoFile=bdtTypesToDoFile+["2lss_0tau_"+bdtTypes[ii]]
            channelToDo=channelToDo+["2lss"]
            print (sources[counter],"rebinning ")
            counter=counter+1
        else : print (source+" "+my_file, "does not exist ")
if channel == "2los_1tau" :
    local="/home/acaan/CMSSW_9_4_0_pre1/src/tth-bdt-training-test/treatDatacards/"+label+"/"
    for ii, bdtType in enumerate(bdtTypes) :
        source=local+"/prepareDatacards_2los_1tau_"+bdtTypes[ii]
        my_file = source+".root"
        if os.path.exists(my_file) :
            sources = sources + [source]
            bdtTypesToDo = bdtTypesToDo +["2los_1tau_"+bdtTypes[ii]]
            bdtTypesToDoFile=bdtTypesToDoFile+["2los_1tau_"+bdtTypes[ii]]
            channelToDo=channelToDo+["2los_1tau"]
            print (sources[counter],"rebinning ")
            counter=counter+1
        else : print (source+" "+my_file, "does not exist ")
if channel == "hh_bb2l" :
    local="/home/acaan/CMSSW_9_4_0_pre1/src/tth-bdt-training-test/treatDatacards/"+label+"/"
    for ii, bdtType in enumerate(bdtTypes) :
        source=local+"/prepareDatacards_hh_bb2l_"+bdtTypes[ii]
        my_file = source+".root"
        if os.path.exists(my_file) :
            sources = sources + [source]
            bdtTypesToDo = bdtTypesToDo +["hh_bb2l_"+bdtTypes[ii]]
            bdtTypesToDoFile=bdtTypesToDoFile+["hh_bb2l_"+bdtTypes[ii]]
            channelToDo=channelToDo+["hh_bb2l"]
            print (sources[counter],"rebinning ")
            counter=counter+1
        else : print (source+" "+my_file, "does not exist ")

if options.BINtype == "regular" or options.BINtype == "ranged" : binstoDo=nbinRegular
if options.BINtype == "quantiles" : binstoDo=nbinQuant

print ("I will rebin",bdtTypesToDoFile,"(",len(sources),") BDT options")

for ns,source in enumerate(sources) :
    for nn,nbins in enumerate(binstoDo) :
        if options.BINtype=="regular" :
            name=source+'_'+str(nbins)+'bins.root'
            nameout=source+'_'+str(nbins)+'bins_dat.root'
        if options.BINtype=="ranged" :
            name=source+'_'+str(nbins)+'bins_ranged.root'
            nameout=source+'_'+str(nbins)+'bins_ranged_dat.root'
        if options.BINtype=="quantiles" :
            name=source+'_'+str(nbins+1)+'bins_quantiles.root'
            nameout=source+'_'+str(nbins+1)+'bins_quantiles_dat.root'
        print ("doing", name)
        shapeVariable=options.variables+'_'+bdtTypesToDoFile[ns]+'_nbin_'+str(nbins)
        if options.BINtype=="ranged" : shapeVariable=shapeVariable+"_ranged"
        if options.BINtype=="quantiles" : shapeVariable=shapeVariable+"_quantiles"
        datacardFile_output = os.path.join(workingDir, local, "ttH_%s" % shapeVariable)
        #run_cmd('%s --input_file=%s --output_file=%s --add_shape_sys=false' % ('WriteDatacards_'+channel, name, datacardFile_output))
        run_cmd('%s --input_file=%s --output_file=%s --add_shape_sys=false' % ('WriteDatacards_'+channelToDo[ns], name, datacardFile_output))

        txtFile = datacardFile_output + ".txt" #.replace(".root", ".txt")
        logFile = datacardFile_output + ".log" #.replace(".root", ".log")
        run_cmd('combine -M Asymptotic -m %s -t -1 %s &> %s' % (str(125), txtFile, logFile))
        run_cmd('rm higgsCombineTest.Asymptotic.mH125.root')
        rootFile = os.path.join(workingDir, local, "ttH_%s_shapes.root" % (shapeVariable))
        run_cmd('PostFitShapes -d %s -o %s -m 125 ' % (txtFile, rootFile))

        if (channel == "0l_2tau" and (options.BINtype == "quantiles" and nbins > 18 and nbins <20) or (channel == "0l_2tau" and options.BINtype == "regular" and nbins > 5 and nbins < 15)) :
            run_cmd('python makePostFitPlots_FromCombine.py --channel "ttH_0l_2tau" --input %s --odir %s --minY 0.1 --maxY 50000 --notFlips --notConversions --fromHavester --useLogPlot --nameOut %s' % (rootFile, local, shapeVariable))

        if (channel == "2los_1tau" and (options.BINtype == "quantiles" and nbins > 18 and nbins <20) or (channel == "2los_1tau" and options.BINtype == "regular" and nbins > 5 and nbins < 15)) :
            run_cmd('python makePostFitPlots_FromCombine.py --channel "ttH_2los_1tau" --input %s --odir %s --minY 0.1 --maxY 50000 --notFlips --notConversions --fromHavester --useLogPlot --nameOut %s' % (rootFile, local, shapeVariable))

        if (channel == "3l_0tau" and (options.BINtype == "quantiles" and nbins > 15 and nbins <20) or (channel == "3l_0tau" and options.BINtype == "quantiles" and nbins > 3 and nbins < 8)) :
            run_cmd('python makePostFitPlots_FromCombine.py --channel "ttH_3l" --input %s --odir %s --minY 0 --maxY 100 --notFlips  --notConversions  --fromHavester --nameOut %s' % (rootFile, local, shapeVariable))

        if (channel == "2lss_0tau" and (options.BINtype == "quantiles" and nbins > 9 and nbins <11) or (channel == "2lss_0tau" and options.BINtype == "regular" and nbins > 7 and nbins < 11)) :
            run_cmd('python makePostFitPlots_FromCombine.py --channel "ttH_2lss" --input %s --odir %s --minY 0 --maxY 180  --fromHavester --nameOut %s' % (rootFile, local, shapeVariable))
