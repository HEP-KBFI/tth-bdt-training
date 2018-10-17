#!/usr/bin/env python
import os, subprocess, sys
workingDir = os.getcwd()

from ROOT import *
from math import sqrt, sin, cos, tan, exp
import numpy as np
from pathlib2 import Path
execfile("../python/data_manager.py")

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# ./rebin_datacards.py --channel "2lss_1tau" --variables "HTT" --BINtype "quantiles" --doLimits
# ./rebin_datacards.py --channel "2l_2tau" --variables "mTauTauVis" --BINtype "mTauTauVis"
# ./rebin_datacards.py --channel "0l_2tau" --BINtype "regular" --variables "teste"
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--channel ", type="string", dest="channel", help="The ones whose variables implemented now are:\n   - 1l_2tau\n   - 2lss_1tau\n It will create a local folder and store the report*/xml", default="2lss_1tau")
parser.add_option("--variables", type="string", dest="variables", help="  Set of variables to use -- it shall be put by hand in the code", default=1000)
parser.add_option("--BDTtype", type="string", dest="BDTtype", help="Variable set", default="1B")
parser.add_option("--BINtype", type="string", dest="BINtype", help="regular / ranged / quantiles", default="regular")
parser.add_option("--doPlots", action="store_true", dest="doPlots", help="If you call this will not do plots with repport", default=False)
parser.add_option("--doLimits", action="store_true", dest="doLimits", help="If you call this will not do plots with repport", default=False)
(options, args) = parser.parse_args()

doLimits=options.doLimits
doPlots=options.doPlots
#user="acaan"
#year="2016"
user="mmaurya"
year="2017"
channel=options.channel
if channel == "2lss_1tau" :
    label="2lss_1tau_2018Feb28_VHbb_TLepMTau_shape" #"2lss_1tau_2018Feb26_VHbb_TLepTTau"
    bdtTypes=["tt","ttV","SUM_T","SUM_M","1B_T","1B_M"] #,"2MEM","2HTT"]
#if channel == "1l_2tau" :
#    label= "1l_2tau_2018Mar02_VHbb_TLepTTau_shape" #"1l_2tau_2018Feb08_VHbb_TightTau" # "1l_2tau_2018Feb02_VHbb_VTightTau" #  "1l_2tau_2018Jan30_VHbb_VVTightTau" # "1l_2tau_2018Jan30_VHbb" # "1l_2tau_2018Jan30_VHbb_VTightTau" #
#    bdtTypes=["ttbar","ttV","SUM_T","SUM_VT","1B_T","1B_VT"] #"1B"] #
if channel == "1l_2tau" :
    label= "2018Jun04_JES_1l2tau"
    bdtTypes=["SUM_VT"] #"1B"] #
if channel == "2l_2tau" :
    label= "2l_2tau_2018Feb20_VHbb_TLepMTau" #
    bdtTypes= ["tt","ttV","SUM_M","SUM_T","SUM_VT","1B_M","1B_T","1B_VT"] #[] #,,
if channel == "3l_1tau" :
    label= "3l_1tau_2018Mar12_VHbb_TLepMTau_shape" #
    bdtTypes= ["tt","ttV","SUM_M","SUM_T","SUM_VT","1B_M","1B_T","1B_VT"] #[] #,,
if channel == "2017" :
    label="datacards_ICHEP"
    bdtTypes= ["plainKin_SUM_VT_noRebin_x"]
    channelsTypes= [ "2l_2tau"]
if channel == "0l_2tau" :
    year="2017"
    label='0l_2tau_datacards_2018Oct07_withBoost'
    #label='0l_2tau_datacards_2018Oct07_withBoost_looseTau'
    bdtTypes=[
    "mva_Updated", #"mva_oldVar", # "mTauTau", # "mTauTauVis",
    #"mva_Boosted_AK12", "mva_Boosted_AK12_basic","mva_Boosted_AK12_noISO",
    "mva_Boosted_AK8", "mva_Boosted_AK8_basic","mva_Boosted_AK8_noISO",
    ] # "mvaOutput_0l_2tau_HTT_sum",
    channelsTypes= [ "0l_2tau" ]
if channel == "3l_0tau" :
    year="2017"
    label='3l_0tau_datacards_2018Oct07_withBoost_multilepCat'
    #label='0l_2tau_datacards_2018Oct07_withBoost_looseTau'
    bdtTypes=[
    #"mva_AK12", "mva_Boosted_AK12_basic", "mva_Boosted_AK12_noISO", "mva_Boosted_AK12",
    #"mva_Boosted_AK8_noISO", "mva_Boosted_AK8",
    "mva_oldVar",
    "mva_Updated"
    ] # "mvaOutput_0l_2tau_HTT_sum",
    channelsTypes= [ "3l_0tau" ]
####
if channel == "2los_1tau" :
    year="2017"
    label='2los_1tau_BDTtraining_Ltau_100bin_10Oct_2018'
    bdtTypes=[
    "mvaOutput_2los_1tau_evtLevelSUM_TTH_19Var",
    ] 
    channelsTypes= [ "2los_1tau" ]

sources=[]
bdtTypesToDo=[]
bdtTypesToDoLabel=[]
bdtTypesToDoFile=[]

import shutil,subprocess
proc=subprocess.Popen(["mkdir "+label],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()
proc=subprocess.Popen(["mkdir "+label+"/"+options.variables],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()
#for test in [1000,900,800,700,600,500,400,300,200,100] : print (test, list(divisorGenerator(test)) )
if channel=="2017" : mom="/home/mmaurya/VHbbNtuples_8_0_x/CMSSW_8_0_21/src/Oct2018/100Bin/"
else : mom="/home/"+user+"/ttHAnalysis/"+year+"/"+label+"/datacards/"+channel

local=workingDir+"/"+options.channel+"_"+label+"/"+options.variables+"/"
originalBinning=100
nbinRegular=np.arange(1, 20)
nbinQuant= np.arange(10,28)
counter=0

if channel == "2lss_1tau" :
    sourceoriginal=mom+"/prepareDatacards_"+channel+"_sumOS_"
    source=local+"/prepareDatacards_"+channel+"_sumOS_"
    print sourceoriginal
    if options.variables=="MEM" :
        my_file = Path(sourceoriginal+"memOutput_LR.root")
        if my_file.exists() :
            proc=subprocess.Popen(["cp "+sourceoriginal+"memOutput_LR.root "+local ],shell=True,stdout=subprocess.PIPE)
            out = proc.stdout.read()
            sources = sources + [source+"memOutput_LR"]
            bdtTypesToDo = bdtTypesToDo +["1D"]
            print ("rebinning ",sources[counter])
        else : print ("does not exist ",sourceoriginal+"memOutput_LR.root")
    else  :
        for bdtType in bdtTypes :
            my_file = Path(sourceoriginal+"mvaOutput_2lss_"+options.variables+"_"+bdtType+".root")
            if my_file.exists() :
                proc=subprocess.Popen(["cp "+
                    sourceoriginal+ "mvaOutput_2lss_"+options.variables+"_"+bdtType+".root " +local
                    ],shell=True,stdout=subprocess.PIPE)
                out = proc.stdout.read()
                sources = sources + [source+"mvaOutput_2lss_"+options.variables+"_"+bdtType]
                bdtTypesToDo = bdtTypesToDo +[bdtType]
                print (sources[counter],"rebinning ")
                counter=counter+1
            else : print (sourceoriginal+"mvaOutput_2lss_"+options.variables+"_"+bdtType+".root","does not exist ")
    bdtTypesToDoLabel=bdtTypesToDo
    bdtTypesToDoFile=bdtTypesToDo
if channel == "1l_2tau" or channel == "2l_2tau" or channel == "3l_1tau" :
    sourceoriginal=mom+"/prepareDatacards_"+channel+"_"
    source=local+"/prepareDatacards_"+channel+"_"
    if options.variables=="oldTrain" :
        oldVar=["1l_2tau_ttbar_Old", "1l_2tau_ttbar_OldVar","ttbar_OldVar"] # "1l_2tau_ttbar",
        typeBDT=[ "oldTrainM","oldVar loose lep" ,"oldVar tight lep"] # "oldTrain",
        for ii,nn in enumerate(oldVar) :
            my_file = Path(sourceoriginal+"mvaOutput_"+nn+".root")
            print sourceoriginal+nn+".root"
            if my_file.exists() :
                proc=subprocess.Popen(['cp '+sourceoriginal+"mvaOutput_"+nn+".root "+local],shell=True,stdout=subprocess.PIPE)
                out = proc.stdout.read()
                sources = sources + [source+"mvaOutput_"+nn]
                bdtTypesToDo = bdtTypesToDo +["1D"]
                bdtTypesToDoLabel = bdtTypesToDoLabel +[typeBDT[ii]]
                bdtTypesToDoFile=bdtTypesToDoFile+[oldVar[ii]]
                print ("rebinning ",sources[counter])
            else : print ("does not exist ",source+nn)
    elif "HTT" in options.variables :
        for bdtType in bdtTypes :
            #if channel == "2l_2tau" :
            fileName=options.variables+"_"+bdtType
            #elif channel == "1l_2tau" : fileName=bdtType+"_"+options.variables
            my_file = Path(sourceoriginal+"mvaOutput_"+fileName+".root")
            if my_file.exists() :
                proc=subprocess.Popen(["cp "+sourceoriginal+"mvaOutput_"+fileName+".root " +local],shell=True,stdout=subprocess.PIPE)
                out = proc.stdout.read()
                sources = sources + [source+"mvaOutput_"+fileName]
                bdtTypesToDo = bdtTypesToDo +[bdtType]
                bdtTypesToDoFile=bdtTypesToDoFile+[fileName]
                print (sources[counter],"rebinning ")
                counter=counter+1
            else : print (sourceoriginal+"mvaOutput_"+fileName+".root","does not exist ")
        bdtTypesToDoLabel=bdtTypesToDo
    elif "mTauTauVis" in options.variables :
        my_file = Path(sourceoriginal+options.variables+".root")
        if my_file.exists() :
            proc=subprocess.Popen(["cp "+sourceoriginal+options.variables+".root " +local],shell=True,stdout=subprocess.PIPE)
            out = proc.stdout.read()
            sources = sources + [source+options.variables]
            bdtTypesToDo = bdtTypesToDo +[options.variables]
            bdtTypesToDoFile=bdtTypesToDoFile+[options.variables]
            print (sources[counter],"rebinning ")
            counter=counter+1
        else : print (sourceoriginal+options.variables+".root","does not exist ")
    else : print ("options",channel,options.variables,"are not compatible")
if channel=="2017" :
    for ii, bdtType in enumerate(bdtTypes) :
        fileName=mom+"prepareDatacards_"+channelsTypes[ii]+"_mvaOutput_"+bdtTypes[ii]+"_noRebin.root"
        my_file = Path(fileName)
        source=local+"/prepareDatacards_"+channelsTypes[ii]+"_mvaOutput_"+bdtTypes[ii]+"_noRebin"
        print fileName
        if my_file.exists() :
            proc=subprocess.Popen(['cp '+fileName+" "+local],shell=True,stdout=subprocess.PIPE)
            out = proc.stdout.read()
            sources = sources + [source]
            bdtTypesToDo = bdtTypesToDo +[channelsTypes[ii]+" "+bdtTypes[ii]]
            bdtTypesToDoLabel = bdtTypesToDoLabel +[channelsTypes[ii]+" "+bdtTypes[ii]]
            bdtTypesToDoFile=bdtTypesToDoFile+[channelsTypes[ii]+"_mvaOutput_"+bdtTypes[ii]]
            print ("rebinning ",sources[counter])
        else : print ("does not exist ",source)
if channel == "0l_2tau" :
    local="/home/acaan/CMSSW_9_4_0_pre1/src/tth-bdt-training-test/treatDatacards/"+label+"/"
    for ii, bdtType in enumerate(bdtTypes) :
        mom = "/home/acaan/ttHAnalysis/2017/"+label+"/datacards/0l_2tau/"
        fileName=mom+"prepareDatacards_0l_2tau_"+bdtTypes[ii]+".root"
        my_file = Path(fileName)
        source=local+"prepareDatacards_0l_2tau_"+bdtTypes[ii]
        print fileName
        if my_file.exists() :
            proc=subprocess.Popen(['cp '+fileName+" "+local],shell=True,stdout=subprocess.PIPE)
            out = proc.stdout.read()
            sources = sources + [source]
            bdtTypesToDo = bdtTypesToDo +["0l_2tau "+bdtTypes[ii]]
            bdtTypesToDoLabel = bdtTypesToDoLabel +["0l_2tau "+bdtTypes[ii]]
            bdtTypesToDoFile=bdtTypesToDoFile+["0l_2tau_"+bdtTypes[ii]]
            print ("rebinning ",sources[counter])
        else : print ("does not exist ",source)
if channel == "3l_0tau" :
    local="/home/acaan/CMSSW_9_4_0_pre1/src/tth-bdt-training-test/treatDatacards/"+label+"/"
    for ii, bdtType in enumerate(bdtTypes) :
        mom = "/home/acaan/ttHAnalysis/2017/"+label+"/datacards/3l/"
        fileName=mom+"prepareDatacards_3l_"+bdtTypes[ii]+".root"
        my_file = Path(fileName)
        source=local+"prepareDatacards_3l_"+bdtTypes[ii]
        print fileName
        if my_file.exists() :
            proc=subprocess.Popen(['cp '+fileName+" "+local],shell=True,stdout=subprocess.PIPE)
            out = proc.stdout.read()
            sources = sources + [source]
            bdtTypesToDo = bdtTypesToDo +["3l "+bdtTypes[ii]]
            bdtTypesToDoLabel = bdtTypesToDoLabel +["3l "+bdtTypes[ii]]
            bdtTypesToDoFile=bdtTypesToDoFile+["3l_0tau_"+bdtTypes[ii]]
            print ("rebinning ",sources[counter])
        else : print ("does not exist ",source)

if channel == "2los_1tau" :
    local="/home/mmaurya/CMSSW_9_4_0_pre1/src/tth-bdt-training/treatDatacards/"+label+"/"
    for ii, bdtType in enumerate(bdtTypes) :
        mom = "/home/mmaurya/ttHAnalysis/2017/"+label+"/datacards/2los_1tau/"
        fileName=mom+"prepareDatacards_2los_1tau_"+bdtTypes[ii]+".root"
        my_file = Path(fileName)
        source=local+"prepareDatacards_2los_1tau_"+bdtTypes[ii]
        print fileName
        if my_file.exists() :
            proc=subprocess.Popen(['cp '+fileName+" "+local],shell=True,stdout=subprocess.PIPE)
            out = proc.stdout.read()
            sources = sources + [source]
            bdtTypesToDo = bdtTypesToDo +["2los_1tau "+bdtTypes[ii]]
            bdtTypesToDoLabel = bdtTypesToDoLabel +["2los_1tau "+bdtTypes[ii]]
            bdtTypesToDoFile=bdtTypesToDoFile+["2los_1tau_"+bdtTypes[ii]]
            print ("rebinning ",sources[counter])
        else : print ("does not exist ",source)
print ("I will rebin",bdtTypesToDoLabel,"(",len(sources),") BDT options")

if options.BINtype == "regular" or options.BINtype == "ranged" or options.BINtype == "mTauTauVis" : binstoDo=nbinRegular
if options.BINtype == "quantiles" : binstoDo=nbinQuant
if options.BINtype == "none" : binstoDo=np.arange(1,originalBinning)
print binstoDo

colorsToDo=['r','g','b','m','y','c', 'fuchsia', "peachpuff",'k','y'] #['r','g','b','m','y','c','k']
if not doLimits:
    #########################################
    ## make rebinned datacards
    fig, ax = plt.subplots(figsize=(5, 5))
    #plt.title(options.BINtype+" binning "+options.variables)
    plt.title(options.BINtype+" in sum of BKG ")
    lastQuant=[]
    xmaxQuant=[]
    xminQuant=[]
    for nn,source in enumerate(sources) :
        errOcont=rebinRegular(source, binstoDo, options.BINtype,originalBinning,doPlots,options.variables,bdtTypesToDo[nn], True)
        print bdtTypesToDo[nn]
        #print ("                 ",nbinRegular)
        #print ("TT-only,  last bin", errOcont[0])
        #print ("TT-only, Plast bin", errOcont[1])
        #print ("TT+TTV,   last bin", errOcont[2])
        #print ("TT+TTV,  Plast bin", errOcont[3])
        #print ("last quantile",len(errOcont[4]),errOcont[4][len(errOcont[4])-1],errOcont[4][len(errOcont[4])-2])
        lastQuant=lastQuant+[errOcont[4]]
        xmaxQuant=xmaxQuant+[errOcont[5]]
        xminQuant=xminQuant+[errOcont[6]]
        #
        print (binstoDo,errOcont[0])
        plt.plot(binstoDo,errOcont[0], color=colorsToDo[nn],linestyle='-') # ,label=bdtTypesToDo[nn]
        plt.plot(binstoDo,errOcont[0], color=colorsToDo[nn],linestyle='-',marker='o',label=bdtTypesToDo[nn]) #
        #plt.plot(binstoDo,errOcont[2], color=colorsToDo[nn],linestyle='--',marker='x')
    ax.set_xlabel('nbins')
    if options.BINtype == "regular" : maxplot =1.0
    #elif options.BINtype == "mTauTauVis" : maxplot=200.
    else : maxplot =1.0 # 0.35
    plt.axis((min(binstoDo),max(binstoDo),0,maxplot))
    #line_up, = plt.plot(binstoDo,linestyle='-',marker='o', color='k',label="fake-only")
    #line_down, = ax.plot(binstoDo,linestyle='--',marker='x', color='k',label="fake+ttV+EWK")
    #legend1 = plt.legend(handles=[line_up], loc='best') # , line_down
    ax.set_ylabel('err/content last bin')
    ax.legend(loc='best', fancybox=False, shadow=False, ncol=1, fontsize=8) #, ncol=3)
    plt.grid(True)
    if options.BINtype == "none" : namefig=local+'/'+options.variables+'_fullsim_ErrOcont_none.pdf'
    if options.BINtype == "quantiles" : namefig=local+'/'+options.variables+'_fullsim_ErrOcont_quantiles.pdf'
    if options.BINtype == "regular" or options.BINtype == "mTauTauVis": namefig=local+'/'+options.variables+'_fullsim_ErrOcont.pdf'
    if options.BINtype == "ranged" : namefig=local+'/'+options.variables+'_fullsim_ErrOcont_ranged.pdf'
    fig.savefig(namefig)
    print ("saved",namefig)
    #########################################
    ## plot quantiles boundaries
    if options.BINtype == "quantiles" :
        fig, ax = plt.subplots(figsize=(5, 5))
        plt.title(options.BINtype+" binning "+options.variables)
        #colorsToDo=['r','g','b','m','y','c']
        for nn,source in enumerate(sources) :
            print (len(binstoDo),len(lastQuant[nn]))
            plt.plot(binstoDo,lastQuant[nn], color=colorsToDo[nn],linestyle='-')
            plt.plot(binstoDo,lastQuant[nn], color=colorsToDo[nn],linestyle='-',marker='o') # ,label=bdtTypesToDo[nn]
            plt.plot(binstoDo,xmaxQuant[nn], color=colorsToDo[nn],linestyle='--',marker='x')
            plt.plot(binstoDo,xminQuant[nn], color=colorsToDo[nn],linestyle='--',marker='.')
        ax.set_xlabel('nbins')
        plt.axis((min(binstoDo),max(binstoDo),0,1.0))
        line_up, = plt.plot(binstoDo, 'o-', color='k',label="last bin low")
        line_down, = ax.plot(binstoDo, 'x--', color='k',label="Max")
        line_d, = ax.plot(binstoDo, '.--', color='k',label="Min")
        legend1 = plt.legend(handles=[line_up, line_down, line_d], loc='best', fontsize=8)
        ax.set_ylabel('boundary')
        #ax.legend(loc='best', fancybox=False, shadow=False, ncol=2)
        plt.grid(True)
        fig.savefig(local+'/'+options.variables+'_fullsim_boundaries_quantiles.pdf')

#########################################
## make limits
print sources
if doLimits :
    print "do limits"
    print sources
    fig, ax = plt.subplots(figsize=(5, 5))
    #plt.title(options.BINtype+" binning")
    #colorsToDo=['r','g','b','m','y','c', 'fuchsia']
    if options.BINtype == "quantiles" : namefig=local+'/'+options.variables+'_fullsim_limits_quantiles'
    if options.BINtype == "regular" or options.BINtype == "mTauTauVis": namefig=local+'/'+options.variables+'_fullsim_limits'
    if options.BINtype == "ranged" : namefig=local+'/'+options.variables+'_fullsim_limits_ranged'
    file = open(namefig+".csv","w")
    for nn,source in enumerate(sources) :
        #options.variables+'_'+bdtTypesToDoFile[ns]+'_nbin_'+str(nbins)
        limits=ReadLimits(bdtTypesToDoFile[nn], binstoDo, options.BINtype,channel,local,0,0)
        print (len(binstoDo),len(limits[0]))
        for jj in limits[0] : file.write(str(jj)+', ')
        file.write('\n')
        plt.plot(binstoDo,limits[0], color=colorsToDo[nn],linestyle='-',marker='o',label=bdtTypesToDoLabel[nn])
        plt.plot(binstoDo,limits[1], color=colorsToDo[nn],linestyle='-')
        plt.plot(binstoDo,limits[3], color=colorsToDo[nn],linestyle='-')
    ax.legend(loc='best', fancybox=False, shadow=False , ncol=1, fontsize=8)
    ax.set_xlabel('nbins')
    ax.set_ylabel('limits')
    maxsum=0
    if channel == "0l_2tau" : maxlim = 10.5
    else : maxlim = 2.0
    plt.axis((min(binstoDo),max(binstoDo),0.5, maxlim))
    #if channel=="2lss_1tau" : plt.axis((min(binstoDo),max(binstoDo),0.7,2.5))
    #if channel=="1l_2tau" :
    #    plt.axis((min(binstoDo),max(binstoDo),2.0,6.5))
    #    #plt.yscale('log')
    #    maxsum=5
    plt.text(11.3, 2.4, options.BINtype+" binning "+" "+options.variables )
    #plt.text(2.3, 10.53+maxsum, "CMS"  ,  fontweight='bold' )
    #plt.text(4.3, 10.53+maxsum, "preliminary" )
    #plt.text(max(binstoDo)-6.0, 2.53+maxsum, "35.9/fb (13 TeV)"   )
    fig.savefig(namefig+'.pdf')
    file.close()
    print ("saved",namefig)
