#!/usr/bin/env python
import os, subprocess, sys
workingDir = os.getcwd()

from ROOT import *
import numpy as np
from pathlib2 import Path

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# ./rebin_datacards.py --channel "2lss_1tau"  --BINtype "regular" --doLimits

from optparse import OptionParser
parser = OptionParser()
parser.add_option("--channel ", type="string", dest="channel", help="The ones whose variables implemented now are:\n   - 1l_2tau\n   - 2lss_1tau\n It will create a local folder and store the report*/xml", default="2lss_1tau")
parser.add_option("--variables", type="string", dest="variables", help="Add convention to file name", default="teste")
parser.add_option("--BINtype", type="string", dest="BINtype", help="regular / ranged / quantiles", default="regular")
parser.add_option("--doPlots", action="store_true", dest="doPlots", help="If you call this will not do plots with repport", default=False)
parser.add_option("--doLimits", action="store_true", dest="doLimits", help="If you call this will not do plots with repport", default=False)
(options, args) = parser.parse_args()

doLimits=options.doLimits
doPlots=options.doPlots
channel=options.channel

channel=options.channel
if channel == "2lss_1tau" : sys.exit("Please make the corresponding input card")
if channel == "1l_2tau"   : sys.exit("Please make the corresponding input card")
if channel == "2l_2tau"   : sys.exit("Please make the corresponding input card")
if channel == "3l_1tau"   : sys.exit("Please make the corresponding input card")
if channel == "0l_2tau"   : sys.exit("Please make the corresponding input card")
if channel == "2lss_0tau" : execfile("../cards/info_2lss_0tau_datacards.py")
if channel == "3l_0tau"   : execfile("../cards/info_3l_0tau_datacards.py")
if channel == "2los_1tau" : sys.exit("Please make the corresponding input card")
if channel == "hh_bb2l"   : sys.exit("Please make the corresponding input card")
if channel == "hh_bb1l"   : sys.exit("Please make the corresponding input card")

info = read_from()
execfile("../python/data_manager.py")

sources=[]
bdtTypesToDo=[]
bdtTypesToDoLabel=[]
bdtTypesToDoFile=[]

import shutil,subprocess
proc=subprocess.Popen(["mkdir " + info["label"]],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()
local = workingDir + "/" + info["label"] 
#local=workingDir+"/"+options.channel+"_"+label+"/"+options.variables+"/"

counter=0
for ii, bdtType in enumerate(info["bdtTypes"]) :
    fileName = info["mom"] + "/prepareDatacards_" + info["ch_nickname"] + "_" + bdtType + ".root"
    my_file = Path(fileName)
    source=local+"/prepareDatacards_" + info["ch_nickname"] + "_" + bdtType
    print (fileName)
    if my_file.exists() :
        proc=subprocess.Popen(['cp ' + fileName + " " + local],shell=True,stdout=subprocess.PIPE)
        out = proc.stdout.read()
        sources = sources + [source]
        bdtTypesToDo = bdtTypesToDo +[channel+" "+bdtType]
        bdtTypesToDoLabel = bdtTypesToDoLabel +[channel+" "+bdtType]
        bdtTypesToDoFile=bdtTypesToDoFile+[channel +"_"+bdtType]
        ++counter
        print ("rebinning ",sources[counter])
    else : print ("does not exist ",source)
print ("I will rebin",bdtTypesToDoLabel,"(",len(sources),") BDT options")

if options.BINtype == "regular" or options.BINtype == "ranged" : binstoDo = info["nbinRegular"]
if options.BINtype == "quantiles" : binstoDo = info["nbinQuant"]
if options.BINtype == "none" : binstoDo=np.arange(1, info["originalBinning"])
print binstoDo

colorsToDo=['r','g','b','m','y','c', 'fuchsia', "peachpuff",'k','orange'] #['r','g','b','m','y','c','k']
if not doLimits : 
    #########################################
    ## make rebinned datacards
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.title(options.BINtype+" in sum of BKG ")
    lastQuant=[]
    xmaxQuant=[]
    xminQuant=[]
    bin_isMoreThan02 = []
    maxplot = -99.
    ncolor = 0
    ncolor2 = 0
    linestyletype = "-"
    for nn,source in enumerate(sources) :
        errOcont=rebinRegular(
            source, 
            binstoDo, 
            options.BINtype,
            info["originalBinning"],
            doPlots,
            bdtTypesToDo[nn], 
            info["withFolder"]
            )
        if max(errOcont[2]) > maxplot : maxplot = max(errOcont[2])
        print bdtTypesToDo[nn]
        lastQuant=lastQuant+[errOcont[4]]
        xmaxQuant=xmaxQuant+[errOcont[5]]
        xminQuant=xminQuant+[errOcont[6]]
        bin_isMoreThan02 = bin_isMoreThan02 + [errOcont[7]]

        print (binstoDo,errOcont[2])
        plt.plot(binstoDo, errOcont[2], color=colorsToDo[ncolor],linestyle=linestyletype,label=bdtTypesToDo[nn].replace("3l_0tau output_NN_3l_ttH_tH_3cat_v8_", "") ) # ,label=bdtTypesToDo[nn]
        #plt.plot(binstoDo, errOcont[2], color=colorsToDo[ncolor],linestyle=linestyletype,marker='o',label=bdtTypesToDo[nn].replace("2lss_0tau output_NN_2lss_ttH_tH_4cat_onlyTHQ_v4_", "") ) #
        ncolor = ncolor + 1
        if ncolor == 10 : 
            ncolor = 0
            ncolor2 = ncolor2 + 1
            if ncolor2 == 0 : linestyletype = "-"
            if ncolor2 == 1 : linestyletype = "-."
            if ncolor2 == 2 : linestyletype = ":"
        #plt.plot(binstoDo,errOcont[2], color=colorsToDo[nn],linestyle='--',marker='x')
        ax.set_xlabel('nbins')
    #if options.BINtype == "regular" : maxplot = 0.3 #0.02
    #elif options.BINtype == "mTauTauVis" : maxplot=200.
    #else : maxplot =1.0 # 0.35
    maxplot = 0.6
    plt.axis((min(binstoDo),max(binstoDo),0,maxplot*1.2))
    #line_up, = plt.plot(binstoDo,linestyle='-',marker='o', color='k',label="fake-only")
    #line_down, = ax.plot(binstoDo,linestyle='--',marker='x', color='k',label="fake+ttV+EWK")
    #legend1 = plt.legend(handles=[line_up], loc='best') # , line_down
    ax.set_ylabel('err/content last bin')
    ax.legend(loc='best', fancybox=False, shadow=False, ncol=3, fontsize=8)
    plt.grid(True)
    namefig = local + '/' + options.variables + '_ErrOcont_' + options.BINtype + '.pdf'
    fig.savefig(namefig)
    print ("saved",namefig)
    print (bin_isMoreThan02)
    #########################################
    ## plot quantiles boundaries
    if options.BINtype == "quantiles" :
        ncolor = 0
        fig, ax = plt.subplots(figsize=(5, 5))
        plt.title(options.BINtype+" binning "+options.variables)
        #colorsToDo=['r','g','b','m','y','c']
        linestyletype = "-"
        for nn,source in enumerate(sources) :
            print (len(binstoDo),len(lastQuant[nn-1]))
            plt.plot(binstoDo,lastQuant[nn], color=colorsToDo[ncolor],linestyle=linestyletype)
            plt.plot(binstoDo,lastQuant[nn], color=colorsToDo[ncolor],linestyle=linestyletype,marker='o') # ,label=bdtTypesToDo[nn]
            plt.plot(binstoDo,xmaxQuant[nn], color=colorsToDo[ncolor],linestyle=linestyletype,marker='x')
            plt.plot(binstoDo,xminQuant[nn], color=colorsToDo[ncolor],linestyle=linestyletype,marker='.')
            ncolor = ncolor + 1
            if ncolor == 10 or ncolor == 20: 
                ncolor = 0
                if ncolor == 10 : linestyletype = "--"
                if ncolor == 20 : linestyletype = ":"
        ax.set_xlabel('nbins')
        ax.set_ylabel('err/content last bin')
        plt.axis((min(binstoDo),max(binstoDo),0,1.0))
        line_up, = plt.plot(binstoDo, 'o-', color='k',label="last bin low")
        line_down, = ax.plot(binstoDo, 'x--', color='k',label="Max")
        line_d, = ax.plot(binstoDo, '.--', color='k',label="Min")
        legend1 = plt.legend(handles=[line_up, line_down, line_d], loc='best', fontsize=8)
        ax.set_ylabel('boundary')
        plt.grid(True)
        fig.savefig(local+'/'+options.variables+'_fullsim_boundaries_quantiles.pdf')

#########################################
## make limits
print sources
if doLimits :
    print "do limits"
    fig, ax = plt.subplots(figsize=(5, 5))
    if options.BINtype == "quantiles" : namefig=local+'/'+options.variables+'_fullsim_limits_quantiles'
    if options.BINtype == "regular" or options.BINtype == "mTauTauVis": namefig=local+'/'+options.variables+'_'+options.channel+'_'+options.subchannel+'_fullsim_limits'
    if options.BINtype == "ranged" : namefig=local+'/'+options.variables+'_fullsim_limits_ranged'
    file = open(namefig+".csv","w")
    #maxlim =-99.
    for nn,source in enumerate(sources) :
        limits=ReadLimits(bdtTypesToDoFile[nn], binstoDo, options.BINtype,channel,local,0,0)
        print (len(binstoDo),len(limits[0]))
        print 'binstoDo= ', binstoDo
        print limits[0]
        for jj in limits[0] : file.write(str(jj)+', ')
        file.write('\n')
        plt.plot(binstoDo,limits[0], color=colorsToDo[nn],linestyle='-',marker='o',label=bdtTypesToDoLabel[nn])
        plt.plot(binstoDo,limits[1], color=colorsToDo[nn],linestyle='-')
        plt.plot(binstoDo,limits[3], color=colorsToDo[nn],linestyle='-')
        #if maxlim < max(limits[3]) : maxlim = max(limits[3])
    ax.legend(loc='best', fancybox=False, shadow=False , ncol=1, fontsize=8)
    ax.set_xlabel('nbins')
    ax.set_ylabel('limits')
    maxsum=0
    if channel in ["0l_2tau", "2los_1tau", "hh_bb2l"] : maxlim = 10.5
    elif channel in ["hh_bb1l"] : maxlim = 200.
    else : maxlim = 2.0
    plt.axis((min(binstoDo),max(binstoDo),0.5, maxlim))
    plt.text(0.3, 1.4, options.BINtype+" binning "+" "+options.variables )
    fig.savefig(namefig+'.png')
    file.close()
    print ("saved",namefig)
