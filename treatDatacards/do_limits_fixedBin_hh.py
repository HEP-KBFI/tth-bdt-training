#!/usr/bin/env python
import os, subprocess, sys
import datetime
from array import array
import CombineHarvester.CombineTools.ch as ch
from ROOT import *
from math import sqrt, sin, cos, tan, exp
import numpy as np
workingDir = os.getcwd()
#from pathlib2 import Path

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# cd /home/acaan/VHbbNtuples_8_0_x/CMSSW_7_4_7/src/ ; cmsenv ; cd -
# python do_limits_fixedBin.py --channel "2lss_1tau" --uni "Tallinn"
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--channel", type="string", dest="channel", help="The ones whose variables implemented now are:\n   - 1l_2tau\n   - 2lss_1tau\n It will create a local folder and store the report*/xml", default="none")
parser.add_option("--uni", type="string", dest="uni", help="  Set of variables to use -- it shall be put by hand in the code", default="Tallinn")
(options, args) = parser.parse_args()

makeDatacardTxt = True
doLimits   = True
doImpacts  = True
doYields   = False
doGOF      = True
doPlots    = True
readLimits = False
blinded    = True

readDatacard = True


#masses = {"400", "700"};
masses = {"400"};
parent_spins = {"radion"}
# parent_spins = {"radion", "graviton"} ## We don't have graviton samples yet


channel = options.channel
university = options.uni

workingDir = os.getcwd()
datacardDir_output = os.path.join(workingDir, "datacards")

os.chdir(os.environ['CMSSW_BASE'] + "/src/tthAnalysis/bdtTraining/treatDatacards/")
execfile("../python/data_manager.py")
print run_cmd('pwd')


if university == "Tallinn_alternative":
    takeRebinedFolder=False
    add_x_prefix=False
    doRebin = True
    divideByBinWidth = "false"
    doKeepBlinded = "true"
    autoMCstats = "true"
    useSyst = "true" # use shape syst
    mom = "/home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun09/"
    local = "Tallinn_alternative/"
    card_prefix = "prepareDatacards_"
    cards = [
    "1l_2tau_mTauTauVis_x",
    "2lss_1tau_sumOS_mTauTauVis_x",
    "2l_2tau_mTauTauVis_x",
    "3l_1tau_mTauTauVis_x",
    #
    "1l_2tau_numJets_x",
    "2lss_1tau_sumOS_numJets_x",
    "2l_2tau_numJets_x",
    "3l_1tau_numJets_x"
    ]

    channels = [
    "1l_2tau",
    "2lss_1tau",
    "2l_2tau",
    "3l_1tau",
    #
    "1l_2tau",
    "2lss_1tau",
    "2l_2tau",
    "3l_1tau"
    ]

if university == "Tallinn_CR":
    takeRebinedFolder=False
    add_x_prefix=False
    doRebin = False
    divideByBinWidth = "false"
    doKeepBlinded = "false"
    autoMCstats = "true"
    useSyst = "true" # use shape syst
    mom = "/home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun09/"
    local = "Tallinn_CR/"
    card_prefix = "prepareDatacards_"
    cards = [
    "ttWctrl_mvaDiscr_2lss",
    "ttWctrl_EventCounter",
    "ttWctrl_numJets",
    "ttZctrl_mvaDiscr_3l",
    "ttZctrl_EventCounter",
    "ttZctrl_mT",
    "ttZctrl_numJets",
    "ttZctrl_mLL"
    ]

    channels = [
    "ttWctrl",
    "ttWctrl",
    "ttWctrl",
    "ttZctrl",
    "ttZctrl",
    "ttZctrl",
    "ttZctrl",
    "ttZctrl",
    ]

if university == "Tallinn_HH":
    takeRebinedFolder=False
    add_x_prefix=False
    doRebin = False
    doKeepBlinded = "true"
    autoMCstats = "true"
    useSyst = "true" # use shape syst
    mom = "/home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun28/"
    local = "Tallinn/"
    card_prefix = "addSystFakeRates_"
    cards = [
    "1l_2tau_OS_mvaOutput_final_x",
    "2lss_1tau_sumOS_mvaOutput_final_x",
    "2l_2tau_sumOS_mvaOutput_final_x",
    "3l_1tau_OS_mvaOutput_final_x",
    "ttWctrl_mvaDiscr_2lss_x",
    "ttZctrl_mvaDiscr_3l_x"
    ]

    channels = [
    "1l_2tau",
    "2lss_1tau",
    "2l_2tau",
    "3l_1tau",
    "ttWctrl",
    "ttZctrl"
    ]

    folders = [
    "ttH_1l_2tau/",
    "",
    "ttH_2l_2tau/",
    "ttH_3l_1tau/",
    "ttH_2lss_1tau/",
    "ttH_1l_2tau/",
    "ttH_1l_2tau/"
    ]

if university == "Tallinn":
    takeRebinedFolder=False
    add_x_prefix=False
    doRebin = False
    doKeepBlinded = "true"
    autoMCstats = "true"
    useSyst = "true" # use shape syst
    mom = "/home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun09/"

    local = "Tallinn/"
    card_prefix = "prepareDatacards_"
    cards = [
    "1l_2tau_mvaOutput_final_x",
    "2lss_1tau_sumOS_mvaOutput_final_x",
    "2l_2tau_mvaOutput_final_regularBin", #"2l_2tau_mvaOutput_final_x",
    "3l_1tau_mvaOutput_final_x_noNeg",
    "2lss_1tau_sumOS_mvaOutput_2lss_1tau_HTT_SUM_M",
    "1l_2tau_mvaOutput_HTT_SUM_VT",
    #
    "1l_2tau_SS_mTauTauVis_x",
    "1l_2tau_SS_mvaOutput_final_x",
    "1l_2tau_SS_EventCounter_x",
    "1l_2tau_SS_EventCounter",
    ]

    channels = [
    "1l_2tau",
    "2lss_1tau",
    "2l_2tau",
    "3l_1tau",
    "2lss_1tau",
    "1l_2tau",
    #
    "1l_2tau",
    "1l_2tau",
    "1l_2tau",
    "1l_2tau",


    ]

    folders = [
    "ttH_1l_2tau/",
    "",
    "ttH_2l_2tau/",
    "ttH_3l_1tau/",
    "ttH_2lss_1tau/",
    "ttH_1l_2tau/",
    #
    "ttH_1l_2tau/",
    "ttH_1l_2tau/",
    "ttH_1l_2tau/",


    ]

##################


if university == "TIFR":
    typeFit = "postfit"
    takeRebinedFolder=False
    add_x_prefix=False
    doRebin = True
    doKeepBlinded = "true"
    autoMCstats = "true"
    useSyst = "false" # use shape syst

    mom = "/home/acaan/CMSSW_9_4_0_pre1/src/tth-bdt-training-test/treatDatacards/"
    mom_original = "/home/mmaurya/VHbbNtuples_8_0_x/CMSSW_8_0_21/src/"
    local = "to_delete_soon/"
    local_original = "Oct2018/"
    card_prefix = "prepareDatacards_"
    cards = [
    "2los_1tau_mvaOutput_2los_1tau_evtLevelSUM_TTH_19Var"
    ]

    channels = [
    "2los_1tau"
    ]

    folders = [
    "ttH_2los_1tau",
    ]


if university == "TIFR" and channel == "hh_3l":
    typeFit = "postfit"
    takeRebinedFolder=False
    add_x_prefix=False
    doRebin = False
    #doKeepBlinded = "true"
    typeKeepBlinded = 'partialBlinded' # 'blinded', 'partialBlinded', 'unblinded'
    autoMCstats = "true"
    useSyst = "true" # use shape syst
    useAsimov = False # combine on Asimov 
    useCmdRMin = False
    
    mom = "/home/ssawant/VHbbNtuples_9_4_x/CMSSW_9_4_6_patch1_rpst2/CMSSW_9_4_6_patch1/src/tth-bdt-training/treatDatacards/"
    mom_original = mom
    local = "hh_3l/"
    local_original = local
    card_prefix = "prepareDatacards_hh_3l_"
    cards = [
    #"hh_3l_mvaOutput_xgb_hh_3l_SUMBk_HH_10bins"
    "hh_3l_mvaOutput_xgb_hh_3l_SUMBk_HH_10bins_SymmetrizeSyst"
    ]

    channels = [
    "hh_3l",
    ]

    folders = [
    "",
    ]

    
elif university == "Cornell":
    takeRebinedFolder=False
    add_x_prefix=False
    doRebin = False
    doKeepBlinded = "true"
    autoMCstats = "true"
    useSyst = "true" # use shape syst
    mom = "/home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun05/"
    local = "Cornell/ch/"
    card_prefix = "datacards_"
    cards = [
    "1l2tau_41p53invfb_Binned_2018jun04",
    "2lss1tau_41p53invfb_Binned_2018jun04",
    "2l2tau_41p53invfb_Binned_2018jun04",
    "3l1tau_41p53invfb_Binned_2018jun04"
    ]

    channels = [
    "1l_2tau",
    "2lss_1tau",
    "2l_2tau",
    "3l_1tau"
    ]


print ("to run this script your CMSSW_base should be the one that CombineHavester installed")


if not readLimits :
    for nn, card in enumerate(cards) :
        if not nn < 4 and channel == "none" : continue
        elif not channel == "none" and not channels[nn] == channel : continue
        #####################################################################
        wdata = "" # to append to WriteDatacard_$channel
        hasConversions = "true"
        if channels[nn] == "1l_2tau" :
            wdata = ""
            hasFlips = "false"
            isSplit = "false"
            max = 2000
            minimim = 0.1
            dolog = "true"
            divideByBinWidth = "false"
        if channels[nn] == "2l_2tau" :
            wdata = ""
            hasFlips = "false"
            isSplit = "false"
            max = 15.0
            minimim = 0.0
            dolog = "false"
            divideByBinWidth = "false"
        if channels[nn] == "3l_1tau" :
            #if "CR" not in university : wdata = "_FRjt_syst"
            #else :
            wdata = ""
            isSplit = "true"
            hasFlips = "false"
            max = 2.5
            minimim = 0.0
            dolog = "false"
            divideByBinWidth = "true"
        if channels[nn] == "2lss_1tau" :
            #if "CR" not in university : wdata = "_FRjt_syst"
            #else :
            wdata = ""
            isSplit = "true"
            hasFlips = "true"
            max = 15.0
            minimim = 0.0
            dolog = "false"
            divideByBinWidth = "true"
        if "ctrl" in channels[nn] :
            wdata = ""
            hasFlips = "true"
            isSplit = "false"
            max = 500000
            minimim = 0.1
            dolog = "true"
            divideByBinWidth = "false"

        if channels[nn] == "2los_1tau" :
            wdata = ""
            hasFlips = "false"
            isSplit = "false"
            max = 15.0
            minimim = 0.0
            dolog = "false"
            divideByBinWidth = "false"

        if channels[nn] == "hh_3l" :
            wdata = ""
            hasFlips = "false"
            isSplit = "false"
            max = 15.0
            minimim = 0.0
            dolog = "true"
            divideByBinWidth = "true"

            
        #####################################################################
        print str(datetime.datetime.now());
        #run_cmd('cd %s' % workingDir)
        os.chdir(workingDir);     print run_cmd('pwd')
        datacardFile_input0 = mom_original+local_original+card_prefix+card+'.root'
        run_cmd('cp %s %s/' % (datacardFile_input0, workingDir))
        datacardFile_input1 = ("%s/%s%s.root" % (workingDir, card_prefix, card))
        if os.path.exists(datacardFile_input0) and os.path.exists(datacardFile_input1):            
            file = TFile(datacardFile_input1,"READ");
            print ("testing ", datacardFile_input1)
            datacardFile_input2 = workingDir+"/"+card_prefix+card+'_noNeg.root' # remove the negatives
            file2 = TFile(datacardFile_input2,"RECREATE");
            file2.cd()
            h2 = TH1F()
            for keyO in file.GetListOfKeys() :
               obj =  keyO.ReadObj()
               if (type(obj) is not TH1F) :
                   if  (type(obj) is not TH1D) :
                       if (type(obj) is not TH1) :
                           if (type(obj) is not TH1I) : continue
               if not takeRebinedFolder :
                   h2=obj.Clone()
               else :
                   h2 = file.Get(folders[nn]+"rebinned/"+str(keyO.GetName())+"_rebinned")
               if doRebin :
                   #print("passed do rebin", channels[nn])
                   if channels[nn] == "2l_2tau" : h2.Rebin(5)
                   if channels[nn] == "3l_1tau" : h2.Rebin(4)
                   ###Rebinning for 2los_1tau:
                   if channels[nn] == "2los_1tau" :
                       h2.Rebin(4)
                       #print("HI")
               for bin in range (0, h2.GetXaxis().GetNbins()) :
                   if h2.GetBinContent(bin) < 0 :
                       h2.AddBinContent(bin, abs(h2.GetBinContent(bin))+0.01)
                   h2.SetBinError(bin, min(h2.GetBinContent(bin), h2.GetBinError(bin)) ) # crop all uncertainties to 100% to avoid negative variations
               #print(h2.GetSize,h2.GetBinContent(bin))
	       if add_x_prefix : h2.SetName("x_"+str(keyO.GetName()))
               h2.Write()
            file2.Close()
            print ("did ", datacardFile_input2)

            for mass in masses:
                for spin in parent_spins:
                    #run_cmd('cd %s' % workingDir)
                    os.chdir(workingDir);    print run_cmd('pwd')
                    run_cmd('mkdir -p %s/%s/%s/%s' %(workingDir, "datacards", spin, mass))
                    #run_cmd('cd %s/%s/%s/%s' %(workingDir, "datacards", spin, mass))
                    os.chdir('%s/%s/%s/%s' %(workingDir, "datacards", spin, mass));    print run_cmd('pwd')

                    datacardFile_input = datacardFile_input2
                    datacardFile_output = card
                    datacard_txt = datacardFile_output+".txt"
                    logFile_dc = card + "_WriteDatacards.log"

                    if makeDatacardTxt:
                        run_cmd('rm %s*' % datacardFile_output)
                        run_cmd('%s --input_file=%s --output_file=%s --add_shape_sys=%s --mass=%s --type=%s &> %s' % ('WriteDatacards_'+channels[nn]+wdata, datacardFile_input, datacardFile_output, useSyst, mass, spin, logFile_dc))
                    print run_cmd('pwd'); print run_cmd('ls -ltrh'); 
                    
                    logFile_ws = datacardFile_output + "_ws.log"
                    datacard_ws = datacardFile_output + "_ws.root"
                    run_cmd('rm %s %s' % (datacard_ws, logFile_ws)) 
                    run_cmd('text2workspace.py %s -m %s -o %s &> %s' % (datacard_txt, str(125), datacard_ws, logFile_ws))
                    print str(datetime.datetime.now()); print run_cmd('pwd'); print run_cmd('ls -ltrh'); sys.stdout.flush();


                    cmdAdd  = ""
                    sAddCmd = ""
                    if useCmdRMin:
                        cmdAdd  += " --rMin =-4.0"
                        sAddCmd += "_rMinm4"

                    cmdAsimov = ""
                    sAsimov   = ""
                    if useAsimov:
                        cmdAsimov = " -t -1"
                        sAsimov   = "_asimov"                        
                    cmdAdd  += cmdAsimov
                    sAddCmd += sAsimov
                    
                        
                    if doLimits :                        
                        run_cmd('combine -M AsymptoticLimits -m %s  --run blind -S 0 %s  %s &> %s' % (str(125), datacard_txt, cmdAdd, datacardFile_output+sAddCmd+"_lmt_wBlind_noSyst.log")) # -t -1
                        run_cmd('combine -M AsymptoticLimits -m %s  --run blind %s  %s &> %s' % (str(125), datacard_txt, cmdAdd, datacardFile_output+sAddCmd+"_lmt_wBlind.log")) # -t -1

                        run_cmd('combineTool.py -M AsymptoticLimits %s -m %s --run blind  %s &> %s' % (datacard_ws, str(125), cmdAdd, datacardFile_output+sAddCmd+"_lmt_wBlind1.log"))
                        run_cmd('combineTool.py -M AsymptoticLimits %s -m %s  %s &> %s' % (datacard_ws, str(125), cmdAdd, datacardFile_output+sAddCmd+"_lmt_woBlind.log"))

                        run_cmd('combineTool.py -M FitDiagnostics %s -m %s --run blind  %s &> %s' % (datacard_ws, str(125), cmdAdd, datacardFile_output+sAddCmd+"_maxl_wBlind.log"))
                        run_cmd('combineTool.py -M FitDiagnostics %s -m %s  %s &> %s' % (datacard_ws, str(125), cmdAdd, datacardFile_output+sAddCmd+"_maxl_woBlind.log"))
                        
                        run_cmd('rm higgsCombineTest.AsymptoticLimits.mH125.root')
                        print str(datetime.datetime.now()); print run_cmd('pwd'); print run_cmd('ls -ltrh'); sys.stdout.flush();
                        

                    if doGOF :
                        run_cmd('rm higgsCombineTest.GoodnessOfFit.mH125*.root') 
                        run_cmd('combine -M GoodnessOfFit --algo=saturated --fixedSignalStrength=1 -m 125 %s &> %s' % (datacard_ws, datacardFile_output+"_gof_data.log"))
                        print str(datetime.datetime.now()); sys.stdout.flush();
                        
                        #run_cmd('combine -M GoodnessOfFit --algo=saturated --fixedSignalStrength=1 -m 125 -t 1000 -s 12345  %s --saveToys --toysFreq &> %s' % (datacard_ws, datacardFile_output+"_gof_toyMC.log"))
                        # run toyMC in steps; and then hadd all
                        nToyEvts = 1000
                        nRuns    = 40
                        seedFirst = 10000
                        nEvtPerRun = int(nToyEvts/nRuns)
                        for seed in range(seedFirst, seedFirst+nRuns):
                            run_cmd('combine -M GoodnessOfFit --algo=saturated --fixedSignalStrength=1 -m 125 %s  -t %i -s %i  --saveToys --toysFreq &> %s' % (datacard_ws, nEvtPerRun, seed, datacardFile_output+"_gof_toyMC_"+str(seed)+".log"))
                            print str(datetime.datetime.now()); sys.stdout.flush();
                        sToyAll = 'higgsCombineTest.GoodnessOfFit.mH125.toys.root'
                        cmdHadd = 'hadd  %s  ' % sToyAll
                        for seed in range(seedFirst, seedFirst+nRuns):
                            cmdHadd += 'higgsCombineTest.GoodnessOfFit.mH125.%i.root ' % seed
                        run_cmd('%s &> %s' % (cmdHadd, datacardFile_output+"_gof_toyMC_hadd.log"));
                        print str(datetime.datetime.now()); print run_cmd('pwd'); print run_cmd('ls -ltrh'); sys.stdout.flush();
                        
                        #run_cmd('combineTool.py -M CollectGoodnessOfFit --input higgsCombineTest.GoodnessOfFit.mH125.root higgsCombineTest.GoodnessOfFit.mH125.12345.root -o GoF_saturated.json &> %s' % (datacardFile_output+"_gof1.log"))
                        run_cmd('combineTool.py -M CollectGoodnessOfFit --input higgsCombineTest.GoodnessOfFit.mH125.root %s -o GoF_saturated.json &> %s' % (sToyAll, datacardFile_output+"_gof1.log"))
                        run_cmd('$CMSSW_BASE/src/CombineHarvester/CombineTools/scripts/plotGof.py --statistic saturated --mass 125.0 GoF_saturated.json -o GoF_saturated &> %s' % (datacardFile_output+"_gof2.log"))
                        run_cmd('mv GoF_saturated.pdf  GoF_saturated_'+channels[nn]+"_"+cards[nn]+"_"+university+'.pdf')
                        run_cmd('mv GoF_saturated.png  GoF_saturated_'+channels[nn]+"_"+cards[nn]+"_"+university+'.png')
                        #run_cmd('rm higgsCombine*root')
                        #run_cmd('rm GoF_saturated.json')
                        print str(datetime.datetime.now()); print run_cmd('pwd'); print run_cmd('ls -ltrh'); sys.stdout.flush();
                        
                        
                    if doImpacts :
                        print str(datetime.datetime.now()); print run_cmd('pwd');  print run_cmd('ls -ltrh');
                        #combineTool.py -M T2W:  Run text2workspace.py on multiple cards or directories
                        #run_cmd('combineTool.py -M T2W -i %s &> %s' % (datacard_txt, datacardFile_output+"_ws_impact.log"))
                        run_cmd('combineTool.py -M Impacts -m 125 -d %s --expectSignal 1 --allPars --parallel 8 --doInitialFit  %s &> %s' % (datacard_ws, cmdAdd, datacardFile_output+sAddCmd+"_impact1.log")) #  -t -1
                        #run_cmd('combineTool.py -M Impacts -m 125 -d %s  --allPars --parallel 8 --doInitialFit  %s &> %s' % (datacard_ws, cmdAdd, datacardFile_output+sAddCmd+"_impact1.log")) #  -t -1
                        run_cmd('combineTool.py -M Impacts -m 125 -d %s --expectSignal 1 --allPars --parallel 8  --robustFit 1 --doFits  %s &> %s' % (datacard_ws, cmdAdd, datacardFile_output+sAddCmd+"_impact2.log")) # -t -1
                        #run_cmd('combineTool.py -M Impacts -m 125 -d %s  --allPars --parallel 8  --robustFit 1 --doFits  %s &> %s' % (datacard_ws, cmdAdd, datacardFile_output+sAddCmd+"_impact2.log")) # -t -1
                        run_cmd('combineTool.py -M Impacts -m 125 -d %s -o impacts.json &> %s' % (datacard_ws, datacardFile_output+sAddCmd+"_impact3.log"))
                        run_cmd('rm higgsCombine_*Fit_*.root')
                        run_cmd('plotImpacts.py -i impacts.json -o  impacts &> %s' % (datacardFile_output+sAddCmd+"_impact4.log"))
                        run_cmd('mv impacts.pdf '+'impacts_'+channels[nn]+"_"+cards[nn]+'_'+university+sAddCmd+'.pdf')
                        print str(datetime.datetime.now()); print run_cmd('pwd'); print run_cmd('ls -ltrh');  sys.stdout.flush();
                        

                    if doYields :
                        #run_cmd('%s --input_file=%s --output_file=%s --add_shape_sys=%s --use_autoMCstats=%s' % ('WriteDatacards_'+channels[nn]+wdata, my_file, datacardFile_output, useSyst, autoMCstats))
                        #run_cmd('%s --input_file=%s --output_file=%s --add_shape_sys=%s ' % ('WriteDatacards_'+channels[nn]+wdata, my_file, datacardFile_output, useSyst))
                        #run_cmd('combine -M FitDiagnostics -d %s  -t -1 --expectSignal 1' % (txtFile))

                        run_cmd('combine -M FitDiagnostics  %s   --expectSignal 1   %s &> %s' % (datacard_ws, cmdAdd, datacardFile_output+"_FitDiagnostics_yield.log")) #  -t -1

                        run_cmd('python $CMSSW_BASE/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py -a fitDiagnostics.root -g plots.root')
                        #run_cmd('combineTool.py  -M T2W -i %s &> %s' % (datacard_txt, datacardFile_output+"_ws_impact.log"))
                        ROOT.PyConfig.IgnoreCommandLineOptions = True
                        gROOT.SetBatch(ROOT.kTRUE)
                        gSystem.Load('libHiggsAnalysisCombinedLimit')
                        #print ("Retrieving yields from: ",datacardFile_output+".root"); sys.stdout.flush()
                        print ("Retrieving yields from: ",datacard_ws); sys.stdout.flush()                        
                        fin = TFile(datacard_ws)
                        wsp = fin.Get('w')
                        cmb = ch.CombineHarvester()
                        cmb.SetFlag("workspaces-use-clone", True)
                        ch.ParseCombineWorkspace(cmb, wsp, 'ModelConfig', 'data_obs', False)
                        mlf = TFile('fitDiagnostics.root')
                        rfr = mlf.Get('fit_s')
                        print 'Pre-fit tables:'
                        filey = open("yields_"+channels[nn]+"_"+university+"_prefit.tex","w")
                        PrintTables(cmb, tuple(), ''+channels[nn], filey, university, channels[nn], blinded)
                        cmb.UpdateParameters(rfr)
                        print 'Post-fit tables:'
                        filey = open("yields_"+channels[nn]+"_"+university+"_postfit.tex","w")
                        PrintTables(cmb, (rfr, 500), ''+channels[nn], filey, university, channels[nn], blinded)
                        print ("the yields are on this file: ", "yields_"+channels[nn]+"_"+university+"_*.tex")


                    if doPlots :
                        print str(datetime.datetime.now()); print run_cmd('pwd');  print run_cmd('ls');  sys.stdout.flush();
                        #rootFile = ""+channels[nn]+"_"+card+"_shapes.root"
                        oFile_pfShapes = datacardFile_output+"_shapes.root"
                        
                        #run_cmd('combine -M FitDiagnostics -d %s -w %s -m 125  --expectSignal 1 &> %s' % (datacard_txt, datacard_ws, datacardFile_output+"_FitDiagnostics.log")) #  -t -1
                        run_cmd('combine -M FitDiagnostics  %s  -m 125  --expectSignal 1  %s &> %s' % (datacard_ws, cmdAdd, datacardFile_output+"_FitDiagnostics.log")) #  -t -1
                        print str(datetime.datetime.now()); print run_cmd('pwd');  print run_cmd('ls -ltrh');  sys.stdout.flush();
                        
                        #run_cmd('PostFitShapes -d %s -o %s -m 125 -f fitDiagnostics.root:fit_s --postfit --sampling --print' % (datacard_txt, rootFile)) # --postfit
                        run_cmd('PostFitShapesFromWorkspace -d %s -w %s -o %s -m 125 --sampling --print  -f fitDiagnostics.root:fit_s --postfit --sampling --print &> %s' % (datacard_txt, datacard_ws, oFile_pfShapes, datacardFile_output+"_pfShapes.log"))
                        print str(datetime.datetime.now()); print run_cmd('pwd');  print run_cmd('ls -ltrh');  sys.stdout.flush();
                        
                        exeScript = "execute_plots"+channels[nn]+"_"+university+".sh"
                        filesh = open(exeScript,"w")
                        filesh.write("#!/bin/bash\n")
                        '''
                        makeplots_prefit=('root -l -b -n -q \''+os.environ['CMSSW_BASE']+'/src/CombineHarvester/ttH_htt/macros/makePostFitPlots_hh_4W.C++("'
                                          +str(card)+'", "'+str("")+'", "'+str(channels[nn])+'", "'+str(".")+'", '+str(dolog)+', '+str(hasFlips)+', '+hasConversions+', "BDT", "", '+str(minimim)+', '+str(max)+', '+isSplit+', "prefit", '+divideByBinWidth+', '+doKeepBlinded+', '+str(mass)+')\'')
                        makeplots_postfit=('root -l -b -n -q \''+os.environ['CMSSW_BASE']+'/src/CombineHarvester/ttH_htt/macros/makePostFitPlots_hh_4W.C++("'
                                          +str(card)+'", "'+str("")+'", "'+str(channels[nn])+'", "'+str(".")+'", '+str(dolog)+', '+str(hasFlips)+', '+hasConversions+', "BDT", "", '+str(minimim)+', '+str(max)+', '+isSplit+', "postfit", '+divideByBinWidth+', '+doKeepBlinded+', '+str(mass)+')\'')
                        '''
                        makeplots_prefit=('root -l -b -n -q \''+os.environ['CMSSW_BASE']+'/src/CombineHarvester/ttH_htt/macros/makePostFitPlots_hh_4W.C++("'
                                          +str(card)+'", "'+str("")+'", "'+str(channels[nn])+'", "'+str(".")+'", '+str(dolog)+', '+str(hasFlips)+', '+hasConversions+', "BDT", "", '+str(minimim)+', '+str(max)+', '+isSplit+', "prefit", '+divideByBinWidth+', "'+typeKeepBlinded+'", '+str(mass)+')\'')
                        makeplots_postfit=('root -l -b -n -q \''+os.environ['CMSSW_BASE']+'/src/CombineHarvester/ttH_htt/macros/makePostFitPlots_hh_4W.C++("'
                                          +str(card)+'", "'+str("")+'", "'+str(channels[nn])+'", "'+str(".")+'", '+str(dolog)+', '+str(hasFlips)+', '+hasConversions+', "BDT", "", '+str(minimim)+', '+str(max)+', '+isSplit+', "postfit", '+divideByBinWidth+', "'+typeKeepBlinded+'", '+str(mass)+')\'')
                        

                        filesh.write(makeplots_prefit+ "\n\n")
                        filesh.write(makeplots_postfit+ "\n\n")
                        filesh.close()
                        print ("to have the plots take the makePlots command from: ",exeScript)
                        #run_cmd('bash %s > %s' % (exeScript, datacardFile_output+"_makePlotFitPlots.log"))
                        #subprocess.call('bash %s > %s' % (exeScript, datacardFile_output+"_makePlotFitPlots.log"))
                        os.chmod(exeScript, 0o755)
                        #subprocess.call("./%s" % exeScript, shell=True)
                        run_cmd('./%s > %s' % (exeScript, datacardFile_output+"_makePlotFitPlots.log"))


        else : print (datacardFile_input0,"does not exist ")
################################################################

## in the future this will also do the plots of limits
if readLimits :
    colorsToDo = np.arange(1,4)
    binstoDo=np.arange(1,4)
    file = open(mom+local+"limits.csv","w")
    for ii in [0] :
        for nn,channel in enumerate(channels) :
            if not nn < 4 : continue
            #options.variables+'_'+bdtTypesToDoFile[ns]+'_nbin_'+str(nbins)
            if ii == 0 : limits=ReadLimits( cards[nn], [1],"" ,channel,mom+local,-1,-1)
            if ii == 1 : limits=ReadLimits( cards[nn], [1],"_noSyst" ,channel,mom+local,-1,-1)
            print (channel, limits)
            for jj in limits[0] : file.write(str(jj)+', ')
            file.write('\n')
            #plt.plot(binstoDo,limits[0], color=colorsToDo[nn],linestyle='-',marker='o',label="bdtTypesToDoLabel[nn]")
            #plt.plot(binstoDo,limits[1], color=colorsToDo[nn],linestyle='-')
            #plt.plot(binstoDo,limits[3], color=colorsToDo[nn],linestyle='-')
        #ax.legend(loc='best', fancybox=False, shadow=False , ncol=1)
        #ax.set_xlabel('nbins')
        #ax.set_ylabel('limits')
        #maxsum=0
        #plt.axis((min(binstoDo),max(binstoDo),0.5,2.5))
        #ax.legend(loc='best', fancybox=False, shadow=False , ncol=1)
        #ax.set_xlabel('nbins')
        #ax.set_ylabel('limits')
        #maxsum=0
        #plt.axis((min(binstoDo),max(binstoDo),0.5,2.5))
