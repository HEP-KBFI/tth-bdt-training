import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from datetime import datetime
import sys , time
#import sklearn_to_tmva
import sklearn
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
try: from sklearn.cross_validation import train_test_split
except : from sklearn.model_selection import train_test_split
import pandas
import matplotlib.mlab as mlab
from scipy.stats import norm
#from pandas import HDFStore,DataFrame
import math
import matplotlib
matplotlib.use('agg')
#matplotlib.use('PS')   # generate postscript output by default
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import numpy as np
import psutil
import os
import pickle
import root_numpy
import xgboost as xgb
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import ROOT
import glob
from collections import OrderedDict

from ROOT import TCanvas, TFile, TProfile, TNtuple, TH1F, TH2F
from ROOT import gROOT, gBenchmark, gRandom, gSystem, Double

from multiprocessing import Pool, Process
p = Pool(16)

def numpyarrayTProfileFill(data_X, data_Y, data_wt, hprof):
    for x,y,w in np.nditer([data_X, data_Y, data_wt]):
        #print("x: {}, y: {}, w: {}".format(x, y, w))
        hprof.Fill(x,y,w)

def MakeTProfile(channel, data, var_name, y_min, y_max):
    data_copy = data.copy(deep=True) ## Making a deep copy of data ( => separate data and index from data)
    data_copy =  data_copy.loc[(data_copy[target]==0)] ## Only take backgrounds
    data_Y  = np.array(data_copy[var_name].values, dtype=np.float)
    data_X  = np.array(data_copy['gen_mHH'].values, dtype=np.float)
    data_wt = np.array(data_copy['totalWeight'].values, dtype=np.float)

    N_x  = len(data_X)
    N_y  = len(data_Y)
    N_wt = len(data_wt)

    # Create a new canvas, and customize it.
    c1 = TCanvas( 'c1', 'Dynamic Filling Example', 200, 10, 700, 500)
    #c1.SetFillColor(42)
    #c1.GetFrame().SetFillColor(21)
    c1.GetFrame().SetBorderSize(6)
    c1.GetFrame().SetBorderMode(-1)


    if((N_x == N_y) and (N_y == N_wt)):
       #print("N_x: {}, N_y: {}, N_wt: {}".format(N_x, N_y, N_wt))
       PlotTitle = 'Profile of '+var_name+' vs gen_mHH'
       hprof  = TProfile( 'hprof', PlotTitle, 17, 250., 1100., y_min, y_max)
       hprof.GetXaxis().SetTitle("gen_mHH (GeV)")
       hprof.GetYaxis().SetTitle(var_name)
       numpyarrayTProfileFill(data_X, data_Y, data_wt, hprof)
       hprof.Draw()
       c1.Modified()
       c1.Update()
       FileName = "{}/{}_{}.root".format(channel, "TProfile", var_name)
       c1.SaveAs(FileName)
    else:
     print('Arrays not of same length')
     print("N_x: {}, N_y: {}, N_wt: {}".format(N_x, N_y, N_wt))


startTime = datetime.now()
execfile("../python/data_manager.py")

from optparse import OptionParser
parser = OptionParser()
parser.add_option("--channel ", type="string", dest="channel", help="The ones whose variables implemented now are:\n   - 1l_2tau\n   - 2lss_1tau\n It will create a local folder and store the report*/xml", default='T')
parser.add_option("--variables", type="string", dest="variables", help="  Set of variables to use -- it shall be put by hand in the code, in the fuction trainVars(all)\n Example to 2ssl_2tau   \n                              all==True -- all variables that should be loaded (training + weights) -- it is used only once\n                               all==False -- only variables of training (not including weights) \n  For the channels implemented I defined 3 sets of variables/each to confront at limit level\n  trainvar=allVar -- all variables that are avaible to training (including lepton IDs, this is here just out of curiosity) \n  trainvar=oldVar -- a minimal set of variables (excluding lepton IDs and lep pt's)\n  trainvar=notForbidenVar -- a maximal set of variables (excluding lepton IDs and lep pt's) \n  trainvar=notForbidenVarNoMEM -- the same as above, but excluding as well MeM variables", default=None)
parser.add_option("--bdtType", type="string", dest="bdtType", help=" evtLevelTT_TTH or evtLevelTTV_TTH", default='T')
parser.add_option("--HypOpt", action="store_true", dest="HypOpt", help="If you call this will not do plots with repport", default=False)
parser.add_option("--doXML", action="store_true", dest="doXML", help="Do save not write the xml file", default=False)
parser.add_option("--doPlots", action="store_true", dest="doPlots", help="Fastsim Loose/Tight vs Fullsim variables plots", default=False)
parser.add_option("--nonResonant", action="store_true", dest="doPlots", help="Fastsim Loose/Tight vs Fullsim variables plots", default=False)
parser.add_option("--ntrees", type="int", dest="ntrees", help="hyp", default=2000) #1000 1500
parser.add_option("--treeDeph", type="int", dest="treeDeph", help="hyp", default=2) #3
parser.add_option("--lr", type="float", dest="lr", help="hyp", default=0.01)
#parser.add_option("--mcw", type="int", dest="mcw", help="hyp", default=1)
parser.add_option("--mcw", type="float", dest="mcw", help="hyp", default=1)
#parser.add_option("--tauID", type="string", dest="tauID", help="sample to pick training events", default='"dR03mvaVLoose"')
parser.add_option("--tauID", type="string", dest="tauID", help="sample to pick training events", default='')
parser.add_option("--Bkg_mass_rand", type="string", dest="Bkg_mass_rand", help="fix gen_mHH randomiz. method for bkg.s", default='"default"')
parser.add_option("--ClassErr_vs_epoch", action="store_true", dest="ClassErr_vs_epoch", help="Makes plots of classification error vs epoch for a grid of hyperparam.s", default=False)
parser.add_option("--gridSearchCV", action="store_true", dest="gridSearchCV", help="Search optimal values for XGB training parameters", default=False)
(options, args) = parser.parse_args()

print "-ntrees ",options.ntrees,"  --treeDeph ",options.treeDeph,"  --lr ",options.lr
tauID=options.tauID
Bkg_mass_rand=options.Bkg_mass_rand
doPlots=options.doPlots
bdtType=options.bdtType
trainvar=options.variables
hyppar=str(options.variables)+"_ntrees_"+str(options.ntrees)+"_deph_"+str(options.treeDeph)+"_mcw_"+str(options.mcw)+"_lr_0o0"+str(int(options.lr*100))

#channel=options.channel+"_HH"
#channel=options.channel+"_HH_"+tauID+"_"+options.Bkg_mass_rand  ## DEF LINE
channel=options.channel+"_HH_"+tauID+"_"+options.Bkg_mass_rand+"_"+options.variables

#if 'evtLevelSUM_HH_bb2l_res' in bdtType :
if 'bb2l' in options.channel:
    execfile("../cards/info_bb2l_HH.py")
    channel = options.channel+"_HH"
if '3l_0tau' in options.channel:
    execfile("../cards/info_3l_0tau_HH.py")
    channel=options.channel+"_HH_BkMassRndm"+options.Bkg_mass_rand+"_Vars"+str(len(trainVars(False, options.variables, options.bdtType)))+"_ntrees_"+str(options.ntrees)+"_deph_"+str(options.treeDeph)+"_mcw_"+str(options.mcw)+"_lr_"+str(options.lr)
    if options.gridSearchCV == True:
        channel += "_gridSearchCV"
if "2l_2tau" in options.channel :
    execfile("../cards/info_2l_2tau_HH.py")


print (startTime)



import shutil,subprocess
proc=subprocess.Popen(['mkdir '+channel],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()

proc=subprocess.Popen(['cp /home/ssawant/VHbbNtuples_9_4_x/CMSSW_9_4_6_patch1_rpst2/CMSSW_9_4_6_patch1/src/tth-bdt-training/EvtLevel/sklearn_Xgboost_evtLevel_HH_parametric_twofold.py '+channel+'/'],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()

weights="totalWeight"
target='target'

output = read_from(Bkg_mass_rand, tauID)

print('output[keys]', output["keys"])

data=load_data_2017(
    output["inputPath"],
    output["channelInTree"],
    trainVars(True),
    [],
    bdtType,
    channel,
    output["keys"],
    output["masses"],
    output["mass_randomization"],
    )
data.dropna(subset=[weights],inplace = True)
data.fillna(0)


print "\n\n\nData all\nEvent table before normalizing:: \n %30s  %10s   %s " % ("Process", "nEvents", "nEventsWeighted")
for key in output["keys"]:
    nEvents = int(len(data.loc[(data['key']==key)]))
    nEventsWtg = float(data.loc[(data['key']==key), [weights]].sum())
    #print " \n\n",key," sum: ",len(data.loc[(data['key']==key)]),", \t ",data.loc[(data['key']==key), [weights]].sum()
    print " %30s  %10i  %10.3f" % (key,nEvents,nEventsWtg)

print("\nnWeighted Events \t\t\t\t\t\t %s \t %s" % ('BDT','Datacards'))    
ttbar_samples = ['TTToSemiLeptonic', 'TTTo2L2Nu']
print("ttbar_samples: %s \t %f \t %f" % (str(ttbar_samples),data.loc[(data['key'].isin(ttbar_samples)), [weights]].sum(),output["TTdatacard"]))
print("DY \t\t\t\t\t\t\t %f \t %f" % (data.loc[(data['key']=='DY'), [weights]].sum(), output["DYdatacard"]))
print("WZ \t\t\t\t\t\t\t %f \t %f" % (data.loc[(data['key']=='WZ'), [weights]].sum(), output["WZdatacard"]))
print("TTZ \t\t\t\t\t\t\t %f \t %f" % (data.loc[(data['key']=='TTZJets'), [weights]].sum(), output["TTZdatacard"]))
print("TTV \t\t\t\t\t\t\t %f \t %f" % (data.loc[(data['key']=='TTWJets'), [weights]].sum(), output["TTWdatacard"]))
for key in output["keys"]:
    if not 'signal' in key: continue
    sProcess = key
    sDatacard = key+'datacard'
    print("%s \t\t\t\t %f \t %f" % (sProcess, data.loc[(data['key']==sProcess), [weights]].sum(), output[sDatacard]))

    
### Plot histograms of training variables

hist_params = {'normed': True, 'histtype': 'bar', 'fill': False , 'lw':5}
if 'evtLevelSUM' in bdtType :
	labelBKG = "SUM BKG"
	#if channel in ["3l_0tau_HH"]:
        if "3l_0tau_HH" in channel:
	    #labelBKG = "WZ+DY+tt+ttV"
            labelBKG = "VV+DY+tt+ttV"
elif 'evtLevelWZ' in bdtType :
	labelBKG = "WZ"
elif 'evtLevelDY' in bdtType :
        labelBKG = "DY"
elif 'evtLevelTT' in bdtType :
        labelBKG = "TT"
elif 'evtLevelSUM_HH_2l_2tau_res' in bdtType :
        labelBKG = "TT+DY+VV"
print "labelBKG: ",labelBKG

printmin=True
plotResiduals=False
plotAll=False
nbins=15
colorFast='g'
colorFastT='b'
BDTvariables=trainVars(plotAll, options.variables, options.bdtType)

print("BDTvariables =>", BDTvariables)
make_plots(BDTvariables, nbins,
    data.ix[data.target.values == 0],labelBKG, colorFast,
    data.ix[data.target.values == 1],'Signal', colorFastT,
    channel+"/"+bdtType+"_"+trainvar+"_Variables_BDT.pdf",
    printmin,
    plotResiduals,
    output["masses_test"],
    output["masses"]
    )

print ("separate datasets odd/even")
data_even = data.loc[(data["event"].values % 2 == 0) ]
data_odd = data.loc[~(data["event"].values % 2 == 0) ]

order_train = [data_odd, data_even]
order_train_name = ["odd","even"]


print ("balance datasets by even/odd chunck")

#for data_do in order_train :
for idx, data_do in enumerate(order_train) :
    print "\n\ndata %i %s \nEvent table before normalizing:: \n %30s  %10s   %s " % (idx,order_train_name[idx],"Process", "nEvents", "nEventsWeighted")
    for key in output["keys"]:
        nEvents = int(len(data_do.loc[(data_do['key']==key)]))
        nEventsWtg = float(data_do.loc[(data_do['key']==key), [weights]].sum())
        #print " \n\n",key," sum: ",len(data.loc[(data['key']==key)]),", \t ",data.loc[(data['key']==key), [weights]].sum()
        print " %30s  %10i  %10.3f" % (key,nEvents,nEventsWtg)

    print("\nnWeighted Events \t\t\t\t\t\t %s \t %s" % ('BDT','Datacards'))
    ttbar_samples = ['TTToSemiLeptonic', 'TTTo2L2Nu']
    print("ttbar_samples: %s \t %f \t %f" % (str(ttbar_samples),data_do.loc[(data_do['key'].isin(ttbar_samples)), [weights]].sum(),output["TTdatacard"]))
    print("DY \t\t\t\t\t\t\t %f \t %f" % (data_do.loc[(data_do['key']=='DY'), [weights]].sum(), output["DYdatacard"]))
    print("WZ \t\t\t\t\t\t\t %f \t %f" % (data_do.loc[(data_do['key']=='WZ'), [weights]].sum(), output["WZdatacard"]))
    print("TTZ \t\t\t\t\t\t\t %f \t %f" % (data_do.loc[(data_do['key']=='TTZJets'), [weights]].sum(), output["TTZdatacard"]))
    print("TTV \t\t\t\t\t\t\t %f \t %f" % (data_do.loc[(data_do['key']=='TTWJets'), [weights]].sum(), output["TTWdatacard"]))
    for key in output["keys"]:
        if not 'signal' in key: continue
        sProcess = key
        sDatacard = key+'datacard'
        print("%s \t\t\t\t %f \t %f" % (sProcess, data_do.loc[(data_do['key']==sProcess), [weights]].sum(), output[sDatacard]))

    
    if 'SUM_HH' in bdtType :
        ttbar_samples = ['TTToSemiLeptonic', 'TTTo2L2Nu'] ## Removed TTToHadronic since zero events selected for this sample
        data_do.loc[(data_do['key'].isin(ttbar_samples)), [weights]]              *= output["TTdatacard"]/data_do.loc[(data_do['key'].isin(ttbar_samples)), [weights]].sum()
        data_do.loc[(data_do['key']=='DY'), [weights]]                            *= output["DYdatacard"]/data_do.loc[(data_do['key']=='DY'), [weights]].sum()
        if "evtLevelSUM_HH_bb1l_res" in bdtType :
            data_do.loc[(data_do['key']=='W'), [weights]]                         *= Wdatacard/data_do.loc[(data_do['key']=='W')].sum()
        if "evtLevelSUM_HH_2l_2tau_res" in bdtType :
            data_do.loc[(data_do['key']=='TTZJets'), [weights]]                       *= output["TTZdatacard"]/data_do.loc[(data_do['key']=='TTZJets'), [weights]].sum() ## TTZJets
            data_do.loc[(data_do['key']=='TTWJets'), [weights]]                       *= output["TTWdatacard"]/data_do.loc[(data_do['key']=='TTWJets'), [weights]].sum() ## TTWJets + TTWW
            data_do.loc[(data_do['key']=='ZZ'), [weights]]                            *= output["ZZdatacard"]/data_do.loc[(data_do['key']=='ZZ'), [weights]].sum() ## ZZ +ZZZ
            data_do.loc[(data_do['key']=='WZ'), [weights]]                            *= output["WZdatacard"]/data_do.loc[(data_do['key']=='WZ'), [weights]].sum() ## WZ + WZZ_4F
            data_do.loc[(data_do['key']=='WW'), [weights]]                            *= output["WWdatacard"]/data_do.loc[(data_do['key']=='WW'), [weights]].sum() ## WW + WWZ + WWW_4F
            #data_do.loc[(data_do['key'].isin(['TTWJets', 'TTZJets'])), [weights]] *= output["TTVdatacard"]/data_do.loc[(data_do['key'].isin(['TTWJets', 'TTZJets'])), [weights]].sum() # consider do separately
            #data_do.loc[(data_do['key'].isin(['WW','WZ','ZZ'])), [weights]]       *= output["VVdatacard"]/data_do.loc[(data_do['key'].isin(['WW','WZ','ZZ'])), [weights]].sum() # consider do separatelly
            #data_do.loc[(data_do['key']=='VH'), [weights]]                        *= output["VHdatacard"]/data_do.loc[(data_do['key']=='VH'), [weights]].sum() # consider removing
            #data_do.loc[(data_do['key']=='TTH'), [weights]]                       *= output["TTHdatacard"]/data_do.loc[(data_do['key']=='TTH'), [weights]].sum() # consider removing
        if "evtLevelSUM_HH_3l_0tau_res" in bdtType :
            data_do.loc[(data_do['key']=='WZ'), [weights]]                            *= output["WZdatacard"]/data_do.loc[(data_do['key']=='WZ'), [weights]].sum()
            data_do.loc[(data_do['key']=='TTZJets'), [weights]]                       *= output["TTZdatacard"]/data_do.loc[(data_do['key']=='TTZJets'), [weights]].sum() ## TTZJets
            data_do.loc[(data_do['key']=='TTWJets'), [weights]]                       *= output["TTWdatacard"]/data_do.loc[(data_do['key']=='TTWJets'), [weights]].sum() ## TTWJets + TTWW
            if 'ZZ' in output["keys"]:
                data_do.loc[(data_do['key']=='ZZ'), [weights]]                            *= output["ZZdatacard"]/data_do.loc[(data_do['key']=='ZZ'), [weights]].sum()
            if 'WW' in output["keys"]:
                data_do.loc[(data_do['key']=='WW'), [weights]]                            *= output["WWdatacard"]/data_do.loc[(data_do['key']=='WW'), [weights]].sum()                
            # Normalize various signal as well
            for key in output["keys"]:
                if not 'signal' in key: continue
                sProcess = key
                sDatacard = key+'datacard'
                #print("%s: wt: %f, datacard: %f" % (sProcess, data_do.loc[(data_do['key']==sProcess), [weights]].sum(), output[sDatacard]))
                data_do.loc[(data_do['key']==sProcess), [weights]]                            *= output[sDatacard]/data_do.loc[(data_do['key']==sProcess), [weights]].sum()

                
            print "\n\ndata %i %s \nEvent table after normalizing:: \n %30s  %10s   %s " % (idx,order_train_name[idx],"Process", "nEvents", "nEventsWeighted")
            for key in output["keys"]:
                nEvents = int(len(data_do.loc[(data_do['key']==key)]))
                nEventsWtg = float(data_do.loc[(data_do['key']==key), [weights]].sum())
                #print " \n\n",key," sum: ",len(data.loc[(data['key']==key)]),", \t ",data.loc[(data['key']==key), [weights]].sum()
                print " %30s  %10i  %10.3f" % (key,nEvents,nEventsWtg)

            print("\nnWeighted Events \t\t\t\t\t\t %s \t %s" % ('BDT','Datacards'))
            ttbar_samples = ['TTToSemiLeptonic', 'TTTo2L2Nu']
            print("ttbar_samples: %s \t %f \t %f" % (str(ttbar_samples),data_do.loc[(data_do['key'].isin(ttbar_samples)), [weights]].sum(),float(output["TTdatacard"])))
            print("DY \t\t\t\t\t\t\t %f \t %f" % (data_do.loc[(data_do['key']=='DY'), [weights]].sum(), output["DYdatacard"]))
            print("WZ \t\t\t\t\t\t\t %f \t %f" % (data_do.loc[(data_do['key']=='WZ'), [weights]].sum(), output["WZdatacard"]))
            print("TTZ \t\t\t\t\t\t\t %f \t %f" % (data_do.loc[(data_do['key']=='TTZJets'), [weights]].sum(), output["TTZdatacard"]))
            print("TTV \t\t\t\t\t\t\t %f \t %f" % (data_do.loc[(data_do['key']=='TTWJets'), [weights]].sum(), output["TTWdatacard"]))
            for key in output["keys"]:
                if not 'signal' in key: continue
                sProcess = key
                sDatacard = key+'datacard'
                print("%s \t\t\t\t %f \t %f" % (sProcess, data_do.loc[(data_do['key']==sProcess), [weights]].sum(), output[sDatacard]))

                
        if "gen_mHH" in trainVars(False, options.variables, options.bdtType):    
            for mass in range(len(output["masses"])) :
                data_do.loc[(data_do[target]==1) & (data_do["gen_mHH"].astype(np.int) == int(output["masses"][mass])),[weights]] *= 100000./data_do.loc[(data_do[target]==1) & (data_do["gen_mHH"]== output["masses"][mass]),[weights]].sum()
                data_do.loc[(data_do[target]==0) & (data_do["gen_mHH"].astype(np.int) == int(output["masses"][mass])),[weights]] *= 100000./data_do.loc[(data_do[target]==0) & (data_do["gen_mHH"]== output["masses"][mass]),[weights]].sum()
        else:
            data_do.loc[data_do['target']==0, [weights]] *= 100000/data_do.loc[data_do['target']==0][weights].sum()
            data_do.loc[data_do['target']==1, [weights]] *= 100000/data_do.loc[data_do['target']==1][weights].sum() 

    else :
        data_do.loc[data_do['target']==0, [weights]] *= 100000/data_do.loc[data_do['target']==0][weights].sum()
        data_do.loc[data_do['target']==1, [weights]] *= 100000/data_do.loc[data_do['target']==1][weights].sum()

    print "\n\ndata %i %s \nEvent table after normalizing and scaling up by 1e5:: \n %30s  %10s   %s " % (idx,order_train_name[idx],"Process", "nEvents", "nEventsWeighted")
    for key in output["keys"]:
        nEvents = int(len(data_do.loc[(data_do['key']==key)]))
        nEventsWtg = float(data_do.loc[(data_do['key']==key), [weights]].sum())
        #print " \n\n",key," sum: ",len(data.loc[(data['key']==key)]),", \t ",data.loc[(data['key']==key), [weights]].sum()
        print " %30s  %10i  %10.3f" % (key,nEvents,nEventsWtg)

    print("\nnWeighted Events \t\t\t\t\t\t %s \t %s" % ('BDT','Datacards'))
    ttbar_samples = ['TTToSemiLeptonic', 'TTTo2L2Nu']
    print("ttbar_samples: %s \t %f \t %f" % (str(ttbar_samples),data_do.loc[(data_do['key'].isin(ttbar_samples)), [weights]].sum(),float(output["TTdatacard"])))
    print("DY \t\t\t\t\t\t\t %f \t %f" % (data_do.loc[(data_do['key']=='DY'), [weights]].sum(), output["DYdatacard"]))
    print("WZ \t\t\t\t\t\t\t %f \t %f" % (data_do.loc[(data_do['key']=='WZ'), [weights]].sum(), output["WZdatacard"]))
    print("TTZ \t\t\t\t\t\t\t %f \t %f" % (data_do.loc[(data_do['key']=='TTZJets'), [weights]].sum(), output["TTZdatacard"]))
    print("TTV \t\t\t\t\t\t\t %f \t %f" % (data_do.loc[(data_do['key']=='TTWJets'), [weights]].sum(), output["TTWdatacard"]))
    for key in output["keys"]:
        if not 'signal' in key: continue
        sProcess = key
        sDatacard = key+'datacard'
        print("%s \t\t\t\t %f \t %f" % (sProcess, data_do.loc[(data_do['key']==sProcess), [weights]].sum(), output[sDatacard]))
    print("\n")
    
    
    
print "\n\nData all\nEvent table after normalizing:: \n %30s  %10s   %s " % ("Process", "nEvents", "nEventsWeighted")
for key in output["keys"]:
    nEvents = int(len(data.loc[(data['key']==key)]))
    nEventsWtg = float(data.loc[(data['key']==key), [weights]].sum())
    #print " \n\n",key," sum: ",len(data.loc[(data['key']==key)]),", \t ",data.loc[(data['key']==key), [weights]].sum()
    print " %30s  %10i  %10.3f" % (key,nEvents,nEventsWtg)

print("\nnWeighted Events \t\t\t\t\t\t %s \t %s" % ('BDT','Datacards'))
ttbar_samples = ['TTToSemiLeptonic', 'TTTo2L2Nu']
print("ttbar_samples: %s \t %f \t %f" % (str(ttbar_samples),data.loc[(data['key'].isin(ttbar_samples)), [weights]].sum(),float(output["TTdatacard"])))
print("DY \t\t\t\t\t\t\t %f \t %f" % (data.loc[(data['key']=='DY'), [weights]].sum(), output["DYdatacard"]))
print("WZ \t\t\t\t\t\t\t %f \t %f" % (data.loc[(data['key']=='WZ'), [weights]].sum(), output["WZdatacard"]))
print("TTZ \t\t\t\t\t\t\t %f \t %f" % (data.loc[(data['key']=='TTZJets'), [weights]].sum(), output["TTZdatacard"]))
print("TTV \t\t\t\t\t\t\t %f \t %f" % (data.loc[(data['key']=='TTWJets'), [weights]].sum(), output["TTWdatacard"]))
for key in output["keys"]:
    if not 'signal' in key: continue
    sProcess = key
    sDatacard = key+'datacard'
    print("%s \t\t\t\t %f \t %f" % (sProcess, data.loc[(data['key']==sProcess), [weights]].sum(), output[sDatacard]))

    
## --- Making TProfile plots --- ###
#if 'evtLevelSUM_HH_bb2l_res' not in bdtType and 'evtLevelSUM_HH_3l_0tau_res' not in bdtType:
if 'evtLevelSUM_HH_bb2l_res' in bdtType :
    MakeTProfile(channel, data, "diHiggsMass", 0., 1000.)
    MakeTProfile(channel, data, "tau1_pt", 0., 1000.)
    MakeTProfile(channel, data, "met_LD", 0., 1000.)
    MakeTProfile(channel, data, "diHiggsVisMass", 0., 1000.)
    MakeTProfile(channel, data, "m_ll", 0., 1000.)
    MakeTProfile(channel, data, "tau2_pt", 0., 1000.)
    MakeTProfile(channel, data, "mTauTau", 0., 1000.)
    MakeTProfile(channel, data, "mT_lep1", 0., 1000.)
    MakeTProfile(channel, data, "mT_lep2", 0., 1000.)
    MakeTProfile(channel, data, "mht", 0., 1000.)
    MakeTProfile(channel, data, "met", 0., 1000.)
    MakeTProfile(channel, data, "dr_lep_tau_min_SS", 0., 1.0)
    MakeTProfile(channel, data, "dr_lep_tau_min_OS", 0., 1.0)
    MakeTProfile(channel, data, "dr_taus", 0., 1.0)
    MakeTProfile(channel, data, "dr_lep1_tau1_tau2_min", 0., 1.0)
    MakeTProfile(channel, data, "dr_lep1_tau1_tau2_max", 0., 1.0)
    MakeTProfile(channel, data, "max_tau_eta", -1.0, 1.0)
    MakeTProfile(channel, data, "max_lep_eta", -1.0, 1.0)
    MakeTProfile(channel, data, "nElectron", 0., 3.)
elif 'evtLevelSUM_HH_3l_0tau_res' not in bdtType:
    MakeTProfile(channel, data, "diHiggsMass", 0., 1000.)
    MakeTProfile(channel, data, "diHiggsVisMass", 0., 1000.)
    


roc_test = []
roc_train = []
estimator = []
for dd, data_do in  enumerate(order_train):
    cls = xgb.XGBClassifier(
    			n_estimators = options.ntrees,
    			max_depth = options.treeDeph,
    			min_child_weight = options.mcw,
    			learning_rate = options.lr,
                        nthread = 8,        
                        #gamma = 0,
                        #subsample = 0.5,
                        #colsample_bytree = 0.5,
                        #reg_alpha = 100,
                        #reg_lambda = 100
    			)

    cls.fit(
    	data_do[trainVars(False, options.variables, options.bdtType)].values,
    	data_do.target.astype(np.bool),
    	sample_weight=(data_do[weights].astype(np.float64))
    	)
    if dd == 0 : val_data = 1
    else : val_data = 0

    print ("XGBoost trained", order_train_name[dd])
    if options.doXML==True :
        pklpath=channel+"/"+channel+"_XGB_"+trainvar+"_"+bdtType+"_"+str(len(trainVars(False, options.variables, options.bdtType)))+"Var_"+order_train_name[dd]
        print ("Date: ", time.asctime( time.localtime(time.time()) ))
        pickle.dump(cls, open(pklpath+".pkl", 'wb'))
        file = open(pklpath+"pkl.log","w")
        file.write(str(trainVars(False, options.variables, options.bdtType))+"\n")
        file.close()
        print ("saved ",pklpath+".pkl")
        print ("variables are: ",pklpath+"_pkl.log")

    proba = cls.predict_proba(data_do[trainVars(False, options.variables, options.bdtType)].values )
    fpr, tpr, thresholds = roc_curve(
        data_do[target].astype(np.bool), proba[:,1],
        sample_weight=(data_do[weights].astype(np.float64))
    )
    train_auc = auc(fpr, tpr, reorder = True)
    roc_train = roc_train + [ { "fpr":fpr, "tpr":tpr, "train_auc":train_auc }]
    print('XGBoost train set:: fpr: {},  tpr": {},  threshold: {}'.format(fpr,tpr,thresholds))
    print("XGBoost train set auc - {}".format(train_auc))

    proba = cls.predict_proba(order_train[val_data][trainVars(False, options.variables, options.bdtType)].values )
    fprt, tprt, thresholds = roc_curve(
        order_train[val_data][target].astype(np.bool), proba[:,1],
        sample_weight=(order_train[val_data][weights].astype(np.float64))
    )
    test_auct = auc(fprt, tprt, reorder = True)
    roc_test = roc_test + [ { "fprt":fprt, "tprt":tprt, "test_auct":test_auct }]
    print('XGBoost test set:: fpr: {},  tpr": {},  threshold: {}'.format(fprt,tprt,thresholds))
    print("XGBoost test set auc - {}".format(test_auct))
    estimator = estimator + [cls]

    print("nVar: {}, var: {}".format(len(trainVars(False, options.variables, options.bdtType)), trainVars(False, options.variables, options.bdtType)))
    ## feature importance plot
    #fig, ax = plt.subplots()
    #f_score_dict =cls.booster().get_fscore()
    fig, ax = plt.subplots()

    fscores = cls.booster().get_fscore()
    importances = np.zeros(len(trainVars(False, options.variables, options.bdtType)))
    for k, v in fscores.iteritems():
        importances[int(k[1:])] = v
        print('k: {}, v: {}, k[1:]: {};\t'.format(k,v,k[1:]))
    print('importances: len {}:  {},   its sum:{}'.format(len(importances), importances, np.sum(importances)))
        
        
    f_score_dict =cls.booster().get_fscore()
    print('f_score_dict 0: len:{}:  {}'.format(len(f_score_dict), f_score_dict))
    f_score_dict = {trainVars(False, options.variables, options.bdtType)[int(k[1:])] : v for k,v in f_score_dict.items()}
    print('f_score_dict 1: len:{}:  {}'.format(len(f_score_dict), f_score_dict))
    feat_imp = pandas.Series(f_score_dict).sort_values(ascending=True)
    feat_imp.plot(kind='barh', title='Feature Importances')
    fig.tight_layout()
    nameout = "{}/{}_{}_{}_{}_{}_{}_XGB_importance.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False, options.variables, options.bdtType))),hyppar, order_train_name[dd], (options.tauID))
    fig.savefig(nameout)
    fig.savefig(nameout.replace(".pdf", ".png"))

###############################
# overall ROC
styleline = ['-', '--']
colorline = ['g', 'r']
fig, ax = plt.subplots(figsize=(6, 6))
for tt, rocs in enumerate(roc_test) :
    ax.plot(
        roc_train[tt]['fpr'],
        roc_train[tt]['tpr'], color = colorline[tt],
        lw = 2, linestyle = '-',
        label = order_train_name[tt] + ' train (area = %0.3f)'%(roc_train[tt]['train_auc'])
        )
    ax.plot(
        roc_test[tt]['fprt'],
        roc_test[tt]['tprt'],
        lw = 2, linestyle = '--', color = colorline[tt],
        label = order_train_name[tt] + ' test (area = %0.3f)'%(roc_test[tt]['test_auct'])
        )
ax.set_ylim([0.0,1.0])
ax.set_xlim([0.0,1.0])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc="lower right")
ax.grid()
nameout = "{}/{}_{}_{}_{}_{}_roc.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False, options.variables, options.bdtType))),hyppar,(options.tauID))
fig.savefig(nameout)
fig.savefig(nameout.replace(".pdf", ".png"))



###############################
# by mass ROC
styleline = ['-', '--', '-.', ':']
#colors_mass = ['m', 'b', 'k', 'r', 'g',  'y', 'c', ]
colors_mass = ['m', 'b', 'k', 'r', 'g',  'y', 'c',
                           'chocolate','teal', 'pink', 'darkkhaki', 'maroon', 'slategray',
                           'orange', 'silver', 'aquamarine', 'lavender', 'goldenrod', 'salmon',
                           'tan', 'lime', 'lightcoral'
            ]
fig, ax = plt.subplots(figsize=(6, 6))
sl = 0
for mm, mass in enumerate(output["masses_test"]) :
    print('\tmass: {}'.format(mass))
    for dd, data_do in  enumerate(order_train) :
        if dd == 0 : val_data = 1
        else : val_data = 0
        print('dd: {}, val_data: {}, order_train_name[dd]: {}'.format(dd,val_data,order_train_name[dd]))
        proba = estimator[dd].predict_proba(
            data_do.loc[(data_do["gen_mHH"].astype(np.int) == int(mass)), trainVars(False, options.variables, options.bdtType)].values
        )
        fpr, tpr, thresholds = roc_curve(
            data_do.loc[(data_do["gen_mHH"].astype(np.int) == int(mass)), target].astype(np.bool), proba[:,1],
            sample_weight=(data_do.loc[(data_do["gen_mHH"].astype(np.int) == int(mass)), weights].astype(np.float64))
        )
        train_auc = auc(fpr, tpr, reorder = True)
        print("train set auc " + str(train_auc) + " (mass = " + str(mass) + ")")
        proba = estimator[dd].predict_proba(
            order_train[val_data].loc[(order_train[val_data]["gen_mHH"].astype(np.int) == int(mass)), trainVars(False, options.variables, options.bdtType)].values
        )
        fprt, tprt, thresholds = roc_curve(
            order_train[val_data].loc[(order_train[val_data]["gen_mHH"].astype(np.int) == int(mass)), target].astype(np.bool), proba[:,1],
            sample_weight=(order_train[val_data].loc[(order_train[val_data]["gen_mHH"].astype(np.int) == int(mass)),weights].astype(np.float64))
        )
        test_auct = auc(fprt, tprt, reorder = True)
        print("test set auc " + str(test_auct) + " (mass = " + str(mass) + ")")
        ax.plot(
            fpr, tpr,
            lw = 2, linestyle = styleline[dd + dd*1], color = colors_mass[mm],
            label = order_train_name[dd] + ' train (area = %0.3f)'%(train_auc) + " (mass = " + str(mass) + ")"
            )
        sl += 1
        ax.plot(
            fprt, tprt,
            lw = 2, linestyle = styleline[dd + 1 + + dd*1], color = colors_mass[mm],
            label = order_train_name[dd] + ' test (area = %0.3f)'%(test_auct) + " (mass = " + str(mass) + ")"
            )
        sl += 1
ax.set_ylim([0.0,1.0])
ax.set_xlim([0.0,1.0])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc="lower right", fontsize = 'small')
ax.grid()
nameout = "{}/{}_{}_{}_{}_{}_roc_by_mass.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False, options.variables, options.bdtType))),hyppar,(options.tauID))
fig.savefig(nameout)
fig.savefig(nameout.replace(".pdf", ".png"))


###############################
## classifier plot by mass
hist_params = {'normed': True, 'bins': 8 , 'histtype':'step', "lw": 2}
for mm, mass in enumerate(output["masses_test"]) :
    plt.clf()
    colorcold = ['g', 'b']
    colorhot = ['r', 'magenta']
    fig, ax = plt.subplots(figsize=(6, 6))
    for dd, data_do in  enumerate(order_train):
        if dd == 0 : val_data = 1
        else : val_data = 0
        y_pred = estimator[dd].predict_proba(
            order_train[val_data].loc[(order_train[val_data].target.values == 0) & (order_train[val_data]["gen_mHH"].astype(np.int) == int(mass)),
            trainVars(False, options.variables, options.bdtType)].values
        )[:, 1]
        y_predS = estimator[dd].predict_proba(
            order_train[val_data].loc[(order_train[val_data].target.values == 1) & (order_train[val_data]["gen_mHH"].astype(np.int) == int(mass)),
            trainVars(False, options.variables, options.bdtType)].values
        )[:, 1]
        y_pred_train = estimator[dd].predict_proba(
            data_do.ix[(data_do.target.values == 0) & (data_do["gen_mHH"].astype(np.int) == int(mass)),
            trainVars(False, options.variables, options.bdtType)].values
        )[:, 1]
        y_predS_train = estimator[dd].predict_proba(
            data_do.ix[(data_do.target.values == 1) & (data_do["gen_mHH"].astype(np.int) == int(mass)),
            trainVars(False, options.variables, options.bdtType)].values
        )[:, 1]
        dict_plot = [
           [y_pred, "-", colorcold[dd], order_train_name[dd] + " test " + labelBKG],
           [y_predS, "-", colorhot[dd], order_train_name[dd] + " test signal"],
           [y_pred_train, "--", colorcold[dd], order_train_name[dd] + " train " + labelBKG],
           [y_predS_train, "--", colorhot[dd], order_train_name[dd] + " train signal"]
        ]
        yMax = 1
        for item in dict_plot :
            values1, bins, _ = ax.hist(
                item[0],
                ls=item[1], color = item[2],
                label=item[3],
                **hist_params
                )
            normed = sum(y_pred)
            mid = 0.5*(bins[1:] + bins[:-1])
            err=np.sqrt(values1*normed)/normed # denominator is because plot is normalized
            plt.errorbar(mid, values1, yerr=err, fmt='none', color= item[2], ecolor= item[2], edgecolor=item[2], lw=2)
            yMax = max(yMax, max(values1))
        #plt.xscale('log')
        #plt.yscale('log')
    #ax.legend(loc='upper center', title="mass = "+str(mass)+" GeV", fontsize = 'small')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.), bbox_transform=fig.transFigure, title="mass = "+str(mass)+" GeV", fontsize = 'small')
    ax.set_ylim(0, yMax*1.2) # set y-axis range of XGBclassifier
    nameout = channel+'/'+bdtType+'_'+trainvar+'_'+str(len(trainVars(False, options.variables, options.bdtType)))+'_'+hyppar+'_mass_'+ str(mass)+'_'+(options.tauID)+'_XGBclassifier.pdf'
    fig.savefig(nameout)
    fig.savefig(nameout.replace(".pdf", ".png"))


###############################
## classifier plot 
'''
plt.clf()
colorcold = ['g', 'b']
colorhot = ['r', 'magenta']
fig, ax = plt.subplots(figsize=(6, 6))
for dd, data_do in  enumerate(order_train):
    if dd == 0 : val_data = 1
    else : val_data = 0
    if "gen_mHH" in trainVars(False, options.variables, options.bdtType): 
        y_pred = estimator[dd].predict_proba(
            order_train[val_data].loc[(order_train[val_data].target.values == 0) & (order_train[val_data]["gen_mHH"].astype(np.int) == int(mass)),
                trainVars(False, options.variables, options.bdtType)].values
        )[:, 1]
        y_predS = estimator[dd].predict_proba(
            order_train[val_data].loc[(order_train[val_data].target.values == 1) & (order_train[val_data]["gen_mHH"].astype(np.int) == int(mass)),
                trainVars(False, options.variables, options.bdtType)].values
        )[:, 1]
        y_pred_train = estimator[dd].predict_proba(
            data_do.ix[(data_do.target.values == 0) & (data_do["gen_mHH"].astype(np.int) == int(mass)),
                trainVars(False, options.variables, options.bdtType)].values
        )[:, 1]
        y_predS_train = estimator[dd].predict_proba(
            data_do.ix[(data_do.target.values == 1) & (data_do["gen_mHH"].astype(np.int) == int(mass)),
                trainVars(False, options.variables, options.bdtType)].values
        )[:, 1]
    else:
        y_pred = estimator[dd].predict_proba(
            order_train[val_data].loc[(order_train[val_data].target.values == 0),
                trainVars(False, options.variables, options.bdtType)].values
        )[:, 1]
        y_predS = estimator[dd].predict_proba(
            order_train[val_data].loc[(order_train[val_data].target.values == 1),
                trainVars(False, options.variables, options.bdtType)].values
        )[:, 1]
        y_pred_train = estimator[dd].predict_proba(
            data_do.ix[(data_do.target.values == 0),
                trainVars(False, options.variables, options.bdtType)].values
        )[:, 1]
        y_predS_train = estimator[dd].predict_proba(
            data_do.ix[(data_do.target.values == 1) & (data_do["gen_mHH"].astype(np.int) == int(mass)),
                trainVars(False, options.variables, options.bdtType)].values
        )[:, 1]
        
    dict_plot = [
       [y_pred, "-", colorcold[dd], order_train_name[dd] + " test " + labelBKG],
       [y_predS, "-", colorhot[dd], order_train_name[dd] + " test signal"],
       [y_pred_train, "--", colorcold[dd], order_train_name[dd] + " train " + labelBKG],
       [y_predS_train, "--", colorhot[dd], order_train_name[dd] + " train signal"]
    ]
    yMax = 1
    for item in dict_plot :
        values1, bins, _ = ax.hist(
            item[0],
            ls=item[1], color = item[2],
            label=item[3],
            **hist_params
            )
        normed = sum(y_pred)
        mid = 0.5*(bins[1:] + bins[:-1])
        err=np.sqrt(values1*normed)/normed # denominator is because plot is normalized
        plt.errorbar(mid, values1, yerr=err, fmt='none', color= item[2], ecolor= item[2], edgecolor=item[2], lw=2)
        yMax = max(yMax, max(values1))
        print('\n\t\tvalue1: {},  max:{}'.format(values1, max(values1)))
    #plt.xscale('log')
    #plt.yscale('log')
#ax.legend(loc='upper center', title="all masses", fontsize = 'small')
ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.), bbox_transform=fig.transFigure, title="all masses", fontsize = 'small')
ax.set_ylim(0, yMax*1.2) # set y-axis range of XGBclassifier
nameout = channel+'/'+bdtType+'_'+trainvar+'_'+str(len(trainVars(False, options.variables, options.bdtType)))+'_'+hyppar+'_AllMass_'+(options.tauID)+'_XGBclassifier.pdf'
fig.savefig(nameout)
fig.savefig(nameout.replace(".pdf", ".png"))
'''


'''
# original
plt.clf()
colorcold = ['g', 'b']
colorhot = ['r', 'magenta']
fig, ax = plt.subplots(figsize=(6, 6))
for dd, data_do in  enumerate(order_train):
    if dd == 0 : val_data = 1
    else : val_data = 0
    y_pred = estimator[dd].predict_proba(
        order_train[val_data].loc[(order_train[val_data].target.values == 0) & (order_train[val_data]["gen_mHH"].astype(np.int) == int(mass)),
        trainVars(False, options.variables, options.bdtType)].values
    )[:, 1]
    y_predS = estimator[dd].predict_proba(
        order_train[val_data].loc[(order_train[val_data].target.values == 1) & (order_train[val_data]["gen_mHH"].astype(np.int) == int(mass)),
        trainVars(False, options.variables, options.bdtType)].values
    )[:, 1]
    y_pred_train = estimator[dd].predict_proba(
        data_do.ix[(data_do.target.values == 0) & (data_do["gen_mHH"].astype(np.int) == int(mass)),
        trainVars(False, options.variables, options.bdtType)].values
    )[:, 1]
    y_predS_train = estimator[dd].predict_proba(
        data_do.ix[(data_do.target.values == 1) & (data_do["gen_mHH"].astype(np.int) == int(mass)),
        trainVars(False, options.variables, options.bdtType)].values
    )[:, 1]
    dict_plot = [
       [y_pred, "-", colorcold[dd], order_train_name[dd] + " test " + labelBKG],
       [y_predS, "-", colorhot[dd], order_train_name[dd] + " test signal"],
       [y_pred_train, "--", colorcold[dd], order_train_name[dd] + " train " + labelBKG],
       [y_predS_train, "--", colorhot[dd], order_train_name[dd] + " train signal"]
    ]
    yMax = 1
    for item in dict_plot :
        values1, bins, _ = ax.hist(
            item[0],
            ls=item[1], color = item[2],
            label=item[3],
            **hist_params
            )
        normed = sum(y_pred)
        mid = 0.5*(bins[1:] + bins[:-1])
        err=np.sqrt(values1*normed)/normed # denominator is because plot is normalized
        plt.errorbar(mid, values1, yerr=err, fmt='none', color= item[2], ecolor= item[2], edgecolor=item[2], lw=2)
        yMax = max(yMax, max(values1))
        print('\n\t\tvalue1: {},  max:{}'.format(values1, max(values1)))
    #plt.xscale('log')
    #plt.yscale('log')
#ax.legend(loc='upper center', title="all masses", fontsize = 'small')
ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.), bbox_transform=fig.transFigure, title="all masses", fontsize = 'small')
ax.set_ylim(0, yMax*1.2) # set y-axis range of XGBclassifier
nameout = channel+'/'+bdtType+'_'+trainvar+'_'+str(len(trainVars(False, options.variables, options.bdtType)))+'_'+hyppar+'_AllMass_'+(options.tauID)+'_XGBclassifier.pdf'
fig.savefig(nameout)
fig.savefig(nameout.replace(".pdf", ".png"))
'''

plt.clf()
colorcold = ['g', 'b']
colorhot = ['r', 'magenta']
fig, ax = plt.subplots(figsize=(6, 6))
for dd, data_do in  enumerate(order_train):
    if dd == 0 : val_data = 1
    else : val_data = 0
    y_pred = estimator[dd].predict_proba(
        order_train[val_data].loc[(order_train[val_data].target.values == 0),
        trainVars(False, options.variables, options.bdtType)].values
    )[:, 1]
    y_predS = estimator[dd].predict_proba(
        order_train[val_data].loc[(order_train[val_data].target.values == 1),
        trainVars(False, options.variables, options.bdtType)].values
    )[:, 1]
    y_pred_train = estimator[dd].predict_proba(
        data_do.ix[(data_do.target.values == 0),
        trainVars(False, options.variables, options.bdtType)].values
    )[:, 1]
    y_predS_train = estimator[dd].predict_proba(
        data_do.ix[(data_do.target.values == 1),
        trainVars(False, options.variables, options.bdtType)].values
    )[:, 1]
    dict_plot = [
       [y_pred, "-", colorcold[dd], order_train_name[dd] + " test " + labelBKG],
       [y_predS, "-", colorhot[dd], order_train_name[dd] + " test signal"],
       [y_pred_train, "--", colorcold[dd], order_train_name[dd] + " train " + labelBKG],
       [y_predS_train, "--", colorhot[dd], order_train_name[dd] + " train signal"]
    ]
    yMax = 1
    for item in dict_plot :
        values1, bins, _ = ax.hist(
            item[0],
            ls=item[1], color = item[2],
            label=item[3],
            **hist_params
            )
        normed = sum(y_pred)
        mid = 0.5*(bins[1:] + bins[:-1])
        err=np.sqrt(values1*normed)/normed # denominator is because plot is normalized
        plt.errorbar(mid, values1, yerr=err, fmt='none', color= item[2], ecolor= item[2], edgecolor=item[2], lw=2)
        yMax = max(yMax, max(values1))
        print('\n\t\tvalue1: {},  max:{}'.format(values1, max(values1)))
    #plt.xscale('log')
    #plt.yscale('log')
#ax.legend(loc='upper center', title="all masses", fontsize = 'small')
ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.), bbox_transform=fig.transFigure, title="all masses", fontsize = 'small')
ax.set_ylim(0, yMax*1.2) # set y-axis range of XGBclassifier
nameout = channel+'/'+bdtType+'_'+trainvar+'_'+str(len(trainVars(False, options.variables, options.bdtType)))+'_'+hyppar+'_AllMass_'+(options.tauID)+'_XGBclassifier.pdf'
fig.savefig(nameout)
fig.savefig(nameout.replace(".pdf", ".png"))








## the correlation matrix we do with all the data
if options.HypOpt==False :
	for ii in [1,2] :
		if ii == 1 :
			datad=data.loc[data[target].values == 1]
			label="signal"
		else :
			datad=data.loc[data[target].values == 0]
			label="BKG"
		datacorr = datad[trainVars(False, options.variables, options.bdtType)].astype(float)  #.loc[:,trainVars(False)] #dataHToNobbCSV[[trainVars(True)]]
		correlations = datacorr.corr()
		fig = plt.figure(figsize=(10, 10))
		ax = fig.add_subplot(111)
		cax = ax.matshow(correlations, vmin=-1, vmax=1)
		ticks = np.arange(0,len(trainVars(False, options.variables, options.bdtType)),1)
		plt.rc('axes', labelsize=8)
		ax.set_xticks(ticks)
		ax.set_yticks(ticks)
		ax.set_xticklabels(trainVars(False, options.variables, options.bdtType),rotation=-90)
		ax.set_yticklabels(trainVars(False, options.variables, options.bdtType))
		fig.colorbar(cax)
		fig.tight_layout()
		nameout = "{}/{}_{}_{}_corr_{}_{}.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False, options.variables, options.bdtType))),label,(options.tauID))
		plt.savefig(nameout)
		plt.savefig(nameout.replace(".pdf",".png"))
		ax.clear()



if options.ClassErr_vs_epoch==True :
   data = data[trainVars(False, options.variables, options.bdtType)+["target","totalWeight"]] ## pandas dataframe with trainVars, target and totalWeight columns
   #data = data.drop(['target'], axis=1)
   #data_target = data.drop(['target'], axis=1)
   #data_target = data[target]
   #data_target = data.loc[[target]]
   data_target = data[[target]]
   print('\n\n data: {}'.format(data))
   print('\n\n data_target: {}'.format(data_target))

   # split data into train and test sets -2
   ## (X_train, X_test) are  pandas dataframes while (y_train, y_test) are numpy arrays
   X_train, X_test, y_train, y_test = train_test_split(data[trainVars(False, options.variables, options.bdtType)+["totalWeight"]], data_target, test_size=0.50, random_state=7)
   #X_train, X_test, y_train, y_test = train_test_split(data[trainVars(False, options.variables, options.bdtType)+["totalWeight"]], test_size=0.50, random_state=7)

   print('\n\n X_train: {}'.format(X_train))
   print('\n\n X_test: {}'.format(X_test))
   print('\n\n y_train: {}'.format(y_train))
   print('\n\n y_test: {}'.format(y_test))

   X_train_weight = np.array(X_train['totalWeight'].values, dtype=np.float)
   X_test_weight = np.array(X_test['totalWeight'].values, dtype=np.float)

   X_train = X_train.drop(['totalWeight'], axis=1)
   X_test = X_test.drop(['totalWeight'], axis=1)


   ## Converting dataframe to numpy array (dtype=object)
   X_train = X_train.values
   X_test = X_test.values

   eval_set = [(X_train, y_train), (X_test, y_test)]


   array_ntrees = [1000, 2000]
   #lr = 10/ntrees ## [1, 0.1, 0.01, 0.005, 0.0033]
   array_depth = [2,3,4]
   array_mcw = [1,5]



   X_label = []
   Acc_label = []
   i = 0
   file_ = open("{}/{}_{}_{}_GSCV111.log".format(channel,bdtType,trainvar,str(len(trainVars(False, options.variables, options.bdtType)))),"w")
   for ntrees in  array_ntrees:
	   lr = 10./float(ntrees) ## so that lr*ntrees = 10
	   for depth in array_depth:
		   for mcw in array_mcw:
			   i = i + 1
			   print('ntrees: %i, lr: %f, depth: %i, mcw: %i' % (ntrees, lr, depth, mcw))
			   #hyppar_test="ntrees_"+str(ntrees)+"_depth_"+str(depth)+"_mcw_"+str(mcw)+lr_to_str(lr)
                           hyppar_test="ntrees_"+str(ntrees)+"_depth_"+str(depth)+"_mcw_"+str(mcw)+"_lr_"+str(lr)
			   X_label.append(hyppar_test)
			   print("file label:", hyppar_test)
			   file_.write('fit %i, %s' % (i, hyppar_test))
			   file_.write('\n')
			   model = xgb.XGBClassifier(
				   n_estimators = ntrees,
				   max_depth = depth,
				   min_child_weight = mcw,
				   learning_rate = lr
				   )
			   print('Fit started')
                           '''
			   model.fit(
				   X_train, y_train,
				   sample_weight=X_train_weight,
				   sample_weight_eval_set=[X_train_weight, X_test_weight],
				   eval_metric=["auc", "error", "logloss"],
				   eval_set=eval_set,
				   verbose=True
				   #,early_stopping_rounds=100
				   #,callbacks=[xgb.callback.print_evaluation(show_stdv=True)] ## Doesn't work here either !!
				   )
                           '''
                           model.fit(
				   X_train, y_train,
				   sample_weight=X_train_weight,
				   eval_set = eval_set,
			           #sample_weight_eval_set = [X_train_weight, X_test_weight],
				   eval_metric = ["auc", "error", "logloss"],
				   verbose=True
				   #,early_stopping_rounds=100
				   #,callbacks=[xgb.callback.print_evaluation(show_stdv=True)] ## Doesn't work here either !!
				   )

                           
			   # make predictions for test data
			   y_pred = model.predict(X_test)
                           print('y_pred: {} \n\ny_test: {}'.format(y_pred,y_test))

			   # evaluate predictions
			   accuracy = accuracy_score(y_test, y_pred)
			   print("Accuracy: %.2f%%" % (accuracy * 100.0))
			   Acc_label.append((accuracy*100.0))

			   # retrieve performance metrics
			   results = model.evals_result()
			   epochs = len(results['validation_0']['error'])
			   x_axis = range(0, epochs)
			   # plot log loss
			   fig1, ax = plt.subplots()
			   ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
			   ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
			   ax.legend()
			   plt.ylabel('Log Loss')
			   plt.title('XGBoost Log Loss')
			   plt.show()
                           #fig1.savefig("Log_loss.pdf")
			   fig1.savefig("{}/Log_loss_{}_{}_{}_{}.pdf".format(channel, bdtType, trainvar, str(len(trainVars(False, options.variables, options.bdtType))), hyppar_test))
			   # plot auc
			   fig1a, ax = plt.subplots()
			   ax.plot(x_axis, results['validation_0']['auc'], label='Train')
			   ax.plot(x_axis, results['validation_1']['auc'], label='Test')
			   ax.legend()
			   plt.ylabel('Area under ROC curve')
			   plt.title('XGBoost AUC')
			   plt.show()
                           #fig1a.savefig('auc.pdf')
			   fig1a.savefig("{}/auc_{}_{}_{}_{}.pdf".format(channel, bdtType, trainvar, str(len(trainVars(False, options.variables, options.bdtType))), hyppar_test))
			   # plot classification error
			   fig2, ax = plt.subplots()
			   ax.plot(x_axis, results['validation_0']['error'], label='Train')
			   ax.plot(x_axis, results['validation_1']['error'], label='Test')
			   ax.legend()
			   plt.ylabel('Classification Error')
			   plt.title('XGBoost Classification Error')
			   plt.show()
                           #fig2.savefig('XGBoost_Classification_error2.pdf')
			   fig2.savefig("{}/XGBoost_Classification_error_{}_{}_{}_{}.pdf".format(channel, bdtType, trainvar, str(len(trainVars(False, options.variables, options.bdtType))), hyppar_test))
			   plt.close()

   fit_label = np.arange(1,31) ## gives us [1,2,....30]
   fig, ax = plt.subplots()
   plt.plot(fit_label, Acc_label, 'ro')
   plt.xlabel('fit index')
   plt.ylabel('Classification Accuracy (%%)')
   plt.axis([0, 35, 60, 100])
   #plt.axis([0, 6, 0, 100])
   #plt.bar(fit_label, height = Acc_label)
   #plt.xticks(fit_label, X_label)
   fig.savefig('Accuracy.pdf')



process = psutil.Process(os.getpid())
print(process.memory_info().rss)
print(datetime.now() - startTime)


process = psutil.Process(os.getpid())
print(process.memory_info().rss)
print(datetime.now() - startTime)


if options.gridSearchCV == True:
    print("\n\nInside gridSearchCV part: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # Normalize Background and signal processes in data collection
    test_size = 0.5
    traindataset1, valdataset1  = train_test_split(data[trainVars(False, options.variables, options.bdtType)+["target","totalWeight","key"]], test_size=test_size, random_state=7)
    #print('\n\ngridSearchCV:: \ntraindataset1: {}'.format(traindataset1))
    #print('valdataset1: {}'.format(valdataset1))
    print("\n Train data fraction: %f,   test data fraction: %f " % ((1-test_size),test_size))
    
    
    print("\n\nCheck how train_test_split data background and signals are normalized::: *****")
    order_train1 = [traindataset1, valdataset1]
    order_train1_name = ["train", "test"]
    for idx, data_do in enumerate(order_train1) :
        print "\n\ndata %i %s \nEvent table :: \n %30s  %10s   %s " % (idx,order_train1_name[idx],"Process", "nEvents", "nEventsWeighted")
        for key in output["keys"]:
            nEvents = int(len(data_do.loc[(data_do['key']==key)]))
            nEventsWtg = float(data_do.loc[(data_do['key']==key), [weights]].sum())
            #print " \n\n",key," sum: ",len(data.loc[(data['key']==key)]),", \t ",data.loc[(data['key']==key), [weights]].sum()
            print " %30s  %10i  %10.3f" % (key,nEvents,nEventsWtg)

        print("\nnWeighted Events \t\t\t\t\t\t %s \t %s \t %s" % ('BDT','Datacards','Datacards*test_size'))
        ttbar_samples = ['TTToSemiLeptonic', 'TTTo2L2Nu']
        print("ttbar_samples: %s \t %f \t %f \t %f" % (str(ttbar_samples),data_do.loc[(data_do['key'].isin(ttbar_samples)), [weights]].sum(),output["TTdatacard"], output["TTdatacard"]*test_size))
        print("DY \t\t\t\t\t\t\t %f \t %f \t %f" % (data_do.loc[(data_do['key']=='DY'), [weights]].sum(), output["DYdatacard"], output["DYdatacard"]*test_size))
        print("WZ \t\t\t\t\t\t\t %f \t %f \t %f" % (data_do.loc[(data_do['key']=='WZ'), [weights]].sum(), output["WZdatacard"], output["WZdatacard"]*test_size))
        print("TTZ \t\t\t\t\t\t\t %f \t %f \t %f" % (data_do.loc[(data_do['key']=='TTZJets'), [weights]].sum(), output["TTZdatacard"], output["TTZdatacard"]*test_size))
        print("TTV \t\t\t\t\t\t\t %f \t %f \t %f" % (data_do.loc[(data_do['key']=='TTWJets'), [weights]].sum(), output["TTWdatacard"], output["TTWdatacard"]*test_size))
        for key in output["keys"]:
            if not 'signal' in key: continue
            sProcess = key
            sDatacard = key+'datacard'
            print("%s \t\t\t\t %f \t %f \t %f" % (sProcess, data_do.loc[(data_do['key']==sProcess), [weights]].sum(), output[sDatacard], output[sDatacard]*test_size))



        if 'SUM_HH' in bdtType :
            ttbar_samples = ['TTToSemiLeptonic', 'TTTo2L2Nu'] ## Removed TTToHadronic since zero events selected for this sample
            data_do.loc[(data_do['key'].isin(ttbar_samples)), [weights]]              *= output["TTdatacard"]/data_do.loc[(data_do['key'].isin(ttbar_samples)), [weights]].sum()
            data_do.loc[(data_do['key']=='DY'), [weights]]                            *= output["DYdatacard"]/data_do.loc[(data_do['key']=='DY'), [weights]].sum()
            if "evtLevelSUM_HH_bb1l_res" in bdtType :
                data_do.loc[(data_do['key']=='W'), [weights]]                         *= Wdatacard/data_do.loc[(data_do['key']=='W')].sum()
            if "evtLevelSUM_HH_2l_2tau_res" in bdtType :
                data_do.loc[(data_do['key']=='TTZJets'), [weights]]                       *= output["TTZdatacard"]/data_do.loc[(data_do['key']=='TTZJets'), [weights]].sum() ## TTZJets
                data_do.loc[(data_do['key']=='TTWJets'), [weights]]                       *= output["TTWdatacard"]/data_do.loc[(data_do['key']=='TTWJets'), [weights]].sum() ## TTWJets + TTWW
                data_do.loc[(data_do['key']=='ZZ'), [weights]]                            *= output["ZZdatacard"]/data_do.loc[(data_do['key']=='ZZ'), [weights]].sum() ## ZZ +ZZZ
                data_do.loc[(data_do['key']=='WZ'), [weights]]                            *= output["WZdatacard"]/data_do.loc[(data_do['key']=='WZ'), [weights]].sum() ## WZ + WZZ_4F
                data_do.loc[(data_do['key']=='WW'), [weights]]                            *= output["WWdatacard"]/data_do.loc[(data_do['key']=='WW'), [weights]].sum() ## WW + WWZ + WWW_4F
                #data_do.loc[(data_do['key'].isin(['TTWJets', 'TTZJets'])), [weights]] *= output["TTVdatacard"]/data_do.loc[(data_do['key'].isin(['TTWJets', 'TTZJets'])), [weights]].sum() # consider do separately
                #data_do.loc[(data_do['key'].isin(['WW','WZ','ZZ'])), [weights]]       *= output["VVdatacard"]/data_do.loc[(data_do['key'].isin(['WW','WZ','ZZ'])), [weights]].sum() # consider do separatelly
                #data_do.loc[(data_do['key']=='VH'), [weights]]                        *= output["VHdatacard"]/data_do.loc[(data_do['key']=='VH'), [weights]].sum() # consider removing
                #data_do.loc[(data_do['key']=='TTH'), [weights]]                       *= output["TTHdatacard"]/data_do.loc[(data_do['key']=='TTH'), [weights]].sum() # consider removing
            if "evtLevelSUM_HH_3l_0tau_res" in bdtType :
                data_do.loc[(data_do['key']=='WZ'), [weights]]                            *= output["WZdatacard"]/data_do.loc[(data_do['key']=='WZ'), [weights]].sum()
                data_do.loc[(data_do['key']=='TTZJets'), [weights]]                       *= output["TTZdatacard"]/data_do.loc[(data_do['key']=='TTZJets'), [weights]].sum() ## TTZJets
                data_do.loc[(data_do['key']=='TTWJets'), [weights]]                       *= output["TTWdatacard"]/data_do.loc[(data_do['key']=='TTWJets'), [weights]].sum() ## TTWJets + TTWW
                if 'ZZ' in output["keys"]:
                    data_do.loc[(data_do['key']=='ZZ'), [weights]]                            *= output["ZZdatacard"]/data_do.loc[(data_do['key']=='ZZ'), [weights]].sum()
                if 'WW' in output["keys"]:
                    data_do.loc[(data_do['key']=='WW'), [weights]]                            *= output["WWdatacard"]/data_do.loc[(data_do['key']=='WW'), [weights]].sum()


                # Normalize various signal as well
                for key in output["keys"]:
                    if not 'signal' in key: continue
                    sProcess = key
                    sDatacard = key+'datacard'
                    #print("%s: wt: %f, datacard: %f" % (sProcess, data_do.loc[(data_do['key']==sProcess), [weights]].sum(), output[sDatacard]))
                    data_do.loc[(data_do['key']==sProcess), [weights]]                            *= output[sDatacard]/data_do.loc[(data_do['key']==sProcess), [weights]].sum()

                
                print "\n\ndata %i %s \nEvent table after normalizing:: \n %30s  %10s   %s " % (idx,order_train_name[idx],"Process", "nEvents", "nEventsWeighted")
                for key in output["keys"]:
                    nEvents = int(len(data_do.loc[(data_do['key']==key)]))
                    nEventsWtg = float(data_do.loc[(data_do['key']==key), [weights]].sum())
                    #print " \n\n",key," sum: ",len(data.loc[(data['key']==key)]),", \t ",data.loc[(data['key']==key), [weights]].sum()
                    print " %30s  %10i  %10.3f" % (key,nEvents,nEventsWtg)
                    
                print("\nnWeighted Events \t\t\t\t\t\t %s \t %s" % ('BDT','Datacards'))
                ttbar_samples = ['TTToSemiLeptonic', 'TTTo2L2Nu']
                print("ttbar_samples: %s \t %f \t %f" % (str(ttbar_samples),data_do.loc[(data_do['key'].isin(ttbar_samples)), [weights]].sum(),float(output["TTdatacard"])))
                print("DY \t\t\t\t\t\t\t %f \t %f" % (data_do.loc[(data_do['key']=='DY'), [weights]].sum(), output["DYdatacard"]))
                print("WZ \t\t\t\t\t\t\t %f \t %f" % (data_do.loc[(data_do['key']=='WZ'), [weights]].sum(), output["WZdatacard"]))
                print("TTZ \t\t\t\t\t\t\t %f \t %f" % (data_do.loc[(data_do['key']=='TTZJets'), [weights]].sum(), output["TTZdatacard"]))
                print("TTV \t\t\t\t\t\t\t %f \t %f" % (data_do.loc[(data_do['key']=='TTWJets'), [weights]].sum(), output["TTWdatacard"]))
                for key in output["keys"]:
                    if not 'signal' in key: continue
                    sProcess = key
                    sDatacard = key+'datacard'
                    print("%s \t\t\t\t %f \t %f" % (sProcess, data_do.loc[(data_do['key']==sProcess), [weights]].sum(), output[sDatacard]))

                        
            if "gen_mHH" in trainVars(False, options.variables, options.bdtType):    
                for mass in range(len(output["masses"])) :
                    data_do.loc[(data_do[target]==1) & (data_do["gen_mHH"].astype(np.int) == int(output["masses"][mass])),[weights]] *= 100000./data_do.loc[(data_do[target]==1) & (data_do["gen_mHH"]== output["masses"][mass]),[weights]].sum()
                    data_do.loc[(data_do[target]==0) & (data_do["gen_mHH"].astype(np.int) == int(output["masses"][mass])),[weights]] *= 100000./data_do.loc[(data_do[target]==0) & (data_do["gen_mHH"]== output["masses"][mass]),[weights]].sum()
            else:
                data_do.loc[data_do['target']==0, [weights]] *= 100000/data_do.loc[data_do['target']==0][weights].sum()
                data_do.loc[data_do['target']==1, [weights]] *= 100000/data_do.loc[data_do['target']==1][weights].sum() 
                    
        else :
            data_do.loc[data_do['target']==0, [weights]] *= 100000/data_do.loc[data_do['target']==0][weights].sum()
            data_do.loc[data_do['target']==1, [weights]] *= 100000/data_do.loc[data_do['target']==1][weights].sum()
                

        print "\n\ndata %i %s \nEvent table after normalizing and scaling up by 1e5:: \n %30s  %10s   %s " % (idx,order_train_name[idx],"Process", "nEvents", "nEventsWeighted")
        for key in output["keys"]:
            nEvents = int(len(data_do.loc[(data_do['key']==key)]))
            nEventsWtg = float(data_do.loc[(data_do['key']==key), [weights]].sum())
            #print " \n\n",key," sum: ",len(data.loc[(data['key']==key)]),", \t ",data.loc[(data['key']==key), [weights]].sum()
            print " %30s  %10i  %10.3f" % (key,nEvents,nEventsWtg)
            
        print("\nnWeighted Events \t\t\t\t\t\t %s \t %s" % ('BDT','Datacards'))
        ttbar_samples = ['TTToSemiLeptonic', 'TTTo2L2Nu']
        print("ttbar_samples: %s \t %f \t %f" % (str(ttbar_samples),data_do.loc[(data_do['key'].isin(ttbar_samples)), [weights]].sum(),float(output["TTdatacard"])))
        print("DY \t\t\t\t\t\t\t %f \t %f" % (data_do.loc[(data_do['key']=='DY'), [weights]].sum(), output["DYdatacard"]))
        print("WZ \t\t\t\t\t\t\t %f \t %f" % (data_do.loc[(data_do['key']=='WZ'), [weights]].sum(), output["WZdatacard"]))
        print("TTZ \t\t\t\t\t\t\t %f \t %f" % (data_do.loc[(data_do['key']=='TTZJets'), [weights]].sum(), output["TTZdatacard"]))
        print("TTV \t\t\t\t\t\t\t %f \t %f" % (data_do.loc[(data_do['key']=='TTWJets'), [weights]].sum(), output["TTWdatacard"]))
        for key in output["keys"]:
            if not 'signal' in key: continue
            sProcess = key
            sDatacard = key+'datacard'
            print("%s \t\t\t\t %f \t %f" % (sProcess, data_do.loc[(data_do['key']==sProcess), [weights]].sum(), output[sDatacard]))
        print("\n")
    
    
    
    print('\n\ngridSearchCV::')
    # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    '''
    # 1st round tunning    
    param_grid = { 
    	'n_estimators': [1000, 1500, 2000],
        'learning_rate': [0.01, 0.05, 0.001],
    	'max_depth': [3],
        'min_child_weight': [1],
        'gamma': [0],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'scale_pos_weight': [1]
    }

    # searchGrid round 2
    param_grid = { 
    	'n_estimators': [1000],
        'learning_rate': [0.01],
    	'max_depth': [3],
        'min_child_weight': [0.5, 1, 3, 5], # 1
        'gamma': [0],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'scale_pos_weight': [1]
    }

    # searchGrid round 3
    param_grid = { 
    	'n_estimators': [1000],
        'learning_rate': [0.01],
    	'max_depth': [3], 
        'min_child_weight': [1],
        'gamma': [i/10.0 for i in range(0,5)], # 0
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'scale_pos_weight': [1]
    }


    # searchGrid round 4 
    param_grid = { 
    	'n_estimators': [1000],
        'learning_rate': [0.01],
    	'max_depth': [3], 
        'min_child_weight': [1], 
        'gamma': [0],
        'subsample': [i/10.0 for i in range(5,10)], # 0.5
        'colsample_bytree': [i/10.0 for i in range(5,10)], # 0.5
        'scale_pos_weight': [1]
    } 

    # searchGrid round 4.1
    param_grid = { 
    	'n_estimators': [1000],
        'learning_rate': [0.01],
    	'max_depth': [3], 
        'min_child_weight': [1], 
        'gamma': [0],
        'subsample': [i/10.0 for i in range(1,7)], #[i/100.0 for i in range(30,70,5)],  #0.75
        'colsample_bytree': [i/10.0 for i in range(1,7)], #[i/100.0 for i in range(30,70,5)], # 0.75
        'scale_pos_weight': [1]
    }

    # searchGrid round 5 scoring = "precision"
    param_grid = { 
    	'n_estimators': [1000],
        'learning_rate': [0.01],
    	'max_depth': [3], 
        'min_child_weight': [1], 
        'gamma': [0],
        'subsample': [0.5],
        'colsample_bytree': [0.5],
        'scale_pos_weight': [1],
        'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100] # 100
    }

    # searchGrid round 5.1 scoring = "precision"
    param_grid = { 
    	'n_estimators': [1000],
        'learning_rate': [0.01],
    	'max_depth': [3], 
        'min_child_weight': [1], 
        'gamma': [0],
        'subsample': [0.5],
        'colsample_bytree': [0.5],
        'scale_pos_weight': [1],
        'reg_alpha':[1, 10, 30, 50, 80, 100, 150] # 0.5
    }
    '''    
    # searchGrid round 6 scoring = "precision"
    param_grid = { 
    	'n_estimators': [1000],
        'learning_rate': [0.01],
    	'max_depth': [3], 
        'min_child_weight': [1], 
        'gamma': [0],
        'subsample': [0.5],
        'colsample_bytree': [0.5],
        'scale_pos_weight': [1],
        'reg_alpha':[0.5],
        'reg_lambda': [0, 1e-5, 1e-2, 0.1, 1, 5, 10, 100] # 1.
    }
    '''        
    # searchGrid round 7.1 scoring = "precision"
    param_grid = { 
    	'n_estimators': [1000],
        'learning_rate': [0.1],
    	'max_depth': [2], 
        'min_child_weight': [3], 
        'gamma': [0],
        'subsample': [0.75],
        'colsample_bytree': [0.75],
        'scale_pos_weight': [1],
        'reg_alpha':[0.5],
        'reg_lambda': [0.5, 1, 1.5, 2, 3, 4] # 3
    }


    # searchGrid round 8 scoring = "precision"
    param_grid = { 
    	'n_estimators': [1000, 1500, 2000],
        'learning_rate': [0.1, 0.05, 0.01],
    	'max_depth': [2], 
        'min_child_weight': [3], 
        'gamma': [0],
        'subsample': [0.75],
        'colsample_bytree': [0.75],
        'scale_pos_weight': [1],
        'reg_alpha':[0.5],
        'reg_lambda': [3]
    }
    '''    
    
    #scoring = "roc_auc"
    #scoring = "accuracy"
    #scoring = "average_precision"
    #scoring = "neg_log_loss"
    scoring = "precision"
    
    early_stopping_rounds = 200 # Will train until validation_0-auc hasn't improved in 100 rounds.
    cv=3
    objective= 'binary:logistic'
    cls = xgb.XGBClassifier(objective= objective, nthread = 8, seed=8)
    fit_params = { "eval_set" : [(valdataset1[trainVars(False, options.variables, options.bdtType)].values,valdataset1[target])],
                   #"eval_metric" : "auc",
                   "eval_metric" : "logloss",
                   "early_stopping_rounds" : early_stopping_rounds,
		   'sample_weight': valdataset1[weights].values }
    print("\nGridSearchCV settings:: \nparam_grid: {} \nscoring: {} \nearly_stopping_rounds: {}".format(param_grid, scoring, early_stopping_rounds))
    print("cv: {} \nobjective: {}\n\n".format(cv, objective))
    gs = GridSearchCV(cls, param_grid, scoring, fit_params, cv = cv, verbose = 0)
    gs.fit(traindataset1[trainVars(False, options.variables, options.bdtType)].values,
	   traindataset1.target.astype(np.bool)
    )    
    for i, param in enumerate(gs.cv_results_["params"]):
	print("params : {} \n    cv auc = {}  +- {} ".format(param,gs.cv_results_["mean_test_score"][i],gs.cv_results_["std_test_score"][i]))
    print("best parameters",gs.best_params_)
    print("best score",gs.best_score_)
    print("grid score: {}".format(gs.grid_scores_))
    #print("best iteration",gs.best_iteration_)
    #print("best ntree limit",gs.best_ntree_limit_)
    file = open("{}/{}_{}_{}_GSCV.log".format(channel,bdtType,trainvar,str(len(trainVars(False, options.variables, options.bdtType)))),"w")
    file.write(
	str(trainVars(False, options.variables, options.bdtType))+"\n"+
	"best parameters"+str(gs.best_params_) + "\n"+
	"best score"+str(gs.best_score_)+ "\n"
	#"best iteration"+str(gs.best_iteration_)+ "\n"+
	#"best ntree limit"+str(gs.best_ntree_limit_)
    )
    for i, param in enumerate(gs.cv_results_["params"]):
	file.write("params : {} \n    cv auc = {}  +- {} {}".format(param,gs.cv_results_["mean_test_score"][i],gs.cv_results_["std_test_score"][i]," \n"))
    file.close()

    best_param = gs.best_params_
    print("\n\nBDT with best parameters of gridSearchCV::\nbest parameters: {}".format(best_param))
    print("max_depth: {};  n_estimators: {};  min_child_weight: {};  learning_rate: {}; ".format(best_param['max_depth'],best_param['n_estimators'],best_param['min_child_weight'],best_param['learning_rate']))
    #print("max_depth: {}".format(best_param['max_depth']))
    #print("max_depth: {}".format())
    cls1 = xgb.XGBClassifier(
                        n_estimators = best_param['n_estimators'],
    			max_depth = best_param['max_depth'],
    			min_child_weight = best_param['min_child_weight'],
    			learning_rate = best_param['learning_rate'],
                        nthread = 8
    )
    cls1.fit(
    	traindataset1[trainVars(False, options.variables, options.bdtType)].values,
    	traindataset1.target.astype(np.bool),
    	sample_weight=(traindataset1[weights].astype(np.float64))
    	)
    proba = cls1.predict_proba(traindataset1[trainVars(False, options.variables, options.bdtType)].values )
    fpr, tpr, thresholds = roc_curve(
        traindataset1[target].astype(np.bool), proba[:,1],
        sample_weight=(traindataset1[weights].astype(np.float64))
    )
    train_auc = auc(fpr, tpr, reorder = True)
    print("XGBoost train set auc - {}".format(train_auc))

    proba = cls1.predict_proba(valdataset1[trainVars(False, options.variables, options.bdtType)].values )
    fprt, tprt, thresholds = roc_curve(
        valdataset1[target].astype(np.bool), proba[:,1],
        sample_weight=(valdataset1[weights].astype(np.float64))
    )
    test_auct = auc(fprt, tprt, reorder = True)
    print("XGBoost test set auc - {}".format(test_auct))
