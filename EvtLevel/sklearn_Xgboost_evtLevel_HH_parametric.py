import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from datetime import datetime
import sys , time
#import sklearn_to_tmva
import sklearn
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
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
from root_numpy import root2array, rec2array, array2root, tree2array

import xgboost as xgb
#import catboost as catboost #import CatBoostRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import ROOT
from tqdm import trange
import glob
#sys.stdout = file_
from collections import OrderedDict
startTime = datetime.now()
execfile("../python/data_manager.py")
#python sklearn_Xgboost_evtLevel_HH_parametric.py --channel 'bb2l' --bdtType 'evtLevelSUM_HH_bb2l_res'  --variables "noTopness"
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--channel ", type="string", dest="channel", help="The ones whose variables implemented now are:\n   - 1l_2tau\n   - 2lss_1tau\n It will create a local folder and store the report*/xml", default='T')
parser.add_option("--variables", type="string", dest="variables", help="  Set of variables to use -- it shall be put by hand in the code, in the fuction trainVars(all)\n Example to 2ssl_2tau   \n                              all==True -- all variables that should be loaded (training + weights) -- it is used only once\n                               all==False -- only variables of training (not including weights) \n  For the channels implemented I defined 3 sets of variables/each to confront at limit level\n  trainvar=allVar -- all variables that are avaible to training (including lepton IDs, this is here just out of curiosity) \n  trainvar=oldVar -- a minimal set of variables (excluding lepton IDs and lep pt's)\n  trainvar=notForbidenVar -- a maximal set of variables (excluding lepton IDs and lep pt's) \n  trainvar=notForbidenVarNoMEM -- the same as above, but excluding as well MeM variables", default=None)
parser.add_option("--bdtType", type="string", dest="bdtType", help=" evtLevelTT_TTH or evtLevelTTV_TTH", default='T')
parser.add_option("--HypOpt", action="store_true", dest="HypOpt", help="If you call this will not do plots with repport", default=False)
parser.add_option("--doXML", action="store_true", dest="doXML", help="Do save not write the xml file", default=False)
parser.add_option("--doPlots", action="store_true", dest="doPlots", help="Fastsim Loose/Tight vs Fullsim variables plots", default=False)
parser.add_option("--nonResonant", action="store_true", dest="doPlots", help="Fastsim Loose/Tight vs Fullsim variables plots", default=False)
parser.add_option("--ntrees ", type="int", dest="ntrees", help="hyp", default=2000) #1500
parser.add_option("--treeDeph", type="int", dest="treeDeph", help="hyp", default=2) #3
parser.add_option("--lr", type="float", dest="lr", help="hyp", default=0.01)
parser.add_option("--mcw", type="float", dest="mcw", help="hyp", default=1)
parser.add_option("--Bkg_mass_rand", type="string", dest="Bkg_mass_rand", help="fix gen_mHH randomiz. method for bkg.s", default='default')
parser.add_option("--gridSearchCV", action="store_true", dest="gridSearchCV", help="Search optimal values for XGB training parameters", default=False)
(options, args) = parser.parse_args()

print "-ntrees ",options.ntrees,"  --treeDeph ",options.treeDeph,"  --lr ",options.lr,"    --mcw ",options.mcw

Bkg_mass_rand=options.Bkg_mass_rand
doPlots=options.doPlots
bdtType=options.bdtType
trainvar=options.variables
#hyppar=str(options.variables)+"_ntrees_"+str(options.ntrees)+"_deph_"+str(options.treeDeph)+"_mcw_"+str(options.mcw)+"_lr_0o0"+str(int(options.lr*100))
hyppar=str(options.variables)+"_ntrees_"+str(options.ntrees)+"_deph_"+str(options.treeDeph)+"_mcw_"+str(options.mcw)+"_lr_"+str(int(options.lr))

## --- OUTPUT DIRECTORY NAME----
#channel=options.channel+"_HH"
# channel=options.channel+"_HH_dR03mvaLoose"
# channel=options.channel+"_HH_dR03mvaVLoose"
#channel=options.channel+"_HH_dR03mvaVVLoose_all"
channel=options.channel+"_HH_BkMassRndm"+options.Bkg_mass_rand+"_"+options.variables

if 'bb2l' in options.channel:
    execfile("../cards/info_bb2l_HH.py")
    channel = options.channel+"_HH"
if '3l_0tau' in options.channel:
    #print("here1 3l_0tau")
    execfile("../cards/info_3l_0tau_HH.py")
    #print("here2 3l_0tau")
    channel=options.channel+"_HH_Simple_BkMassRndm"+options.Bkg_mass_rand+"_Vars"+str(len(trainVars(False, options.variables, options.bdtType)))+"_ntrees_"+str(options.ntrees)+"_deph_"+str(options.treeDeph)+"_mcw_"+str(options.mcw)+"_lr_"+str(options.lr)
    if options.gridSearchCV == True:
        channel += "_gridSearchCV"
if "2l_2tau" in options.channel :
    execfile("../cards/info_2l_2tau_HH.py")


#if resonant bdtype
file_ = open('roc_%s.log'%channel,'w+')


weights="totalWeight"
target='target'
variable =''
if bdtType.find('nonres') ==-1 :
        variable = 'gen_mHH'
else :
        variable = 'node'

import shutil,subprocess
proc=subprocess.Popen(['mkdir '+channel],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()

output = read_from(Bkg_mass_rand)
#print "inputPath: ",inputPath,", channelInTree: ",channelInTree; sys.stdout.flush()
print 'output[keys]: ', output["keys"]; sys.stdout.flush()
print("output[masses]: {}".format(output["masses"]))
print("output[masses_test]: {}".format(output["masses_test"]))
readIpMasses = output["masses"] + list(set(output["masses_test"]) - set(output["masses"]))
#readIpMasses = readIpMasses.sort()
print("readIpMasses: {}".format(readIpMasses))


####################################################################################################
## Load data
#data=load_data_2017(inputPath,channelInTree,trainVars(True),[],bdtType)
if "bb2l" in channel   : data=load_data_2017_HH(inputPath,channelInTree,trainVars(True),[],bdtType)
elif "bb1l" in channel : data=load_data_2017_HH(inputPath,channelInTree,trainVars(True),[],bdtType)
elif "2l_2tau" in channel: data=load_data_2017_HH_2l_2tau(inputPath,channelInTree,trainVars(True),[],bdtType)
elif "3l_0tau" in channel:
        data=load_data_2017(
                inputPath = output["inputPath"],
                channelInTree = output["channelInTree"],
                variables = trainVars(True),
                criteria = [],
                bdtType = bdtType,
                channel = channel,
                keys = output["keys"],
                masses = readIpMasses, #output["masses"],
                mass_randomization = output["mass_randomization"],
        )
else : data=load_data_2017(inputPath,channelInTree,trainVars(True),[],bdtType)
#**********************

#################################################################################
#print ("Sum of weights:", data.loc[data['target']==0][weights].sum())

dataOriginal = data.copy() 


## Balance datasets
#https://stackoverflow.com/questions/34803670/pandas-conditional-multiplication
if "evtLevelSUM_HH_bb1l_res" in bdtType : variables = [250,270,280,320,350,400,450,500,600,650,750,800,850,900,1000]
elif "evtLevelSUM_HH_bb2l_res" in bdtType : variables = [250,270,280,320,350,400,450,500,600,650,750,800,850,900,1000]
elif "evtLevelSUM_HH_bb2l_nonres" in bdtType : variables = [2,3,7,9,12,20]
elif "evtLevelSUM_HH_2l_2tau_res" in bdtType : variables = [250,260,270,280,300,350,400,450,500, 550,600,650,700,750,800,850,900,1000]
elif "evtLevelSUM_HH_3l_0tau_res" in bdtType : variables = [400, 700]
#elif "evtLevelSUM_HH_2l_2tau_res" in bdtType : variables = [500]
else : print '****************** please define your mass point**************'



print "Weights: DY: tree.sumWeights: ",data.loc[(data['key']=='DY'), [weights]].sum(), ",  datacard: ",output["DYdatacard"]
print "Weights: WZ: tree.sumWeights: ",data.loc[(data['key']=='WZ'), [weights]].sum(), ",  datacard: ",output["WZdatacard"]
print "Weights: TTZJets: tree.sumWeights: ",data.loc[(data['key']=='TTZJets'), [weights]].sum(), ",  datacard: ",output["TTZdatacard"]
print "Weights: TTWJets: tree.sumWeights: ",data.loc[(data['key']=='TTWJets'), [weights]].sum(), ",  datacard: ",output["TTWdatacard"]
ttbar_samples = ['TTToSemiLeptonic', 'TTTo2L2Nu', 'TTToHadronic']
print "Weights: TT: tree.sumWeights: ",data.loc[(data['key'].isin(ttbar_samples)), [weights]].sum(), ",  datacard: ",output["TTdatacard"]

print "\nWeights: signal_ggf_spin0_400_hh_4v: tree.sumWeights: ",data.loc[(data['key']=='signal_ggf_spin0_400_hh_4v'), [weights]].sum()
print "Weights: signal_ggf_spin0_700_hh_4v: tree.sumWeights: ",data.loc[(data['key']=='signal_ggf_spin0_700_hh_4v'), [weights]].sum()
print "Weights: signal_ggf_spin0_400_hh_4t: tree.sumWeights: ",data.loc[(data['key']=='signal_ggf_spin0_400_hh_4t'), [weights]].sum()
print "Weights: signal_ggf_spin0_700_hh_4t: tree.sumWeights: ",data.loc[(data['key']=='signal_ggf_spin0_700_hh_4t'), [weights]].sum()
print "Weights: signal_ggf_spin0_400_hh_2v2t: tree.sumWeights: ",data.loc[(data['key']=='signal_ggf_spin0_400_hh_2v2t'), [weights]].sum()
print "Weights: signal_ggf_spin0_700_hh_2v2t: tree.sumWeights: ",data.loc[(data['key']=='signal_ggf_spin0_700_hh_2v2t'), [weights]].sum()

print "\n\nEvent table before normalizing:: \n %30s  %10s   %s " % ("Process", "nEvents", "nEventsWeighted")
for key in output["keys"]:
    nEvents = int(len(data.loc[(data['key']==key)]))
    nEventsWtg = float(data.loc[(data['key']==key), [weights]].sum())
    #print " \n\n",key," sum: ",len(data.loc[(data['key']==key)]),", \t ",data.loc[(data['key']==key), [weights]].sum()
    print " %30s  %10i  %10.3f" % (key,nEvents,nEventsWtg)

ttbar_samples = ['TTToSemiLeptonic', 'TTTo2L2Nu']
print("ttbar_samples: {} \t {} \t {}".format(ttbar_samples,data.loc[(data['key'].isin(ttbar_samples)), [weights]].sum(),output["TTdatacard"]))
print("DY \t\t {} \t {}".format(data.loc[(data['key']=='DY'), [weights]].sum(), output["DYdatacard"]))
print("WZ \t\t {} \t {}".format(data.loc[(data['key']=='WZ'), [weights]].sum(), output["WZdatacard"]))
print("TTZ \t\t {} \t {}".format(data.loc[(data['key']=='TTZJets'), [weights]].sum(), output["TTZdatacard"]))
print("TTV \t\t {} \t {}".format(data.loc[(data['key']=='TTWJets'), [weights]].sum(), output["TTWdatacard"]))



if 'SUM_HH' in bdtType :
        ttbar_samples = ['TTToSemiLeptonic', 'TTTo2L2Nu'] ## Removed TTToHadronic since zero events selected for this sample
        #ttbar_samples = ['TTToSemiLeptonic', 'TTTo2L2Nu', 'TTToHadronic'] ## Removed TTToHadronic since zero events selected for this sample
        data.loc[(data['key'].isin(ttbar_samples)), [weights]]              *= output["TTdatacard"]/data.loc[(data['key'].isin(ttbar_samples)), [weights]].sum()
        data.loc[(data['key']=='DY'), [weights]]                            *= output["DYdatacard"]/data.loc[(data['key']=='DY'), [weights]].sum()
        if "evtLevelSUM_HH_bb1l_res" in bdtType :
            data.loc[(data['key']=='W'), [weights]]                         *= Wdatacard/data.loc[(data['key']=='W')].sum()
        if "evtLevelSUM_HH_2l_2tau_res" in bdtType :
            data.loc[(data['key']=='TTZJets'), [weights]]                       *= output["TTZdatacard"]/data.loc[(data['key']=='TTZJets'), [weights]].sum() ## TTZJets
            data.loc[(data['key']=='TTWJets'), [weights]]                       *= output["TTWdatacard"]/data.loc[(data['key']=='TTWJets'), [weights]].sum() ## TTWJets + TTWW
            data.loc[(data['key']=='ZZ'), [weights]]                            *= output["ZZdatacard"]/data.loc[(data['key']=='ZZ'), [weights]].sum() ## ZZ +ZZZ
            data.loc[(data['key']=='WZ'), [weights]]                            *= output["WZdatacard"]/data.loc[(data['key']=='WZ'), [weights]].sum() ## WZ + WZZ_4F
            data.loc[(data['key']=='WW'), [weights]]                            *= output["WWdatacard"]/data.loc[(data['key']=='WW'), [weights]].sum() ## WW + WWZ + WWW_4F
            #data.loc[(data['key'].isin(['TTWJets', 'TTZJets'])), [weights]] *= output["TTVdatacard"]/data.loc[(data['key'].isin(['TTWJets', 'TTZJets'])), [weights]].sum() # consider do separately
            #data.loc[(data['key'].isin(['WW','WZ','ZZ'])), [weights]]       *= output["VVdatacard"]/data.loc[(data['key'].isin(['WW','WZ','ZZ'])), [weights]].sum() # consider do separatelly
            #data.loc[(data['key']=='VH'), [weights]]                        *= output["VHdatacard"]/data.loc[(data['key']=='VH'), [weights]].sum() # consider removing
            #data.loc[(data['key']=='TTH'), [weights]]                       *= output["TTHdatacard"]/data.loc[(data['key']=='TTH'), [weights]].sum() # consider removing
        if "evtLevelSUM_HH_3l_0tau_res" in bdtType :
            data.loc[(data['key']=='WZ'), [weights]]                            *= output["WZdatacard"]/data.loc[(data['key']=='WZ'), [weights]].sum()
            data.loc[(data['key']=='TTZJets'), [weights]]                       *= output["TTZdatacard"]/data.loc[(data['key']=='TTZJets'), [weights]].sum() ## TTZJets
            data.loc[(data['key']=='TTWJets'), [weights]]                       *= output["TTWdatacard"]/data.loc[(data['key']=='TTWJets'), [weights]].sum() ## TTWJets + TTWW
            if 'ZZ' in output["keys"]:
                data.loc[(data['key']=='ZZ'), [weights]]                            *= output["ZZdatacard"]/data.loc[(data['key']=='ZZ'), [weights]].sum()
            if 'WW' in output["keys"]:
                data.loc[(data['key']=='WW'), [weights]]                            *= output["WWdatacard"]/data.loc[(data['key']=='WW'), [weights]].sum()
            # Normalize various signal as well
            for key in output["keys"]:
                if not 'signal' in key: continue
                sProcess = key
                sDatacard = key+'datacard'
                print("%s: wt: %f, datacard: %f" % (sProcess, data.loc[(data['key']==sProcess), [weights]].sum(), output[sDatacard]))
                data.loc[(data['key']==sProcess), [weights]]                            *= output[sDatacard]/data.loc[(data['key']==sProcess), [weights]].sum()

                
            print "\n\nEvent table after normalizing:: \n %30s  %10s   %s " % ("Process", "nEvents", "nEventsWeighted")
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

            print("nEvents for BDT: %i,  bk: %i,  signal: %i" % (len(data), len(data.loc[data['target']==0]), len(data.loc[data['target']==1])))
            print("nEventsweight  : %f,  bk: %f,  signal: %f" % (data[weights].sum(), data.loc[data['target']==0][weights].sum(), data.loc[data['target']==1][weights].sum()))

                  
        if "gen_mHH" in trainVars(False, options.variables, options.bdtType):
            print("output[masses]: {}".format(output["masses"]))
            print("Bk and signal total weights per mass point before bk-signal normalization")
            for mass in range(len(output["masses"])) :
                data.loc[(data[target]==1) & (data["gen_mHH"].astype(np.int) == int(output["masses"][mass])),[weights]] *= 100000./data.loc[(data[target]==1) & (data["gen_mHH"]== output["masses"][mass]),[weights]].sum()
                data.loc[(data[target]==0) & (data["gen_mHH"].astype(np.int) == int(output["masses"][mass])),[weights]] *= 100000./data.loc[(data[target]==0) & (data["gen_mHH"]== output["masses"][mass]),[weights]].sum()
                print("mass: %i,  bk weight: %f,  signal weight: %f" % (mass, data.loc[(data[target]==0) & (data["gen_mHH"]== output["masses"][mass]),[weights]].sum(), data.loc[(data[target]==1) & (data["gen_mHH"]== output["masses"][mass]),[weights]].sum()))
        else:
            data.loc[data['target']==0, [weights]] *= 100000/data.loc[data['target']==0][weights].sum()
            data.loc[data['target']==1, [weights]] *= 100000/data.loc[data['target']==1][weights].sum() 
else :
        data.loc[data['target']==0, [weights]] *= 100000/data.loc[data['target']==0][weights].sum()
        data.loc[data['target']==1, [weights]] *= 100000/data.loc[data['target']==1][weights].sum()
                        
                  
        

print "\n\nEvent table after normalizing and scale up by 1e5:: \n %30s  %10s   %s " % ("Process", "nEvents", "nEventsWeighted")
for key in output["keys"]:
    nEvents = int(len(data.loc[(data['key']==key)]))
    nEventsWtg = float(data.loc[(data['key']==key), [weights]].sum())
    #print " \n\n",key," sum: ",len(data.loc[(data['key']==key)]),", \t ",data.loc[(data['key']==key), [weights]].sum()
    print " %30s  %10i  %10.3f" % (key,nEvents,nEventsWtg)

print "\n\nAfter weight scaling"
print "Weights: DY: tree.sumWeights: ",data.loc[(data['key']=='DY'), [weights]].sum(), ",  datacard: ",output["DYdatacard"]
print "Weights: WZ: tree.sumWeights: ",data.loc[(data['key']=='WZ'), [weights]].sum(), ",  datacard: ",output["WZdatacard"]
print "Weights: TTZJets: tree.sumWeights: ",data.loc[(data['key']=='TTZJets'), [weights]].sum(), ",  datacard: ",output["TTZdatacard"]
print "Weights: TTWJets: tree.sumWeights: ",data.loc[(data['key']=='TTWJets'), [weights]].sum(), ",  datacard: ",output["TTWdatacard"]
ttbar_samples = ['TTToSemiLeptonic', 'TTTo2L2Nu', 'TTToHadronic']
print "Weights: TT: tree.sumWeights: ",data.loc[(data['key'].isin(ttbar_samples)), [weights]].sum(), ",  datacard: ",output["TTdatacard"]

print "\nWeights: signal_ggf_spin0_400_hh_4v: tree.sumWeights: ",data.loc[(data['key']=='signal_ggf_spin0_400_hh_4v'), [weights]].sum()
print "Weights: signal_ggf_spin0_700_hh_4v: tree.sumWeights: ",data.loc[(data['key']=='signal_ggf_spin0_700_hh_4v'), [weights]].sum()
print "Weights: signal_ggf_spin0_400_hh_4t: tree.sumWeights: ",data.loc[(data['key']=='signal_ggf_spin0_400_hh_4t'), [weights]].sum()
print "Weights: signal_ggf_spin0_700_hh_4t: tree.sumWeights: ",data.loc[(data['key']=='signal_ggf_spin0_700_hh_4t'), [weights]].sum()
print "Weights: signal_ggf_spin0_400_hh_2v2t: tree.sumWeights: ",data.loc[(data['key']=='signal_ggf_spin0_400_hh_2v2t'), [weights]].sum()
print "Weights: signal_ggf_spin0_700_hh_2v2t: tree.sumWeights: ",data.loc[(data['key']=='signal_ggf_spin0_700_hh_2v2t'), [weights]].sum()

print("After normalizing and scaling::")        
print("nEvents for BDT: %i,  bk: %i,  signal: %i" % (len(data), len(data.loc[data['target']==0]), len(data.loc[data['target']==1])))
print("nEventsweight  : %f,  bk: %f,  signal: %f" % (data[weights].sum(), data.loc[data['target']==0][weights].sum(), data.loc[data['target']==1][weights].sum()))
print("Bk and signal total weights per mass point after bk-signal normalization")
for mass in range(len(output["masses"])) :
    print("mass: %i,  bk weight: %f,  signal weight: %f" % (mass, data.loc[(data[target]==0) & (data["gen_mHH"]== output["masses"][mass]),[weights]].sum(), data.loc[(data[target]==1) & (data["gen_mHH"]== output["masses"][mass]),[weights]].sum()))


totTT =  data.loc[(data['key']=='TTToHadronic_PSweights') & (data[variable]==400),[weights]].sum()+data.loc[(data['key']=='TTToSemiLeptonic_PSweights') & (data[variable]==400),[weights]].sum()+data.loc[(data['key']=='TTTo2L2Nu_PSweights') & (data[variable]==400), [weights]].sum()
totDY = data.loc[(data['key']=='DY') & (data[variable]==400), [weights]].sum()
totW =  data.loc[(data['key']=='W') & (data[variable]==400) , [weights]].sum()
print 'ratioTTWDY = ', totTT/(totTT+totW+totDY),'\t', ':', totW/(totTT+totW+totDY),'\t', ':', totDY/(totTT+totW+totDY)
#print 'data to list = ', data.columns.values.tolist()
#print ("norm bk:", data.loc[data[target]==0][weights].sum(),",  sig:",data.loc[data[target]==1][weights].sum())

# drop events with NaN weights - for safety
#data.replace(to_replace=np.inf, value=np.NaN, inplace=True)
#data.replace(to_replace=np.inf, value=np.zeros, inplace=True)
#data = data.apply(lambda x: pandas.to_numeric(x,errors='ignore'))
data.dropna(subset=[weights],inplace = True) # data
data.fillna(0)

print "length of sig, bkg without NaN: bk:", len(data.loc[data.target.values == 0]),", sig:", len(data.loc[data.target.values == 1])
#################################################################################
### Plot histograms of training variables
nbins=8
colorFast='g'
colorFastT='b'
colorFull='r'
hist_params = {'normed': True, 'histtype': 'bar', 'fill': False , 'lw':5}
#plt.figure(figsize=(60, 60))
if 'evtLevelSUM' in bdtType : 
	labelBKG = "tt+ttV"
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
elif 'evtLevelSUM_HH_res' in bdtType :
        labelBKG = "TT+DY+VV"
print "labelBKG: ",labelBKG
printmin=True
plotResiduals=False
plotAll=False
BDTvariables=trainVars(plotAll, options.variables, options.bdtType)
print "~~~>> BDTvariables: ",BDTvariables
print "\n\n\n trainVars(True): ",trainVars(True, options.variables, options.bdtType)
print "\n\n\n trainVars(False): ",trainVars(False, options.variables, options.bdtType)
make_plots(BDTvariables,nbins,
    data.ix[data.target.values == 0],labelBKG, colorFast,
    data.ix[data.target.values == 1],'Signal', colorFastT,
    channel+"/"+bdtType+"_"+trainvar+"_Variables_BDT.pdf",
    printmin,
    plotResiduals,
    output["masses_test"],
    output["masses"]       
    )

#########################################################################################
# Take fraction of sginal and/or background
'''
print("\n\nStart with dataOriginal read from ntuples")
data = dataOriginal
print 'nEvents for BDT in full       data: %i,  bk: %i,  signal: %i' % (len(data), len(data.loc[data.target.values == 0]), len(data.loc[data.target.values == 1]) )

dataBk = data.loc[data['target']==0]
dataSignal = data.loc[data['target']==1]
print 'nEvents for BDT in full       dataBk: %i,  bk: %i,  signal: %i' % (len(dataBk), len(dataBk.loc[dataBk.target.values == 0]), len(dataBk.loc[dataBk.target.values == 1]) )
print 'nEvents for BDT in ful    dataSignal: %i,  bk: %i,  signal: %i' % (len(dataSignal), len(dataSignal.loc[dataSignal.target.values == 0]), len(dataSignal.loc[dataSignal.target.values == 1]) )

useBkFraction     = 1
useSignalFraction = 1
print("Make dataset with background fraction %f and signal fraction %f" % (useBkFraction,useSignalFraction))
dataBk1,     dataBk2     = train_test_split(dataBk[trainVars(False, options.variables, options.bdtType)+["target","totalWeight","key"]], test_size=(1.0-useBkFraction), random_state=7)
dataSignal1, dataSignal2 = train_test_split(dataSignal[trainVars(False, options.variables, options.bdtType)+["target","totalWeight","key"]], test_size=(1.0-useSignalFraction), random_state=7)

print 'nEvents for BDT in full     data bk1: %i,  bk: %i,  signal: %i' % (len(dataBk1), len(dataBk1.loc[dataBk1.target.values == 0]), len(dataBk1.loc[dataBk1.target.values == 1]) )
print 'nEvents for BDT in full     data bk2: %i,  bk: %i,  signal: %i' % (len(dataBk2), len(dataBk2.loc[dataBk2.target.values == 0]), len(dataBk2.loc[dataBk2.target.values == 1]) )
print 'nEvents for BDT in full data signal1: %i,  bk: %i,  signal: %i' % (len(dataSignal1), len(dataSignal1.loc[dataSignal1.target.values == 0]), len(dataSignal1.loc[dataSignal1.target.values == 1]) )
print 'nEvents for BDT in full data signal2: %i,  bk: %i,  signal: %i' % (len(dataSignal2), len(dataSignal2.loc[dataSignal2.target.values == 0]), len(dataSignal2.loc[dataSignal2.target.values == 1]) )

data_new = pandas.concat([dataBk1, dataSignal1], axis=0)
data_original = data
data = data_new
print 'nEvents for BDT in          data_new: %i,  bk: %i,  signal: %i' % (len(data_new), len(data_new.loc[data_new.target.values == 0]), len(data_new.loc[data_new.target.values == 1]) )

print "\n\nEvent table for the data (cropped/chopped) :: \n %30s  %10s   %s " % ("Process", "nEvents", "nEventsWeighted")
for key in output["keys"]:
    nEvents = int(len(data.loc[(data['key']==key)]))
    nEventsWtg = float(data.loc[(data['key']==key), [weights]].sum())
    #print " \n\n",key," sum: ",len(data.loc[(data['key']==key)]),", \t ",data.loc[(data['key']==key), [weights]].sum()
    print " %30s  %10i  %10.3f" % (key,nEvents,nEventsWtg)

print("After normalizing and scaling::")        
print("nEvents for BDT: %i,  bk: %i,  signal: %i" % (len(data), len(data.loc[data['target']==0]), len(data.loc[data['target']==1])))
print("nEventsweight  : %f,  bk: %f,  signal: %f" % (data[weights].sum(), data.loc[data['target']==0][weights].sum(), data.loc[data['target']==1][weights].sum()))
print("Bk and signal total weights per mass point after bk-signal normalization")
for mass in range(len(output["masses"])) :
    print("mass: %i,  bk weight: %f,  signal weight: %f" % (mass, data.loc[(data[target]==0) & (data["gen_mHH"]== output["masses"][mass]),[weights]].sum(), data.loc[(data[target]==1) & (data["gen_mHH"]== output["masses"][mass]),[weights]].sum()))
'''    
     
'''
print("\n\n\ndata: {}".format(data))
print("\n\n\ndataBk: {}".format(dataBk))
print("\n\n\ndataSignal: {}".format(dataSignal))
print("\n\n\ndataBk1: {}".format(dataBk1))
print("\n\n\ndataBk2: {}".format(dataBk2))
print("\n\n\ndataSignal1: {}".format(dataSignal1))
print("\n\n\ndataSignal2: {}".format(dataSignal2))
print("\n\n\ndata_new: {}".format(data_new))
'''





 


#########################################################################################
# Split dataset for training and validation
'''traindataset, valdataset1  = train_test_split(data[trainVars(False)+["target","totalWeight"]], test_size=0.2, random_state=7)
valdataset = valdataset1.loc[valdataset1['gen_mHH']==400]
valdataset.loc[valdataset[target]==1,[weights]] *= valdataset1.loc[valdataset1[target]==1]["totalWeight"].sum()/valdataset.loc[valdataset[target]==1]["totalWeight"].sum()
valdataset.loc[valdataset[target]==0,[weights]]*= valdataset1.loc[valdataset1[target]==0]["totalWeight"].sum()/valdataset.loc[valdataset[target]==0]["totalWeight"].sum()'''

test_size = 0.2
print("\n\nTraining dataset: {},   Validation dataest: {}".format((1-test_size), test_size))
traindataset1, valdataset1  = train_test_split(data[trainVars(False, options.variables, options.bdtType)+["target","totalWeight","key"]], test_size=test_size, random_state=7)

#traindataset = traindataset1
#valdataset = valdataset1

traindataset = traindataset1.loc[((traindataset1[target]==1) & (traindataset1["gen_mHH"].isin(output["masses"]) )) | (traindataset1[target]==0)] # masses_test
valdataset = valdataset1.loc[((valdataset1[target]==1) & (valdataset1["gen_mHH"].isin(output["masses_test"]) )) | (valdataset1[target]==0)] 

#print("\n\ntraindataset: {}".format(traindataset))
#print("\n\nvaldataset: {}".format(valdataset))


print 'nEvents for BDT in full       data: %i,  bk: %i,  signal: %i' % (len(data), len(data.loc[data.target.values == 0]), len(data.loc[data.target.values == 1]) )
print 'nEvents for BDT in training   data: %i,  bk: %i,  signal: %i' % (len(traindataset), len(traindataset.loc[traindataset.target.values == 0]), len(traindataset.loc[traindataset.target.values == 1]) )
print 'nEvents for BDT in validation data: %i,  bk: %i,  signal: %i' % (len(valdataset), len(valdataset.loc[valdataset.target.values == 0]), len(valdataset.loc[valdataset.target.values == 1]) )

print 'Tot weight of train and validation for signal= ', traindataset.loc[traindataset[target]==1]["totalWeight"].sum(), valdataset.loc[valdataset[target]==1]["totalWeight"].sum()
print 'Tot weight of train and validation for bkg= ', traindataset.loc[traindataset[target]==0]['totalWeight'].sum(),valdataset.loc[valdataset[target]==0]['totalWeight'].sum()


#########################################################################################
# Normalize with 
NormalizeTrainTestData = True
if NormalizeTrainTestData:    
    print("\n\nCheck how train_test_split data background and signals are normalized::: *****")
    order_train1 = [traindataset, valdataset]
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

                
                print "\n\ndata %i %s \nEvent table after normalizing:: \n %30s  %10s   %s " % (idx,order_train1_name[idx],"Process", "nEvents", "nEventsWeighted")
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

                print("nEvents for BDT: %i,  bk: %i,  signal: %i" % (len(data), len(data.loc[data['target']==0]), len(data.loc[data['target']==1])))
                print("nEventsweight  : %f,  bk: %f,  signal: %f" % (data[weights].sum(), data.loc[data['target']==0][weights].sum(), data.loc[data['target']==1][weights].sum()))
            
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
                

        print "\n\ndata %i %s \nEvent table after normalizing and scaling up by 1e5:: \n %30s  %10s   %s " % (idx,order_train1_name[idx],"Process", "nEvents", "nEventsWeighted")
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

        print("After normalizing and scaling::")        
        print("nEvents for BDT: %i,  bk: %i,  signal: %i" % (len(data_do), len(data_do.loc[data_do['target']==0]), len(data_do.loc[data_do['target']==1])))
        print("nEventsweight  : %f,  bk: %f,  signal: %f" % (data_do[weights].sum(), data_do.loc[data_do['target']==0][weights].sum(), data_do.loc[data_do['target']==1][weights].sum()))
        print("Bk and signal total weights per mass point after bk-signal normalization")
        for mass in range(len(output["masses"])) :
            print("mass: %i,  bk weight: %f,  signal weight: %f" % (mass, data_do.loc[(data[target]==0) & (data_do["gen_mHH"]== output["masses"][mass]),[weights]].sum(), data_do.loc[(data[target]==1) & (data_do["gen_mHH"]== output["masses"][mass]),[weights]].sum()))
                        







## to GridSearchCV the test_size should not be smaller than 0.4 == it is used for cross validation!
## to final BDT fit test_size can go down to 0.1 without sign of overtraining
#############################################################################################
## Training parameters
if options.HypOpt==True :
	# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
	param_grid = {
    			'n_estimators': [200,500,800, 1000,2500],
    			'min_child_weight': [1,100],
    			'max_depth': [1,2,3,4],
    			'learning_rate': [0.01,0.02,0.03]
				}
	scoring = "roc_auc"
	early_stopping_rounds = 200 # Will train until validation_0-auc hasn't improved in 100 rounds.
	cv=3
	cls = xgb.XGBClassifier()
	fit_params = { "eval_set" : [(valdataset[trainVars(False)].values,valdataset[target])],
                           "eval_metric" : "auc",
                           "early_stopping_rounds" : early_stopping_rounds,
						   'sample_weight': valdataset[weights].values }
	gs = GridSearchCV(cls, param_grid, scoring, fit_params, cv = cv, verbose = 0)
	gs.fit(traindataset[trainVars(False)].values,
	traindataset.target.astype(np.bool)
	)
	for i, param in enumerate(gs.cv_results_["params"]):
		print("params : {} \n    cv auc = {}  +- {} ".format(param,gs.cv_results_["mean_test_score"][i],gs.cv_results_["std_test_score"][i]))
	print("best parameters",gs.best_params_)
	print("best score",gs.best_score_)
	#print("best iteration",gs.best_iteration_)
	#print("best ntree limit",gs.best_ntree_limit_)
	file = open("{}/{}_{}_{}_GSCV.log".format(channel,bdtType,trainvar,str(len(trainVars(False)))),"w")
	file.write(
				str(trainVars(False))+"\n"+
				"best parameters"+str(gs.best_params_) + "\n"+
				"best score"+str(gs.best_score_)+ "\n"
				#"best iteration"+str(gs.best_iteration_)+ "\n"+
				#"best ntree limit"+str(gs.best_ntree_limit_)
				)
	for i, param in enumerate(gs.cv_results_["params"]):
		file.write("params : {} \n    cv auc = {}  +- {} {}".format(param,gs.cv_results_["mean_test_score"][i],gs.cv_results_["std_test_score"][i]," \n"))
	file.close()

#############################################################################################        
cls = xgb.XGBClassifier(
			n_estimators = options.ntrees,
			max_depth = options.treeDeph,
			min_child_weight = options.mcw, # min_samples_leaf
			learning_rate = options.lr,
			#max_features = 'sqrt',
			#min_samples_leaf = 100
			#objective='binary:logistic', #booster='gbtree',
			#gamma=0, #min_child_weight=1,
			#max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, #random_state=0
			)
cls.fit(
	traindataset[trainVars(False, options.variables, options.bdtType)].values,
	traindataset.target.astype(np.bool),
	sample_weight=(traindataset[weights].astype(np.float64))
	# more diagnosis, in case
	#eval_set=[(traindataset[trainVars(False)].values,  traindataset.target.astype(np.bool),traindataset[weights].astype(np.float64)),
	#(valdataset[trainVars(False)].values,  valdataset.target.astype(np.bool), valdataset[weights].astype(np.float64))] ,
	#verbose=True,eval_metric="auc"
	)

#print trainVars(False)
print 'traindataset[trainVars(False)].columns.values.tolist() : ', traindataset[trainVars(False, options.variables, options.bdtType)].columns.values.tolist()

print ("XGBoost trained")
proba = cls.predict_proba(traindataset[trainVars(False, options.variables, options.bdtType)].values )
fpr, tpr, thresholds = roc_curve(traindataset[target], proba[:,1],
	sample_weight=(traindataset[weights].astype(np.float64)) )
train_auc = auc(fpr, tpr, reorder = True)
print("XGBoost train set auc - {}".format(train_auc))
proba = cls.predict_proba(valdataset[trainVars(False, options.variables, options.bdtType)].values )
fprt, tprt, thresholds = roc_curve(valdataset[target], proba[:,1], sample_weight=(valdataset[weights].astype(np.float64))  )
test_auct = auc(fprt, tprt, reorder = True)
print("XGBoost test set auc - {}".format(test_auct))
file_.write("XGBoost_train = %0.8f\n" %train_auc)
file_.write("XGBoost_test = %0.8f\n" %test_auct)
fig, ax = plt.subplots()
f_score_dict =cls.get_booster().get_fscore()

pklpath=channel+"/"+channel+"_XGB_"+trainvar+"_"+bdtType+"_"+str(len(trainVars(False, options.variables, options.bdtType)))+"Var"
print ("Done  ",pklpath,hyppar)
if options.doXML==True :
	print ("Date: ", time.asctime( time.localtime(time.time()) ))
	pickle.dump(cls, open(pklpath+".pkl", 'wb'))
	file = open(pklpath+"pkl.log","w")
	file.write(str(trainVars(False))+"\n")
	file.close()
	print ("saved ",pklpath+".pkl")
	print ("variables are: ",pklpath+"_pkl.log")
##################################################
## Draw ROC curve
fig, ax = plt.subplots(figsize=(8, 8))
train_auc = auc(fpr, tpr, reorder = True)
#ax.plot(fpr, tpr, lw=1, color='g',label='XGB train all mass excluding 350GeV(area = %0.5f)'%(train_auc))
#ax.plot(fprt, tprt, lw=1, ls='--',color='g',label='XGB test excluding 350GeV(area = %0.5f)'%(test_auct))
#ax.plot(fprtightF, tprtightF, lw=1, label='XGB test - Fullsim All (area = %0.3f)'%(test_auctightF))
ax.plot(fpr, tpr, lw=1, color='g',label='XGB train all mass (area = %0.5f)'%(train_auc))
ax.plot(fprt, tprt, lw=1, ls='--',color='g',label='XGB test all mass (area = %0.5f)'%(test_auct))


#ax.set_ylim([0.0,1.0])
#ax.set_xlim([0.0,1.0])
#ax.set_xlabel('False Positive Rate')
#ax.set_ylabel('True Positive Rate')
#ax.legend(loc="lower right")
#ax.grid()
#fig.savefig("{}/{}_{}_{}_{}_roc.png".format(channel,bdtType,trainvar,str(len(trainVars(False))),hyppar))
#fig.savefig("{}/{}_{}_{}_{}_roc.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False))),hyppar))

if variable == 'node' : variables=[20]
else :
    #variables = [350]
    variables = output["masses"]
colors = ['b', 'g', 'r']
'''for vv, var in enumerate(variables) :
	valdataset1= valdataset.loc[(valdataset[variable]==var) & (valdataset["target"]==0) & ((valdataset["key"] == "TTToHadronic_PSweights") | (valdataset["key"] == "TTToSemiLeptonic_PSweights") | (valdataset["key"] == "TTTo2L2Nu_PSweights"))]
	valdataset1=valdataset1.append(valdataset.loc[(valdataset[variable]==var) & (valdataset["target"]==1)])
	traindataset1= traindataset.loc[(traindataset[variable]==var) & (traindataset["target"]==0) & ((traindataset["key"] == "TTToHadronic_PSweights") | (traindataset["key"] == "TTToSemiLeptonic_PSweights") |(traindataset["key"] == "TTTo2L2Nu_PSweights"))]
	traindataset1=traindataset1.append(traindataset.loc[(traindataset[variable]==var) & (traindataset["target"]==1)])
	proba = cls.predict_proba(valdataset1[trainVars(False)].values )
	fprt, tprt, thresholds = roc_curve(valdataset1[target], proba[:,1], sample_weight=(valdataset1[weights].astype(np.float64))  )
	test_auct = auc(fprt, tprt, reorder = True)
	proba = cls.predict_proba(traindataset1[trainVars(False)].values )
	fpr, tpr, thresholds = roc_curve(traindataset1[target], proba[:,1],sample_weight=(traindataset1[weights].astype(np.float64)) )
	train_auc = auc(fpr, tpr, reorder = True)
	file_.write('traintt_auc= %0.8f\n' %train_auc)
	file_.write('testtt_auc= %0.8f\n' %test_auct)
	file_.write('xtrain_tt= ')
	file_.write(str(fpr.tolist()))
	file_.write('\n')
	file_.write('ytrain_tt= ')
	file_.write(str(tpr.tolist()))
	file_.write('\n')
	file_.write('xval_tt= ')
	file_.write(str(fprt.tolist()))
	file_.write('\n')
	file_.write('yval_tt = ')
	file_.write(str(tprt.tolist()))
	file_.write('\n')'''

'''for vv, var in enumerate(variables) :
	valdataset2= valdataset1.loc[(valdataset1[variable]==var)]
	traindataset2= traindataset1.loc[(traindataset1[variable]==var)]
        proba = cls.predict_proba(valdataset2[trainVars(False)].values )
        fprt, tprt, thresholds = roc_curve(valdataset2[target], proba[:,1], sample_weight=(valdataset2[weights].astype(np.float64))  )
        test_auct = auc(fprt, tprt, reorder = True)
        proba = cls.predict_proba(traindataset2[trainVars(False)].values )
        fpr, tpr, thresholds = roc_curve(traindataset2[target], proba[:,1],sample_weight=(traindataset2[weights].astype(np.float64)) )
        train_auc = auc(fpr, tpr, reorder = True)
	ax.plot(fpr, tpr, lw=1, color=colors[vv],label='XGB train 350GeV (area = %0.5f)'%(train_auc))
	ax.plot(fprt, tprt, lw=1, ls='--',color=colors[vv],label='XGB test 350GeV(area = %0.5f)'%(test_auct))         
	#ax.plot(fpr, tpr, lw=1, color=colors[vv],label='XGB train (area = %0.5f), 400GeV'%(train_auc))
	#ax.plot(fprt, tprt, lw=1, ls='--',color=colors[vv],label='XGB test (area = %0.5f), 400 GeV'%(test_auct))
	print("XGBoost train set auc 400 GeV- {}".format(train_auc))
	print("XGBoost train set auc 400 GeV- {}".format(test_auct))
	file_.write('train_auc= %0.8f\n' %train_auc)
	file_.write('test_auc= %0.8f\n' %test_auct)
	file_.write('xtrain= ')
	file_.write(str(fpr.tolist()))
	file_.write('\n')
	file_.write('ytrain= ')
	file_.write(str(tpr.tolist()))
	file_.write('\n')
	file_.write('xval= ')
	file_.write(str(fprt.tolist()))
	file_.write('\n')
	file_.write('yval = ')
	file_.write(str(tprt.tolist()))'''
ax.set_ylim([0.0,1.0])
ax.set_xlim([0.0,1.0])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc="lower right")
ax.grid()

fig.savefig("{}/{}_{}_{}_{}_roc.png".format(channel,bdtType,trainvar,str(len(trainVars(False, options.variables, options.bdtType))),hyppar)) 
fig.savefig("{}/{}_{}_{}_{}_roc.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False, options.variables, options.bdtType))),hyppar))


##################################################
## Draw ROC curve - by mass
styleline = ['-', '--', '-.', ':']
#colors_mass = ['m', 'b', 'k', 'r', 'g',  'y', 'c', ]
colors_mass = ['m', 'b', 'k', 'r', 'g',  'y', 'c',
                           'chocolate','teal', 'pink', 'darkkhaki', 'maroon', 'slategray',
                           'orange', 'silver', 'aquamarine', 'lavender', 'goldenrod', 'salmon',
                           'tan', 'lime', 'lightcoral'
            ]
fig, ax = plt.subplots(figsize=(6, 6))
for mm, mass in enumerate(output["masses_test"]) :
    proba = cls.predict_proba(
        traindataset.loc[(traindataset["gen_mHH"].astype(np.int) == int(mass)),
                        trainVars(False, options.variables, options.bdtType)].values )
    fpr, tpr, thresholds = roc_curve(
        traindataset.loc[(traindataset["gen_mHH"].astype(np.int) == int(mass)), target], proba[:,1],
	sample_weight=(traindataset.loc[(traindataset["gen_mHH"].astype(np.int) == int(mass)), weights].astype(np.float64)) )
    train_auc = auc(fpr, tpr, reorder = True)
    print("mass: {},  train_auc: {}".format(mass,train_auc))
    print("Train: proba: {}".format(proba))
    print("fpr: {}".format(fpr))
    print("tpr: {}".format(tpr))
    print("traindataset.loc[(traindataset[gen_mHH].astype(np.int) == int(mass)), trainVars(False, options.variables, options.bdtType)]: {}".format(traindataset.loc[(traindataset["gen_mHH"].astype(np.int) == int(mass)),
                        trainVars(False, options.variables, options.bdtType)]))

    proba = cls.predict_proba(
        valdataset.loc[(valdataset["gen_mHH"].astype(np.int) == int(mass)),
                       trainVars(False, options.variables, options.bdtType)].values )
    fprt, tprt, thresholds = roc_curve(
        valdataset.loc[(valdataset["gen_mHH"].astype(np.int) == int(mass)), target], proba[:,1],
        sample_weight=(valdataset.loc[(valdataset["gen_mHH"].astype(np.int) == int(mass)), weights].astype(np.float64))  )
    test_auct = auc(fprt, tprt, reorder = True)
    print("mass: {},  test_auc: {}".format(mass,test_auct))
    print("Train: proba: {}".format(proba))
    print("fprt: {}".format(fprt))
    print("tprt: {}".format(tprt))
    print("valdataset.loc[(valdataset[gen_mHH].astype(np.int) == int(mass)), trainVars(False, options.variables, options.bdtType)]: {}".format(valdataset.loc[(valdataset["gen_mHH"].astype(np.int) == int(mass)),
                       trainVars(False, options.variables, options.bdtType)]))
    
    ax.plot(
        fpr, tpr,
        lw = 2, linestyle = styleline[0], color = colors_mass[mm],
        label = 'train (area = %0.3f)'%(train_auc) + " (mass = " + str(mass) + ")"
    )
    ax.plot(
        fprt, tprt,
        lw = 2, linestyle = styleline[1], color = colors_mass[mm],
        label = 'test (area = %0.3f)'%(test_auct) + " (mass = " + str(mass) + ")"
    )    
ax.set_ylim([0.0,1.0])
ax.set_xlim([0.0,1.0])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc="lower right", fontsize = 'small')
ax.grid()
nameout = "{}/{}_{}_{}_{}_roc_by_mass.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False, options.variables, options.bdtType))),hyppar)
fig.savefig(nameout)
fig.savefig(nameout.replace(".pdf", ".png"))    



###########################################################################
## feature importance plot
fig, ax = plt.subplots()
f_score_dict =cls.get_booster().get_fscore()
print("f_score_dict: {}".format(f_score_dict))
f_score_dict = {trainVars(False, options.variables, options.bdtType)[int(k[1:])] : v for k,v in f_score_dict.items()}
feat_imp = pandas.Series(f_score_dict).sort_values(ascending=True)
feat_imp.plot(kind='barh', title='Feature Importances')
fig.tight_layout()
fig.savefig("{}/{}_{}_{}_{}_XGB_importance.png".format(channel,bdtType,trainvar,str(len(trainVars(False, options.variables, options.bdtType))),hyppar))
fig.savefig("{}/{}_{}_{}_{}_XGB_importance.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False, options.variables, options.bdtType))),hyppar))

###########################################################################
## BDT classifier - all
hist_params = {'normed': True, 'bins': 10 , 'histtype':'step'}
plt.clf()
y_pred = cls.predict_proba(valdataset.ix[valdataset.target.values == 0, trainVars(False, options.variables, options.bdtType)].values)[:, 1] #
y_predS = cls.predict_proba(valdataset.ix[valdataset.target.values == 1, trainVars(False, options.variables, options.bdtType)].values)[:, 1] #
'''for indx in range(0,len(valdataset)) :
	test = valdataset.take([indx])
	#print indx, '\t', 'test data : '
	#print test
	pre = cls.predict_proba(test[trainVars(False)].values )[:, 1]
	#print 'predict for test data : ',
	#print pre'''
plt.figure('XGB',figsize=(6, 6))
#values, bins, _ = plt.hist(y_pred , label="TT (XGB)", **hist_params)
values, bins, _ = plt.hist(y_pred , label=("%s (XGB)" % labelBKG), **hist_params)
values, bins, _ = plt.hist(y_predS , label="signal", **hist_params )
#plt.xscale('log')
#plt.yscale('log')
plt.legend(loc='best')
plt.savefig(channel+'/'+bdtType+'_'+trainvar+'_'+str(len(trainVars(False, options.variables, options.bdtType)))+'_'+hyppar+'_XGBclassifier.pdf')
plt.savefig(channel+'/'+bdtType+'_'+trainvar+'_'+str(len(trainVars(False, options.variables, options.bdtType)))+'_'+hyppar+'_XGBclassifier.png')

###########################################################################
## BDT classifier - by mass
hist_params = {'normed': True, 'bins': 8 , 'histtype':'step', "lw": 2}
for mm, mass in enumerate(output["masses_test"]) :    
    #y_pred = cls.predict_proba(valdataset.ix[valdataset.target.values == 0, trainVars(False, options.variables, options.bdtType)].values)[:, 1] #
    y_pred = cls.predict_proba(valdataset.ix[(valdataset.target.values == 0) & (valdataset["gen_mHH"].astype(np.int) == int(mass)),
        trainVars(False, options.variables, options.bdtType)].values)[:, 1] #
    y_predS = cls.predict_proba(valdataset.ix[(valdataset.target.values == 1) & (valdataset["gen_mHH"].astype(np.int) == int(mass)),
        trainVars(False, options.variables, options.bdtType)].values)[:, 1]

    y_pred_train = cls.predict_proba(traindataset.ix[(traindataset.target.values == 0) & (traindataset["gen_mHH"].astype(np.int) == int(mass)),
        trainVars(False, options.variables, options.bdtType)].values)[:, 1] #
    y_predS_train = cls.predict_proba(traindataset.ix[(traindataset.target.values == 1) & (traindataset["gen_mHH"].astype(np.int) == int(mass)),
        trainVars(False, options.variables, options.bdtType)].values)[:, 1]
     
    colorcold = ['g', 'b']
    colorhot = ['r', 'magenta']
    plt.clf()
    fig, ax = plt.subplots(figsize=(6, 6))
    dict_plot = [
        [y_pred, "-", colorcold[0], ("test %s" % labelBKG)],
        [y_predS, "-", colorhot[0], "test signal"],
        [y_pred_train, "--", colorcold[1], ("train %s" % labelBKG)],
        [y_predS_train, "--", colorhot[1], "train signal"]
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
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.), bbox_transform=fig.transFigure, title="mass = "+str(mass)+" GeV", fontsize = 'small')
    ax.set_ylim(0, yMax*1.2) # set y-axis range of XGBclassifier
    nameout = channel+'/'+bdtType+'_'+trainvar+'_'+str(len(trainVars(False, options.variables, options.bdtType)))+'_'+hyppar+'_mass_'+str(mass)+'_XGBclassifier.pdf'
    fig.savefig(nameout)
    fig.savefig(nameout.replace(".pdf", ".png"))

    

    

###########################################################################
# plot correlation matrix
if options.HypOpt==False :
	for ii in [1,2] :
		if ii == 1 :
			datad=traindataset.loc[traindataset[target].values == 1]
			label="signal"
		else :
			datad=traindataset.loc[traindataset[target].values == 0]
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
		#plt.subplots_adjust(left=0.9, right=0.9, top=0.9, bottom=0.1)
		plt.savefig("{}/{}_{}_{}_corr_{}.png".format(channel,bdtType,trainvar,str(len(trainVars(False, options.variables, options.bdtType))),label))
		plt.savefig("{}/{}_{}_{}_corr_{}.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False, options.variables, options.bdtType))),label))
		ax.clear()
process = psutil.Process(os.getpid())
print(process.memory_info().rss)
print(datetime.now() - startTime)
