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
from sklearn.metrics import roc_curve, auc, accuracy_score

import ROOT
ROOT.gROOT.SetBatch(True)

import glob
from collections import OrderedDict

from ROOT import TCanvas, TFile, TProfile, TNtuple, TH1F, TH2F, TH1D, TH2D, THStack, TF1
from ROOT import gBenchmark, gRandom, gSystem, Double, gPad, TFitResultPtr, TMath

import copy


##--- Example as to how to run the code for 2l_2tau channel ----- ###
##python sklearn_Xgboost_evtLevel_HH_parametric_twofold_2l_2tau_diagnostics.py --channel "2l_2tau" --bdtType "evtLevelSUM_HH_2l_2tau_res" --variables $VAR --tauID dR03mvaVLoose --Bkg_mass_rand oversampling --TrainMode 0 --BDT_Diagnostics_2l_2tau True  --ReweightVars True --doXML True  


def lr_to_str(lr):
    if(lr >= 1.0):
        lr_label = "_lr_"+str(int(lr*1))
    elif((lr >= 0.1) & (lr < 1.0)):
        lr_label = "_lr_0o"+str(int(lr*10))
    elif((lr >= 0.01) & (lr < 0.1)):
        lr_label = "_lr_0o0"+str(int(lr*100))
    elif((lr >= 0.001) & (lr < 0.01)):
        lr_label = "_lr_0o00"+str(int(lr*1000))
    else:
        lr_label = "_indeter"
    return lr_label


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
parser.add_option("--ntrees", type="int", dest="ntrees", help="hyp", default=1000)
parser.add_option("--treeDeph", type="int", dest="treeDeph", help="hyp", default=2)
parser.add_option("--lr", type="float", dest="lr", help="hyp", default=0.01)
parser.add_option("--mcw", type="int", dest="mcw", help="hyp", default=1)
parser.add_option("--tauID", type="string", dest="tauID", help="sample to pick training events", default='"dR03mvaVLoose"')
parser.add_option("--Bkg_mass_rand", type="string", dest="Bkg_mass_rand", help="fix gen_mHH randomiz. method for bkg.s", default='"default"')
parser.add_option("--ClassErr_vs_epoch", action="store_true", dest="ClassErr_vs_epoch", help="Makes scans of classification error vs epoch for a fixed grid of hyperparam.s", default=False)
parser.add_option("--SplitMode", type="int", dest="SplitMode", help="Selects Mode for splitting data while scanning hyperparam.s (Use only when --ClassErr_vs_epoch = True) \n                                                              SplitMode==0 -- dataframe will be randomly split into two equal halves using test_train_split() function \n                                                                                                                SplitMode==1 -- dataframe will be split into halves having odd and even numbered events (rows) respectively with odd (even) half being used for testing (training) \n                                                      SplitMode==2 -- dataframe will be split into halves having odd and even numbered events (rows) respectively with even (odd) half being used for testing (training) \n", default=-1)
parser.add_option("--ScanMode", type="int", dest="ScanMode", help="Selects Mode for scanning hyperparam.s (Use only when --ClassErr_vs_epoch = True and --SplitMode = 0/1/2 as specified above) \n                                           ScanMode==1 -- Hyper-Param.s will be scanned for fixed value of lr*ntrees (fixed to 10 in the code right now) \n                                                                                                           ScanMode==2 -- Hyper-Param.s will be scanned for ntrees for fixed lr and depth (See values in the code) \n", default=-1)
parser.add_option("--BDT_Diagnostics_2l_2tau", action="store_true", dest="BDT_Diagnostics_2l_2tau", help="Makes useful plots for Input var.s for 2l_2tau channel", default=False)
parser.add_option("--ReweightVars", action="store_true", dest="ReweightVars", help="Reweight Input variables with Fits", default=False)
parser.add_option("--TrainMode", type="int", dest="TrainMode", help="Select the input mass range for the training \n 0: to include all masses \n 1: to include only low masses (<= 400 GeV) \n 2: to include only high masses (> 400 GeV)", default=0)
(options, args) = parser.parse_args()

TrainMode=options.TrainMode
do_2l_2tau_diagnostics=options.BDT_Diagnostics_2l_2tau
do_ReweightVars=options.ReweightVars
tauID=options.tauID
Bkg_mass_rand=options.Bkg_mass_rand
doPlots=options.doPlots
bdtType=options.bdtType
trainvar=options.variables
#hyppar=str(options.variables)+"_ntrees_"+str(options.ntrees)+"_deph_"+str(options.treeDeph)+"_mcw_"+str(options.mcw)+"_lr_0o0"+str(int(options.lr*100))
hyppar=str(options.variables)+"_ntrees_"+str(options.ntrees)+"_deph_"+str(options.treeDeph)+"_mcw_"+str(options.mcw)+lr_to_str(options.lr)


print("do_ReweightVars", do_ReweightVars)
print("do_2l_2tau_diagnostics", do_2l_2tau_diagnostics)



if(TrainMode == 0):
    channel=options.channel+"_HH_"+tauID+"_"+options.Bkg_mass_rand+"_"+options.variables+"_Train_all_Masses4"
elif(TrainMode == 1):
    channel=options.channel+"_HH_"+tauID+"_"+options.Bkg_mass_rand+"_"+options.variables+"_Train_Low_masses_only4"
elif(TrainMode == 2):
    channel=options.channel+"_HH_"+tauID+"_"+options.Bkg_mass_rand+"_"+options.variables+"_Train_High_masses_only4"
else:
    channel=options.channel+"_HH_"+tauID+"_"+options.Bkg_mass_rand+"_"+options.variables+"_Train_all_Masses4"

print("TrainMode", TrainMode)

print("do_ReweightVars:")
print(do_ReweightVars)

if(do_2l_2tau_diagnostics): channel += "_2l_2tau_diagnostics"

if(do_ReweightVars): 
    channel += "_with_reweighting"
else:
    channel += "_wo_reweighting"

print (startTime)

if "bb2l" in channel   : execfile("../cards/info_bb2l_HH.py")
if "2l_2tau" in channel: execfile("../cards/info_2l_2tau_HH.py")


if(options.ClassErr_vs_epoch == True):
    channel+"_SplitMode_"+str(options.SplitMode)+"_ScanMode_"+str(options.ScanMode)
    log_file_name=channel+".log"
    if 'evtLevelSUM_HH_2l_2tau_res' in bdtType :
        file1_ = open(log_file_name, 'w+')
    else : file1_ = open('roc.log','w+')


import shutil,subprocess
proc=subprocess.Popen(['mkdir '+channel],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()


weights="totalWeight"
target='target'

output = read_from(Bkg_mass_rand, tauID)

execfile("../cards/auxiliary_functions_2l_2tau_HH.py") ## Loading all the functions needed for fitting

##--- Choice of train mass ranges implemented
if( TrainMode == 0 ): ## All Masses included in training
    print("Training on all masses: 250-1000 GeV")
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
    mass_list = output["masses"]
    test_masses = output["masses_test"]
elif( TrainMode == 1 ): ## Only Low Masses (<= 400 GeV) included in training
    print("Training on Low masses only: 250-400 GeV")
    data=load_data_2017(
        output["inputPath"],
        output["channelInTree"],
        trainVars(True),
        [],
        bdtType,
        channel,
        output["keys"],
        output["masses_low"],
        output["mass_randomization"],
        )
    mass_list = output["masses_low"]
    test_masses = output["masses_test_low"]
elif( TrainMode == 2 ): ## Only High Masses (> 400 GeV) included in training
    print("Training on High masses only: 450-1000 GeV")
    data=load_data_2017(
        output["inputPath"],
        output["channelInTree"],
        trainVars(True),
        [],
        bdtType,
        channel,
        output["keys"],
        output["masses_high"],
        output["mass_randomization"],
        )
    mass_list = output["masses_high"]
    test_masses = output["masses_test_high"]
else:
    print("TrainMode neither 0/1/2 switching back to Training on all masses: 250-1000 GeV")
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
    mass_list = output["masses"]
    test_masses = output["masses_test"]
    
data.dropna(subset=[weights],inplace = True)
data.fillna(0)

#print("data", data)


### Plot histograms of training variables
hist_params = {'normed': True, 'histtype': 'bar', 'fill': False , 'lw':5}
if 'evtLevelSUM' in bdtType :
	labelBKG = "SUM BKG"
	if channel in ["3l_0tau_HH"]:
		labelBKG = "WZ+tt+ttV"
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
#print("BDTvariables", BDTvariables)

## Removing gen_mHH from the list of input variables
BDTvariables_wo_gen_mHH = copy.deepcopy(BDTvariables)  ## Works
BDTvariables_wo_gen_mHH.remove("gen_mHH")
#print("BDTvariables_wo_gen_mHH", BDTvariables_wo_gen_mHH)


make_plots(BDTvariables, nbins,
    data.ix[data.target.values == 0],labelBKG, colorFast,
    data.ix[data.target.values == 1],'Signal', colorFastT,
    channel+"/"+bdtType+"_"+trainvar+"_Variables_BDT.pdf",
    printmin,
    plotResiduals,
    test_masses,
    mass_list
    )


if(do_2l_2tau_diagnostics == True):
    if(do_ReweightVars):
        DoFits = True
        print DoFits
        print("Perfoming Fits to TProfile plots for signal")
    else:
        DoFits = False
        print DoFits
        print("Not Perfoming Fits to TProfile plots for signal")


    ## --- Making TProfile plots with fits (Signal) --- ###
    MakeTProfile_New(channel, data, BDTvariables_wo_gen_mHH, 1, DoFits, "before", TrainMode, mass_list)
    
    ## --- Making TProfile plots (background) --- ###
    MakeTProfile_New(channel, data, BDTvariables_wo_gen_mHH, 0, False, "before", TrainMode, mass_list)

    ## --- Making 1D Histo plots (background) --- ###
    MakeHisto1D_New(channel, data, BDTvariables, "before")

    ## --- Making 1D THStack plots (background) --- ###
    MakeTHStack_New(channel, data, BDTvariables, "before")

    if(do_ReweightVars): 
        ## ----- SCALING I/P VAR.S IN DATA USING THE FITS DONE ABOVE---- ###
        ReweightDataframe_New(data, channel, BDTvariables_wo_gen_mHH, mass_list)
else:
    print("No plots and fits will be made for 2l_2tau diagnostics")


print ("separate datasets odd/even")
data_even = data.loc[(data["event"].values % 2 == 0) ]
data_odd = data.loc[~(data["event"].values % 2 == 0) ]

#data_even_test_800 = data.loc[((data['gen_mHH'] == 800) & (data['event'] == 101964) & (data['target'] == 1))]
#print('data_even_test_800', data_even_test_800)

#print("diHiggsMass: {}, diHiggsVisMass: {}, tau1_pt: {}, nBJet_medium: {}, nElectron: {}, dr_lep_tau_min_SS: {}, met_LD: {}, tau2_pt: {}, dr_lep_tau_min_OS: {}, gen_mHH: {}".format(data_even_test_800["diHiggsMass"], data_even_test_800["diHiggsVisMass"], data_even_test_800["tau1_pt"], data_even_test_800["nBJet_medium"], data_even_test_800["nElectron"], data_even_test_800["dr_lep_tau_min_SS"], data_even_test_800["met_LD"], data_even_test_800["tau2_pt"], data_even_test_800["dr_lep_tau_min_OS"], data_even_test_800["gen_mHH"]) )


order_train = [data_odd, data_even]
order_train_name = ["odd","even"]

print ("balance datasets by even/odd chunck")
for data_do in order_train : ## the sample lists are now prepared from the info_2l_2tau_HH.py file (Please modify the info file for bb2l accordingly)
    if 'SUM_HH' in bdtType :
        ttbar_samples = ['TTToSemiLeptonic', 'TTTo2L2Nu'] ## Removed TTToHadronic since zero events selected for this sample
        data_do.loc[(data_do['key'].isin(ttbar_samples)), [weights]]              *= output["TTdatacard"]/data_do.loc[(data_do['key'].isin(ttbar_samples)), [weights]].sum()
        data_do.loc[(data_do['key']=='DY'), [weights]]                            *= output["DYdatacard"]/data_do.loc[(data_do['key']=='DY'), [weights]].sum()
        if "evtLevelSUM_HH_bb1l_res" in bdtType :
            data_do.loc[(data_do['key']=='W'), [weights]]                         *= Wdatacard/data_do.loc[(data_do['key']=='W')].sum() ## Saswati check please !!!
        if "evtLevelSUM_HH_2l_2tau_res" in bdtType :
            data_do.loc[(data_do['key']=='TTZJets'), [weights]]                       *= output["TTZdatacard"]/data_do.loc[(data_do['key']=='TTZJets'), [weights]].sum() ## TTZJets
            data_do.loc[(data_do['key']=='TTWJets'), [weights]]                       *= output["TTWdatacard"]/data_do.loc[(data_do['key']=='TTWJets'), [weights]].sum() ## TTWJets + TTWW
            data_do.loc[(data_do['key']=='ZZ'), [weights]]                            *= output["ZZdatacard"]/data_do.loc[(data_do['key']=='ZZ'), [weights]].sum() ## ZZ +ZZZ
            data_do.loc[(data_do['key']=='WZ'), [weights]]                            *= output["WZdatacard"]/data_do.loc[(data_do['key']=='WZ'), [weights]].sum() ## WZ + WZZ_4F
            data_do.loc[(data_do['key']=='WW'), [weights]]                            *= output["WWdatacard"]/data_do.loc[(data_do['key']=='WW'), [weights]].sum() ## WW + WWZ + WWW_4F
            #data_do.loc[(data_do['key']=='VH'), [weights]]                        *= output["VHdatacard"]/data_do.loc[(data_do['key']=='VH'), [weights]].sum() # consider removing
            #data_do.loc[(data_do['key']=='TTH'), [weights]]                       *= output["TTHdatacard"]/data_do.loc[(data_do['key']=='TTH'), [weights]].sum() # consider removing
            #print('TTH keys modified', data_do.loc[(data_do['key'].isin(tth_samples)), ['key']] )
        for mass in range(len(output["masses"])) :
            data_do.loc[(data_do[target]==1) & (data_do["gen_mHH"].astype(np.int) == int(output["masses"][mass])),[weights]] *= 100000./data_do.loc[(data_do[target]==1) & (data_do["gen_mHH"]== output["masses"][mass]),[weights]].sum()
            data_do.loc[(data_do[target]==0) & (data_do["gen_mHH"].astype(np.int) == int(output["masses"][mass])),[weights]] *= 100000./data_do.loc[(data_do[target]==0) & (data_do["gen_mHH"]== output["masses"][mass]),[weights]].sum()
    else :
        data_do.loc[data_do['target']==0, [weights]] *= 100000/data_do.loc[data_do['target']==0][weights].sum()
        data_do.loc[data_do['target']==1, [weights]] *= 100000/data_do.loc[data_do['target']==1][weights].sum()



if(do_2l_2tau_diagnostics == True):
    ## ----Merging the odd and even data-sets ---------------------------------------####
    ## ---(reweighting the merged dataframe by 0.5 as it is derived from 2 halves) ---###
    data_odd_copy = data_odd.copy(deep=True) ## Making sure we do not alter the halves used for roc curves later
    data_even_copy = data_even.copy(deep=True) ## Making sure we do not alter the halves used for roc curves later
    data_do = data_odd_copy.append(data_even_copy, ignore_index=True) 
    data_do.loc[data_do['target']==0, [weights]] *= 0.5
    data_do.loc[data_do['target']==1, [weights]] *= 0.5
    label = "after"


    ## --- Making TProfile plots w/o fitting (Signal) --- ###
    MakeTProfile_New(channel, data, BDTvariables_wo_gen_mHH, 1, False, label, TrainMode, mass_list)

    ## --- Making TProfile plots (background) --- ###
    MakeTProfile_New(channel, data, BDTvariables_wo_gen_mHH, 0, False, label, TrainMode, mass_list)

    ## --- Making 1D Histo plots (background) --- ###
    MakeHisto1D_New(channel, data,  BDTvariables, label)

    ## --- Making 1D THStack plots (background) --- ###
    MakeTHStack_New(channel, data, BDTvariables, label)
    
    make_plots(BDTvariables, nbins,
               data.ix[data_do.target.values == 0],labelBKG, colorFast,
               data.ix[data_do.target.values == 1],'Signal', colorFastT,
               channel+"/"+bdtType+"_"+trainvar+"_Variables_BDT_after.pdf",
               printmin,
               plotResiduals,
               test_masses,
               mass_list
               )
               
else:
    print("No plots will be made for 2l_2tau diagnostics")



roc_test = []
roc_train = []
estimator = []
for dd, data_do in  enumerate(order_train):
    cls = xgb.XGBClassifier(
    			n_estimators = options.ntrees,
    			max_depth = options.treeDeph,
    			min_child_weight = options.mcw,
    			learning_rate = options.lr,
    			nthread = 8
    			)

    cls.fit(
    	data_do[trainVars(False, options.variables, options.bdtType)].values,
    	data_do.target.astype(np.bool),
    	sample_weight=(data_do[weights].astype(np.float64))
    	)
    if dd == 0 : val_data = 1
    else : val_data = 0

    print ("XGBoost trained", order_train_name[dd])

    #if(dd == 0): ## Odd train Even test 
    #    proba_even_test_800 = cls.predict_proba(data_even_test_800[trainVars(False, options.variables, options.bdtType)].values)
    #    print('proba_even_test_800[:,1]', proba_even_test_800[:,1])

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
    print("XGBoost train set auc - {}".format(train_auc))

    proba = cls.predict_proba(order_train[val_data][trainVars(False, options.variables, options.bdtType)].values )
    fprt, tprt, thresholds = roc_curve(
        order_train[val_data][target].astype(np.bool), proba[:,1],
        sample_weight=(order_train[val_data][weights].astype(np.float64))
    )

    test_auct = auc(fprt, tprt, reorder = True)
    roc_test = roc_test + [ { "fprt":fprt, "tprt":tprt, "test_auct":test_auct }]
    print("XGBoost test set auc - {}".format(test_auct))
    estimator = estimator + [cls]

    ## feature importance plot
    fig, ax = plt.subplots()
    f_score_dict =cls.get_booster().get_fscore()
    fig, ax = plt.subplots()
    f_score_dict =cls.get_booster().get_fscore()
    f_score_dict = {trainVars(False, options.variables, options.bdtType)[int(k[1:])] : v for k,v in f_score_dict.items()}
    feat_imp = pandas.Series(f_score_dict).sort_values(ascending=True)
    feat_imp.plot(kind='barh', title='Feature Importances')
    fig.tight_layout()
    nameout = "{}/{}_{}_{}_{}_{}_{}_XGB_importance.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False, options.variables, options.bdtType))),hyppar, order_train_name[dd], (options.tauID))
    fig.savefig(nameout)
    #fig.savefig(nameout.replace(".pdf", ".png"))

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
#fig.savefig(nameout.replace(".pdf", ".png"))

###############################
# by mass ROC
styleline = ['-', '--', '-.', ':']
colors_mass = ['m', 'b', 'k', 'r', 'g',  'y', 'c', ]
fig, ax = plt.subplots(figsize=(6, 6))
sl = 0


#for mm, mass in enumerate(output["masses_test"]) :
for mm, mass in enumerate(test_masses) :
    for dd, data_do in  enumerate(order_train) :
        if dd == 0 : val_data = 1
        else : val_data = 0
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

###############################
## classifier plot by mass
hist_params = {'normed': True, 'bins': 8 , 'histtype':'step', "lw": 2}
#for mm, mass in enumerate(output["masses_test"]) :
for mm, mass in enumerate(test_masses) :
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
        #plt.xscale('log')
        #plt.yscale('log')
    ax.legend(loc='upper center', title="mass = "+str(mass)+" GeV", fontsize = 'small')
    nameout = channel+'/'+bdtType+'_'+trainvar+'_'+str(len(trainVars(False, options.variables, options.bdtType)))+'_'+hyppar+'_mass_'+ str(mass)+'_'+(options.tauID)+'_XGBclassifier.pdf'
    fig.savefig(nameout)
    #fig.savefig(nameout.replace(".pdf", ".png"))

###############################
## classifier plot by mass
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
    #plt.xscale('log')
    #plt.yscale('log')
ax.legend(loc='upper center', title="all masses", fontsize = 'small')
nameout = channel+'/'+bdtType+'_'+trainvar+'_'+str(len(trainVars(False, options.variables, options.bdtType)))+'_'+hyppar+'_AllMass_'+(options.tauID)+'_XGBclassifier.pdf'
fig.savefig(nameout)
#fig.savefig(nameout.replace(".pdf", ".png"))




#### ----- Interpolation check log files ----####
#for mm, mass in enumerate(output["masses_test"]): ## Loop over the test masses
for mm, mass in enumerate(test_masses): ## Loop over the test masses
    print("mass", mass)  
    MASS = int(mass)
    print("MASS", MASS)
    MASS_STR = "{}".format(MASS)
    print("MASS_STR", MASS_STR)

    ## For writing the interpolation check roc curves ##
    log_file2_name = "{}/{}_{}.log".format(channel,channel,MASS_STR)
    file2_ = open(log_file2_name, 'w+')

    for dd, data_do in  enumerate(order_train):   ## Loop over the odd and even dataset halves
        if dd == 0 : val_data = 1
        else : val_data = 0
        
        if(Bkg_mass_rand == "default"): 
            print("Using the new logic") 
            traindataset1 = data_do.loc[~((data_do["gen_mHH"]==mass) & (data_do[target]==1))]  ## Training for all masses except the one under study only for signal process
            #valdataset1   = order_train[val_data].loc[~((order_train[val_data]["gen_mHH"]==mass) & (order_train[val_data][target]==1))]  ## Testing for all masses except the one under study for signal process
            valdataset1 = order_train[val_data].loc[~((order_train[val_data]["gen_mHH"]!=mass) & (order_train[val_data][target]==1))] ## Testing on only the mass under study for signal
        else:
            print("Using the old logic") 
            traindataset1 = data_do.loc[~(data_do["gen_mHH"]==mass)]  ## Training for all masses except the one under study
            #valdataset1 = order_train[val_data].loc[~(order_train[val_data]["gen_mHH"]==mass)]  ## Testing for all masses except the one under study
            valdataset1 = order_train[val_data].loc[(order_train[val_data]["gen_mHH"]==mass)] ## Testing on only the mass under study

        print("traindataset1", traindataset1)
        print("valdataset1", valdataset1)

        if(mass == 300):
            if(TrainMode == 0):
                masses1 = [250,260,270,280,350,400,450,500,550,600,650,700,750,800,850,900,1000] ## All masses except 300
                masses_test1 = [250,1000]
            elif(TrainMode == 1):
                masses1 = [250,260,270,280,350,400] ## Low mass only training (except 300)
                masses_test1 = [250]
            masses2 = [300]
            masses_test2 = [300]
        elif(mass == 500):    
            if(TrainMode == 0):
                masses1 = [250,260,270,280,300,350,400,450,550,600,650,700,750,800,850,900,1000] ## All masses except 500
                masses_test1 = [250,1000]
            elif(TrainMode == 2):
                masses1 = [450,550,600,650,700,750,800,850,900,1000] ## High mass only training (except 500)
                masses_test1 = [1000]
            masses2 = [500]            
            masses_test2 = [500]
        elif(mass == 800):    
            if(TrainMode == 0):
                masses1 = [250,260,270,280,300,350,400,450,500,550,600,650,700,750,850,900,1000] ## All masses except 800
                masses_test1 = [250,1000]
            elif(TrainMode == 2):
                masses1 = [450,500,550,600,650,700,750,850,900,1000] ## High mass only training (except 800)
                masses_test1 = [1000]
            masses2 = [800]
            masses_test2 = [800]

        plot_name_train1 = "_Variables_traindataset1_BDT_"+MASS_STR+".pdf"
        plot_name_val1 = "_Variables_valdataset1_BDT_"+MASS_STR+".pdf"

        make_plots(BDTvariables, nbins,
                   traindataset1.ix[traindataset1.target.values == 0],labelBKG, colorFast,
                   traindataset1.ix[traindataset1.target.values == 1],'Signal', colorFastT,
                   channel+"/"+bdtType+"_"+trainvar+plot_name_train1,
                   printmin,
                   plotResiduals,
                   masses_test1,
                   masses1
                   )

        make_plots(BDTvariables, nbins,
                   valdataset1.ix[valdataset1.target.values == 0],labelBKG, colorFast,
                   valdataset1.ix[valdataset1.target.values == 1],'Signal', colorFastT,
                   channel+"/"+bdtType+"_"+trainvar+plot_name_val1,
                   printmin,
                   plotResiduals,
                   masses_test2,
                   [250.,260.,270.,280.,300.,350.,400.,450.,500.,550.,600.,650.,700.,750.,800.,850.,900.,1000.]
                   )

        ################# WITH gen_mHH #####################
        cls2 = xgb.XGBClassifier(
            n_estimators = options.ntrees,
            max_depth = options.treeDeph,
            min_child_weight = options.mcw,
            learning_rate = options.lr,
            )

        cls2.fit(
            traindataset1[trainVars(False, options.variables, options.bdtType)].values,
            traindataset1.target.astype(np.bool),
            sample_weight=(traindataset1[weights].astype(np.float64))
            )

        ## --- ROC curve ----##                                                                                                                                                                       
        proba = cls2.predict_proba(valdataset1[trainVars(False, options.variables, options.bdtType)].values )
        #proba = cls2.predict_proba(valdataset1[BDTvariables_wo_gen_mHH].values )
        fprt, tprt, thresholds = roc_curve(valdataset1[target].astype(np.bool), proba[:,1], sample_weight=(valdataset1[weights].astype(np.float64))  )                                                                  
        test_auct = auc(fprt, tprt, reorder = True)                                                                                                                                                  
 
        proba = cls2.predict_proba(traindataset1[trainVars(False, options.variables, options.bdtType)].values )                                                                               
        fpr, tpr, thresholds = roc_curve(traindataset1[target].astype(np.bool), proba[:,1], sample_weight=(traindataset1[weights].astype(np.float64)) )                                                                  
        train_auc = auc(fpr, tpr, reorder = True)                                                                                                                                                     

        ## ---- BDT Output Distributions ----- ###
        y_pred = cls2.predict_proba(valdataset1.ix[valdataset1.target.values == 0, trainVars(False, options.variables, options.bdtType)].values)[:, 1]                                        
        y_predS = cls2.predict_proba(valdataset1.ix[valdataset1.target.values == 1, trainVars(False, options.variables, options.bdtType)].values)[:, 1]                                                 
        #y_pred = cls2.predict_proba(valdataset1.ix[valdataset1.target.values == 0, BDTvariables_wo_gen_mHH].values)[:, 1]                                        
        #y_predS = cls2.predict_proba(valdataset1.ix[valdataset1.target.values == 1, BDTvariables_wo_gen_mHH].values)[:, 1]                                                 

        y_pred_train = cls2.predict_proba(traindataset1.ix[traindataset1.target.values == 0, trainVars(False, options.variables, options.bdtType)].values)[:, 1]                            
        y_predS_train = cls2.predict_proba(traindataset1.ix[traindataset1.target.values == 1, trainVars(False, options.variables, options.bdtType)].values)[:, 1]      

        file2_.write('#------ Training for all masses except %0.1f GeV-------\n' %mass)                                                                                                                
        file2_.write('#------ Testing only for mass %0.1f GeV (using other dataset half)--------\n' %mass)                                                                                    
        if(dd == 0): ## Train->odd; Test/val -> even                                                                                                                                             
            file2_.write('train_auc_ODD0_%s = %0.8f\n' % (MASS_STR, train_auc))       
            file2_.write('test_auc_EVEN0_%s  = %0.8f\n' % (MASS_STR, test_auct))
            file2_.write('xtrain_ODD0_%s = ' % (MASS_STR))                                                                                                                                               
            file2_.write(str(fpr.tolist()))
            file2_.write('\n')
            file2_.write('ytrain_ODD0_%s = ' % (MASS_STR))
            file2_.write(str(tpr.tolist()))                                                                                                                                                            
            file2_.write('\n')
            file2_.write('xval_EVEN0_%s = ' % (MASS_STR))
            file2_.write(str(fprt.tolist()))                                                                                                                                                           
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('yval_EVEN0_%s = ' % (MASS_STR))                                                                                                                                               
            file2_.write(str(tprt.tolist()))                                                                                                                                                           
            file2_.write('\n')
            file2_.write('y_pred_test_EVEN0_%s = ' % (MASS_STR))
            file2_.write(str(y_pred.tolist()))                                                                                                                                                         
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('y_predS_test_EVEN0_%s = ' % (MASS_STR))                                                                                                                                          
            file2_.write(str(y_predS.tolist()))                                                                                                                                                        
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('y_pred_train_ODD0_%s = ' % (MASS_STR))                                                                                                                                         
            file2_.write(str(y_pred_train.tolist()))                                                                                                                                                   
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('y_predS_train_ODD0_%s = ' % (MASS_STR))                                                                                                                                         
            file2_.write(str(y_predS_train.tolist()))                                                                                                                                                  
            file2_.write('\n')
        else: ## Train->even; Test/val -> odd                                                                                                                                                  
            file2_.write('train_auc_EVEN0_%s = %0.8f\n' %  (MASS_STR, train_auc))                                                                                                                      
            file2_.write('test_auc_ODD0_%s  = %0.8f\n' %  (MASS_STR, test_auct)) 
            file2_.write('xtrain_EVEN0_%s = ' % (MASS_STR)) 
            file2_.write(str(fpr.tolist()))                                                                                                                                                            
            file2_.write('\n')
            file2_.write('ytrain_EVEN0_%s = ' % (MASS_STR))
            file2_.write(str(tpr.tolist()))                                                                                                                                                            
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('xval_ODD0_%s = ' % (MASS_STR))                                                                                                                             
            file2_.write(str(fprt.tolist()))                                                                                                                                                           
            file2_.write('\n')                         
            file2_.write('yval_ODD0_%s = ' % (MASS_STR))                                                                                                                                                
            file2_.write(str(tprt.tolist()))                                                                                                                                                           
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('y_pred_test_ODD0_%s = ' % (MASS_STR))                                                                                                                                        
            file2_.write(str(y_pred.tolist()))                                                                                                                                                         
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('y_predS_test_ODD0_%s = ' % (MASS_STR))                                                                                                                                      
            file2_.write(str(y_predS.tolist()))                                                                                                                                                        
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('y_pred_train_EVEN0_%s = ' % (MASS_STR))                                                                                                                                    
            file2_.write(str(y_pred_train.tolist()))                                                                                                                                                   
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('y_predS_train_EVEN0_%s = ' % (MASS_STR))                                                                                                                                     
            file2_.write(str(y_predS_train.tolist()))                                                                                                                                                  
            file2_.write('\n')                                                                                                                                                                         
        file2_.write('##-------------------------------------------##\n')                                                                                                                               
        ###################################################

        ################# W/O gen_mHH ##################### 
        cls2a = xgb.XGBClassifier(
            n_estimators = options.ntrees,
            max_depth = options.treeDeph,
            min_child_weight = options.mcw,
            learning_rate = options.lr,
            )

        cls2a.fit(
            traindataset1[BDTvariables_wo_gen_mHH].values,
            traindataset1.target.astype(np.bool),
            sample_weight=(traindataset1[weights].astype(np.float64))
            )

        ## --- ROC curve ----##
        proba_a = cls2a.predict_proba(valdataset1[BDTvariables_wo_gen_mHH].values )
        fprt_a, tprt_a, thresholds = roc_curve(valdataset1[target].astype(np.bool), proba_a[:,1], sample_weight=(valdataset1[weights].astype(np.float64))  )                                                                  
        test_auct_a = auc(fprt_a, tprt_a, reorder = True)

        proba_a = cls2a.predict_proba(traindataset1[BDTvariables_wo_gen_mHH].values )                                                                               
        fpr_a, tpr_a, thresholds = roc_curve(traindataset1[target].astype(np.bool), proba_a[:,1], sample_weight=(traindataset1[weights].astype(np.float64)) )                                                                  
        train_auc_a = auc(fpr_a, tpr_a, reorder = True)                                                                                                                                                     

        ## ---- BDT Output Distributions ----- ###
        y_pred_a = cls2a.predict_proba(valdataset1.ix[valdataset1.target.values == 0, BDTvariables_wo_gen_mHH].values)[:, 1]                                        
        y_predS_a = cls2a.predict_proba(valdataset1.ix[valdataset1.target.values == 1, BDTvariables_wo_gen_mHH].values)[:, 1]                                                 

        y_pred_train_a = cls2a.predict_proba(traindataset1.ix[traindataset1.target.values == 0, BDTvariables_wo_gen_mHH].values)[:, 1]                     
        y_predS_train_a = cls2a.predict_proba(traindataset1.ix[traindataset1.target.values == 1, BDTvariables_wo_gen_mHH].values)[:, 1]

        file2_.write('#------ Training for all masses except %0.1f GeV (w/o gen_mHH)-------\n' %mass)                                                                                                                
        file2_.write('#------ Testing only for mass %0.1f GeV using other dataset half (w/o gen_mHH)--------\n' %mass)                                                                                    
        if(dd == 0): ## Train->odd; Test/val -> even                                                                                                                                             
            file2_.write('train_auc_ODD0a_%s = %0.8f\n' % (MASS_STR, train_auc_a))       
            file2_.write('test_auc_EVEN0a_%s  = %0.8f\n' % (MASS_STR, test_auct_a))
            file2_.write('xtrain_ODD0a_%s = ' % (MASS_STR))                                                                                                                                               
            file2_.write(str(fpr_a.tolist()))
            file2_.write('\n')
            file2_.write('ytrain_ODD0a_%s = ' % (MASS_STR))
            file2_.write(str(tpr_a.tolist()))                                                                                                                                                            
            file2_.write('\n')
            file2_.write('xval_EVEN0a_%s = ' % (MASS_STR))
            file2_.write(str(fprt_a.tolist()))                                                                                                                                                           
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('yval_EVEN0a_%s = ' % (MASS_STR))                                                                                                                                               
            file2_.write(str(tprt_a.tolist()))                                                                                                                                                           
            file2_.write('\n')
            file2_.write('y_pred_test_EVEN0a_%s = ' % (MASS_STR))
            file2_.write(str(y_pred_a.tolist()))                                                                                                                                                         
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('y_predS_test_EVEN0a_%s = ' % (MASS_STR))                                                                                                                                          
            file2_.write(str(y_predS_a.tolist()))                                                                                                                                                        
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('y_pred_train_ODD0a_%s = ' % (MASS_STR))                                                                                                                                         
            file2_.write(str(y_pred_train_a.tolist()))                                                                                                                                                   
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('y_predS_train_ODD0a_%s = ' % (MASS_STR))                                                                                                                                         
            file2_.write(str(y_predS_train_a.tolist()))                                                                                                                                                  
            file2_.write('\n')
        else: ## Train->even; Test/val -> odd                                                                                                                                                  
            file2_.write('train_auc_EVEN0a_%s = %0.8f\n' %  (MASS_STR, train_auc_a))                                                                                                                      
            file2_.write('test_auc_ODD0a_%s  = %0.8f\n' %  (MASS_STR, test_auct_a)) 
            file2_.write('xtrain_EVEN0a_%s = ' % (MASS_STR)) 
            file2_.write(str(fpr_a.tolist()))                                                                                                                                                            
            file2_.write('\n')
            file2_.write('ytrain_EVEN0a_%s = ' % (MASS_STR))
            file2_.write(str(tpr_a.tolist()))                                                                                                                                                            
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('xval_ODD0a_%s = ' % (MASS_STR))                                                                                                                             
            file2_.write(str(fprt_a.tolist()))                                                                                                                                                           
            file2_.write('\n')                         
            file2_.write('yval_ODD0a_%s = ' % (MASS_STR))                                                                                                                                                
            file2_.write(str(tprt_a.tolist()))                                                                                                                                                           
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('y_pred_test_ODD0a_%s = ' % (MASS_STR))                                                                                                                                        
            file2_.write(str(y_pred_a.tolist()))                                                                                                                                                         
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('y_predS_test_ODD0a_%s = ' % (MASS_STR))                                                                                                                                      
            file2_.write(str(y_predS_a.tolist()))                                                                                                                                                        
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('y_pred_train_EVEN0a_%s = ' % (MASS_STR))                                                                                                                                    
            file2_.write(str(y_pred_train_a.tolist()))                                                                                                                                                   
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('y_predS_train_EVEN0a_%s = ' % (MASS_STR))                                                                                                                                     
            file2_.write(str(y_predS_train_a.tolist()))                                                                                                                                                  
            file2_.write('\n')                                                                                                                                                                         
        file2_.write('##-------------------------------------------##\n')                                                                                                                               
        ###################################################

        ## --- ROC curve ----##
        if(Bkg_mass_rand == "default"): 
            print("Using the new logic") 
            #traindataset2 = data_do.loc[(data_do["gen_mHH"]==mass)]  ## Training for only the mass point under study                                                                                       
            traindataset2 = data_do.loc[ ~((data_do["gen_mHH"]!=mass) & (data_do[target]==1)) ]  ## Training for only the mass point under study (for the signal process)                
            valdataset2 = order_train[val_data].loc[ ~((order_train[val_data]["gen_mHH"]!=mass) & (order_train[val_data][target]==1)) ]  ## Testing for only the mass point under study (for the signal process)
        else:    
            print("Using the old logic") 
            traindataset2 = data_do.loc[(data_do["gen_mHH"]==mass)]  ## Training for only the mass point under study                                                                                       
            valdataset2 = order_train[val_data].loc[(order_train[val_data]["gen_mHH"]==mass)]  ## Testing for only the mass point under study
         
        #traindataset2 = data_do.loc[(data_do["gen_mHH"]==mass)]  ## Training for only the mass point under study                                                                                       
        #valdataset2 = order_train[val_data].loc[(order_train[val_data]["gen_mHH"]==mass)]  ## Testing for only the mass point under study                               

        plot_name_train2 = "_Variables_traindataset2_BDT_"+MASS_STR+".pdf"
        plot_name_val2 = "_Variables_valdataset2_BDT_"+MASS_STR+".pdf"

        make_plots(BDTvariables, nbins,
                   traindataset2.ix[traindataset2.target.values == 0],labelBKG, colorFast,
                   traindataset2.ix[traindataset2.target.values == 1],'Signal', colorFastT,
                   channel+"/"+bdtType+"_"+trainvar+plot_name_train2,
                   printmin,
                   plotResiduals,
                   masses_test2,
                   masses2
                   )

        make_plots(BDTvariables, nbins,
                   valdataset2.ix[valdataset2.target.values == 0],labelBKG, colorFast,
                   valdataset2.ix[valdataset2.target.values == 1],'Signal', colorFastT,
                   channel+"/"+bdtType+"_"+trainvar+plot_name_val2,
                   printmin,
                   plotResiduals,
                   masses_test2,
                   masses2
                   )

        cls3 = xgb.XGBClassifier(
            n_estimators = options.ntrees,
            max_depth = options.treeDeph,
            min_child_weight = options.mcw,
            learning_rate = options.lr,
            )

        cls3.fit(
            traindataset2[BDTvariables_wo_gen_mHH].values, ## Since we are doing single mass train no need for gen_mHH
            traindataset2.target.astype(np.bool),
            sample_weight=(traindataset2[weights].astype(np.float64))
            )

        #proba = cls3.predict_proba(valdataset2[trainVars(False, options.variables, options.bdtType)].values )
        proba = cls3.predict_proba(valdataset2[BDTvariables_wo_gen_mHH].values ) ## Since we are doing single mass train no need for gen_mHH  
        fprt, tprt, thresholds = roc_curve(valdataset2[target].astype(np.bool), proba[:,1], sample_weight=(valdataset2[weights].astype(np.float64))  )                                                                  
        print("fprt", fprt)
        print("tprt", tprt)
        test_auct = auc(fprt, tprt, reorder = True)                                                                                                                                                   

        #proba = cls3.predict_proba(traindataset2[trainVars(False, options.variables, options.bdtType)].values )
        proba = cls3.predict_proba(traindataset2[BDTvariables_wo_gen_mHH].values ) ## Since we are doing single mass train no need for gen_mHH  
        fpr, tpr, thresholds = roc_curve(traindataset2[target].astype(np.bool), proba[:,1], sample_weight=(traindataset2[weights].astype(np.float64)) )                                                                  
        train_auc = auc(fpr, tpr, reorder = True)                                                                                                                                                     

        ## ---- BDT Output Distributions ----- ###  
        y_pred = cls3.predict_proba(valdataset2.ix[valdataset2.target.values == 0, BDTvariables_wo_gen_mHH].values)[:, 1]   ## Since we are doing single mass train no need for gen_mHH
        y_predS = cls3.predict_proba(valdataset2.ix[valdataset2.target.values == 1, BDTvariables_wo_gen_mHH].values)[:, 1]  ## Since we are doing single mass train no need for gen_mHH

        y_pred_train = cls3.predict_proba(traindataset2.ix[traindataset2.target.values == 0, BDTvariables_wo_gen_mHH].values)[:, 1]  ## Since we are doing single mass train no need for gen_mHH
        y_predS_train = cls3.predict_proba(traindataset2.ix[traindataset2.target.values == 1, BDTvariables_wo_gen_mHH].values)[:, 1] ## Since we are doing single mass train no need for gen_mHH   

        file2_.write('#------ Training for only mass %0.1f GeV-------\n' %mass)                                                                                                                        
        file2_.write('#------ Testing for only mass %0.1f GeV (using other data half)--------\n' %mass)                                                                                              
        if(dd == 0): ## Train->odd; Test/val -> even                                                                                                                                                            
            file2_.write('train_auc_ODD1_%s = %0.8f\n' % (MASS_STR, train_auc))                                                                                                                                    
            file2_.write('test_auc_EVEN1_%s  = %0.8f\n' % (MASS_STR, test_auct))                                                                                                                                  
            file2_.write('xtrain_ODD1_%s = ' % (MASS_STR))                                                                                                                                                  
            file2_.write(str(fpr.tolist()))                                                                                                                                                            
            file2_.write('\n')               
            file2_.write('ytrain_ODD1_%s = ' % (MASS_STR)) 
            file2_.write(str(tpr.tolist())) 
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('xval_EVEN1_%s = ' % (MASS_STR))                                                                                                                                                       
            file2_.write(str(fprt.tolist()))                                                                                                                                                           
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('yval_EVEN1_%s = ' % (MASS_STR))                                                                                                                                                 
            file2_.write(str(tprt.tolist()))                                                                                                                                                           
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('y_pred_test_EVEN1_%s = ' % (MASS_STR))                                                                                                                                                 
            file2_.write(str(y_pred.tolist()))                                                                                                                                                         
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('y_predS_test_EVEN1_%s = ' % (MASS_STR))                                                                                                                                       
            file2_.write(str(y_predS.tolist()))                                                                                                                                                        
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('y_pred_train_ODD1_%s = ' % (MASS_STR))                                                                                                                                     
            file2_.write(str(y_pred_train.tolist()))                                                                                                                                                   
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('y_predS_train_ODD1_%s = ' % (MASS_STR))                                                                                                                                      
            file2_.write(str(y_predS_train.tolist()))                                                                                                                                                  
            file2_.write('\n')                                                                                                                                                                         
        else: ## Train->even; Test/val -> odd         
            file2_.write('train_auc_EVEN1_%s = %0.8f\n' %  (MASS_STR, train_auc))                                                                                                                                 
            file2_.write('test_auc_ODD1_%s  = %0.8f\n' % (MASS_STR, test_auct))                                                                                                                                 
            file2_.write('xtrain_EVEN1_%s = ' % (MASS_STR))                                                                                                                                                        
            file2_.write(str(fpr.tolist()))                                                                                                                                                            
            file2_.write('\n')           
            file2_.write('ytrain_EVEN1_%s = ' % (MASS_STR))                                                                                                                                                       
            file2_.write(str(tpr.tolist()))                                                                                                                                                            
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('xval_ODD1_%s = ' % (MASS_STR))                                                                                                                                          
            file2_.write(str(fprt.tolist()))                                                                                                                                                           
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('yval_ODD1_%s = ' % (MASS_STR))                                                                                                                                              
            file2_.write(str(tprt.tolist()))                                                                                                                                                           
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('y_pred_test_ODD1_%s = ' % (MASS_STR))                                                                                                                                                
            file2_.write(str(y_pred.tolist()))                                                                                                                                                         
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('y_predS_test_ODD1_%s = ' % (MASS_STR))                                                                                                                                      
            file2_.write(str(y_predS.tolist()))                                                                                                                                                        
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('y_pred_train_EVEN1_%s = ' % (MASS_STR))                                                                                                                                    
            file2_.write(str(y_pred_train.tolist()))                                                                                                                                                   
            file2_.write('\n')                                                                                                                                                                         
            file2_.write('y_predS_train_EVEN1_%s = ' % (MASS_STR))                                                                                                                                     
            file2_.write(str(y_predS_train.tolist()))                                                                                                                                                  
            file2_.write('\n')                                                                                                                                                                         
        file2_.write('##-------------------------------------------##\n')
    file2_.close()
### ---------------------------------------- ####



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
		#plt.savefig(namesave.replace(".pdf",".png"))
		ax.clear()

## Scanning the classification error vs epoch (=ntrees) for a given set of hyper-parameter optimization (RUN IN CMSSW_10_X_Y ENVIRON.)
if options.ClassErr_vs_epoch==True :
   if options.SplitMode == 0:
       ### ----- RANDOM SPLIT OF PANDAS DATAFRAME "data" ---- ####
       data_total = data_odd.append(data_even, ignore_index=True) ## Adding the odd and even dataframes together to form one dataframe
       data = data_total[trainVars(False, options.variables, options.bdtType)+["target","totalWeight"]] ## pandas dataframe with trainVars, target and totalWeight columns

       data_target = np.array(data['target'].values, dtype=np.int32)
       data = data.drop(['target'], axis=1)

       # split data into train and test sets -2
       ## (X_train, X_test) are  pandas dataframes while (y_train, y_test) are numpy arrays
       X_train, X_test, y_train, y_test = train_test_split(data[trainVars(False, options.variables, options.bdtType)+["totalWeight"]], data_target, test_size=0.50, random_state=7) ## The only random part of the code
       print('Using Random split')
   elif options.SplitMode == 1:
       ### ----- ODD-EVEN SPLIT OF PANDAS DATAFRAME "data" ---- ####
       data_even = data_even[trainVars(False, options.variables, options.bdtType)+["target","totalWeight"]] ## pandas dataframe with trainVars, target and totalWeight columns
       data_even_target = np.array(data_even['target'].values, dtype=np.int32)
       data_even = data_even.drop(['target'], axis=1)

       data_odd = data_odd[trainVars(False, options.variables, options.bdtType)+["target","totalWeight"]] ## pandas dataframe with trainVars, target and totalWeight columns
       data_odd_target = np.array(data_odd['target'].values, dtype=np.int32)
       data_odd = data_odd.drop(['target'], axis=1)

       ## (X_train, X_test) are  pandas dataframes while (y_train, y_test) are numpy arrays
       X_train = data_even
       y_train = data_even_target
       X_test  = data_odd
       y_test  = data_odd_target
       print('Using Odd train Even test split')
   elif options.SplitMode == 2:
       ### ----- ODD-EVEN SPLIT OF PANDAS DATAFRAME "data" ---- ####
       data_even = data_even[trainVars(False, options.variables, options.bdtType)+["target","totalWeight"]] ## pandas dataframe with trainVars, target and totalWeight columns
       data_even_target = np.array(data_even['target'].values, dtype=np.int32)
       data_even = data_even.drop(['target'], axis=1)

       data_odd = data_odd[trainVars(False, options.variables, options.bdtType)+["target","totalWeight"]] ## pandas dataframe with trainVars, target and totalWeight columns
       data_odd_target = np.array(data_odd['target'].values, dtype=np.int32)
       data_odd = data_odd.drop(['target'], axis=1)

       ## (X_train, X_test) are  pandas dataframes while (y_train, y_test) are numpy arrays
       X_train = data_odd
       y_train = data_odd_target
       X_test  = data_even
       y_test  = data_even_target
       print('Using Even train Odd test split')
   else:
       print("Invalid option SplitMode: Please give values 0/1/2")
       raise ValueError("Invalid parameter SplitMode = '%i' !!" % options.SplitMode)



   X_train_weight = np.array(X_train['totalWeight'].values, dtype=np.float)
   X_test_weight = np.array(X_test['totalWeight'].values, dtype=np.float)

   X_train = X_train.drop(['totalWeight'], axis=1)
   X_test = X_test.drop(['totalWeight'], axis=1)


   ## Converting dataframe to numpy array (dtype=object)
   X_train = X_train.values
   X_test = X_test.values

   eval_set = [(X_train, y_train), (X_test, y_test)]


   if (options.ScanMode == 1):
       # ---- For scanning depth at fixed "ntrees*lr = 10" -- ##
       array_ntrees = [10, 100, 1000, 2000, 3000]
       array_lr = [0.1]
       array_depth = [2,3,4]
       array_mcw = [1]
       N = len(array_ntrees)*len(array_lr)*len(array_depth)*len(array_mcw)
       print('N =', N)
   elif (options.ScanMode == 2):
       # ---- For scanning ntrees at fixed lr and depth -- ##
       array_ntrees = [1000, 1500, 2000, 2500, 3000]
       array_lr = [0.01]
       array_depth = [2]
       array_mcw = [1]
       N = len(array_ntrees)*len(array_lr)*len(array_depth)*len(array_mcw)
       print('N =', N)
   else:
       print("Invalid option ScanMode: Please give values 1/2")
       raise ValueError("Invalid parameter ScanMode = '%i' !!" % options.ScanMode)


   X_label = []
   Acc_label = []
   i = 0
   for ntrees in  array_ntrees:
       for lr in array_lr:
           if (options.ScanMode == 1):   lr = 10./float(ntrees) ## so that lr*ntrees = 10
           for depth in array_depth:
               for mcw in array_mcw:
			   i = i + 1
			   print('ntrees: %i, lr: %f, depth: %i, mcw: %i' % (ntrees, lr, depth, mcw))
			   hyppar_test="ntrees_"+str(ntrees)+"_depth_"+str(depth)+"_mcw_"+str(mcw)+lr_to_str(lr)
			   X_label.append(hyppar_test)
			   print("file label:", hyppar_test)
			   file1_.write('fit %i, %s' % (i, hyppar_test))
			   file1_.write('\n')
			   model = xgb.XGBClassifier(
				   n_estimators = ntrees,
				   max_depth = depth,
				   min_child_weight = mcw,
				   learning_rate = lr
				   )
			   print('Fit started')
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

			   # make predictions for test data
			   y_pred = model.predict(X_test)

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
                           fig1.savefig("{}/Log_loss_{}_{}_{}_{}.pdf".format(channel, bdtType, trainvar, str(len(trainVars(False, options.variables, options.bdtType))), hyppar_test))

                           # plot auc
			   fig1a, ax = plt.subplots()
			   ax.plot(x_axis, results['validation_0']['auc'], label='Train')
			   ax.plot(x_axis, results['validation_1']['auc'], label='Test')
			   ax.legend()
			   plt.ylabel('Area under ROC curve')
			   plt.title('XGBoost AUC')
			   plt.show()
			   fig1a.savefig("{}/auc_{}_{}_{}_{}.pdf".format(channel, bdtType, trainvar, str(len(trainVars(False, options.variables, options.bdtType))), hyppar_test))

			   # plot classification error
			   fig2, ax = plt.subplots()
			   ax.plot(x_axis, results['validation_0']['error'], label='Train')
			   ax.plot(x_axis, results['validation_1']['error'], label='Test')
			   ax.legend()
			   plt.ylabel('Classification Error')
			   plt.title('XGBoost Classification Error')
			   plt.show()
			   fig2.savefig("{}/XGBoost_Classification_error_{}_{}_{}_{}.pdf".format(channel, bdtType, trainvar, str(len(trainVars(False, options.variables, options.bdtType))), hyppar_test))
			   plt.close() 
   file1_.close()
   fit_label = np.arange(1,(N+1)) ## gives us [1,2,....,N]                                                                                                                                                  
   fig, ax = plt.subplots()
   plt.plot(fit_label, Acc_label, 'ro')
   plt.xlabel('fit index')
   plt.ylabel('Classification Accuracy (%%)')
   plt.axis([0, (N+5), 60, 100])
   fig.savefig("{}/Accuracy.pdf".format(channel))


process = psutil.Process(os.getpid())
print(process.memory_info().rss)
print(datetime.now() - startTime)


process = psutil.Process(os.getpid())
print(process.memory_info().rss)
print(datetime.now() - startTime)
