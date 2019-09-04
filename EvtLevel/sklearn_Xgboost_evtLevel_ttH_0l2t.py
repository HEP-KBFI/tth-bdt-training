import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys , time
import sklearn_to_tmva
import sklearn
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
import pandas
import matplotlib.mlab as mlab
from scipy.stats import norm
#from pandas import HDFStore,DataFrame
import math
#import sklearn_to_tmva
import xgboost2tmva
import skTMVA
import matplotlib
matplotlib.use('agg')
#matplotlib.use('PS')   # generate postscript output by default
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import numpy as np
import psutil
import os
from rep.estimators import TMVAClassifier

import pickle

from sklearn.externals import joblib
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

from keras.models import Sequential, model_from_json
import json

from collections import OrderedDict

#from tth-bdt-training-test.data_manager import load_data
#dm = __import__("tth-bdt-training-test.data_manager.py")

#import imp
#dm = imp.load_module("dm_name", "tth-bdt-training-test/data_manager.py")
print 'starting job'
execfile("../python/data_manager.py")
# we have many trees
# https://stackoverflow.com/questions/38238139/python-prevent-ioerror-errno-5-input-output-error-when-running-without-stdo

#"""
#run command
#run:  python sklearn_Xgboost_evtLevel_ttH_0l2t.py --channel 0l_2tau --variables NoHTT --bdtType evtLevelSUM_TTH
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--channel ", type="string", dest="channel", help="The ones whose variables implemented now are:\n   - 1l_2tau\n   - 2lss_1tau\n It will create a local folder and store the report*/xml", default='T')
parser.add_option("--variables", type="string", dest="variables", help="  Set of variables to use ", default='HTT')
parser.add_option("--bdtType", type="string", dest="bdtType", help=" evtLevelTT_TTH or evtLevelTTV_TTH", default='T')
parser.add_option("--HypOpt", action="store_true", dest="HypOpt", help="If you call this will not do plots with repport", default=False)
parser.add_option("--doXML", action="store_true", dest="doXML", help="Do save not write the xml file", default=False)
parser.add_option("--doPlots", action="store_true", dest="doPlots", help="Fastsim Loose/Tight vs Fullsim variables plots", default=False)
parser.add_option("--oldNtuple", action="store_true", dest="oldNtuple", help="use Matthias", default=False)
parser.add_option("--ntrees ", type="int", dest="ntrees", help="hyp", default=1000)
parser.add_option("--treeDeph", type="int", dest="treeDeph", help="hyp", default=2)
parser.add_option("--lr", type="float", dest="lr", help="hyp", default=0.01)
parser.add_option("--mcw", type="int", dest="mcw", help="hyp", default=1)
(options, args) = parser.parse_args()
#""" bdtType=="evtLevelTTV_TTH"

doPlots=options.doPlots
bdtType=options.bdtType
trainvar=options.variables
hyppar=str(options.variables)+"_ntrees_"+str(options.ntrees)+"_deph_"+str(options.treeDeph)+"_mcw_"+str(options.mcw)+"_lr_0o0"+str(int(options.lr*100))

channel=options.channel

if channel=='0l_2tau':
	channelInTree='0l_2tau_OS_forBDTtraining'
	inputPath='/hdfs/local/arun/ttHAnalysis/2017/2018Sep30_BDT/histograms/0l_2tau/forBDTtraining_OS/'
	criteria=[]
	testtruth1="bWj1Wj2_isGenMatchedWithKinFit_top1"
	testtruth2="bWj1Wj2_isGenMatchedWithKinFit_top2"
	FullsimWP="MediumTau"

print "reading "+inputPath

import shutil,subprocess
proc=subprocess.Popen(['mkdir '+options.channel],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()

def trainVars(all):

        if channel=="0l_2tau" and all==True :return [
		"mindr_tau1_jet",
		"mindr_tau2_jet",
		"avg_dr_jet",
		"ptmiss",
		"htmiss",
		"tau1_pt",
		"tau2_pt",
		"tau1_eta",
		"tau2_eta",
		"dr_taus",
		"mT_tau1",
		"mT_tau2",
		"mTauTauVis",
		"mTauTau",
		"HTT_wKinFit_top1",
		"dr_HadTop1_tau_lead",
		"dr_HadTop1_tau_sublead",
		"dr_HadTop1_tautau",
		"HadTop1_pt",
		"HadTop1_eta",
		"HTT_wKinFit_top2",
		"dr_HadTop2_tau_lead",
		"dr_HadTop2_tau_sublead",
		"dr_HadTop2_tautau",
		"HadTop2_pt",
		"HadTop2_eta",
		"ncombo_top1",
		"ncombo_top2",
		"nJet" ,
		"nBJetLoose" , "nBJetMedium"
		]

        if channel=="0l_2tau" and bdtType=="evtLevelTT_TTH" and trainvar=="HTTWithKinFit" and all==False :return [
		"mindr_tau1_jet",
                "mindr_tau2_jet",
                "avg_dr_jet",
                "ptmiss",
                "htmiss",
                "tau1_pt",
                "tau2_pt",
                "tau1_eta",
                "tau2_eta",
                "dr_taus",
                "mT_tau1",
                "mT_tau2",
                "mTauTauVis",
		"mTauTau",
                "HTT_wKinFit_top1",
                "dr_HadTop1_tau_lead",
                "dr_HadTop1_tau_sublead",
                "dr_HadTop1_tautau",
                "HadTop1_pt",
                "HadTop1_eta",
                "HTT_wKinFit_top2",
                "dr_HadTop2_tau_lead",
                "dr_HadTop2_tau_sublead",
                "dr_HadTop2_tautau",
                "HadTop2_pt",
                "HadTop2_eta",
                "ncombo_top1",
                "ncombo_top2",
                "nJet" ,
                "nBJetLoose" , "nBJetMedium"
		]
	if channel=="0l_2tau" and bdtType=="evtLevelTT_TTH" and trainvar=="HTTWithKinFitReduced" and all==False :return [
                "mindr_tau1_jet",
                "mindr_tau2_jet",
                "avg_dr_jet",
                "ptmiss",
                "htmiss",
                "tau1_pt",
                "tau2_pt",
                "tau1_eta",
                "tau2_eta",
                "dr_taus",
                "mT_tau1",
                "mT_tau2",
                "mTauTauVis",
		"mTauTau",
                "HTT_wKinFit_top1",
                "HadTop1_pt",
                "HadTop1_eta",
                "HTT_wKinFit_top2",
                "HadTop2_pt",
                "nJet"
                ]
        if channel=="0l_2tau" and bdtType=="evtLevelTT_TTH" and trainvar=="Reduced" and all==False :return [
                "mindr_tau1_jet",
                "mindr_tau2_jet",
                "avg_dr_jet",
                "ptmiss",
                "htmiss",
                "tau1_pt",
                "tau2_pt",
                "tau1_eta",
                "tau2_eta",
                "dr_taus",
                "mT_tau1",
                "mT_tau2",
                "mTauTauVis",
		"mTauTau",
                "nJet"
                ]
	if channel=="0l_2tau" and bdtType=="evtLevelTT_TTH" and all==False :return [
		"mindr_tau1_jet",
                "mindr_tau2_jet",
                "avg_dr_jet",
                "ptmiss",
                "htmiss",
                "tau1_pt",
                "tau2_pt",
                "tau1_eta",
                "tau2_eta",
                "dr_taus",
                "mT_tau1",
                "mT_tau2",
                "mTauTauVis",
		"mTauTau",
                "nJet" ,
                "nBJetLoose",
		"nBJetMedium"
                ]
	if channel=="0l_2tau"  and bdtType=="evtLevelTTV_TTH" and trainvar=="HTTWithKinFit" and all==False :return [
                "mindr_tau1_jet",
                "mindr_tau2_jet",
                "avg_dr_jet",
                "ptmiss",
                "htmiss",
                "tau1_pt",
                "tau2_pt",
                "tau1_eta",
                "tau2_eta",
                "dr_taus",
                "mT_tau1",
                "mT_tau2",
                "mTauTauVis",
		"mTauTau",
                "HTT_wKinFit_top1",
                "dr_HadTop1_tau_lead",
                "dr_HadTop1_tau_sublead",
                "dr_HadTop1_tautau",
                "HadTop1_pt",
                "HadTop1_eta",
                "HTT_wKinFit_top2",
                "dr_HadTop2_tau_lead",
                "dr_HadTop2_tau_sublead",
                "dr_HadTop2_tautau",
                "HadTop2_pt",
                "HadTop2_eta",
                "ncombo_top1",
                "ncombo_top2",
                "nJet" ,
                "nBJetLoose",
		"nBJetMedium"
		]
	if channel=="0l_2tau"  and bdtType=="evtLevelTTV_TTH" and trainvar=="HTTWithKinFitReduced" and all==False :return [
                "mindr_tau1_jet",
                "mindr_tau2_jet",
                "avg_dr_jet",
                "ptmiss",
                "htmiss",
                "tau1_pt",
                "tau2_pt",
                "tau1_eta",
                "tau2_eta",
                "dr_taus",
                "mT_tau1",
                "mT_tau2",
                "mTauTauVis",
		"mTauTau",
                "HTT_wKinFit_top1",
                "HTT_wKinFit_top2",
                "nJet"
                ]
	if channel=="0l_2tau"  and bdtType=="evtLevelTTV_TTH" and trainvar=="Reduced" and all==False :return [
                "mindr_tau1_jet",
                "mindr_tau2_jet",
                "avg_dr_jet",
                "ptmiss",
                "htmiss",
                "tau1_pt",
                "tau2_pt",
                "tau1_eta",
                "tau2_eta",
                "dr_taus",
                "mT_tau1",
                "mT_tau2",
                "mTauTauVis",
		"mTauTau",
                "nJet"
                ]
        if channel=="0l_2tau"  and bdtType=="evtLevelTTV_TTH" and all==False :return [
                "mindr_tau1_jet",
                "mindr_tau2_jet",
                "avg_dr_jet",
                "ptmiss",
                "htmiss",
                "tau1_pt",
                "tau2_pt",
                "tau1_eta",
                "tau2_eta",
                "dr_taus",
                "mT_tau1",
                "mT_tau2",
                "mTauTauVis",
		"mTauTau",
                "nJet" ,
                "nBJetLoose",
		"nBJetMedium"
                ]
	if channel=="0l_2tau"  and bdtType=="evtLevelSUM_TTH" and trainvar=="HTTWithKinFit" and all==False :return [
                "mindr_tau1_jet",
                "mindr_tau2_jet",
                "avg_dr_jet",
                "ptmiss",
                "htmiss",
                "tau1_pt",
                "tau2_pt",
                "tau1_eta",
                "tau2_eta",
		"dr_taus",
                "mT_tau1",
                "mT_tau2",
                "mTauTauVis",
		"mTauTau",
                "HTT_wKinFit_top1",
                "dr_HadTop1_tau_lead",
                "dr_HadTop1_tau_sublead",
                "dr_HadTop1_tautau",
                "HadTop1_pt",
                "HadTop1_eta",
                "HTT_wKinFit_top2",
                "dr_HadTop2_tau_lead",
                "dr_HadTop2_tau_sublead",
                "dr_HadTop2_tautau",
                "HadTop2_pt",
                "HadTop2_eta",
                "ncombo_top1",
                "ncombo_top2",
                "nJet" ,
                "nBJetLoose" , "nBJetMedium"
                ]
        if channel=="0l_2tau" and bdtType=="evtLevelSUM_TTH" and trainvar=="HTTWithKinFitReduced" and all==False :return [
                "mindr_tau1_jet",
                "mindr_tau2_jet",
                "avg_dr_jet",
                "ptmiss",
                "htmiss",
                "tau1_pt",
                "tau2_pt",
                "tau1_eta",
                "tau2_eta",
                "dr_taus",
                "mT_tau1",
                "mT_tau2",
                "mTauTauVis",
		"mTauTau",
                "HTT_wKinFit_top1",
                "HadTop1_pt",
                "HadTop1_eta",
                "HTT_wKinFit_top2",
                "HadTop2_pt",
                "nJet"
		#"nBJetLoose",
		#"nBJetMedium"
                ]
        if channel=="0l_2tau"  and bdtType=="evtLevelSUM_TTH" and trainvar=="Reduced" and all==False :return [
                "mindr_tau1_jet",
                "mindr_tau2_jet",
                "avg_dr_jet",
                "ptmiss",
                "htmiss",
                "tau1_pt",
                "tau2_pt",
                "tau1_eta",
                "tau2_eta",
                "dr_taus",
                "mT_tau1",
                "mT_tau2",
                "mTauTauVis",
		"mTauTau",
                "nJet"
                ]
        if channel=="0l_2tau"  and bdtType=="evtLevelSUM_TTH" and all==False :return [
                "mindr_tau1_jet",
                "mindr_tau2_jet",
                "avg_dr_jet",
                "ptmiss",
                "htmiss",
                "tau1_pt",
                "tau2_pt",
                "tau1_eta",
                "tau2_eta",
                "dr_taus",
                "mT_tau1",
                "mT_tau2",
                "mTauTauVis",
		"mTauTau",
                "nJet" ,
                "nBJetLoose" , "nBJetMedium"
                ]
	if channel=="0l_2tau" and bdtType=="evtLevelDY_TTH" and trainvar=="HTTWithKinFitReduced" and all==False :return [
                "mindr_tau1_jet",
                "mindr_tau2_jet",
                "avg_dr_jet",
                "ptmiss",
                "htmiss",
                "tau1_pt",
                "tau2_pt",
                "tau1_eta",
                "tau2_eta",
                "dr_taus",
                "mT_tau1",
                "mT_tau2",
		"mTauTauVis",
                "mTauTau",
                "HTT_wKinFit_top1",
                "HadTop1_pt",
                "HadTop1_eta",
                "HTT_wKinFit_top2",
                "HadTop2_pt",
                "nJet",
                "nBJetLoose",
                "nBJetMedium"
                ]

####################################################################################################
## Load data
data=load_data_2017(inputPath,channelInTree,trainVars(True),[],bdtType)
weights="totalWeight"
target='target'

#################################################################################
## Balance datasets
#https://stackoverflow.com/questions/34803670/pandas-conditional-multiplication
data.loc[data['target']==0, ['totalWeight']] *= 100000/data.loc[data['target']==0]["totalWeight"].sum()
data.loc[data['target']==1, ['totalWeight']] *= 100000/data.loc[data['target']==1]["totalWeight"].sum()

print ("norm", data.loc[data[target]==0][weights].sum(),data.loc[data[target]==1][weights].sum())
### TT-sample is usually much more than fakes
TTdatacard=197.762+348.568
TTVdatacard=1.0
fastsimTT=418.63
fastsimTTV=1.0
data.loc[(data['key']=='TTTo2L2Nu') | (data['key']=='TTToSemilepton'), [weights]]*=TTdatacard/fastsimTT
data.loc[(data['key']=='TTWJetsToLNu') | (data['key']=='TTZToLLNuNu'), [weights]]*=TTVdatacard/fastsimTTV
data.loc[data[target]==0, [weights]] *= 100000/data.loc[data[target]==0][weights].sum()
data.loc[data[target]==1, [weights]] *= 100000/data.loc[data[target]==1][weights].sum()
print data.columns.values.tolist()

# drop events with NaN weights - for safety
#data.replace(to_replace=np.inf, value=np.NaN, inplace=True)
#data.replace(to_replace=np.inf, value=np.zeros, inplace=True)
#data = data.apply(lambda x: pandas.to_numeric(x,errors='ignore'))
data.dropna(subset=[weights],inplace = True) # data
data.fillna(0)

nS = len(data.loc[data.target.values == 1])
nB = len(data.loc[data.target.values == 0])
print "length of sig, bkg without NaN: ", nS, nB

#################################################################################
### Plot histograms of training variables
nbins=8
colorFast='g'
colorFastT='b'
colorFull='r'
hist_params = {'normed': True, 'histtype': 'bar', 'fill': False , 'lw':5}
#plt.figure(figsize=(60, 60))
if 'evtLevelSUM_TTH' in bdtType : labelBKG = "tt+ttV"
if bdtType=='evtLevelTT_TTH' : labelBKG = "tt"
if bdtType=='evtLevelTTV_TTH' : labelBKG = "ttV"
if bdtType=='evtLevelDY_TTH' : labelBKG = "DY"
printmin=True
plotResiduals=False
plotAll=False
BDTvariables=trainVars(plotAll)
make_plots(BDTvariables,nbins,
    data.ix[data.target.values == 0],labelBKG, colorFast,
    data.ix[data.target.values == 1],'Signal', colorFastT,
    channel+"/"+bdtType+"_"+trainvar+"_Variables_BDT_fullsim"+FullsimWP+".pdf",
    printmin, plotResiduals
    )

### Plot aditional histograms
if 1<0 : # channel=="1l_2tau" :
	BDTvariables=['mvaOutput_hadTopTaggerWithKinFit_top1', 'mvaOutput_hadTopTaggerWithKinFit_top2']
	make_plots(BDTvariables,nbins,
    data.ix[data.target.values == 0],labelBKG, colorFast,
    data.ix[data.target.values == 1],'Signal', colorFastT,
    channel+"/"+bdtType+"_"+trainvar+"_BDTVariables_fullsim"+FullsimWP+".pdf",
    printmin
    )

#########################################################################################
traindataset, valdataset  = train_test_split(data[trainVars(False)+["target","totalWeight"]], test_size=0.2, random_state=7)
## to GridSearchCV the test_size should not be smaller than 0.4 == it is used for cross validation!
## to final BDT fit test_size can go down to 0.1 without sign of overtraining
#############################################################################################
## Training parameters
if options.HypOpt==True :
	# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
	param_grid = {
    			'n_estimators': [500,800,1000,2000],
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

cls = xgb.XGBClassifier(
			n_estimators = options.ntrees,
			max_depth = options.treeDeph,
			min_child_weight = options.mcw, # min_samples_leaf
			learning_rate = options.lr,
			nthread = 8
			#max_features = 'sqrt',
			#min_samples_leaf = 100
			#objective='binary:logistic', #booster='gbtree',
			#gamma=0, #min_child_weight=1,
			#max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, #random_state=0
			)
cls.fit(
	traindataset[trainVars(False)].values,
	traindataset.target.astype(np.bool),
	sample_weight=(traindataset[weights].astype(np.float64))
	# more diagnosis, in case
	#eval_set=[(traindataset[trainVars(False)].values,  traindataset.target.astype(np.bool),traindataset[weights].astype(np.float64)),
	#(valdataset[trainVars(False)].values,  valdataset.target.astype(np.bool), valdataset[weights].astype(np.float64))] ,
	#verbose=True,eval_metric="auc"
	)
print trainVars(False)
print traindataset[trainVars(False)].columns.values.tolist()
print ("XGBoost trained")
proba = cls.predict_proba(traindataset[trainVars(False)].values )
fpr, tpr, thresholds = roc_curve(traindataset[target], proba[:,1],
	sample_weight=(traindataset[weights].astype(np.float64)) )
train_auc = auc(fpr, tpr, reorder = True)
print("XGBoost train set auc - {}".format(train_auc))
proba = cls.predict_proba(valdataset[trainVars(False)].values )
fprt, tprt, thresholds = roc_curve(valdataset[target], proba[:,1], sample_weight=(valdataset[weights].astype(np.float64))  )
test_auct = auc(fprt, tprt, reorder = True)
print("XGBoost test set auc - {}".format(test_auct))

pklpath=channel+"/"+channel+"_XGB_"+trainvar+"_"+bdtType+"_"+str(len(trainVars(False)))+"Var"
print ("Done  ",pklpath,hyppar)
if options.doXML==True :
	print ("Date: ", time.asctime( time.localtime(time.time()) ))
	pickle.dump(cls, open(pklpath+".pkl", 'wb'))
	file = open(pklpath+"_pkl.log","w")
	file.write(str(trainVars(False))+"\n")
	file.close()
	print ("saved ",pklpath+".pkl")
	print ("variables are: ",pklpath+"_pkl.log")
	# save the model in file 'xgb.model.dump'
	#model = cls.get_booster().get_dump(fmap='', with_stats=False) #.get_dump() #pickle.dumps(cls)
	#xmlfile=channel+"/"+channel+"_XGB_"+trainvar+"_"+bdtType+".xml"
##################################################
fig, ax = plt.subplots(figsize=(6, 6))
## ROC curve
#ax.plot(fprf, tprf, lw=1, label='GB train (area = %0.3f)'%(train_aucf))
#ax.plot(fprtf, tprtf, lw=1, label='GB test (area = %0.3f)'%(test_auctf))
ax.plot(fpr, tpr, lw=1, label='XGB train (area = %0.3f)'%(train_auc))
ax.plot(fprt, tprt, lw=1, label='XGB test (area = %0.3f)'%(test_auct))
ax.set_ylim([0.0,1.0])
ax.set_xlim([0.0,1.0])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc="lower right")
ax.grid()
fig.savefig("{}/{}_{}_{}_{}_roc.png".format(channel,bdtType,trainvar,str(len(trainVars(False))),hyppar))
fig.savefig("{}/{}_{}_{}_{}_roc.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False))),hyppar))
###########################################################################
## feature importance plot
fig, ax = plt.subplots()
f_score_dict =cls.get_booster().get_fscore()
f_score_dict = {trainVars(False)[int(k[1:])] : v for k,v in f_score_dict.items()}
feat_imp = pandas.Series(f_score_dict).sort_values(ascending=True)
feat_imp.plot(kind='barh', title='Feature Importances')
fig.tight_layout()
fig.savefig("{}/{}_{}_{}_{}_XGB_importance.png".format(channel,bdtType,trainvar,str(len(trainVars(False))),hyppar))
fig.savefig("{}/{}_{}_{}_{}_XGB_importance.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False))),hyppar))
###########################################################################
#print (list(valdataset))
hist_params = {'normed': True, 'bins': 10 , 'histtype':'step'}
plt.clf()
y_pred = cls.predict_proba(valdataset.ix[valdataset.target.values == 0, trainVars(False)].values)[:, 1] #
y_predS = cls.predict_proba(valdataset.ix[valdataset.target.values == 1, trainVars(False)].values)[:, 1] #
plt.figure('XGB',figsize=(6, 6))
values, bins, _ = plt.hist(y_pred , label="TT (XGB)", **hist_params)
values, bins, _ = plt.hist(y_predS , label="signal", **hist_params )
#plt.xscale('log')
#plt.yscale('log')
plt.legend(loc='best')
plt.savefig(channel+'/'+bdtType+'_'+trainvar+'_'+str(len(trainVars(False)))+'_'+hyppar+'_XGBclassifier.pdf')
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
		datacorr = datad[trainVars(False)] #.loc[:,trainVars(False)] #dataHToNobbCSV[[trainVars(True)]]
		correlations = datacorr.corr()
		fig = plt.figure(figsize=(10, 10))
		ax = fig.add_subplot(111)
		cax = ax.matshow(correlations, vmin=-1, vmax=1)
		ticks = np.arange(0,len(trainVars(False)),1)
		plt.rc('axes', labelsize=8)
		ax.set_xticks(ticks)
		ax.set_yticks(ticks)
		ax.set_xticklabels(trainVars(False),rotation=-90)
		ax.set_yticklabels(trainVars(False))
		fig.colorbar(cax)
		fig.tight_layout()
		#plt.subplots_adjust(left=0.9, right=0.9, top=0.9, bottom=0.1)
		plt.savefig("{}/{}_{}_{}_corr_{}.png".format(channel,bdtType,trainvar,str(len(trainVars(False))),label))
		plt.savefig("{}/{}_{}_{}_corr_{}.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False))),label))
		ax.clear()
process = psutil.Process(os.getpid())
print(process.memory_info().rss)
