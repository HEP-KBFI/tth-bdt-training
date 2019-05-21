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
parser.add_option("--ntrees ", type="int", dest="ntrees", help="hyp", default=1500)
parser.add_option("--treeDeph", type="int", dest="treeDeph", help="hyp", default=3)
parser.add_option("--lr", type="float", dest="lr", help="hyp", default=0.01)
parser.add_option("--mcw", type="int", dest="mcw", help="hyp", default=1)
(options, args) = parser.parse_args()

doPlots=options.doPlots
bdtType=options.bdtType
trainvar=options.variables
hyppar=str(options.variables)+"_ntrees_"+str(options.ntrees)+"_deph_"+str(options.treeDeph)+"_mcw_"+str(options.mcw)+"_lr_0o0"+str(int(options.lr*100))

## --- OUTPUT DIRECTORY NAME----
channel=options.channel+"_HH"
# channel=options.channel+"_HH_dR03mvaLoose"
# channel=options.channel+"_HH_dR03mvaVLoose"
#channel=options.channel+"_HH_dR03mvaVVLoose_all"


#if resonant bdtype
file_ = open('roc_%s.log'%channel,'w+')
if "bb2l" in channel   : execfile("../cards/info_bb2l_HH.py")
elif "bb1l" in channel : execfile("../cards/info_bb1l_HH.py")
if "2l_2tau" in channel : execfile("../cards/info_2l_2tau_HH.py")
if "3l_0tau" in channel : execfile("../cards/info_3l_0tau_HH.py")

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

print "inputPath: ",inputPath,", channelInTree: ",channelInTree; sys.stdout.flush()
####################################################################################################
## Load data
#data=load_data_2017(inputPath,channelInTree,trainVars(True),[],bdtType)
if "bb2l" in channel   : data=load_data_2017_HH(inputPath,channelInTree,trainVars(True),[],bdtType)
elif "bb1l" in channel : data=load_data_2017_HH(inputPath,channelInTree,trainVars(True),[],bdtType)
elif "2l_2tau" in channel: data=load_data_2017_HH_2l_2tau(inputPath,channelInTree,trainVars(True),[],bdtType)
else : data=load_data_2017(inputPath,channelInTree,trainVars(True),[],bdtType)
#**********************

#################################################################################
#print ("Sum of weights:", data.loc[data['target']==0][weights].sum())



## Balance datasets
#https://stackoverflow.com/questions/34803670/pandas-conditional-multiplication
if "evtLevelSUM_HH_bb1l_res" in bdtType : variables = [250,270,280,320,350,400,450,500,600,650,750,800,850,900,1000]
elif "evtLevelSUM_HH_bb2l_res" in bdtType : variables = [250,270,280,320,350,400,450,500,600,650,750,800,850,900,1000]
elif "evtLevelSUM_HH_bb2l_nonres" in bdtType : variables = [2,3,7,9,12,20]
elif "evtLevelSUM_HH_2l_2tau_res" in bdtType : variables = [250,260,270,280,300,350,400,450,500, 550,600,650,700,750,800,850,900,1000]
#elif "evtLevelSUM_HH_2l_2tau_res" in bdtType : variables = [500]
else : print '****************** please define your mass point**************'


if 'evtLevelSUM' in bdtType :
	data.loc[(data['key']=='TTToHadronic_PSweights') | (data['key']=='TTToSemiLeptonic_PSweights') | (data['key']=='TTTo2L2Nu_PSweights'), [weights]]*=TTdatacard/fastsimTT
	#fastsimTT=data.loc[(data['key']=='TTToHadronic_PSweights'),[weights]].sum()+ data.loc[(data['key']=='TTToSemiLeptonic_PSweights'),[weights]].sum() + data.loc[(data['key']=='TTTo2L2Nu_PSweights'), [weights]].sum()
	#print 'fastsimTT = ', fastsimTT
	#data.loc[(data['key']=='TTToHadronic_PSweights') | (data['key']=='TTToSemiLeptonic_PSweights') | (data['key']=='TTTo2L2Nu_PSweights'), [weights]]*=TTdatacard/fastsimTT
        #data.loc[(data['key']=='TTWJetsToLNu') | (data['key']=='TTZ'), [weights]]*=TTVdatacard/fastsimTTV
	data.loc[(data['key']=='DY'), [weights]]*=DYdatacard/fastsimDY
	#fastsimDY=data.loc[(data['key']=='DY'), [weights]].sum()
	#print 'fastsimDY= ', fastsimDY
        #data.loc[(data['key']=='DY'), [weights]]*=DYdatacard/fastsimDY
	if "evtLevelSUM_HH_bb1l" in bdtType : 
		#fastsimW=data.loc[(data['key']=='W'), [weights]].sum()
		#print 'fastsimW= ', fastsimW
		#data.loc[(data['key']=='W'), [weights]]*=Wdatacard/fastsimW
		data.loc[(data['key']=='W'), [weights]]*=Wdatacard/fastsimW


	if"evtLevelSUM_HH_2l_2tau_res" in bdtType : 
	        data.loc[(data['key']=='TTToHadronic') | (data['key']=='TTToSemileptonic') | (data['key']=='TTTo2L2Nu'), [weights]]*=TTdatacard/fastsimTT
                data.loc[(data['key']=='TTWJets') | (data['key']=='TTZJets'), [weights]]*=TTVdatacard/fastsimTTV
                data.loc[(data['key']=='DY'), [weights]]*=DYdatacard/fastsimDY
                data.loc[(data['key']=='VH'), [weights]]*=VHdatacard/fastsimVH
                data.loc[(data['key']=='TTH'), [weights]]*=TTHdatacard/fastsimTTH
                data.loc[(data['key']=='WW') | (data['key']=='WZ') | (data['key']=='ZZ'), [weights]]*=VVdatacard/fastsimVV
                
	for var in range(len(variables)) :
                data.loc[(data[target]==1) & (data[variable] == variables[var]),[weights]] *= 100000./data.loc[(data[target]==1) & (data[variable]== variables[var]),[weights]].sum()
                data.loc[(data[target]==0) & (data[variable] == variables[var]),[weights]] *= 100000./data.loc[(data[target]==0) & (data[variable]== variables[var]),[weights]].sum()

else :
	data.loc[data['target']==0, [weights]] *= 100000/data.loc[data['target']==0][weights].sum()
	data.loc[data['target']==1, [weights]] *= 100000/data.loc[data['target']==1][weights].sum()


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
	if channel in ["3l_0tau_HH"]:
		labelBKG = "WZ+tt+ttV"
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
BDTvariables=trainVars(plotAll)
make_plots(BDTvariables,nbins,
    data.ix[data.target.values == 0],labelBKG, colorFast,
    data.ix[data.target.values == 1],'Signal', colorFastT,
    channel+"/"+bdtType+"_"+trainvar+"_Variables_BDT.png",
    printmin,
    plotResiduals
    )

#########################################################################################
'''traindataset, valdataset1  = train_test_split(data[trainVars(False)+["target","totalWeight"]], test_size=0.2, random_state=7)
valdataset = valdataset1.loc[valdataset1['gen_mHH']==400]
valdataset.loc[valdataset[target]==1,[weights]] *= valdataset1.loc[valdataset1[target]==1]["totalWeight"].sum()/valdataset.loc[valdataset[target]==1]["totalWeight"].sum()
valdataset.loc[valdataset[target]==0,[weights]]*= valdataset1.loc[valdataset1[target]==0]["totalWeight"].sum()/valdataset.loc[valdataset[target]==0]["totalWeight"].sum()'''
traindataset1, valdataset1  = train_test_split(data[trainVars(False)+["target","totalWeight","key"]], test_size=0.2, random_state=7)
traindataset = traindataset1.loc[(traindataset1["gen_mHH"] ==350)]# | (traindataset["gen_mHH"] ==500) | (traindataset["gen_mHH"] == 900)]
valdataset = valdataset1.loc[(valdataset1["gen_mHH"] ==350)]
totTT =  traindataset.loc[(traindataset['key']=='TTToHadronic_PSweights') & (traindataset[variable]==320),[weights]].sum()+traindataset.loc[(traindataset['key']=='TTToSemiLeptonic_PSweights') & (traindataset[variable]==320),[weights]].sum()+traindataset.loc[(traindataset['key']=='TTTo2L2Nu_PSweights') & (traindataset[variable]==320), [weights]].sum()
totDY = traindataset.loc[(traindataset['key']=='DY') & (traindataset[variable]==320), [weights]].sum()
totW =  traindataset.loc[(traindataset['key']=='W') & (traindataset[variable]==320) , [weights]].sum()
print 'ratioTTWDY = ', totTT/(totTT+totW+totDY),'\t', ':', totW/(totTT+totW+totDY),'\t', ':', totDY/(totTT+totW+totDY)

print 'Tot weight of train and validation for signal= ', traindataset.loc[traindataset[target]==1]["totalWeight"].sum(), valdataset.loc[valdataset[target]==1]["totalWeight"].sum()
print 'Tot weight of train and validation for bkg= ', traindataset.loc[traindataset[target]==0]['totalWeight'].sum(),valdataset.loc[valdataset[target]==0]['totalWeight'].sum()

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
	traindataset[trainVars(False)].values,
	traindataset.target.astype(np.bool),
	sample_weight=(traindataset[weights].astype(np.float64))
	# more diagnosis, in case
	#eval_set=[(traindataset[trainVars(False)].values,  traindataset.target.astype(np.bool),traindataset[weights].astype(np.float64)),
	#(valdataset[trainVars(False)].values,  valdataset.target.astype(np.bool), valdataset[weights].astype(np.float64))] ,
	#verbose=True,eval_metric="auc"
	)

#print trainVars(False)
print 'traindataset[trainVars(False)].columns.values.tolist() : ', traindataset[trainVars(False)].columns.values.tolist()

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
file_.write("XGBoost_train = %0.8f\n" %train_auc)
file_.write("XGBoost_test = %0.8f\n" %test_auct)
fig, ax = plt.subplots()
f_score_dict =cls.booster().get_fscore()

pklpath=channel+"/"+channel+"_XGB_"+trainvar+"_"+bdtType+"_"+str(len(trainVars(False)))+"Var"
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
ax.plot(fpr, tpr, lw=1, color='g',label='XGB train all mass excluding 350GeV(area = %0.5f)'%(train_auc))
ax.plot(fprt, tprt, lw=1, ls='--',color='g',label='XGB test excluding 350GeV(area = %0.5f)'%(test_auct))
#ax.plot(fprtightF, tprtightF, lw=1, label='XGB test - Fullsim All (area = %0.3f)'%(test_auctightF))                                                                                                        
#ax.set_ylim([0.0,1.0])
#ax.set_xlim([0.0,1.0])
#ax.set_xlabel('False Positive Rate')
#ax.set_ylabel('True Positive Rate')
#ax.legend(loc="lower right")
#ax.grid()
#fig.savefig("{}/{}_{}_{}_{}_roc.png".format(channel,bdtType,trainvar,str(len(trainVars(False))),hyppar))
#fig.savefig("{}/{}_{}_{}_{}_roc.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False))),hyppar))

if variable == 'node' : variables=[20]
else : variables = [350]
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

fig.savefig("{}/{}_{}_{}_{}_350roc.png".format(channel,bdtType,trainvar,str(len(trainVars(False))),hyppar)) 
fig.savefig("{}/{}_{}_{}_{}_350roc.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False))),hyppar)) 
###########################################################################
## feature importance plot
fig, ax = plt.subplots()
f_score_dict =cls.booster().get_fscore()
f_score_dict = {trainVars(False)[int(k[1:])] : v for k,v in f_score_dict.items()}
feat_imp = pandas.Series(f_score_dict).sort_values(ascending=True)
feat_imp.plot(kind='barh', title='Feature Importances')
fig.tight_layout()
fig.savefig("{}/{}_{}_{}_{}_XGB_importance.png".format(channel,bdtType,trainvar,str(len(trainVars(False))),hyppar))
fig.savefig("{}/{}_{}_{}_{}_XGB_importance.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False))),hyppar))
###########################################################################
hist_params = {'normed': True, 'bins': 10 , 'histtype':'step'}
plt.clf()
y_pred = cls.predict_proba(valdataset.ix[valdataset.target.values == 0, trainVars(False)].values)[:, 1] #
y_predS = cls.predict_proba(valdataset.ix[valdataset.target.values == 1, trainVars(False)].values)[:, 1] #
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
plt.savefig(channel+'/'+bdtType+'_'+trainvar+'_'+str(len(trainVars(False)))+'_'+hyppar+'_XGBclassifier.pdf')
plt.savefig(channel+'/'+bdtType+'_'+trainvar+'_'+str(len(trainVars(False)))+'_'+hyppar+'_XGBclassifier.png')
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
		datacorr = datad[trainVars(False)].astype(float)  #.loc[:,trainVars(False)] #dataHToNobbCSV[[trainVars(True)]]
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
print(datetime.now() - startTime)
