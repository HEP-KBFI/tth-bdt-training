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
import xgboost as xgb
from sklearn.metrics import roc_curve, auc
from root_numpy import tree2array

import ROOT
import glob
from collections import OrderedDict
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
parser.add_option("--ntrees ", type="int", dest="ntrees", help="hyp", default=1000)
parser.add_option("--treeDeph", type="int", dest="treeDeph", help="hyp", default=2)
parser.add_option("--lr", type="float", dest="lr", help="hyp", default=0.01)
parser.add_option("--mcw", type="int", dest="mcw", help="hyp", default=1)
(options, args) = parser.parse_args()

doPlots=options.doPlots
bdtType=options.bdtType
trainvar=options.variables
hyppar=str(options.variables)+"_ntrees_"+str(options.ntrees)+"_deph_"+str(options.treeDeph)+"_mcw_"+str(options.mcw)+"_lr_0o0"+str(int(options.lr*100))

channel=options.channel+"_HH"
print (startTime)

if "bb2l" in channel   :
    execfile("../cards/info_bb2l_HH.py")
    masses = [400,300,750]


import shutil,subprocess
proc=subprocess.Popen(['mkdir '+channel],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()

weights="totalWeight"
target='target'

data=load_data_2017_HH(inputPath,channelInTree,trainVars(True),[],bdtType)
data.dropna(subset=[weights],inplace = True)
data.fillna(0)

### Plot histograms of training variables

hist_params = {'normed': True, 'histtype': 'bar', 'fill': False , 'lw':5}
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

printmin=False
plotResiduals=False
plotAll=False
nbins=8
colorFast='g'
colorFastT='b'
BDTvariables=trainVars(plotAll)
make_plots(BDTvariables, nbins,
    data.ix[data.target.values == 0],labelBKG, colorFast,
    data.ix[data.target.values == 1],'Signal', colorFastT,
    channel+"/"+bdtType+"_"+trainvar+"_Variables_BDT.pdf",
    printmin,
    plotResiduals
    )

print ("separate datasets odd/even")
data_even = data.loc[(data["event"].values % 2 == 0) ]
data_odd = data.loc[~(data["event"].values % 2 == 0) ]

print ("balance datasets by even/odd chunck")
for data_do in [data_odd, data_even] :
    if 'evtLevelSUM' in bdtType :
        data_do.loc[(data_do['key'].isin(['TTToHadronic_PSweights', 'TTToSemiLeptonic_PSweights', 'TTTo2L2Nu_PSweights'])), [weights]]*=TTdatacard/data_do.loc[(data_do['key'].isin(['TTToHadronic_PSweights', 'TTToSemiLeptonic_PSweights', 'TTTo2L2Nu_PSweights'])), [weights]].sum()
        data_do.loc[(data_do['key']=='DY'), [weights]]*=DYdatacard/data_do.loc[(data_do['key']=='DY'), [weights]].sum()
        if "evtLevelSUM_HH_bb1l_res" in bdtType :
            data_do.loc[(data_do['key']=='W'), [weights]]*=Wdatacard/data_do.loc[(data_do['key']=='W')].sum()
        if "evtLevelSUM_HH_res" in bdtType :
            data_do.loc[(data_do['key'].isin(['TTToHadronic', 'TTToSemileptonic', 'TTTo2L2Nu'])), [weights]]*=TTdatacard/data_do.loc[(data_do['key'].isin(['TTToHadronic', 'TTToSemileptonic', 'TTTo2L2Nu'])), [weights]]/sum()
            data_do.loc[(data_do['key'].isin(['TTWJets', 'TTZJets'])), [weights]]*=TTVdatacard/data_do.loc[(data_do['key'].isin(['TTWJets', 'TTZJets'])), [weights]].sum()
            data_do.loc[(data_do['key']=='DY'), [weights]]*=DYdatacard/data_do.loc[(data_do['key']=='DY'), [weights]].sum()
            data_do.loc[(data_do['key']=='VH'), [weights]]*=VHdatacard/data_do.loc[(data_do['key']=='VH'), [weights]].sum()
            data_do.loc[(data_do['key']=='TTH'), [weights]]*=TTHdatacard/data_do.loc[(data_do['key']=='TTH'), [weights]].sum()
            data_do.loc[(data_do['key'].isin(['WW','WZ','ZZ'])), [weights]]*=VVdatacard/data_do.loc[(data_do['key'].isin(['WW','WZ','ZZ']))]
            data_do.loc[data_do[target]==1, [weights]] *= 100000./data_do.loc[data_do[target]==1][weights].sum()
        for mass in range(len(masses)) :
            data_do.loc[(data_do[target]==1) & (data_do["gen_mHH"] == masses[mass]),[weights]] *= 100000./data_do.loc[(data_do[target]==1) & (data_do["gen_mHH"]== masses[mass]),[weights]].sum()
            data_do.loc[(data_do[target]==0) & (data_do["gen_mHH"] == masses[mass]),[weights]] *= 100000./data_do.loc[(data_do[target]==0) & (data_do["gen_mHH"]== masses[mass]),[weights]].sum()
    else :
        data_do.loc[data_do['target']==0, [weights]] *= 100000/data_do.loc[data_do['target']==0][weights].sum()
        data_do.loc[data_do['target']==1, [weights]] *= 100000/data_do.loc[data_do['target']==1][weights].sum()


order_train = [data_odd, data_even]
order_train_name = ["odd","even"]

for dd, data_do in  enumerate(order_train):
    cls = xgb.XGBClassifier(
    			n_estimators = options.ntrees,
    			max_depth = options.treeDeph,
    			min_child_weight = options.mcw,
    			learning_rate = options.lr,
    			)

    cls.fit(
    	data_do[trainVars(False)].values,
    	data_do.target.astype(np.bool),
    	sample_weight=(data_do[weights].astype(np.float64))
    	)
    if dd == 0 : val_data = 1
    else : val_data = 0

    print ("XGBoost trained", order_train_name[dd])
    if options.doXML==True :
        pklpath=channel+"/"+channel+"_XGB_"+trainvar+"_"+bdtType+"_"+str(len(trainVars(False)))+"Var_"+order_train_name[dd]
        print ("Date: ", time.asctime( time.localtime(time.time()) ))
        pickle.dump(cls, open(pklpath+".pkl", 'wb'))
        file = open(pklpath+"pkl.log","w")
        file.write(str(trainVars(False))+"\n")
        file.close()
        print ("saved ",pklpath+".pkl")
        print ("variables are: ",pklpath+"_pkl.log")

    proba = cls.predict_proba(data_do[trainVars(False)].values )
    fpr, tpr, thresholds = roc_curve(
        data_do[target], proba[:,1],
        sample_weight=(data_do[weights].astype(np.float64))
    )
    train_auc = auc(fpr, tpr, reorder = True)
    print("XGBoost train set auc - {}".format(train_auc))

    proba = cls.predict_proba(order_train[val_data][trainVars(False)].values )
    fprt, tprt, thresholds = roc_curve(
        order_train[val_data][target], proba[:,1],
        sample_weight=(order_train[val_data][weights].astype(np.float64))
    )
    test_auct = auc(fprt, tprt, reorder = True)
    print("XGBoost test set auc - {}".format(test_auct))

    # overall ROC
    fig, ax = plt.subplots(figsize=(6, 6))
    train_auc = auc(fpr, tpr, reorder = True)
    ax.plot(fpr, tpr, lw=1, label='XGB train (area = %0.3f)'%(train_auc))
    ax.plot(fprt, tprt, lw=1, label='XGB test (area = %0.3f)'%(test_auct))
    ax.set_ylim([0.0,1.0])
    ax.set_xlim([0.0,1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    ax.grid()
    fig.savefig("{}/{}_{}_{}_{}_{}_roc.png".format(channel,bdtType,trainvar,str(len(trainVars(False))),hyppar,order_train_name[dd]))
    fig.savefig("{}/{}_{}_{}_{}_{}_roc.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False))),hyppar, order_train_name[dd]))

    ## feature importance plot
    fig, ax = plt.subplots()
    f_score_dict =cls.booster().get_fscore()
    fig, ax = plt.subplots()
    f_score_dict =cls.booster().get_fscore()
    f_score_dict = {trainVars(False)[int(k[1:])] : v for k,v in f_score_dict.items()}
    feat_imp = pandas.Series(f_score_dict).sort_values(ascending=True)
    feat_imp.plot(kind='barh', title='Feature Importances')
    fig.tight_layout()
    fig.savefig("{}/{}_{}_{}_{}_{}_XGB_importance.png".format(channel,bdtType,trainvar,str(len(trainVars(False))),hyppar, order_train_name[dd]))
    fig.savefig("{}/{}_{}_{}_{}_{}_XGB_importance.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False))),hyppar, order_train_name[dd]))

    ## classifier plot
    hist_params = {'normed': True, 'bins': 10 , 'histtype':'step', "lw": 2}
    plt.clf()
    y_pred = cls.predict_proba(order_train[val_data].ix[order_train[val_data].target.values == 0, trainVars(False)].values)[:, 1]
    y_predS = cls.predict_proba(order_train[val_data].ix[order_train[val_data].target.values == 1, trainVars(False)].values)[:, 1]
    y_pred_train = cls.predict_proba(data_do.ix[data_do.target.values == 0, trainVars(False)].values)[:, 1]
    y_predS_train = cls.predict_proba(data_do.ix[data_do.target.values == 1, trainVars(False)].values)[:, 1]
    fig = plt.figure('XGB',figsize=(6, 6))
    plt.hist(y_pred, ls="-", color = "g" , label=("test %s" % labelBKG), **hist_params)
    plt.hist(y_predS, ls="-", color = "b" , label="test signal", **hist_params)
    plt.hist(y_pred_train, ls="--", color = "g" , label="train", **hist_params)
    plt.hist(y_predS_train, ls="--", color = "b" , **hist_params )
    #plt.xscale('log')
    #plt.yscale('log')
    plt.legend(loc='best')
    plt.savefig(channel+'/'+bdtType+'_'+trainvar+'_'+str(len(trainVars(False)))+'_'+hyppar+'_'+order_train_name[dd]+'_XGBclassifier.pdf')
    plt.savefig(channel+'/'+bdtType+'_'+trainvar+'_'+str(len(trainVars(False)))+'_'+hyppar+'_'+order_train_name[dd]+'_XGBclassifier.png')

## the correlation matrix we do with all the data
if options.HypOpt==False :
	for ii in [1,2] :
		if ii == 1 :
			datad=data.loc[data[target].values == 1]
			label="signal"
		else :
			datad=data.loc[data[target].values == 0]
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


process = psutil.Process(os.getpid())
print(process.memory_info().rss)
print(datetime.now() - startTime)
