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
parser.add_option("--ntrees", type="int", dest="ntrees", help="hyp", default=1000)
parser.add_option("--treeDeph", type="int", dest="treeDeph", help="hyp", default=2)
parser.add_option("--lr", type="float", dest="lr", help="hyp", default=0.01)
parser.add_option("--mcw", type="int", dest="mcw", help="hyp", default=1)
parser.add_option("--tauID", type="string", dest="tauID", help="sample to pick training events", default='"dR03mvaVLoose"')
(options, args) = parser.parse_args()

tauID=options.tauID
doPlots=options.doPlots
bdtType=options.bdtType
trainvar=options.variables
hyppar=str(options.variables)+"_ntrees_"+str(options.ntrees)+"_deph_"+str(options.treeDeph)+"_mcw_"+str(options.mcw)+"_lr_0o0"+str(int(options.lr*100))

channel=options.channel+"_HH"
print (startTime)

if "bb2l" in channel       : execfile("../cards/info_bb2l_HH.py")
if channel == '2l_2tau_HH' : execfile("../cards/info_2l_2tau_HH.py")

import shutil,subprocess
proc=subprocess.Popen(['mkdir '+channel],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()

weights="totalWeight"
target='target'

output = read_from(tauID)
data=load_data_2017(
    output["inputPath"],
    output["channelInTree"],
    trainVars(True),
    [],
    bdtType,
    channel,
    output["keys"],
    output["masses"],
)
data.dropna(subset=[weights],inplace = True)
data.fillna(0)

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
elif 'evtLevelSUM_HH_res' in bdtType :
        labelBKG = "TT+DY+VV"
print "labelBKG: ",labelBKG

printmin=True
plotResiduals=False
plotAll=False
nbins=15
colorFast='g'
colorFastT='b'
BDTvariables=trainVars(plotAll)
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
for data_do in order_train :
    if 'SUM_HH' in bdtType :
        ttbar_samples = ['TTToHadronic_PSweights', 'TTToSemiLeptonic_PSweights', 'TTTo2L2Nu_PSweights']
        data_do.loc[(data_do['key'].isin(ttbar_samples)), [weights]]              *= output["TTdatacard"]/data_do.loc[(data_do['key'].isin(ttbar_samples)), [weights]].sum()
        data_do.loc[(data_do['key']=='DY'), [weights]]                            *= output["DYdatacard"]/data_do.loc[(data_do['key']=='DY'), [weights]].sum()
        if "evtLevelSUM_HH_bb1l_res" in bdtType :
            data_do.loc[(data_do['key']=='W'), [weights]]                         *= Wdatacard/data_do.loc[(data_do['key']=='W')].sum()
        if "evtLevelSUM_HH_res" in bdtType :
            data_do.loc[(data_do['key'].isin(['TTWJets', 'TTZJets'])), [weights]] *= output["TTVdatacard"]/data_do.loc[(data_do['key'].isin(['TTWJets', 'TTZJets'])), [weights]].sum() # consider do separatelly
            data_do.loc[(data_do['key']=='VH'), [weights]]                        *= ["VHdatacard"]/data_do.loc[(data_do['key']=='VH'), [weights]].sum() # consider removing
            data_do.loc[(data_do['key']=='TTH'), [weights]]                       *= ["TTHdatacard"]/data_do.loc[(data_do['key']=='TTH'), [weights]].sum() # consider removing
            data_do.loc[(data_do['key'].isin(['WW','WZ','ZZ'])), [weights]]       *= ["VVdatacard"]/data_do.loc[(data_do['key'].isin(['WW','WZ','ZZ']))] # consider do separatelly
        for mass in range(len(output["masses"])) :
            data_do.loc[(data_do[target]==1) & (data_do["gen_mHH"].astype(np.int) == int(output["masses"][mass])),[weights]] *= 100000./data_do.loc[(data_do[target]==1) & (data_do["gen_mHH"]== output["masses"][mass]),[weights]].sum()
            data_do.loc[(data_do[target]==0) & (data_do["gen_mHH"].astype(np.int) == int(output["masses"][mass])),[weights]] *= 100000./data_do.loc[(data_do[target]==0) & (data_do["gen_mHH"]== output["masses"][mass]),[weights]].sum()
    else :
        data_do.loc[data_do['target']==0, [weights]] *= 100000/data_do.loc[data_do['target']==0][weights].sum()
        data_do.loc[data_do['target']==1, [weights]] *= 100000/data_do.loc[data_do['target']==1][weights].sum()

roc_test = []
roc_train = []
estimator = []
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
        data_do[target].astype(np.bool), proba[:,1],
        sample_weight=(data_do[weights].astype(np.float64))
    )
    train_auc = auc(fpr, tpr, reorder = True)
    roc_train = roc_train + [ { "fpr":fpr, "tpr":tpr, "train_auc":train_auc }]
    print("XGBoost train set auc - {}".format(train_auc))

    proba = cls.predict_proba(order_train[val_data][trainVars(False)].values )
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
    f_score_dict =cls.booster().get_fscore()
    fig, ax = plt.subplots()
    f_score_dict =cls.booster().get_fscore()
    f_score_dict = {trainVars(False)[int(k[1:])] : v for k,v in f_score_dict.items()}
    feat_imp = pandas.Series(f_score_dict).sort_values(ascending=True)
    feat_imp.plot(kind='barh', title='Feature Importances')
    fig.tight_layout()
    nameout = "{}/{}_{}_{}_{}_{}_XGB_importance.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False))),hyppar, order_train_name[dd])
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
nameout = "{}/{}_{}_{}_{}_roc.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False))),hyppar)
fig.savefig(nameout)
#fig.savefig(nameout.replace(".pdf", ".png"))

###############################
# by mass ROC
styleline = ['-', '--', '-.', ':']
colors_mass = ['m', 'b', 'k', 'r', 'g',  'y', 'c', ]
fig, ax = plt.subplots(figsize=(6, 6))
sl = 0
for mm, mass in enumerate(output["masses_test"]) :
    for dd, data_do in  enumerate(order_train) :
        if dd == 0 : val_data = 1
        else : val_data = 0
        proba = estimator[dd].predict_proba(
            data_do.loc[(data_do["gen_mHH"].astype(np.int) == int(mass)), trainVars(False)].values
        )
        fpr, tpr, thresholds = roc_curve(
            data_do.loc[(data_do["gen_mHH"].astype(np.int) == int(mass)), target].astype(np.bool), proba[:,1],
            sample_weight=(data_do.loc[(data_do["gen_mHH"].astype(np.int) == int(mass)), weights].astype(np.float64))
        )
        train_auc = auc(fpr, tpr, reorder = True)
        print("train set auc " + str(train_auc) + " (mass = " + str(mass) + ")")
        proba = estimator[dd].predict_proba(
            order_train[val_data].loc[(order_train[val_data]["gen_mHH"].astype(np.int) == int(mass)), trainVars(False)].values
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
nameout = "{}/{}_{}_{}_{}_roc_by_mass.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False))),hyppar)
fig.savefig(nameout)

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
            trainVars(False)].values
        )[:, 1]
        y_predS = estimator[dd].predict_proba(
            order_train[val_data].loc[(order_train[val_data].target.values == 1) & (order_train[val_data]["gen_mHH"].astype(np.int) == int(mass)),
            trainVars(False)].values
        )[:, 1]
        y_pred_train = estimator[dd].predict_proba(
            data_do.ix[(data_do.target.values == 0) & (data_do["gen_mHH"].astype(np.int) == int(mass)),
            trainVars(False)].values
        )[:, 1]
        y_predS_train = estimator[dd].predict_proba(
            data_do.ix[(data_do.target.values == 1) & (data_do["gen_mHH"].astype(np.int) == int(mass)),
            trainVars(False)].values
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
    nameout = channel+'/'+bdtType+'_'+trainvar+'_'+str(len(trainVars(False)))+'_'+hyppar+'_mass_'+ str(mass)+'_XGBclassifier.pdf'
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
        trainVars(False)].values
    )[:, 1]
    y_predS = estimator[dd].predict_proba(
        order_train[val_data].loc[(order_train[val_data].target.values == 1) & (order_train[val_data]["gen_mHH"].astype(np.int) == int(mass)),
        trainVars(False)].values
    )[:, 1]
    y_pred_train = estimator[dd].predict_proba(
        data_do.ix[(data_do.target.values == 0) & (data_do["gen_mHH"].astype(np.int) == int(mass)),
        trainVars(False)].values
    )[:, 1]
    y_predS_train = estimator[dd].predict_proba(
        data_do.ix[(data_do.target.values == 1) & (data_do["gen_mHH"].astype(np.int) == int(mass)),
        trainVars(False)].values
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
nameout = channel+'/'+bdtType+'_'+trainvar+'_'+str(len(trainVars(False)))+'_'+hyppar+'_AllMass_'+'_XGBclassifier.pdf'
fig.savefig(nameout)
#fig.savefig(nameout.replace(".pdf", ".png"))

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
		nameout = "{}/{}_{}_{}_corr_{}.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False))),label)
		plt.savefig(nameout)
		#plt.savefig(namesave.replace(".pdf",".png"))
		ax.clear()
process = psutil.Process(os.getpid())
print(process.memory_info().rss)
print(datetime.now() - startTime)


process = psutil.Process(os.getpid())
print(process.memory_info().rss)
print(datetime.now() - startTime)
