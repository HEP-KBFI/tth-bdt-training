# python map_2D_evtLevel_ttH.py --channel '2lss_1tau' --variables "oldVar" --nbins-start 5 --nbins-target 5
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--channel ", type="string", dest="channel", help="The ones whose variables implemented now are:\n   - 1l_2tau\n   - 2lss_1tau\n It will create a local folder and store the report*/xml", default='T')
parser.add_option("--variables", type="string", dest="variables", help="  Set of variables to use -- it shall be put by hand in the code", default=1000)

parser.add_option("--nbins-start ", type="int", dest="start", help="for the squared 2D histogram", default=20)
parser.add_option("--nbins-target", type="int", dest="target", help="hyp", default=5)

parser.add_option("--relaxedLepID", action="store_true", dest="relaxedLepID", help="Do with input data with relaxed lepID", default=False)

parser.add_option("--doBDT", action="store_true", dest="doBDT", help=" do a BDT from tt and ttV. If false does 2D binning", default=False)
parser.add_option("--BDTtype", type="string", dest="BDTtype", help="Variables to joint BDT", default="1B")
parser.add_option("--doXML", action="store_true", dest="doXML", help="Do save not write the xml file", default=False)
parser.add_option("--ntrees ", type="int", dest="ntrees", help="hyp", default=2000)
parser.add_option("--treeDeph", type="int", dest="treeDeph", help="hyp", default=2)
parser.add_option("--lr", type="float", dest="lr", help="hyp", default=0.01)
parser.add_option("--mcw", type="int", dest="mcw", help="hyp", default=1)


(options, args) = parser.parse_args()
nbins= options.start # 15
nbinsout= options.target #5
BDTvar=options.variables #"oldVar"

#channel="2lss_1tau"
channel=options.channel #"1l_2tau"
if channel=='1l_2tau':
	channelInTree='1l_2tau_OS_Tight'
	inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Dec-BDT-noMEM-LooseLepMedTau-TagT-fakeR/histograms/1l_2tau/forBDTtraining_OS/'#  - tight lepton, medium tau

if channel=='2lss_1tau':
	if options.relaxedLepID==True :
		channelInTree='2lss_1tau_lepSS_sumOS_Loose'
		#inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Dec13-BDT-noMEM-LooseLepMedTau-TagT-fakeR/histograms/2lss_1tau/forBDTtraining_SS_OS/' # with charge tag, 2017Dec13-BDT-noMEM-LooseLepMedTau-TagT-fakeR
		#inputPath='/hdfs/local/acaan/ttHAnalysis/2016/20172017Dec30-VHbb-wMEM-LooseLepMedTau/histograms/2lss_1tau/forBDTtraining_SS_OS/'
		inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Dec30-VHbb-wMEM-LooseLepMedTau/histograms/2lss_1tau/forBDTtraining_SS_OS/'
		criteria=['lep1_isTight', 'lep2_isTight','tau_isTight',"failsTightChargeCut"]
		# 2017Dec-BDT-withMEM-LooseLepMedTau == with oldTrain
		#
	else :
		channelInTree='2lss_1tau_lepSS_sumOS_Tight'
		inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Dec08-BDT-noMEM-tighLep/histograms/2lss_1tau/forBDTtraining_SS_OS/'
	#else :
	channelInTreeTight='2lss_1tau_lepSS_sumOS_Tight'
	inputPathTight='/hdfs/local/acaan/ttHAnalysis/2016/2017Dec08-BDT-noMEM-tighLep/histograms/2lss_1tau/forBDTtraining_SS_OS/' #  2017Dec08-BDT-noMEM-tighLep
	inputPathTightFS='/hdfs/local/acaan/ttHAnalysis/2016/2018Jan_BDT_fromVHbb_tightL_mediumTau/histograms/2lss_1tau/forBDTtraining_SS_OS/'

	#channelInTree='2lss_1tau_lepSS_sumOS_Fakeable_wFakeRateWeights'
	#inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Dec08-BDT-noMEM-fakableLepLooseTau/histograms/2lss_1tau/forBDTtraining_SS_OS/' # 2017Dec08-BDT-noMEM-fakableLepMedTau

from sklearn import svm
import sys , time
#import sklearn_to_tmva
from ROOT import TMVA
import sklearn
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
import pandas
import ROOT
#from pandas import HDFStore,DataFrame
import sklearn_to_tmva
import xgboost2tmva
import skTMVA
import matplotlib
matplotlib.use('agg')
#matplotlib.use('PS')   # generate postscript output by default
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from matplotlib import colors as colors
import math , array
import numpy as np
import seaborn as sns
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
ROOT.gStyle.SetOptStat(0)
from tqdm import trange
import glob

from keras.models import Sequential, model_from_json
import json

from collections import OrderedDict
execfile("../python/data_manager.py")


import shutil,subprocess
proc=subprocess.Popen(['mkdir '+options.channel],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()

Variables_all=[
'HadTop_eta', 'HadTop_pt',
'MT_met_lep1', 'avg_dr_jet', 'avg_dr_lep', 'dr_lep1_HTfitted', 'dr_lep1_HTunfitted', 'dr_lep1_tau',
'dr_lep2_HTfitted', 'dr_lep2_HTunfitted', 'dr_lep2_tau', 'dr_leps',
'dr_tau_HTfitted', 'dr_tau_HTunfitted', 'evtWeight', 'fitHTptoHTmass', 'fitHTptoHTpt',
'fittedHadTop_eta', 'fittedHadTop_pt', 'hadtruth', 'htmiss', 'lep1_conePt', 'lep1_eta',
'lep1_frWeight', 'lep1_genLepPt', 'lep1_pt', 'lep1_tth_mva', 'lep2_conePt', 'lep2_eta',
'lep2_frWeight', 'lep2_genLepPt', 'lep2_pt', 'lep2_tth_mva', 'mT_lep1',
'mT_lep2', 'mTauTauVis1', 'mTauTauVis2', 'mass_lep1_HTfitted',
'mass_lep2_HTfitted', 'max_lep_eta', 'mbb',
 'min(met_pt,400)', 'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet',
'mvaOutput_hadTopTaggerWithKinFit', 'nJet25_Recl',
'ptmiss', 'tau_eta',
'tau_mva', 'tau_pt', 'unfittedHadTop_pt',
 'lep1_isTight', 'lep2_isTight', 'nBJetLoose', 'nJet', 'nLep',  'tau_isTight',
 "mvaOutput_2lss_1tau_ttV",
 "mvaOutput_2lss_1tau_ttbar"
]


def trainVarsTT(trainvar):

        if trainvar=="noHTT" and channel=="2lss_1tau" :
			return [
		'avg_dr_jet', 'dr_lep1_tau', 'dr_lep2_tau',
		'dr_leps',
		'lep1_conePt', 'lep2_conePt',
		'mT_lep1',
		'mT_lep2', 'mTauTauVis1', 'mTauTauVis2', 'max_lep_eta',
		'mbb',
		'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet',
		'nJet25_Recl',
		'ptmiss', 'tau_pt',
		]

        if trainvar=="HTT" and channel=="2lss_1tau" :
			return [
			'dr_lep1_tau', 'dr_lep2_tau', 'dr_leps',
			'mT_lep1', 'mT_lep2', 'mTauTauVis1', 'mTauTauVis2', 'mindr_lep1_jet',
			'mindr_lep2_jet', 'mindr_tau_jet', 'ptmiss', 'tau_pt',
			'mvaOutput_hadTopTaggerWithKinFit', 'mvaOutput_Hj_tagger', 'unfittedHadTop_pt',
			'nJet25_Recl', 'avg_dr_jet'
			]


        if trainvar=="HTT_LepID" and channel=="2lss_1tau" :
			return [
		'mTauTauVis1', 'mTauTauVis2',
		'tau_pt',
		'mvaOutput_hadTopTaggerWithKinFit',
		'lep1_tth_mva',
		'lep2_tth_mva'
		]

        if trainvar=="oldVar"  and channel=="2lss_1tau" :return [
		"max_lep_eta",
		"nJet25_Recl",
		"mindr_lep1_jet",
		"mindr_lep2_jet",
		"min(met_pt,400)",
		"avg_dr_jet",
		"MT_met_lep1"
		]

        if trainvar=="oldTrainCSV"  and channel=="2lss_1tau" :return [
		"nJet","mindr_lep1_jet","avg_dr_jet",
		"TMath::Max(TMath::Abs(lep1_eta),TMath::Abs(lep2_eta))",
		"lep2_conePt","dr_leps","tau_pt","dr_lep1_tau"
		]

        if trainvar=="oldTrainN"  and channel=="2lss_1tau" :return [
		"nJet","mindr_lep1_jet","avg_dr_jet",
		"max_lep_eta",
		"lep2_conePt","dr_leps","tau_pt","dr_lep1_tau"
		]

        if trainvar=="oldVarA"  and channel=="2lss_1tau"  :return [
 		"nJet","mindr_lep1_jet","avg_dr_jet",
 		"max_lep_eta",
 		"lep2_conePt","dr_leps","tau_pt","dr_lep1_tau"
 		]

        if trainvar=="HTTMEM" and channel=="2lss_1tau" :
			return [
			'dr_lep1_tau',"memOutput_LR", 'dr_lep2_tau', 'dr_leps',
			'mT_lep1', 'mT_lep2', 'mTauTauVis1', 'mTauTauVis2', 'mindr_lep1_jet',
			'mindr_lep2_jet', 'mindr_tau_jet', 'ptmiss', 'tau_pt',
			'mvaOutput_hadTopTaggerWithKinFit', 'mvaOutput_Hj_tagger', 'unfittedHadTop_pt',
			'nJet25_Recl', 'avg_dr_jet'
			]

def trainVarsTTV(trainvar):

        if trainvar=="oldVar"  and channel=="2lss_1tau" :return [
		"max_lep_eta",
		"MT_met_lep1",
		"nJet25_Recl",
		"mindr_lep1_jet",
		"mindr_lep2_jet",
		"lep1_conePt",
		"lep2_conePt"
		]

        if trainvar=="HTT_LepID" and channel=="2lss_1tau"  :
			return [
		'avg_dr_jet', 'dr_lep1_tau', 'dr_lep2_tau', 'dr_leps',
		'lep1_conePt', 'lep2_conePt',
		'mT_lep1', 'mT_lep2', 'mTauTauVis1', 'mTauTauVis2',
		'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet', 'ptmiss', 'tau_pt'
		]

			"""
		[
		'avg_dr_jet', 'dr_lep1_tau', 'dr_lep2_tau', 'dr_leps',
		'lep1_conePt', 'lep2_conePt', 'mT_lep1', 'mT_lep2',
		'mTauTauVis1', 'mTauTauVis2', 'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet',
		'ptmiss', 'tau_pt',
		'mvaOutput_hadTopTaggerWithKinFit', 'mvaOutput_Hj_tagger'
		]
			"""

        if trainvar=="HTT" and channel=="2lss_1tau" :
			return [
		'avg_dr_jet', 'dr_lep1_tau', 'dr_lep2_tau', 'dr_leps',
		'lep1_conePt', 'lep2_conePt',
		'mT_lep1', 'mT_lep2', 'mTauTauVis1', 'mTauTauVis2',
		'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet', 'ptmiss', 'tau_pt'
		]
			"""
		[
		'avg_dr_jet', 'dr_lep1_tau', 'dr_lep2_tau', 'dr_leps',
		'lep1_conePt', 'lep2_conePt', 'mT_lep1', 'mT_lep2',
		'mTauTauVis1', 'mTauTauVis2', 'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet',
		'ptmiss', 'tau_pt',
		'mvaOutput_hadTopTaggerWithKinFit', 'mvaOutput_Hj_tagger'
		]
			"""

        if trainvar=="noHTT" and channel=="2lss_1tau":
			return [
		'avg_dr_jet', 'dr_lep1_tau', 'dr_lep2_tau', 'dr_leps',
		'lep1_conePt', 'lep2_conePt',
		'mT_lep1', 'mT_lep2', 'mTauTauVis1', 'mTauTauVis2',
		'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet', 'ptmiss', 'tau_pt'
		]

        if trainvar=="oldTrainCSV"  and channel=="2lss_1tau" :return [
		"mindr_lep1_jet","mindr_lep2_jet", "avg_dr_jet", "TMath::Max(TMath::Abs(lep1_eta),TMath::Abs(lep2_eta))",
		"lep1_conePt", "lep2_conePt", "mT_lep1", "dr_leps", "mTauTauVis1", "mTauTauVis2"
		]

        if trainvar=="oldTrainN"  and channel=="2lss_1tau" :return [
		"mindr_lep1_jet","mindr_lep2_jet", "avg_dr_jet", "max_lep_eta",
		"lep1_conePt", "lep2_conePt", "mT_lep1", "dr_leps", "mTauTauVis1", "mTauTauVis2"
		]

        if trainvar=="oldVarA"  and channel=="2lss_1tau" :return [
		"mindr_lep1_jet","mindr_lep2_jet", "avg_dr_jet", "max_lep_eta",
		"lep1_conePt", "lep2_conePt", "mT_lep1", "dr_leps", "mTauTauVis1", "mTauTauVis2"
		]

        if trainvar=="HTTMEM" and channel=="2lss_1tau"  :
			return [
			'avg_dr_jet', 'dr_lep1_tau', 'dr_lep2_tau', 'dr_leps',
			'lep1_conePt', 'lep2_conePt', 'mT_lep1', 'mT_lep2',
			'mTauTauVis1', 'mTauTauVis2', 'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet',
			'ptmiss', 'tau_pt',
			'mvaOutput_hadTopTaggerWithKinFit', 'mvaOutput_Hj_tagger', 'mvaOutput_Hjj_tagger',"memOutput_LR"
			]


####################################################################################################
## Load data
#my_cols_list=Variables_all+['key','target','file']
testtruth="bWj1Wj2_isGenMatchedWithKinFit"
weights="totalWeight"
#keys=['ttHToNonbb','TTTo2L2Nu','TTToSemilepton','TTZToLLNuNu','TTWJetsToLNu']


doCSVfile=False
loadTight=True
doInFS=False
if options.variables!="oldTrainCSV" or doCSVfile : #options.variables!="oldTrainCSV"
	#
	dataTightFS=load_data_fullsim(inputPathTightFS,channelInTreeTight,Variables_all,[],testtruth,"all")
	if doInFS : data= dataTightFS
	elif options.relaxedLepID==True : data=load_data(inputPath,channelInTree,Variables_all,[],testtruth,"all")
	else : data=load_data(inputPathTight,channelInTreeTight,Variables_all,[],testtruth,"all") #
	if doCSVfile : data=load_data_xml(data)
	#if loadTight : dataTight=load_data(inputPathTight,channelInTreeTight,Variables_all,[],testtruth,"all")

	#elif options.variables=="oldTrain":
	#	data=load_data(inputPath,channelInTree,Variables_all,[],testtruth,"arun")
else :
	data = pandas.read_csv('arun_xml_2lss_1tau/arun_xml_2lss_1tau_FromAnalysis.csv')
	keys=['ttHToNonbb','TTZToLLNuNu','TTWJetsToLNu','TTTo2L2Nu','TTToSemilepton']
	for folderName in keys :
		nS = len(data.ix[(data.target.values == 0) & (data.key.values==folderName)])
		nB = len(data.ix[(data.target.values == 1) & (data.key.values==folderName)])
		print folderName,"length of sig, bkg: ", nS, nB
#########################################################################################
## Load the BDTs - do 2D plot
if channel=="2lss_1tau" :
	basepkl="/hdfs/local/acaan/ttHAnalysis/2016/2lss_1tau_opt1"
	if BDTvar=="oldVar" : ttV_file=basepkl+"/2lss_1tau_XGB_oldVar_evtLevelTTV_TTH_7Var.pkl"
	if BDTvar=="oldVarA" : ttV_file=basepkl+"/2lss_1tau_XGB_oldVarA_evtLevelTTV_TTH_10Var.pkl"
	#if BDTvar=="HTT_LepID" : ttV_file=basepkl+"/2lss_1tau_XGB_HTT_evtLevelTTV_TTH_17Var.pkl"
	#if BDTvar=="HTT" : ttV_file=basepkl+"/2lss_1tau_XGB_HTT_evtLevelTTV_TTH_17Var.pkl"
	if BDTvar=="HTT_LepID" : ttV_file=basepkl+"/2lss_1tau_XGB_noHTT_evtLevelTTV_TTH_15Var.pkl"
	if BDTvar=="HTT" : ttV_file=basepkl+"/2lss_1tau_XGB_noHTT_evtLevelTTV_TTH_15Var.pkl"
	if BDTvar=="noHTT" : ttV_file=basepkl+"/2lss_1tau_XGB_noHTT_evtLevelTTV_TTH_15Var.pkl"
	if BDTvar=="HTTMEM" : ttV_file=basepkl+"/2lss_1tau_XGB_HTTMEM_evtLevelTTV_TTH_19Var.pkl"
	# 2lss_1tau/      2lss_1tau/
	#if BDTvar=="oldTrain" : ttV_file="" # ROOT.TMVA.Tools.Instance()

	if BDTvar=="oldVar" : tt_file=basepkl+"/2lss_1tau_XGB_oldVar_evtLevelTT_TTH_7Var.pkl"
	if BDTvar=="oldVarA" : tt_file=basepkl+"/2lss_1tau_XGB_oldVarA_evtLevelTT_TTH_8Var.pkl"
	if BDTvar=="HTT_LepID" : tt_file=basepkl+"/2lss_1tau_XGB_HTT_LepID_evtLevelTT_TTH_6Var.pkl"
	if BDTvar=="HTT" : tt_file=basepkl+"/2lss_1tau_XGB_HTT_evtLevelTT_TTH_17Var.pkl"
	if BDTvar=="noHTT" : tt_file=basepkl+"/2lss_1tau_XGB_noHTT_evtLevelTT_TTH_18Var.pkl"
	if BDTvar=="HTTMEM" : tt_file=basepkl+"/2lss_1tau_XGB_HTTMEM_evtLevelTT_TTH_18Var.pkl"
	#if BDTvar=="oldTrain" : tt_file=basepkl+"/2lss_1tau_XGB_noHTT_evtLevelTT_TTH_18Var.pkl"

if doInFS :
	dataTT =dataTightFS.ix[(dataTightFS.proces.values=='TT')]
	#print ("dataTT",len(dataTT), dataTT.target.values)
	dataTTV=dataTightFS.ix[(dataTightFS.proces.values=='TTZ') | (dataTightFS.key.values=='TTW')]
	#print ("dataTTV",len(dataTTV), dataTTV.target.values)
	dataTTH=dataTightFS.ix[(dataTightFS.proces.values=='signal')]
	#print ("dataTTH",len(dataTTH), dataTTH.target.values)
elif options.variables!="oldTrainCSV" :
	dataTT =data.ix[(data.key.values=='TTTo2L2Nu') | (data.key.values=='TTToSemilepton')]
	dataTTV=data.ix[(data.key.values=='TTZToLLNuNu') | (data.key.values=='TTWJetsToLNu')]
	dataTTH=data.ix[(data.key.values=='ttHToNonbb')]
else :
	dataTT =data.ix[((data.key.values=='TTTo2L2Nu') | (data.key.values=='TTToSemilepton'))]
	print ("lenght TT", len(dataTT))
	dataTT=dataTT.ix[ (dataTT.oldTrainTMVA_tt.values<=1)  & (dataTT.oldTrainTMVA_tt.values>-1) & (dataTT.oldTrainTMVA_ttV.values<=1)  & (dataTT.oldTrainTMVA_ttV.values>-1)]
	print ("not rejected TT", len(dataTT))
	dataTTV=data.ix[((data.key.values=='TTZToLLNuNu') | (data.key.values=='TTWJetsToLNu')) ]
	print ("lenght TTV", len(dataTTV))
	dataTTV=dataTTV.ix[ (dataTTV.oldTrainTMVA_tt.values<=1)  & (dataTTV.oldTrainTMVA_tt.values>-1) & (dataTTV.oldTrainTMVA_ttV.values<=1)  & (dataTTV.oldTrainTMVA_ttV.values>-1)]
	print ("not rejected TTV", len(dataTTV))
	dataTTH=data.ix[(data.key.values=='ttHToNonbb')] # data.oldTrainTMVA_tt.values<=1  & data.oldTrainTMVA_tt.values>-1
	print ("lenght TTH", len(dataTTH))
	dataTTH=dataTTH.ix[(dataTTH.oldTrainTMVA_tt.values<=1)  & (dataTTH.oldTrainTMVA_tt.values>-1) & (dataTTH.oldTrainTMVA_ttV.values<=1)  & (dataTTH.oldTrainTMVA_ttV.values>-1)]
	print ("not rejected TTH", len(dataTTH))
	print dataTTV[["oldTrainTMVA_tt","oldTrainTMVA_ttV"]]

bdtmax=1.0
if "oldTrain" in BDTvar : bdtmin=-1.
else : bdtmin=0.
factor=1.-bdtmin #2. | 1.


if BDTvar=="oldTrainCSV" :
	ttBDT="oldTrainTMVA_tt"
	ttVBDT="oldTrainTMVA_ttV"
elif BDTvar=="oldTrain" :
	ttBDT="mvaOutput_2lss_ttbar"
	ttVBDT="mvaOutput_2lss_ttV"

if "oldTrain" in BDTvar :
	tt_datainTT =dataTT[ttBDT].values
	ttV_datainTT=dataTTV[ttBDT].values
	ttH_datainTT=dataTTH[ttBDT].values
	#
	tt_target =dataTT["target"].values
	ttV_target =dataTT["target"].values
	ttH_target =dataTT["target"].values
	#
	tt_datainTTV =dataTT[ttVBDT].values
	ttV_datainTTV=dataTTV[ttVBDT].values
	ttH_datainTTV=dataTTH[ttVBDT].values
	#print (tt_datainTT,tt_datainTTV)
	#testROC=np.concatenate(tt_datainTT,ttH_datainTT)
	fpr, tpr, thresholds = roc_curve(dataTT["target"].append(dataTTH["target"]), dataTT[ttBDT].append(dataTTH[ttBDT]) )
	print ("ROC TT data in TT bdt",BDTvar,auc(fpr, tpr, reorder = True))
	#testROC=np.concatenate(ttV_datainTTV,ttH_datainTT)
	fpr, tpr, thresholds = roc_curve(dataTTV["target"].append(dataTTH["target"]), dataTTV[ttBDT].append(dataTTH[ttBDT]) )
	print ("ROC TTV data in TTV in bdt",BDTvar,auc(fpr, tpr, reorder = True))
	fpr, tpr, thresholds = roc_curve(dataTTV["target"].append(dataTTH["target"].append(dataTT["target"])), dataTTV[ttBDT].append(dataTTH[ttBDT]).append(dataTT[ttBDT] ))
	print ("ROC all data in TTV in bdt",BDTvar,auc(fpr, tpr, reorder = True))
	fpr, tpr, thresholds = roc_curve(dataTTV["target"].append(dataTTH["target"].append(dataTT["target"])), dataTTV[ttVBDT].append(dataTTH[ttVBDT]).append(dataTT[ttVBDT] ) )
	print ("ROC all data in TT in bdt",BDTvar,auc(fpr, tpr, reorder = True))
else :
	clsTT=pickle.load(open(tt_file,'rb'))
	#dataTT =data.ix[(data.key.values=='TTTo2L2Nu') | (data.key.values=='TTToSemilepton')]
	#dataTTV=data.ix[(data.key.values=='TTZToLLNuNu') | (data.key.values=='TTWJetsToLNu')]
	#dataTTH=data.ix[(data.key.values=='ttHToNonbb')]
	####
	tt_datainTT =clsTT.predict_proba(dataTT[trainVarsTT(BDTvar)].values)[:, 1]
	ttV_datainTT=clsTT.predict_proba(dataTTV[trainVarsTT(BDTvar)].values)[:, 1]
	ttH_datainTT=clsTT.predict_proba(dataTTH[trainVarsTT(BDTvar)].values)[:, 1]
	###
	#dataTT =data.ix[(data.key.values=='TTTo2L2Nu') | (data.key.values=='TTToSemilepton')]
	#dataTTV=data.ix[(data.key.values=='TTZToLLNuNu') | (data.key.values=='TTWJetsToLNu')]
	#dataTTH=data.ix[(data.key.values=='ttHToNonbb')]
	clsTTV=pickle.load(open(ttV_file,'rb'))
	tt_datainTTV =clsTTV.predict_proba(dataTT[trainVarsTTV(BDTvar)].values)[:, 1]
	ttV_datainTTV=clsTTV.predict_proba(dataTTV[trainVarsTTV(BDTvar)].values)[:, 1]
	ttH_datainTTV=clsTTV.predict_proba(dataTTH[trainVarsTTV(BDTvar)].values)[:, 1]

	testTT=dataTT.append(dataTTH)
	testTTV=dataTTV.append(dataTTH)
	testAll=testTTV.append(dataTT)
	print (len(testAll), testAll.target.values)
	testROC=clsTT.predict_proba(testTT[trainVarsTT(BDTvar)].values)
	fpr, tpr, thresholds = roc_curve(testTT["target"].values, testROC[:,1] )
	print ("ROC TT BDT in TT sample",BDTvar,auc(fpr, tpr, reorder = True))
	testROC=clsTT.predict_proba(testTTV[trainVarsTT(BDTvar)].values)
	fpr, tpr, thresholds = roc_curve(testTTV["target"].values, testROC[:,1] )
	print ("ROC TT BDT in TTV sample",BDTvar,auc(fpr, tpr, reorder = True))
	testROC=clsTT.predict_proba(testAll[trainVarsTT(BDTvar)].values)
	fpr, tpr, thresholds = roc_curve(testAll["target"].values, testROC[:,1] )
	print ("ROC TT BDT in all",BDTvar,auc(fpr, tpr, reorder = True))
	testROCttV=clsTTV.predict_proba(testTT[trainVarsTTV(BDTvar)].values)
	fpr, tpr, thresholds = roc_curve(testTT["target"].values, testROCttV[:,1])
	print ("ROC TTV BDT in TT sample",BDTvar,auc(fpr, tpr, reorder = True))
	testROCttV=clsTTV.predict_proba(testTTV[trainVarsTTV(BDTvar)].values)
	fpr, tpr, thresholds = roc_curve(testTTV["target"].values, testROCttV[:,1])
	print ("ROC TTV BDT in TTV",BDTvar,auc(fpr, tpr, reorder = True))
	testROCttV=clsTTV.predict_proba(testAll[trainVarsTTV(BDTvar)].values)
	fpr, tpr, thresholds = roc_curve(testAll["target"].values, testROCttV[:,1])
	print ("ROC TTV BDT in all",BDTvar,auc(fpr, tpr, reorder = True))
##############
plt.figure(figsize=(20, 3))
fig, ax = plt.subplots(figsize=(18, 6))
plt.subplot(1, 3, 0+1)
histtt, xbins, ybins, im  = plt.hist2d(tt_datainTT, tt_datainTTV,
									weights=dataTT["totalWeight"].values.astype(np.float64),
									bins=nbins,
									cmap=reverse_colourmap(cm.hot))
plt.xlabel('BDT for tt')
plt.ylabel('BDT for ttV')
plt.text(0.05+bdtmin, 0.92, "tt sample",  fontweight='bold')
plt.text((0.05+bdtmin), 0.8, BDTvar,  fontweight='bold')
plt.axis((bdtmin,bdtmax,bdtmin,bdtmax))
plt.colorbar()
plt.subplot(1, 3, 1+1)
histttV, xbins, ybins, im  = plt.hist2d(ttV_datainTT, ttV_datainTTV,
										weights=dataTTV["totalWeight"].values.astype(np.float64),
										bins=nbins,
										cmap=reverse_colourmap(cm.hot))
plt.xlabel('BDT for tt')
plt.ylabel('BDT for ttV')
plt.axis((bdtmin,bdtmax,bdtmin,bdtmax))
plt.text(0.05+bdtmin, 0.92, "ttV sample",  fontweight='bold')
plt.text((0.05+bdtmin), 0.80, BDTvar,  fontweight='bold')
plt.colorbar()
plt.subplot(1, 3, 2+1)
histttH, xbins, ybins, im  = plt.hist2d(ttH_datainTT,ttH_datainTTV,
					weights=dataTTH["totalWeight"].values.astype(np.float64),
					bins=nbins,
					cmap=reverse_colourmap(cm.hot)
					)
#plt.subplot(1, 3, 2+1)
#histttH, xbins, ybins, im = sns.kdeplot(ttH_datainTT,ttH_datainTTV, shade=True)

plt.xlabel('BDT for tt')
plt.ylabel('BDT for ttV')
plt.axis((bdtmin,1.,bdtmin,1.))
plt.text(0.05+bdtmin, 0.92, "ttH sample",  fontweight='bold')
plt.text((0.05+bdtmin), 0.8, BDTvar,  fontweight='bold')
plt.colorbar()
plt.savefig(channel+"/"+BDTvar+"_2D_"+str(nbins)+"bins.pdf")
plt.clf()
###########################################################
plt.figure(figsize=(20, 3))
fig, ax = plt.subplots(figsize=(18, 6))

plt.subplot(1, 3, 0+1)
plt.hist(tt_datainTT, weights=dataTT["totalWeight"].values.astype(np.float64),
									bins=nbins,normed=True,alpha=0.4,label='TT bdt')
plt.hist(tt_datainTTV, weights=dataTT["totalWeight"].values.astype(np.float64),
									bins=nbins,normed=True,alpha=0.4,label='TTV bdt')
plt.legend(loc='best')
plt.ylabel('discriminant')
plt.text(0.05+bdtmin, 0.92, "tt sample",  fontweight='bold')
plt.text((0.05+bdtmin), 0.8, BDTvar,  fontweight='bold')

plt.subplot(1, 3, 1+1)
plt.hist(ttV_datainTT, weights=dataTTV["totalWeight"].values.astype(np.float64),
									bins=nbins,normed=True,alpha=0.4,label='TT bdt')
plt.hist(ttV_datainTTV, weights=dataTTV["totalWeight"].values.astype(np.float64),
									bins=nbins,normed=True,alpha=0.4,label='TTV bdt')
plt.legend(loc='best')
plt.ylabel('discriminant')
plt.text(0.05+bdtmin, 0.92, "ttV sample",  fontweight='bold')
plt.text((0.05+bdtmin), 0.80, BDTvar,  fontweight='bold')

plt.subplot(1, 3, 2+1)
plt.hist(ttH_datainTT, weights=dataTTH["totalWeight"].values.astype(np.float64),
									bins=nbins,normed=True,alpha=0.4,label='TT bdt')
plt.hist(ttH_datainTTV, weights=dataTTH["totalWeight"].values.astype(np.float64),
									bins=nbins,normed=True,alpha=0.4,label='TTV bdt')
plt.legend(loc='best')
plt.ylabel('discriminant')
plt.text(0.05+bdtmin, 0.92, "ttH sample",  fontweight='bold')
plt.text((0.05+bdtmin), 0.8, BDTvar,  fontweight='bold')

plt.savefig(channel+"/"+BDTvar+"_2times1D_"+str(nbins)+"bins.pdf")
plt.clf()
if options.doBDT == False :
	#################################
	### not doing: Smooth(1,"k5b");
	#################################
	## 1D simply linearized
	fig, ax = plt.subplots(figsize=(18, 6))
	hist1DttH = []
	hist1DttV = []
	hist1Dtt = []
	for i in range(0,len(ybins)-1):
		for j in range(0,len(xbins)-1):
			hist1DttH.append(max(1e-5,histttH[i][j]+histttV[i][j]+histtt[i][j]))
			hist1DttV.append(max(1e-5,histttV[i][j]+histtt[i][j]))
			hist1Dtt.append( max(1e-5,histtt[i][j]))
	xaxis = np.arange(1, len(hist1DttH)+1, 1)
	#print ("size 1D",len(hist1DttH),len(xaxis) ,hist1DttH[0])
	plt.text(2, 18, BDTvar+" (stacked plots)",  fontweight='bold')
	plt.step(xaxis,hist1DttH, label="ttH", lw=3, color='r')
	plotttttV=plt.step(xaxis,hist1DttV, label="ttV",  color= 'g', lw=3)
	plottt=plt.step(xaxis,hist1Dtt, label="tt", color= 'b', lw=3)
	#plt.fill_between(plotttttV[0].get_xdata(orig=False), plotttttV[0].get_ydata(orig=False),
	#plottt[0].get_ydata(orig=False),  color= 'g')
	#plt.fill_between(plottt[0].get_xdata(orig=False), 0, plottt[0].get_ydata(orig=False), color= 'b')
	plt.axis((0.,len(hist1DttH),-0.5,25.0))
	plt.legend(loc='upper right')
	plt.savefig(channel+"/"+BDTvar+"_1D_"+str(nbins)+"_relLepID"+str(options.relaxedLepID)+"bins.pdf")
	plt.clf()
	######################################################################
	## do cummulative
	#norm=(max((histtt+histttV).flatten()))/max(histttH.flatten())
	#histRatio= histttH/(histtt+histttV)
	#histRatio = np.nan_to_num(histRatio)
	#histRatio[histRatio<0]=0
	weightTTV=dataTTV["totalWeight"].values
	weightTT=dataTT["totalWeight"].values
	weightTTH=dataTTH["totalWeight"].values
	hBkg = ROOT.TH2F("hBkg","",nbins,bdtmin,1.,nbins,bdtmin,1.)
	for ii in range(0,len(ttV_datainTT)) : hBkg.Fill(ttV_datainTT[ii], ttV_datainTTV[ii],weightTTV[ii])
	for ii in range(0,len(tt_datainTT)) : hBkg.Fill(tt_datainTT[ii], tt_datainTTV[ii],weightTT[ii])
	hSig = ROOT.TH2F("hSig","",nbins,bdtmin,1.,nbins,bdtmin,1.)
	for ii in range(0,len(ttH_datainTT)) : hSig.Fill(ttH_datainTT[ii],ttH_datainTTV[ii],weightTTH[ii])
	#hSig.Scale( 1. / hSig.GetMaximum());
	#hBkg.Scale( 1. / hBkg.GetMaximum());
	hBkg.Smooth(1,"k5b");
	hSig.Smooth(1,"k5b");
	"""
	for x in range(1,hSig.GetNbinsX() + 1):
		for y in range(1,hSig.GetNbinsY() + 1):
			if y>10 : print ("sig,bkg", hSig.GetYaxis().GetBinCenter(x),hSig.GetXaxis().GetBinCenter(y), hSig.GetBinContent(x,y),hBkg.GetBinContent(x,y))
			#if hSig.GetBinContent(x,y) ==0 and hBkg.GetBinContent(x,y) ==0:
			#	hSig.SetBinContent(x,y,0.0000001)
			if hBkg.GetBinContent(x,y) ==0 :
				hBkg.SetBinContent(x,y,0.0000001)
				hSig.SetBinContent(x,y,0.000000005)
	#"""

	####
	####
	hRtio = hSig.Clone("hRtio");
	hRtio.Divide(hBkg);
	totalSig=hSig.Integral()
	totalBkg=hBkg.Integral()
	#"""
	for x in range(1,hRtio.GetNbinsX()):
		for y in range(1,hRtio.GetNbinsY()):
			#if hRtio.GetYaxis().GetBinCenter(y) >0.7 : print ("ratio", hRtio.GetYaxis().GetBinCenter(y), hRtio.GetXaxis().GetBinCenter(x), hRtio.GetBinContent(x,y),hBkg.GetBinContent(x,y)/totalBkg, hSig.GetBinContent(x,y)/totalSig)
			if hSig.GetBinContent(x,y) < 0.000005*totalSig or hBkg.GetBinContent(x,y)< 0.000005*totalBkg: #  and
				hRtio.SetBinContent(x,y,0.0001)
				#print "changed"
			#hRtio.SetBinContent(x,y, hRtio.GetBinContent(x,y)-hRtio.GetBinContent(1,1))
	#"""
	cRatio = ROOT.TCanvas("cRatio","",1600,600);
	cRatio.Divide(3,1,0,0);
	cRatio.cd(1)
	ROOT.gPad.SetLogz()
	ROOT.gPad.SetRightMargin(0.01)
	hRtio.Draw("colz");
	cRatio.cd(2)
	ROOT.gPad.SetRightMargin(0.01)
	hSig.Draw("colz")
	cRatio.cd(3)
	ROOT.gPad.SetRightMargin(0.01)
	hBkg.Draw("colz")
	cRatio.SaveAs(channel+"/"+BDTvar+"_2D_"+str(nbins)+"bins_fromROOT.pdf")
	#############
	h = ROOT.TH1F("h","",1000,0,1.0)
	h.GetXaxis().SetTitle("Likelihood ratio [tth/(tt+ttV)]");
	h.GetYaxis().SetTitle("Cumulative Likelihood ratio");
	#yTT= tt_datainTT #clsTT.predict_proba(dataTT[trainVarsTT(BDTvar)].values)[:, 1]
	#xTT= tt_datainTTV #clsTTV.predict_proba(dataTT[trainVarsTTV(BDTvar)].values)[:, 1]
	#print tt_datainTT
	binxTT=np.trunc(nbins*(-bdtmin+tt_datainTTV)/factor)
	binyTT=np.trunc(nbins*(-bdtmin+tt_datainTT)/factor)
	#print (len(tt_datainTT),binxTT,binyTT)
	for ii in range(0,len(tt_datainTT)) :
		if int(binxTT[ii]) >= 0  and int(binxTT[ii]) < nbins :
			#h.Fill(histRatio[int(binxTT[ii])][int(binyTT[ii])])
			h.Fill(hRtio.GetBinContent(int(binxTT[ii]),int(binyTT[ii])))
		else : print ("binning went wrong",ii,int(binxTT[ii]),int(binyTT[ii]))
	#xTTV=ttV_datainTTV #clsTTV.predict_proba(dataTTV[trainVarsTTV(BDTvar)].values)[:, 1]
	#yTTV=ttV_datainTT #clsTT.predict_proba(dataTTV[trainVarsTT(BDTvar)].values)[:, 1]
	binxTTV=np.trunc(nbins*(-bdtmin+ttV_datainTTV)/factor)
	binyTTV=np.trunc(nbins*(-bdtmin+ttV_datainTT)/factor)
	#print (len(tt_datainTTV),binxTTV,binyTTV)
	for ii in range(0,len(ttV_datainTT)) :
		if int(binxTTV[ii]) >= 0 and int(binyTTV[ii]) < nbins :
			#h.Fill(histRatio[int(binxTTV[ii])][int(binyTTV[ii])])
			h.Fill(hRtio.GetBinContent(int(binxTTV[ii]),int(binyTTV[ii])))
		else : print ("binning went wrong",ii,int(binxTTV[ii]),int(binyTTV[ii]))
	# int is used here is because I want to fill the histogram with the bin position, not in the actual value
	# as the BDT values range from bdtmin-1 , int(nbins*(bdtmin+value)/(bdtmax-bdtmin)) is the bin position
	#############################################
	c = ROOT.TCanvas("c1","",200,200)
	h.Scale(1./ h.Integral());
	h.SetLineWidth(3)
	h.SetLineColor(6)
	h.GetCumulative().Draw();
	h.GetYaxis().SetRangeUser(0.,1.)
	h.SetMinimum(0.0)
	nq=int(nbinsout)
	xq= array.array('d', [0.] * (nq+1)) #[ii/nq for i in range(0,nq-1)] #np.empty(nq+1, dtype=object)
	yq= array.array('d', [0.] * (nq+1)) # [0]*nq #np.empty(nq+1, dtype=object)
	for  ii in range(0,nq) : xq[ii]=(float(ii)/nq)
	xq[nq]=0.999999999
	h.GetQuantiles(nq+1,yq,xq)
	print ("quantiles",nq,len(xq),len(yq))
	line = [None for point in range(nq)]
	line2 = [None for point in range(nq)]
	for  jj in range(0,nq) :
			line[jj] = ROOT.TLine(0,xq[jj],yq[jj],xq[jj]);
			line[jj].SetLineColor(ROOT.kRed);
			line[jj].Draw("same")
			#
			line2[jj] = ROOT.TLine(yq[jj],0,yq[jj],xq[jj]);
			line2[jj].SetLineColor(ROOT.kRed);
			line2[jj].Draw("same")
			print (xq[jj],yq[jj])
	hAuxHisto = ROOT.TH1F("hAuxHisto","",nq,yq)
	latex1= ROOT.TLatex();
	latex1.SetTextSize(0.04);
	latex1.SetTextAlign(13);  #//align at top
	latex1.SetTextFont(42);
	latex1.DrawLatexNDC(0.1,.95,"from ("+str(nbins)+"^2) to "+str(nbinsout)+" bins");
	latex1.DrawLatexNDC(0.8,.95,BDTvar);
	if options.relaxedLepID == True : latex1.DrawLatexNDC(0.5,.95,"looseLep")
	c.Modified();
	c.Update();
	c.SaveAs(channel+"/"+BDTvar+"_from"+str(nbins)+"_to_"+str(nbinsout)+"bins_relLepID"+str(options.relaxedLepID)+"_Cumulative.pdf")
	#################################################
	## to fed analysis code
	hTargetBinning = ROOT.TH2F("hTargetBinning","",100,bdtmin,1.,100,bdtmin,1.)
	for ix in range(0,hTargetBinning.GetXaxis().GetNbins()  ):
		for iy in range(1,hTargetBinning.GetYaxis().GetNbins()) :
			bin1 = hTargetBinning.GetBin(ix,iy)
			biny=int(nbins*(-bdtmin+hTargetBinning.GetXaxis().GetBinCenter(iy))/factor)
			binx=int(nbins*(-bdtmin+hTargetBinning.GetYaxis().GetBinCenter(ix))/factor)
			#content=histRatio[binx][biny] #GetLikeLiHood(ii))
			content=hRtio.GetBinContent(binx,biny)
			bin = hAuxHisto.FindBin(content)-1;
			if bin < 0 : bin=0;
			if bin+1 > hAuxHisto.GetNbinsX() : bin = hAuxHisto.GetNbinsX()-1
			hTargetBinning.SetBinContent(bin1,bin)
	c2 = ROOT.TCanvas("c2","",600,600)
	hTargetBinning.GetXaxis().SetTitle("BDT(ttH,tt)");
	hTargetBinning.GetYaxis().SetTitle("BDT(ttH,ttV)");
	hTargetBinning.Draw("colz")
	latex5= ROOT.TLatex();
	latex5.SetTextSize(0.035);
	latex5.SetTextAlign(13);  #//align at top
	latex5.SetTextFont(62);
	latex5.DrawLatexNDC(.1,.93,"from ("+str(nbins)+"^2) to "+str(nbinsout)+" bins");
	latex5.DrawLatexNDC(0.75,.93,BDTvar);
	c2.Modified();
	c2.Update();
	latex5= ROOT.TLatex();
	latex5.SetTextSize(0.035);
	latex5.SetTextAlign(13);  #//align at top
	latex5.SetTextFont(62);
	latex5.DrawLatexNDC(.1,.93,"from ("+str(nbins)+"^2) to "+str(nbinsout)+" bins");
	latex5.DrawLatexNDC(0.75,.93,BDTvar);
	if options.relaxedLepID == True : latex5.DrawLatexNDC(0.5,.93,"looseLep")
	c2.SaveAs(channel+"/"+BDTvar+"_from"+str(nbins)+"_to_"+str(nbinsout)+"bins_relLepID"+
					str(options.relaxedLepID)+"_CumulativeBins.pdf")
	binning = ROOT.TFile(channel+"/"+BDTvar+"_from"+str(nbins)+"_to_"+str(nbinsout)+"bins_relLepID"+
					str(options.relaxedLepID)+"_CumulativeBins.root","recreate")
	binning.cd()
	hTargetBinning.Write()
	binning.Close()
	#################################################
	## VoronoiPlot() 2D
	doVoronoi2D=False
	if doVoronoi2D :
		c3 =ROOT.TCanvas("c1","",600,600);
		c3.cd();
		hDummy = ROOT.TH1F("hDummy","",2,bdtmin,1);
		hDummy.SetLineColor(ROOT.kWhite);
		hDummy.GetYaxis().SetRangeUser(bdtmin,1.);
		hDummy.GetXaxis().SetRangeUser(bdtmin,1.);
		hDummy.GetXaxis().SetTitle("BDT(ttH,tt)");
		hDummy.GetYaxis().SetTitle("BDT(ttH,ttV)");
		hDummy.Draw();
		print ("len auxiliary", hAuxHisto.GetNbinsX()+1)
		XX=[array.array( 'd' ) for count in xrange(int(hAuxHisto.GetNbinsX()+1))] #
		YY=[array.array( 'd' ) for count in xrange(int(hAuxHisto.GetNbinsX()+1))] #
		granularity=1000
		for x in range(int(bdtmin),granularity) :
			for y in range(int(bdtmin),granularity) :
				#content=histRatio[int(nbins*x/float(granularity))][int(nbins*y/float(granularity))] #GetLikeLiHood(ii))
				content=hRtio.GetBinContent(int(nbins*(x-bdtmin)/float(granularity*factor)),int(nbins*(y-bdtmin)/float(granularity*factor)))
				bin = hAuxHisto.FindBin(content)-1;
				if bin < 0 : bin=0;
				if bin+1 > hAuxHisto.GetNbinsX() : bin = hAuxHisto.GetNbinsX()-1
				XX[bin].append(x/float(granularity));
				YY[bin].append(y/float(granularity));
		print ("nbins" , len(XX)-1,hAuxHisto.GetNbinsX())
		graphs=[None for count in xrange(int(hAuxHisto.GetNbinsX()+1))]
		for k in range(0,hAuxHisto.GetNbinsX()) :
			graphs[k]=ROOT.TGraph(len(XX[k]),  XX[k], YY[k] ); #
			graphs[k].SetMarkerColor(int(k+1));
			graphs[k].SetMarkerStyle(6);
			graphs[k].Draw("PSAME");
		latex= ROOT.TLatex();
		latex.SetTextSize(0.04);
		latex.SetTextAlign(13);  #//align at top
		latex.SetTextFont(62);
		latex1.DrawLatexNDC(.1,.95,"from ("+str(nbins)+"^2) to "+str(nbinsout)+" bins");
		latex1.DrawLatexNDC(0.8,.95,BDTvar);
		if options.relaxedLepID == True : latex1.DrawLatexNDC(0.5,.95,"looseLep")
		c3.Modified();
		c3.Update();
		c3.SaveAs(channel+"/"+BDTvar+"_from"+str(nbins)+"_to_"+str(nbinsout)+"bin_relLepID"+
							str(options.relaxedLepID)+"_Voronoi.pdf")
		print ("s/B in last bin",h3.GetBinContent(h3.GetNbinsX()),h3.GetBinContent(h3.GetNbinsX()-1))
		#c3.SaveAs(channel+"/"+BDTvar+"_from"+str(nbins)+"_to_"+str(nbinsout)+"bin_relLepID"+str(options.relaxedLepID)+"_Voronoi.png")
	################################################################
	## VoronoiPlot1D() - 'tight lep'
	if 1>0: #and "oldTrain" not in BDTvar:
			if "oldTrain" not in BDTvar:
				"""
				tt_datainTTVless =clsTTV.predict_proba(dataTight.loc[
									(dataTight.key.values=='TTTo2L2Nu') | (dataTight.key.values=='TTToSemilepton') ,
									trainVarsTTV(BDTvar)].values)[:, 1]
				ttV_datainTTVless=clsTTV.predict_proba(dataTight.loc[
									(dataTight.key.values=='TTZToLLNuNu') | (dataTight.key.values=='TTWJetsToLNu'),
									trainVarsTTV(BDTvar)].values)[:, 1]
				ttH_datainTTVless=clsTTV.predict_proba(dataTight.loc[
									(dataTight.key.values=='ttHToNonbb'),
									trainVarsTTV(BDTvar)].values)[:, 1]
				tt_datainTTless =clsTT.predict_proba(dataTight.loc[
									(dataTight.key.values=='TTTo2L2Nu') | (dataTight.key.values=='TTToSemilepton') ,
									trainVarsTT(BDTvar)].values)[:, 1]
				ttV_datainTTless=clsTT.predict_proba(dataTight.loc[
									(dataTight.key.values=='TTZToLLNuNu') | (dataTight.key.values=='TTWJetsToLNu'),
									trainVarsTT(BDTvar)].values)[:, 1]
				ttH_datainTTless=clsTT.predict_proba(dataTight.loc[
									(dataTight.key.values=='ttHToNonbb'),
									trainVarsTT(BDTvar)].values)[:, 1]
				"""
				tt_datainTTVless =clsTTV.predict_proba(dataTightFS.loc[(dataTightFS.proces.values=='TT'), trainVarsTTV(BDTvar)].values)[:, 1]
				ttV_datainTTVless=clsTTV.predict_proba(dataTightFS.loc[(dataTightFS.proces.values=='TTZ') | (dataTightFS.proces.values=='TTW'), trainVarsTTV(BDTvar)].values)[:, 1]
				ttH_datainTTVless=clsTTV.predict_proba(dataTightFS.loc[(dataTightFS.proces.values=='signal'), trainVarsTTV(BDTvar)].values)[:, 1]
				EWK_datainTTVless=clsTTV.predict_proba(dataTightFS.loc[(dataTightFS.proces.values=='EWK'), trainVarsTTV(BDTvar)].values)[:, 1]
				rares_datainTTVless=clsTTV.predict_proba(dataTightFS.loc[(dataTightFS.proces.values=='Rares'), trainVarsTTV(BDTvar)].values)[:, 1]
				tt_datainTTless =clsTT.predict_proba(dataTightFS.loc[(dataTightFS.proces.values=='TT'),trainVarsTT(BDTvar)].values)[:, 1]
				ttV_datainTTless=clsTT.predict_proba(dataTightFS.loc[(dataTightFS.proces.values=='TTZ') | (dataTightFS.proces.values=='TTW'), trainVarsTT(BDTvar)].values)[:, 1]
				ttH_datainTTless=clsTT.predict_proba(dataTightFS.loc[(dataTightFS.proces.values=='signal'), trainVarsTT(BDTvar)].values)[:, 1]
				EWK_datainTTless=clsTT.predict_proba(dataTightFS.loc[(dataTightFS.proces.values=='EWK'), trainVarsTT(BDTvar)].values)[:, 1]
				rares_datainTTless=clsTT.predict_proba(dataTightFS.loc[(dataTightFS.proces.values=='Rares'), trainVarsTT(BDTvar)].values)[:, 1]
			else :
				tt_datainTTVless = dataTightFS.loc[(dataTightFS.proces.values=='TT'), ttVBDT].values
				ttV_datainTTVless=dataTightFS.loc[(dataTightFS.proces.values=='TTZ') | (dataTightFS.proces.values=='TTW'), ttVBDT].values
				ttH_datainTTVless=dataTightFS.loc[(dataTightFS.proces.values=='signal'),ttVBDT].values
				EWK_datainTTVless=dataTightFS.loc[(dataTightFS.proces.values=='EWK'),ttVBDT].values
				rares_datainTTVless=dataTightFS.loc[(dataTightFS.proces.values=='Rares'),ttVBDT].values
				tt_datainTTless =dataTightFS.loc[(dataTightFS.proces.values=='TT'), ttBDT].values
				ttV_datainTTless=dataTightFS.loc[(dataTightFS.proces.values=='TTZ') | (dataTightFS.proces.values=='TTW'), ttBDT].values
				ttH_datainTTless=dataTightFS.loc[(dataTightFS.proces.values=='signal'),ttBDT].values
				EWK_datainTTless=dataTightFS.loc[(dataTightFS.proces.values=='EWK'),ttBDT].values
				rares_datainTTless=dataTightFS.loc[(dataTightFS.proces.values=='Rares'),ttBDT].values
			#if BDTvar!="oldTrain" :
			hTT = ROOT.TH1F("hTTbarless","",hAuxHisto.GetNbinsX(), -0.5, hAuxHisto.GetNbinsX()-0.5);
			hTTW   = ROOT.TH1F("hTTWless"  ,"",hAuxHisto.GetNbinsX(), -0.5, hAuxHisto.GetNbinsX()-0.5);
			hTTH   = ROOT.TH1F("hTTHless"  ,"",hAuxHisto.GetNbinsX(), -0.5, hAuxHisto.GetNbinsX()-0.5);
			hEWK   = ROOT.TH1F("hEWKless"  ,"",hAuxHisto.GetNbinsX(), -0.5, hAuxHisto.GetNbinsX()-0.5);
			#hRares   = ROOT.TH1F("hRaresless"  ,"",hAuxHisto.GetNbinsX(), -0.5, hAuxHisto.GetNbinsX()-0.5);
			mc  = ROOT.THStack("mc","");
			weightTTV=dataTightFS.loc[(dataTightFS.proces.values=='TTZ') | (dataTightFS.proces.values=='TTW'),"totalWeight"].values
			weightTT=dataTightFS.loc[(dataTightFS.proces.values=='TT'),"totalWeight"].values
			weightTTH=dataTightFS.loc[(dataTightFS.proces.values=='signal'),"totalWeight"].values
			weightEWK=dataTightFS.loc[(dataTightFS.proces.values=='EWK'),"totalWeight"].values
			weightRares=dataTightFS.loc[(dataTightFS.proces.values=='Rares'),"totalWeight"].values
			print ("tight yield TT,TTV,TTH",weightTT.sum(),weightTTV.sum(),weightTTH.sum(),weightEWK.sum(),weightRares.sum(),len(EWK_datainTTVless),len(weightEWK))

			for ii in range(0,max(len(ttV_datainTTVless),len(tt_datainTTVless),len(ttH_datainTTless),len(EWK_datainTTless),len(rares_datainTTless))) :
				#if ii<20 : print (ttH_datainTTV[ii],ttH_datainTT[ii],hTargetBinning.FindBin(ttH_datainTTV[ii],ttH_datainTT[ii]), hTargetBinning.GetBinContent(hTargetBinning.FindBin(ttH_datainTTV[ii],ttH_datainTT[ii])))
				if ii<len(weightTTV) : hTTW.Fill(hTargetBinning.GetBinContent(hTargetBinning.FindBin(ttV_datainTTless[ii],ttV_datainTTVless[ii])),weightTTV[ii])
				if ii<len(weightTT) : hTT.Fill(hTargetBinning.GetBinContent(hTargetBinning.FindBin(tt_datainTTless[ii],tt_datainTTVless[ii])),weightTT[ii])
				if ii<len(weightTTH) : hTTH.Fill(hTargetBinning.GetBinContent(hTargetBinning.FindBin(ttH_datainTTless[ii],ttH_datainTTVless[ii])),weightTTH[ii]) # +0.01
				if ii<len(weightEWK) : hEWK.Fill(hTargetBinning.GetBinContent(hTargetBinning.FindBin(EWK_datainTTless[ii],EWK_datainTTVless[ii])),weightEWK[ii])
				#if ii<len(weightRares) : hRares.Fill(hTargetBinning.GetBinContent(hTargetBinning.FindBin(rares_datainTTless[ii],rares_datainTTVless[ii])),weightRares[ii])
				#yTTH=ttH_datainTT #clsTT.predict_proba(dataTTH[trainVarsTT(BDTvar)].values)[:, 1]
				#xTTH=ttH_datainTTV #clsTTV.predict_proba(dataTTH[trainVarsTTV(BDTvar)].values)[:, 1]
			hTT.SetFillColor( 17 );
			hTTH.SetFillColor( ROOT.kRed );
			hTTW.SetFillColor( 8 );
			hEWK.SetFillColor( 65 );
			#hRares.SetFillColor( 5 );
			mc.Add(hEWK);
			#mc.Add(hRares);
			mc.Add(hTTW);
			mc.Add(hTTH);
			mc.Add( hTT );
			c4 = ROOT.TCanvas("c5","",500,500);
			c4.cd();
			c4.Divide(1,2,0,0);
			c4.cd(1)
			ROOT.gPad.SetLogy()
			#c5.SetLogy()
			ROOT.gPad.SetBottomMargin(0.001)
			ROOT.gPad.SetTopMargin(0.065)
			ROOT.gPad.SetRightMargin(0.01)
			ROOT.gPad.SetLeftMargin(0.12)
			#ROOT.gPad.SetLabelSize(.4, "XY")
			mc.Draw("HIST");
			mc.SetMaximum(15* mc.GetMaximum());
			mc.SetMinimum(max(0.04* mc.GetMinimum(),0.01));
			mc.GetYaxis().SetRangeUser(0.01,110);
			mc.GetHistogram().GetYaxis().SetTitle("Expected events/bin");
			mc.GetHistogram().GetXaxis().SetTitle("Bin in the bdt1#times bdt2 plane");
			mc.GetHistogram().GetXaxis().SetTitleSize(0.06);
			mc.GetHistogram().GetXaxis().SetLabelSize(.06); #SetTitleOffset(1.1);
			mc.GetHistogram().GetYaxis().SetTitleSize(0.06);
			mc.GetHistogram().GetYaxis().SetLabelSize(.06);
			#mc.GetHistogram().GetYaxis().SetTitleOffset(1.1);
			l = ROOT.TLegend(0.16,0.6,0.3,0.9);
			l.AddEntry(hTTH  , "ttH", "f");
			l.AddEntry(hTTW  , "ttV"       , "f");
			l.AddEntry(hTT, "tt"        , "f");
			#l.AddEntry(hRares, "rares"        , "f");
			l.AddEntry(hEWK, "EWK"        , "f");
			l.Draw();
			latex= ROOT.TLatex();
			latex.SetTextSize(0.065);
			latex.SetTextAlign(13);  #//align at top
			latex.SetTextFont(62);
			latex.DrawLatexNDC(.15,1.0,"CMS Simulation");
			latex.DrawLatexNDC(.8,1.0,"#it{36 fb^{-1}}");
			#latex1.DrawLatexNDC(0.5,.77,"looseLep")
			latex.DrawLatexNDC(.55,.8,"from ("+str(nbins)+"^2) to "+str(nbinsout)+" bins");
			latex.DrawLatexNDC(.55,.9,BDTvar);
			#"""
			c4.cd(2)
			#ROOT.gPad.SetLogy()
			ROOT.gStyle.SetHatchesSpacing(100)
			#c5.SetLogy(1)
			ROOT.gPad.SetLeftMargin(0.12)
			ROOT.gPad.SetBottomMargin(0.12)
			ROOT.gPad.SetTopMargin(0.001)
			ROOT.gPad.SetRightMargin(0.005)
			#ROOT.gPad.SetLabelSize(.1, "XY")
			if not hTT.GetSumw2N() : hTT.Sumw2()
			#if not hTTW.GetSumw2N() : hTTW.Sumw2()
			h2=hTT.Clone()
			h2.Add(hTTW)
			hBKG1D=h2.Clone()
			h3=hTTH.Clone()
			h4=hTT.Clone()
			#h3=Divide(h3,h2)
			#"""
			#h3=hTT.Clone()
			if not h2.GetSumw2N() : h2.Sumw2()
			if not h3.GetSumw2N() : h3.Sumw2()
			for binn in range(0,h2.GetNbinsX()+1) :
				ratio=0
				ratio3=0
				if h2.GetBinContent(binn) >0 :
					ratio=h2.GetBinError(binn)/h2.GetBinContent(binn)
					h2.SetBinContent(binn,ratio)
				if h3.GetBinContent(binn) > 0 :
					ratio3=h3.GetBinContent(binn)/hBKG1D.GetBinContent(binn)
					h3.SetBinContent(binn,ratio3)
				if h4.GetBinContent(binn) > 0 : h4.SetBinContent(binn,h4.GetBinError(binn)/h4.GetBinContent(binn))
				print (binn,ratio,ratio3)
			#"""
			h2.SetLineWidth(3)
			h2.SetLineColor(2)
			h2.SetFillStyle(3690)
			h3.GetYaxis().SetRangeUser(0.01,1.0);
			h3.SetLineWidth(3)
			h3.SetFillStyle(3690)
			h3.SetLineColor(28)
			h4.SetLineWidth(3)
			h4.SetFillStyle(3690)
			h4.SetLineColor(6)
			h3.Draw("HIST")
			h3.GetYaxis().SetTitle("S/B");
			#latex.DrawLatexNDC(.15,.65,"S/B");
			h3.GetXaxis().SetTitle("Bin in the bdt1#times bdt2 plane");
			h3.GetYaxis().SetTitleSize(0.06);
			h3.GetYaxis().SetLabelSize(.06)
			h3.GetXaxis().SetTitleSize(0.06);
			h3.GetXaxis().SetLabelSize(.06)
			l2 = ROOT.TLegend(0.16,0.77,0.4,0.98);
			l2.AddEntry(h3  , "S/B" , "l");
			l2.AddEntry(h2  , "ttV + tt err/cont", "l");
			l2.AddEntry(h4  , "tt err/cont", "l");
			l2.Draw("same");
			h2.Draw("HIST,SAME")
			h4.Draw("HIST,SAME")
			#c5.SetLogy(1)
			#"""
			c4.Modified();
			c4.Update();
			print ("s/B in last bin (tight)",
					h3.GetBinContent(h3.GetNbinsX()), # /hBKG1D.GetBinContent(h3.GetNbinsX())
					h3.GetBinContent(h3.GetNbinsX()-1), #/hBKG1D.GetBinContent(h3.GetNbinsX()-1)
					h2.GetBinContent(h3.GetNbinsX())
					)

			c4.SaveAs(channel+"/"+BDTvar+"_from"+str(nbins)+"_to_"+str(nbinsout)+"bins_relLepID"+
							str(options.relaxedLepID)+"_lessVoronoi1D.pdf")
			#c4.SaveAs(channel+"/"+BDTvar+"_from"+str(nbins)+"_to_"+str(nbinsout)+"bins_Voronoi1D.png")
	##############################################
	## VoronoiPlot1D() loose lep
	hTT = ROOT.TH1F("hTTbar","",hAuxHisto.GetNbinsX(), -0.5+1, 1+hAuxHisto.GetNbinsX()-0.5);
	hTTW   = ROOT.TH1F("hTTW"  ,"",hAuxHisto.GetNbinsX(), -0.5+1, 1+hAuxHisto.GetNbinsX()-0.5);
	hTTH   = ROOT.TH1F("hTTH"  ,"",hAuxHisto.GetNbinsX(), -0.5+1, 1+hAuxHisto.GetNbinsX()-0.5);
	mc  = ROOT.THStack("mc","mc");
	weightTTV=dataTTV["totalWeight"].values
	weightTT=dataTT["totalWeight"].values
	weightTTH=dataTTH["totalWeight"].values
	print ("loose yield TT,TTV,TTH",weightTT.sum(),weightTTV.sum(),weightTTH.sum())
	for ii in range(0,max(len(dataTTV),len(dataTT),len(dataTTH))) :
		#if ii<20 : print (ttH_datainTTV[ii],ttH_datainTT[ii],hTargetBinning.FindBin(ttH_datainTTV[ii],ttH_datainTT[ii]), hTargetBinning.GetBinContent(hTargetBinning.FindBin(ttH_datainTTV[ii],ttH_datainTT[ii])))
		if ii <len(ttV_datainTT) : hTTW.Fill(hTargetBinning.GetBinContent(hTargetBinning.FindBin(ttV_datainTT[ii],ttV_datainTTV[ii]))+1,weightTTV[ii]) # +0.0001
		if ii < len(tt_datainTT) : hTT.Fill(hTargetBinning.GetBinContent(hTargetBinning.FindBin(tt_datainTT[ii],tt_datainTTV[ii]))+1,weightTT[ii]) #+0.01
		if ii < len(ttH_datainTT) : hTTH.Fill(hTargetBinning.GetBinContent(hTargetBinning.FindBin(ttH_datainTT[ii],ttH_datainTTV[ii]))+1,weightTTH[ii]) # +0.01
		#yTTH=ttH_datainTT #clsTT.predict_proba(dataTTH[trainVarsTT(BDTvar)].values)[:, 1]
		#xTTH=ttH_datainTTV #clsTTV.predict_proba(dataTTH[trainVarsTTV(BDTvar)].values)[:, 1]
	hTT.SetFillColor( 17 );
	hTTH.SetFillColor( ROOT.kRed );
	hTTW.SetFillColor( 8 );
	mc.Add(hTTW);
	mc.Add(hTTH);
	mc.Add( hTT );
	c5 = ROOT.TCanvas("c5","",500,500);
	c5.cd();
	c5.Divide(1,2,0,0);
	c5.cd(1)
	ROOT.gPad.SetLogy()
	#c5.SetLogy()
	ROOT.gPad.SetBottomMargin(0.001)
	ROOT.gPad.SetTopMargin(0.065)
	ROOT.gPad.SetRightMargin(0.01)
	#ROOT.gPad.SetLabelSize(.4, "XY")
	mc.Draw("HIST");
	mc.SetMaximum(1.5* mc.GetMaximum());
	mc.SetMinimum(0.1);
	mc.GetHistogram().GetYaxis().SetTitle("Expected events/bin");
	mc.GetHistogram().GetXaxis().SetTitle("Bin in the bdt1#times bdt2 plane");
	mc.GetHistogram().GetXaxis().SetTitleSize(0.05);
	mc.GetHistogram().GetXaxis().SetLabelSize(.06); #SetTitleOffset(1.1);
	mc.GetHistogram().GetYaxis().SetTitleSize(0.05);
	mc.GetHistogram().GetYaxis().SetLabelSize(.06);
	#mc.GetHistogram().GetYaxis().SetTitleOffset(1.1);
	l = ROOT.TLegend(0.15,0.7,0.47,0.9);
	l.AddEntry(hTTH  , "ttH ", "f");
	l.AddEntry(hTTW  , "ttV"       , "f");
	l.AddEntry(hTT, "tt"        , "f");
	l.Draw();
	latex= ROOT.TLatex();
	latex.SetTextSize(0.065);
	latex.SetTextAlign(13);  #//align at top
	latex.SetTextFont(62);
	latex.DrawLatexNDC(.15,1.0,"CMS Simulation");
	latex.DrawLatexNDC(.8,1.0,"#it{36 fb^{-1}}");
	#latex1.DrawLatexNDC(0.5,.77,"looseLep")
	latex.DrawLatexNDC(.55,.8,"from ("+str(nbins)+"^2) to "+str(nbinsout)+" bins");
	if options.relaxedLepID : latex.DrawLatexNDC(.55,.9,BDTvar+" loose lep ");
	else : latex.DrawLatexNDC(.55,.9,BDTvar+" tight lep ")
	#"""
	c5.cd(2)
	ROOT.gPad.SetLogy()
	#c5.SetLogy(1)
	ROOT.gPad.SetTopMargin(0.001)
	ROOT.gPad.SetRightMargin(0.01)
	#ROOT.gPad.SetLabelSize(.1, "XY")
	if not hTT.GetSumw2N() : hTT.Sumw2()
	#if not hTTW.GetSumw2N() : hTTW.Sumw2()
	h2=hTT.Clone()
	h2.Add(hTTW)
	hBKG1D=h2.Clone()
	h3=hTTH.Clone()
	#h3=Divide(h3,h2)
	#"""
	#h3=hTT.Clone()
	if not h2.GetSumw2N() : h2.Sumw2()
	if not h3.GetSumw2N() : h3.Sumw2()
	for binn in range(0,h2.GetNbinsX()+1) :
		ratio=0
		ratio3=0
		if h2.GetBinContent(binn) >0 :
			ratio=h2.GetBinError(binn)/h2.GetBinContent(binn)
			h2.SetBinContent(binn,ratio)
		if h3.GetBinContent(binn) > 0 :
			ratio3=h3.GetBinContent(binn)/hBKG1D.GetBinContent(binn)
			h3.SetBinContent(binn,ratio3)
		print (binn,ratio,ratio3)
	#"""
	h2.SetLineWidth(3)
	h2.SetLineColor(8)
	h2.SetFillStyle(3001)
	h3.GetYaxis().SetRangeUser(0.01,1.4);
	h3.SetLineWidth(3)
	h3.SetFillStyle(3001)
	h3.SetLineColor(28)
	h3.Draw("HIST")
	h3.GetYaxis().SetTitle("S/B");
	#latex.DrawLatexNDC(.15,.65,"S/B");
	h3.GetXaxis().SetTitle("Bin in the bdt1#times bdt2 plane");
	h3.GetYaxis().SetTitleSize(0.05);
	h3.GetYaxis().SetLabelSize(.06)
	h3.GetXaxis().SetTitleSize(0.05);
	h3.GetXaxis().SetLabelSize(.06)
	#l2 = ROOT.TLegend(0.12,0.8,0.32,0.98);
	#l2.AddEntry(h3  , "S/B" , "l");
	#l2.AddEntry(h2  , "ttV + tt err/content", "l");
	#l2.Draw("same");
	#h2.Draw("HIST,SAME")
	#c5.SetLogy(1)
	#"""
	c5.Modified();
	c5.Update();
	print ("s/B in last bin",
			h3.GetBinContent(h3.GetNbinsX()), # /hBKG1D.GetBinContent(h3.GetNbinsX())
			h3.GetBinContent(h3.GetNbinsX()-1), #/hBKG1D.GetBinContent(h3.GetNbinsX()-1)
			h2.GetBinContent(h3.GetNbinsX())
			)
	c5.SaveAs(channel+"/"+BDTvar+"_from"+str(nbins)+"_to_"+str(nbinsout)+"bins_relLepID"+str(options.relaxedLepID)+"_Voronoi1D.pdf")
#################################################################
if options.doBDT == True :
	## do BDT from tt and ttV_file
	print "training joint-BDT"
	BDTtype=options.BDTtype
	if BDTtype=="1B" : trainVars=["BDTtt","BDTttV"]
	if BDTtype=="2MEM" : trainVars=["BDTtt","BDTttV",'mvaOutput_hadTopTaggerWithKinFit', 'mvaOutput_Hj_tagger', 'unfittedHadTop_pt','memOutput_LR']
	if BDTtype=="noMEM" : trainVars=["BDTtt","BDTttV",'mvaOutput_hadTopTaggerWithKinFit', 'mvaOutput_Hj_tagger', 'unfittedHadTop_pt','memOutput_LR']
	data["BDTtt"] = clsTT.predict_proba(data[trainVarsTT(BDTvar)].values)[:, 1]
	data["BDTttV"] = clsTTV.predict_proba(data[trainVarsTTV(BDTvar)].values)[:, 1]
	dataTightFS["BDTtt"] = clsTT.predict_proba(dataTightFS[trainVarsTT(BDTvar)].values)[:, 1]
	dataTightFS["BDTttV"] = clsTTV.predict_proba(dataTightFS[trainVarsTTV(BDTvar)].values)[:, 1]
	traindataset, valdataset  = train_test_split(data[trainVars+["target","totalWeight"]], test_size=0.2, random_state=7)
	cls =  xgb.XGBClassifier(
				n_estimators = options.ntrees,
				max_depth = options.treeDeph,
				min_child_weight = options.mcw, # min_samples_leaf
				learning_rate = options.lr
				#max_features = 'sqrt',
				#min_samples_leaf = 100
				)
	cls.fit(
		traindataset[trainVars].values,
		traindataset.target.astype(np.bool),
		sample_weight=(traindataset[weights].astype(np.float64))
		# more diagnosis, in case
		#eval_set=[(traindataset[trainVars(False)].values,  traindataset.target.astype(np.bool),traindataset[weights].astype(np.float64)),
		#(valdataset[trainVars(False)].values,  valdataset.target.astype(np.bool), valdataset[weights].astype(np.float64))] ,
		#verbose=True,eval_metric="auc"
		)
	pklpath=channel+"/"+channel+"_XGB_JointBDT_"+BDTvar+"_"+BDTtype
	print ("Done  ",pklpath)
	if options.doXML==True :
		print ("Date: ", time.asctime( time.localtime(time.time()) ))
		pickle.dump(cls, open(pklpath+".pkl", 'wb'))
		file = open(pklpath+"_pkl.log","w")
		file.write(str(trainVars)+"\n")
		file.close()
		print ("saved ",pklpath+".pkl")
		print ("variables are: ",pklpath+"_pkl.log")
	print ("XGBoost trained")
	proba = cls.predict_proba(traindataset[trainVars].values  )
	fpr, tpr, thresholds = roc_curve(traindataset["target"], proba[:,1] )
	train_auc = auc(fpr, tpr, reorder = True)
	print("XGBoost train set auc - {}".format(train_auc))
	proba = cls.predict_proba(valdataset[trainVars].values)
	fprt, tprt, thresholds = roc_curve(valdataset["target"], proba[:,1] )
	test_auct = auc(fprt, tprt, reorder = True)
	print("XGBoost test set auc - {}".format(test_auct))
	proba = cls.predict_proba(dataTightFS[trainVars].values)
	fprtf, tprtf, thresholds = roc_curve(dataTightFS["target"], proba[:,1] )
	test_auctf = auc(fprtf, tprtf, reorder = True)
	print("XGBoost fulsim auc - {}".format(test_auctf))
	f_score_dict =cls.booster().get_fscore()
	f_score_dict = {trainVars[int(k[1:])] : v for k,v in f_score_dict.items()}
	print("importance", f_score_dict)
	###########################################################################
	#print (list(valdataset))
	#plt.clf()
	hist_params = {'normed': True, 'bins': 7 , 'histtype':'step', 'lw': 3}
	fig, ax = plt.subplots(figsize=(5, 5))
	# plt.subplot(221)
	y_pred = cls.predict_proba(valdataset.ix[valdataset.target.values == 0, trainVars].values)[:, 1] #
	y_predS = cls.predict_proba(valdataset.ix[valdataset.target.values == 1, trainVars].values)[:, 1] #
	#plt.figure('XGB',figsize=(6, 6))
	ax.hist(y_pred , label="TT + TTV", **hist_params)
	ax.hist(y_predS , label="signal", **hist_params )
	#plt.xscale('log')
	#plt.yscale('log')
	ax.legend(loc='upper left')
	plt.show()
	fig.savefig(channel+'/'+BDTvar+'_jointBDT_XGBclassifier.pdf')
	plt.clf()
	##################################################
	fig, ax = plt.subplots(figsize=(5, 5))
	# plt.subplot(221)
	## ROC curve
	#ax.plot(fprf, tprf, lw=1, label='GB train (area = %0.3f)'%(train_aucf))
	#ax.plot(fprtf, tprtf, lw=1, label='GB test (area = %0.3f)'%(test_auctf))
	ax.plot(fpr, tpr, lw=1, label='XGB train (area = %0.3f)'%(train_auc))
	ax.plot(fprt, tprt, lw=1, label='XGB test (area = %0.3f)'%(test_auct))
	#ax.plot(fprc, tprc, lw=1, label='CB train (area = %0.3f)'%(train_aucc))
	ax.plot(fprtf, tprtf, lw=1, label='FullSim (area = %0.3f)'%(test_auctf))
	ax.set_ylim([0.0,1.0])
	ax.set_xlabel('False Positive Rate')
	ax.set_ylabel('True Positive Rate')
	ax.legend(loc="lower right")
	#ax.grid()
	fig.savefig(channel+'/'+BDTvar+'_jointBDT_'+BDTtype+'_roc.pdf')
	plt.clf()
	###########################################################################
	for ii in [1,2] :
		if ii == 1 :
			datad=traindataset.loc[traindataset['target'].values == 1]
			label="signal"
		else :
			datad=traindataset.loc[traindataset['target'].values == 0]
			label="BKG"
		datacorr = datad[trainVars] #.loc[:,trainVars(False)] #dataHToNobbCSV[[trainVars(True)]]
		correlations = datacorr.corr()
		fig = plt.figure(figsize=(5, 5))
		ax = fig.add_subplot(111)
		cax = ax.imshow(correlations, vmin=-1, vmax=1, cmap=cm.coolwarm)
		ticks = np.arange(0,len(trainVars),1)
		plt.rc('axes', labelsize=8)
		ax.set_xticks(ticks)
		ax.set_yticks(ticks)
		ax.set_xticklabels(trainVars,rotation=-90)
		ax.set_yticklabels(trainVars)
		fig.colorbar(cax)
		fig.tight_layout()
		#plt.subplots_adjust(left=0.9, right=0.9, top=0.9, bottom=0.1)
		fig.savefig(channel+'/'+BDTvar+'_'+label+'_jointBDT'+BDTtype+'_corr.pdf')
		print (label, correlations)
		ax.clear()
	plt.clf()
	####################################################
	tt_datainJoint =cls.predict_proba(dataTightFS.loc[(dataTightFS.proces.values=='TT'),trainVars].values  )[:,1]
	ttV_datainJoint=cls.predict_proba(dataTightFS.loc[(dataTightFS.proces.values=='TTZ') | (dataTightFS.proces.values=='TTW'),trainVars].values  )[:,1]
	ttH_datainJoint=cls.predict_proba(dataTightFS.loc[(dataTightFS.proces.values=='signal'),trainVars].values  )[:,1]
	EWK_datainJoint=cls.predict_proba(dataTightFS.loc[(dataTightFS.proces.values=='EWK'),trainVars].values  )[:,1]
	rares_datainJoint=cls.predict_proba(dataTightFS.loc[(dataTightFS.proces.values=='Rares'),trainVars].values  )[:,1]
	##
	tt_datainTTV = clsTTV.predict_proba(dataTightFS.loc[(dataTightFS.proces.values=='TT'), trainVarsTTV(BDTvar)].values)[:, 1]
	ttV_datainTTV=clsTTV.predict_proba(dataTightFS.loc[(dataTightFS.proces.values=='TTZ') | (dataTightFS.proces.values=='TTW'), trainVarsTTV(BDTvar)].values)[:, 1]
	ttH_datainTTV=clsTTV.predict_proba(dataTightFS.loc[(dataTightFS.proces.values=='signal'), trainVarsTTV(BDTvar)].values)[:, 1]
	EWK_datainTTV=clsTTV.predict_proba(dataTightFS.loc[(dataTightFS.proces.values=='EWK'), trainVarsTTV(BDTvar)].values)[:, 1]
	rares_datainTTV=clsTTV.predict_proba(dataTightFS.loc[(dataTightFS.proces.values=='Rares'), trainVarsTTV(BDTvar)].values)[:, 1]
	##
	tt_datainTT = clsTT.predict_proba(dataTightFS.loc[(dataTightFS.proces.values=='TT'), trainVarsTT(BDTvar)].values)[:, 1]
	ttV_datainTT=clsTT.predict_proba(dataTightFS.loc[(dataTightFS.proces.values=='TTZ') | (dataTightFS.proces.values=='TTW'), trainVarsTT(BDTvar)].values)[:, 1]
	ttH_datainTT=clsTT.predict_proba(dataTightFS.loc[(dataTightFS.proces.values=='signal'), trainVarsTT(BDTvar)].values)[:, 1]
	EWK_datainTT=clsTT.predict_proba(dataTightFS.loc[(dataTightFS.proces.values=='EWK'), trainVarsTT(BDTvar)].values)[:, 1]
	rares_datainTT=clsTT.predict_proba(dataTightFS.loc[(dataTightFS.proces.values=='Rares'), trainVarsTT(BDTvar)].values)[:, 1]
	##
	weightTTV=dataTightFS.loc[(dataTightFS.proces.values=='TTZ') | (dataTightFS.proces.values=='TTW'),"totalWeight"].values
	weightTT=dataTightFS.loc[(dataTightFS.proces.values=='TT'),"totalWeight"].values
	weightTTH=dataTightFS.loc[(dataTightFS.proces.values=='signal'),"totalWeight"].values
	weightEWK=dataTightFS.loc[(dataTightFS.proces.values=='EWK'),"totalWeight"].values
	weightRares=dataTightFS.loc[(dataTightFS.proces.values=='Rares'),"totalWeight"].values
	errOcont=[]
	errOcontTT=[]
	sOb=[]
	errOcont_tt=[]
	errOcontTT_tt=[]
	sOb_tt=[]
	errOcont_ttV=[]
	errOcontTT_ttV=[]
	sOb_ttV=[]
	maxdataJoint=max(max(ttV_datainJoint),max(tt_datainJoint),max(ttH_datainJoint))
	maxdataTT=max(max(ttV_datainTT),max(tt_datainTT),max(ttH_datainTT))
	maxdataTTV=max(max(ttV_datainTTV),max(tt_datainTTV),max(ttH_datainTTV))
	print ("max data Joint/TT/TTV",maxdataJoint,maxdataTTV,maxdataTT)
	mindataJoint=min(min(ttV_datainJoint),min(tt_datainJoint),min(ttH_datainJoint))
	mindataTT=min(max(ttV_datainTT),min(tt_datainTT),min(ttH_datainTT))
	mindataTTV=min(min(ttV_datainTTV),min(tt_datainTTV),min(ttH_datainTTV))
	print ("min data Joint/TT/TTV",mindataJoint,mindataTTV,mindataTT)
	nmax=11
	for ntarget in range(1,nmax) :
		hTT = ROOT.TH1F("hTTbarless"+str(ntarget),"",ntarget, mindataJoint, maxdataJoint);
		hTTW   = ROOT.TH1F("hTTWless"+str(ntarget)  ,"",ntarget, mindataJoint, maxdataJoint);
		hTTH   = ROOT.TH1F("hTTHless"+str(ntarget)  ,"",ntarget, mindataJoint, maxdataJoint);
		hEWK   = ROOT.TH1F("hEWKless"+str(ntarget)  ,"",ntarget, mindataJoint, maxdataJoint);
		hRares   = ROOT.TH1F("hRaresless"+str(ntarget)  ,"",ntarget, mindataJoint, maxdataJoint);
		##
		hTT_ttV = ROOT.TH1F("hTTbarlessttV"+str(ntarget),"",ntarget, mindataTTV, maxdataTTV);
		hTTW_ttV   = ROOT.TH1F("hTTWlessttV"+str(ntarget)  ,"",ntarget, mindataTTV, maxdataTTV);
		hTTH_ttV   = ROOT.TH1F("hTTHlessttV"+str(ntarget)  ,"",ntarget, mindataTTV, maxdataTTV);
		hEWK_ttV   = ROOT.TH1F("hEWKlessttV"+str(ntarget)  ,"",ntarget, mindataTTV, maxdataTTV);
		hRares_ttV   = ROOT.TH1F("hRareslessttV"+str(ntarget)  ,"",ntarget, mindataTTV, maxdataTTV);
		##
		hTT_tt = ROOT.TH1F("hTTbarlesstt"+str(ntarget),"",ntarget, mindataTT, maxdataTT);
		hTTW_tt   = ROOT.TH1F("hTTWlesstt"+str(ntarget)  ,"",ntarget, mindataTT, maxdataTT);
		hTTH_tt   = ROOT.TH1F("hTTHlesstt"+str(ntarget)  ,"",ntarget, mindataTT, maxdataTT);
		hEWK_tt   = ROOT.TH1F("hEWKlesstt"+str(ntarget)  ,"",ntarget, mindataTT, maxdataTT);
		hRares_tt   = ROOT.TH1F("hRareslesstt"+str(ntarget)  ,"",ntarget, mindataTT, maxdataTT);
		##
		print ttV_datainJoint
		print ("tight yield TT,TTV,TTH",weightTT.sum(),weightTTV.sum(),weightTTH.sum(),weightEWK.sum(),weightRares.sum())
		for ii in range(0,max(len(ttV_datainJoint),len(tt_datainJoint),len(ttH_datainJoint),len(EWK_datainJoint),len(rares_datainJoint))) :
			if ii<len(weightTTV) :
				hTTW.Fill(ttV_datainJoint[ii],weightTTV[ii])
				hTTW_tt.Fill(ttV_datainTT[ii],weightTTV[ii])
				hTTW_ttV.Fill(ttV_datainTTV[ii],weightTTV[ii])
			if ii<len(weightTT) :
				hTT.Fill(tt_datainJoint[ii],weightTT[ii])
				hTT_tt.Fill(tt_datainTT[ii],weightTT[ii])
				hTT_ttV.Fill(tt_datainTTV[ii],weightTT[ii])
			if ii<len(weightTTH) :
				hTTH.Fill(ttH_datainJoint[ii],weightTTH[ii]) # +0.01
				hTTH_tt.Fill(ttH_datainTT[ii],weightTTH[ii])
				hTTH_ttV.Fill(ttH_datainTTV[ii],weightTTH[ii])
			if ii<len(weightEWK) :
				hEWK.Fill(EWK_datainJoint[ii],weightEWK[ii])
				hEWK_tt.Fill(EWK_datainTT[ii],weightEWK[ii])
				hEWK_ttV.Fill(EWK_datainTTV[ii],weightEWK[ii])
			if ii<len(weightRares) :
				hRares.Fill(rares_datainJoint[ii],weightRares[ii])
				hRares_tt.Fill(rares_datainTT[ii],weightRares[ii])
				hRares_ttV.Fill(rares_datainTTV[ii],weightRares[ii])
			#yTTH=ttH_datainTT #clsTT.predict_proba(dataTTH[trainVarsTT(BDTvar)].values)[:, 1]
			#xTTH=ttH_datainTTV #clsTTV.predict_proba(dataTTH[trainVarsTTV(BDTvar)].values)[:, 1]

		if hTTW.GetBinContent(ntarget) >0 :
			#print (ntarget,hTT.GetBinContent(ntarget),hTT.GetBinContent(ntarget)+hTTW.GetBinContent(ntarget),(hTTH.GetBinContent(ntarget))/(hTT.GetBinContent(ntarget)+hTTW.GetBinContent(ntarget)))
			errOcont.append((hTT.GetBinError(ntarget)+hTTW.GetBinError(ntarget))/(hTT.GetBinContent(ntarget)+hTTW.GetBinContent(ntarget)))
			if hTT.GetBinContent(ntarget) >0 : errOcontTT.append((hTT.GetBinError(ntarget))/(hTT.GetBinContent(ntarget)))
			else : errOcontTT.append(1)
			sOb.append((hTTH.GetBinContent(ntarget))/(hTT.GetBinContent(ntarget)+hTTW.GetBinContent(ntarget)))
		else :
			errOcont.append(1)
			errOcontTT.append(1)
			sOb.append(1)
		##
		if hTTW_tt.GetBinContent(ntarget) >0 :
			#print (ntarget,hTT_tt.GetBinContent(ntarget),hTT_tt.GetBinContent(ntarget)+hTTW_tt.GetBinContent(ntarget),(hTTH_tt.GetBinContent(ntarget))/(hTT_tt.GetBinContent(ntarget)+hTTW_tt.GetBinContent(ntarget)))
			errOcont_tt.append((hTT_tt.GetBinError(ntarget)+hTTW_tt.GetBinError(ntarget))/(hTT_tt.GetBinContent(ntarget)+hTTW_tt.GetBinContent(ntarget)))
			if hTT_tt.GetBinContent(ntarget) >0 : errOcontTT_tt.append((hTT_tt.GetBinError(ntarget))/(hTT_tt.GetBinContent(ntarget)))
			else: errOcontTT_tt.append(1)
			sOb_tt.append((hTTH_tt.GetBinContent(ntarget))/(hTT_tt.GetBinContent(ntarget)+hTTW_tt.GetBinContent(ntarget)))
		else :
			errOcont_tt.append(1)
			errOcontTT_tt.append(1)
			sOb_tt.append(1)
		##
		if hTTW_ttV.GetBinError(ntarget) >0 :
			#print (ntarget,hTT_ttV.GetBinContent(ntarget),hTT_ttV.GetBinContent(ntarget)+hTTW_ttV.GetBinContent(ntarget),(hTTH_ttV.GetBinContent(ntarget))/(hTT_ttV.GetBinContent(ntarget)+hTTW_ttV.GetBinContent(ntarget)))
			errOcont_ttV.append((hTT_ttV.GetBinError(ntarget)+hTTW_ttV.GetBinError(ntarget))/(hTT_ttV.GetBinContent(ntarget)+hTTW_tt.GetBinContent(ntarget)))
			if hTT_ttV.GetBinError(ntarget) >0 :errOcontTT_ttV.append((hTT_ttV.GetBinError(ntarget))/(hTT_ttV.GetBinContent(ntarget)))
			else : errOcontTT_ttV.append(1)
			sOb_ttV.append((hTTH_ttV.GetBinContent(ntarget))/(hTT_ttV.GetBinContent(ntarget)+hTTW_ttV.GetBinContent(ntarget)))
		else :
			errOcont_ttV.append(1)
			errOcontTT_ttV.append(1)
			sOb_ttV.append(1)
		if 0>1 : #if  ntarget in range(3,nmax) :
			hTT.SetFillColor( 17 );
			hTTH.SetFillColor( ROOT.kRed );
			hTTW.SetFillColor( 8 );
			hEWK.SetFillColor( 65 );
			hRares.SetFillColor( 65 );
			mc  = ROOT.THStack("mc","");
			mc.Add(hEWK);
			#mc.Add(hRares);
			mc.Add(hTTW);
			mc.Add(hTTH);
			mc.Add( hTT );
			c4 = ROOT.TCanvas("c5","",500,500);
			c4.cd();
			c4.Divide(1,2,0,0);
			c4.cd(1)
			ROOT.gPad.SetLogy()
			#c5.SetLogy()
			ROOT.gPad.SetBottomMargin(0.001)
			ROOT.gPad.SetTopMargin(0.065)
			ROOT.gPad.SetRightMargin(0.01)
			ROOT.gPad.SetLeftMargin(0.12)
			#ROOT.gPad.SetLabelSize(.4, "XY")
			mc.SetMinimum(max(0.04* mc.GetMinimum(),0.01));
			mc.Draw("HIST");
			mc.SetMaximum(15* mc.GetMaximum());
			mc.SetMinimum(max(0.04* mc.GetMinimum(),0.01));
			mc.GetYaxis().SetRangeUser(0.01,110);
			mc.GetHistogram().GetYaxis().SetTitle("Expected events/bin");
			mc.GetHistogram().GetXaxis().SetTitle("Bin in the bdt1#times bdt2 plane");
			mc.GetHistogram().GetXaxis().SetTitleSize(0.06);
			mc.GetHistogram().GetXaxis().SetLabelSize(.06); #SetTitleOffset(1.1);
			mc.GetHistogram().GetYaxis().SetTitleSize(0.06);
			mc.GetHistogram().GetYaxis().SetLabelSize(.06);
			#mc.GetHistogram().GetYaxis().SetTitleOffset(1.1);
			l = ROOT.TLegend(0.16,0.6,0.3,0.9);
			l.AddEntry(hTTH  , "ttH", "f");
			l.AddEntry(hTTW  , "ttV"       , "f");
			l.AddEntry(hTT, "tt"        , "f");
			#l.AddEntry(hRares, "rares"        , "f");
			l.AddEntry(hEWK, "Rares"        , "f");
			l.Draw();
			latex= ROOT.TLatex();
			latex.SetTextSize(0.065);
			latex.SetTextAlign(13);  #//align at top
			latex.SetTextFont(62);
			latex.DrawLatexNDC(.15,1.0,"CMS Simulation");
			latex.DrawLatexNDC(.8,1.0,"#it{36 fb^{-1}}");
			#latex1.DrawLatexNDC(0.5,.77,"looseLep")
			#latex.DrawLatexNDC(.55,.8,"from ("+str(nbins)+"^2) to "+str(nbinsout)+" bins");
			latex.DrawLatexNDC(.55,.9,BDTvar);
			#"""
			c4.cd(2)
			#ROOT.gPad.SetLogy()
			ROOT.gStyle.SetHatchesSpacing(100)
			#c5.SetLogy(1)
			ROOT.gPad.SetLeftMargin(0.12)
			ROOT.gPad.SetBottomMargin(0.12)
			ROOT.gPad.SetTopMargin(0.001)
			ROOT.gPad.SetRightMargin(0.005)
			#ROOT.gPad.SetLabelSize(.1, "XY")
			if not hTT.GetSumw2N() : hTT.Sumw2()
			#if not hTTW.GetSumw2N() : hTTW.Sumw2()
			h2=hTT.Clone()
			h2.Add(hTTW)
			hBKG1D=h2.Clone()
			h3=hTTH.Clone()
			h4=hTT.Clone()
			#h3=Divide(h3,h2)
			#"""
			#h3=hTT.Clone()
			if not h2.GetSumw2N() : h2.Sumw2()
			if not h3.GetSumw2N() : h3.Sumw2()
			for binn in range(0,h2.GetNbinsX()+1) :
				ratio=0
				ratio3=0
				if h2.GetBinContent(binn) >0 :
					if h2.GetBinContent(binn) > 0 :ratio=h2.GetBinError(binn)/h2.GetBinContent(binn)
					h2.SetBinContent(binn,ratio)
				if h3.GetBinContent(binn) > 0 :
					if hBKG1D.GetBinContent(binn)> 0 :ratio3=h3.GetBinContent(binn)/hBKG1D.GetBinContent(binn)
					h3.SetBinContent(binn,ratio3)
				if h4.GetBinContent(binn) > 0 : h4.SetBinContent(binn,h4.GetBinError(binn)/h4.GetBinContent(binn))
				#print (binn,ratio,ratio3)
			#"""
			h2.SetLineWidth(3)
			h2.SetLineColor(2)
			h2.SetFillStyle(3690)
			h3.GetYaxis().SetRangeUser(0.01,4.5);
			h3.SetLineWidth(3)
			h3.SetFillStyle(3690)
			h3.SetLineColor(28)
			h4.SetLineWidth(3)
			h4.SetFillStyle(3690)
			h4.SetLineColor(6)
			h3.Draw("HIST")
			h3.GetYaxis().SetTitle("S/B");
			#latex.DrawLatexNDC(.15,.65,"S/B");
			h3.GetXaxis().SetTitle("BDT");
			h3.GetYaxis().SetTitleSize(0.06);
			h3.GetYaxis().SetLabelSize(.06)
			h3.GetXaxis().SetTitleSize(0.06);
			h3.GetXaxis().SetLabelSize(.06)
			l2 = ROOT.TLegend(0.16,0.77,0.4,0.98);
			l2.AddEntry(h3  , "S/B" , "l");
			l2.AddEntry(h2  , "ttV + tt err/cont", "l");
			l2.AddEntry(h4  , "tt err/cont", "l");
			l2.Draw("same");
			h2.Draw("HIST,SAME")
			h4.Draw("HIST,SAME")
			#c5.SetLogy(1)
			#"""
			c4.Modified();
			c4.Update();
			print ("s/B in last bin (tight)",
					h3.GetBinContent(h3.GetNbinsX()), # /hBKG1D.GetBinContent(h3.GetNbinsX())
					h3.GetBinContent(h3.GetNbinsX()-1), #/hBKG1D.GetBinContent(h3.GetNbinsX()-1)
					h2.GetBinContent(h3.GetNbinsX())
					)
			c4.SaveAs(channel+'/'+BDTvar+'_ntarget'+str(ntarget)+'_jointBDT_fullsim.pdf')
	"""
	print ("err/ct Joint",errOcont)
	print ("err/ct Joint, only TT",errOcontTT)
	print ("S/B Joint",sOb)
	#
	print ("err/ct ttBDT",errOcont_tt)
	print ("err/ct ttBDT, only TT",errOcontTT_tt)
	print ("S/B ttBDT",sOb_tt)
	#
	print ("err/ct ttVBDT",errOcont_ttV)
	print ("err/ct ttVBDT, only TT",errOcontTT_ttV)
	print ("S/B ttVBDT",sOb_ttV)
	"""

	xbinstarget=np.arange(1, nmax)
	print (len(xbinstarget),len(sOb))
	fig, ax = plt.subplots(figsize=(4, 4))
	# plt.subplot(221)
	plt.plot(xbinstarget,sOb,'ro-',label="Joint-BDT")
	plt.plot(xbinstarget,sOb_tt,'go-',label="tt-BDT")
	plt.plot(xbinstarget,sOb_ttV,'bo-',label="ttV-BDT")
	ax.legend(loc="best")
	ax.set_xlabel('nbins')
	ax.set_ylabel('ttH/(ttV+tt)')
	plt.grid(True)
	fig.savefig(channel+'/'+BDTvar+'_jointBDT_'+BDTtype+'_fullsim_SoB.pdf')
	plt.clf()
	###
	fig, ax = plt.subplots(figsize=(4, 4))
	# plt.subplot(221)
	plt.plot(xbinstarget,errOcont,'ro-',label="Joint-BDT (TT+TTV)")
	plt.plot(xbinstarget,errOcont_tt,'go-',label="tt-BDT (TT+TTV)")
	plt.plot(xbinstarget,errOcont_ttV,'bo-',label="ttV-BDT (TT+TTV)")
	plt.plot(xbinstarget,errOcontTT, 'ro--',label="Joint-BDT (TT only)")
	plt.plot(xbinstarget,errOcontTT_tt, 'go--',label="tt-BDT (TT only)")
	plt.plot(xbinstarget,errOcontTT_ttV, 'bo--',label="ttV-BDT (TT only)")
	plt.plot(xbinstarget,np.array([0.17]*len(xbinstarget)), 'k-')
	plt.plot(xbinstarget,np.array([0.41]*len(xbinstarget)), 'k-')
	#ax.set_ylim(ymin=0.009)
	#ax.set_xlim(xmin=0.01)
	#ax.set_yscale('log')
	#ax.set_xscale('log')
	ax.set_xlabel('nbins')
	ax.set_ylabel('err/content')
	ax.legend(loc="best")
	plt.grid(True)
	fig.savefig(channel+'/'+BDTvar+'_jointBDT_'+BDTtype+'_fullsim_ErrOcont.pdf')
	plt.clf()
	"""
	###########################################################################
	plt.figure(figsize=(15, 5))

	plt.subplot(1, 3, 0+1)
	plt.hist2d(data.ix[(data.key.values=='TTTo2L2Nu') | (data.key.values=='TTToSemilepton'),"BDTttV"].values.astype(np.float64) ,
				data.ix[(data.key.values=='TTTo2L2Nu') | (data.key.values=='TTToSemilepton'),"BDTtt"].values.astype(np.float64) ,
				weights= data.ix[(data.key.values=='TTTo2L2Nu') | (data.key.values=='TTToSemilepton'), weights].values.astype(np.float64) ,
				bins=10,normed=True)
	plt.title("ttbar (SL+DL) sample")

	plt.subplot(1, 3, 1+1)
	plt.hist2d(data.ix[(data.key.values=='TTZToLLNuNu') | (data.key.values=='TTWJetsToLNu'),"BDTttV"].values.astype(np.float64) ,
				data.ix[(data.key.values=='TTZToLLNuNu') | (data.key.values=='TTWJetsToLNu'),"BDTtt"].values.astype(np.float64) ,
				weights= data.ix[(data.key.values=='TTZToLLNuNu') | (data.key.values=='TTWJetsToLNu'), weights].values.astype(np.float64) ,
				bins=10,normed=True)
	plt.title("ttW+ttZ sample")

	plt.subplot(1, 3, 2+1)
	plt.hist2d(data.ix[(data.key.values=='ttHToNonbb'),"BDTttV"].values.astype(np.float64) ,
					 	data.ix[(data.key.values=='ttHToNonbb'),"BDTtt"].values.astype(np.float64) ,
						weights= data.ix[(data.key.values=='ttHToNonbb'), weights].values.astype(np.float64) ,
						bins=10, normed=True)
	plt.title("ttH sample")

	plt.savefig(channel+'/'+BDTvar+'_jointBDT_2D.pdf')
	plt.clf()


	plt.figure(figsize=(10, 5))
	plt.subplot(1, 2, 0+1)
	plt.hist(data.ix[(data.key.values=='TTTo2L2Nu') | (data.key.values=='TTToSemilepton'),"BDTttV"].values.astype(np.float64) ,
				weights= data.ix[(data.key.values=='TTTo2L2Nu') | (data.key.values=='TTToSemilepton'), weights].values.astype(np.float64) ,
				bins=10,normed=True,label="tt sample")
	plt.hist(data.ix[(data.key.values=='TTZToLLNuNu') | (data.key.values=='TTWJetsToLNu'),"BDTttV"].values.astype(np.float64) ,
				weights= data.ix[(data.key.values=='TTZToLLNuNu') | (data.key.values=='TTWJetsToLNu'), weights].values.astype(np.float64) ,
				bins=10,normed=True,label="ttV sample")
	plt.hist(data.ix[(data.key.values=='ttHToNonbb'),"BDTttV"].values.astype(np.float64) ,
				weights= data.ix[(data.key.values=='ttHToNonbb'), weights].values.astype(np.float64) ,
				bins=10,normed=True,label="ttH")
	plt.legend(loc='best')
	plt.subplot(1, 2, 1+1)
	plt.hist(data.ix[(data.key.values=='TTTo2L2Nu') | (data.key.values=='TTToSemilepton'),"BDTtt"].values.astype(np.float64) ,
				weights= data.ix[(data.key.values=='TTTo2L2Nu') | (data.key.values=='TTToSemilepton'), weights].values.astype(np.float64) ,
				bins=30,normed=True,label="tt sample")
	plt.hist(data.ix[(data.key.values=='TTZToLLNuNu') | (data.key.values=='TTWJetsToLNu'),"BDTtt"].values.astype(np.float64) ,
				weights= data.ix[(data.key.values=='TTZToLLNuNu') | (data.key.values=='TTWJetsToLNu'), weights].values.astype(np.float64) ,
				bins=30,normed=True,label="ttV sample")
	plt.hist(data.ix[(data.key.values=='ttHToNonbb'),"BDTtt"].values.astype(np.float64) ,
				weights= data.ix[(data.key.values=='ttHToNonbb'), weights].values.astype(np.float64) ,
				bins=30,normed=True,label="ttH")
	plt.legend(loc='best')
	plt.savefig(channel+'/'+BDTvar+'_jointBDT_1D.pdf')
	plt.clf()


	x=data.ix[data.target.values == 0, "BDTtt"].values
	y=data.ix[data.target.values == 1, "BDTttV"].values
	x_min, x_max = x.min() - 1, x.max() + 1
	y_min, y_max = y.min() - 1, y.max() + 1
	#h=.02
	xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
	Z = cls.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

	plt.hist2d(data.ix[data.target.values == 0, "BDTtt"].values , data.ix[data.target.values == 0, "BDTttV"].values ,
							   weights= abs(data.ix[data.target.values == 0, weights].values.astype(np.float64)) ,
                               cmap="Oranges",normed=True)
	plt.hist2d(data.ix[data.target.values == 1, "BDTttV"].values , data.ix[data.target.values == 1, "BDTtt"].values ,
							   weights= abs(data.ix[data.target.values == 1, weights].values.astype(np.float64)) ,
							   cmap="Purples",normed=True)
	plt.title("test")
	plt.savefig(channel+"/"+BDTvar+"_2D_BDT.pdf")


	xx, yy = make_meshgrid(X0, X1)

	#for clf, title, ax in zip(models, titles, sub.flatten()):
	plot_contours(ax, clf, xx, yy,cmap=plt.cm.coolwarm, alpha=0.8)
	ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
	ax.set_xlim(xx.min(), xx.max())
	ax.set_ylim(yy.min(), yy.max())
	ax.set_xlabel('Sepal length')
	ax.set_ylabel('Sepal width')
	ax.set_xticks(())
	ax.set_yticks(())
	ax.set_title("test")
	"""
