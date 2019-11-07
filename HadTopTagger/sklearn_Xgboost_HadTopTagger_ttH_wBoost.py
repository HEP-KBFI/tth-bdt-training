import sys , time

#######
"""
python sklearn_Xgboost_HadTopTagger_ttH_wBoost.py --process "HTT" --ntrees 2500 --treeDeph 5 --lr 0.03 --mcw 100 --doXML --cat 3 --withKinFit --btagSort > HadTopTagger_2017_boosted_final/HTT_cat3_sel5_withKinFit.log &
python sklearn_Xgboost_HadTopTagger_ttH_wBoost.py --process "HTT" --ntrees 2500 --treeDeph 5 --lr 0.03 --mcw 10 --doXML --cat 3 --btagSort  > HadTopTagger_2017_boosted_final/HTT_cat3_sel5.log &
python sklearn_Xgboost_HadTopTagger_ttH_wBoost.py --process "HTT" --ntrees 2500 --treeDeph 5 --lr 0.1 --mcw 10 --doXML --cat 3 > HadTopTagger_2017_boosted_final/HTT_cat3_sel6.log &
python sklearn_Xgboost_HadTopTagger_ttH_wBoost.py --process "HTT" --ntrees 2500 --treeDeph 4 --lr 0.1 --mcw 100 --doXML --withKinFit --cat 3 > HadTopTagger_2017_boosted_final/HTT_cat3_sel6_withKinFit.log &
##
python sklearn_Xgboost_HadTopTagger_ttH_wBoost.py --process "HTT" --ntrees 2500 --treeDeph 5 --lr 0.03 --mcw 100 --doXML --cat 3 --withKinFit --btagSort > HadTopTagger_2017_boosted_final/HTT_cat3_sel9_withKinFit.log &
python sklearn_Xgboost_HadTopTagger_ttH_wBoost.py --process "HTT" --ntrees 2500 --treeDeph 5 --lr 0.03 --mcw 10 --doXML --cat 3 --btagSort  > HadTopTagger_2017_boosted_final/HTT_cat3_sel9.log &
##
python sklearn_Xgboost_HadTopTagger_ttH_wBoost.py --process "HTT" --ntrees 2500 --treeDeph 4 --lr 0.03 --mcw 100 --cat 2 --doXML --btagSort  > HadTopTagger_2017_boosted_final/HTT_cat2_sel5.log &
python sklearn_Xgboost_HadTopTagger_ttH_wBoost.py --process "HTT" --ntrees 2500 --treeDeph 4 --lr 0.03 --mcw 100 --cat 2 --withKinFit --doXML --btagSort  > HadTopTagger_2017_boosted_final/HTT_cat2_sel5_withKinFit.log &
---
python sklearn_Xgboost_HadTopTagger_ttH_wBoost.py --process "HTT" --ntrees 2500 --treeDeph 4 --lr 0.03 --mcw 100 --cat 2 --doXML --btagSort  > HadTopTagger_2017_boosted_final/HTT_cat2_sel9.log &
python sklearn_Xgboost_HadTopTagger_ttH_wBoost.py --process "HTT" --ntrees 2500 --treeDeph 4 --lr 0.03 --mcw 100 --cat 2 --withKinFit --doXML --btagSort  > HadTopTagger_2017_boosted_final/HTT_cat2_sel9_withKinFit.log &
###
python sklearn_Xgboost_HadTopTagger_ttH_wBoost.py --process "HTT" --ntrees 2500 --treeDeph 4 --lr 0.03 --mcw 100 --cat 1 --doXML  > HadTopTagger_2017_boosted_final/HTT_cat1_sel3.log &
python sklearn_Xgboost_HadTopTagger_ttH_wBoost.py --process "HTT" --ntrees 2500 --treeDeph 4 --lr 0.03 --mcw 100 --cat 1 --withKinFit --doXML > HadTopTagger_2017_boosted_final/HTT_cat1_sel3_withKinFit.log &
---
python sklearn_Xgboost_HadTopTagger_ttH_wBoost.py --process "HTT" --ntrees 2500 --treeDeph 4 --lr 0.03 --mcw 100 --cat 1 --doXML  > HadTopTagger_2017_boosted_final/HTT_cat1_sel1.log &
python sklearn_Xgboost_HadTopTagger_ttH_wBoost.py --process "HTT" --ntrees 2500 --treeDeph 4 --lr 0.03 --mcw 100 --cat 1 --withKinFit --doXML > HadTopTagger_2017_boosted_final/HTT_cat1_sel1_withKinFit.log &
###
#run: python sklearn_Xgboost_HadTopTagger_ttH_wBoost.py --process HTTwboost_Summer2018 --ntrees 500 --treeDeph 3 --lr 0.01' --evaluateFOM --HypOpt --doXML &
"""
import os
import sklearn
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import pandas
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import numpy as np
import pickle
import root_numpy
from root_numpy import root2array, rec2array, array2root, tree2array
import ROOT
import glob
import xgboost as xgb
execfile("../python/data_manager.py")
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--process", type="string", dest="process", help="process", default='ttH')
parser.add_option("--evaluateFOM", action="store_true", dest="evaluateFOM", help="evaluateFOM", default=False)
parser.add_option("--HypOpt", action="store_true", dest="HypOpt", help="If you call this will not do plots with repport", default=False)
parser.add_option("--withKinFit", action="store_true", dest="withKinFit", help="BDT variables with kinfit", default=False)
parser.add_option("--doXML", action="store_true", dest="doXML", help="BDT variables with kinfit", default=False)
parser.add_option("--ntrees", type="int", dest="ntrees", help="hyp", default=2000)
parser.add_option("--treeDeph", type="int", dest="treeDeph", help="hyp", default=5)
parser.add_option("--lr", type="float", dest="lr", help="hyp", default=0.01)
parser.add_option("--mcw", type="float", dest="mcw", help="hyp", default=10)
parser.add_option("--cat", type="int", dest="cat", help="hyp", default=1)
parser.add_option("--btagSort", action="store_true", dest="btagSort", help="the BKG selection", default=False)
(options, args) = parser.parse_args()

channel="HadTopTagger_2017_final_nomasscut" # "HadTopTagger_wBoost" #options.channel #"1l_2tau"
inputPath='structured/'
process=options.process
keys=['ttHToNonbb','TTToSemilepton','TTZToLLNuNu','TTWJetsToLNu']
if options.btagSort :
    bdtType="btagSort3rd"
    CSVsort=True
else :
    bdtType="btagHighest"
    CSVsort=False
withKinFit=options.withKinFit
doXML=options.doXML
trainvar="ntrees_"+str(options.ntrees)+"_deph_"+str(options.treeDeph)+"_lr_0o0"+str(int(options.lr*100))

inputTree="TCVARSbfilter"
target="bWj1Wj2_isGenMatched"

HypOpt=options.HypOpt
category = options.cat
jet = 12

makeplots = True
# reduction = 0.97

import shutil,subprocess
proc=subprocess.Popen(['mkdir '+channel],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()

def trainVars(cat,train):
	if cat==1 and train==False :
		return [
        #'genTopPt',
        #'bjet_tag_position',
        'massTop',
        'tau32Top',
        #"btagDisc",
        "drT_gen", "drWj1_gen", "drWj2_gen", "drB_gen", "drW_gen",
        "etaWj1_gen", "etaWj2_gen", "etaB_gen",
        "ptWj1_gen", "ptWj2_gen", "ptB_gen",
        "dr_b_wj1", "dr_b_wj2", "dr_wj1_wj2",
        "genFatPtAll", "genFatEtaAll"
        #'collectionSize',
        #"fatjet_isGenMatched"
		]

	if cat==2 and train==False :
		return [
        'genTopPt',
        #'bWj1Wj2_isGenMatched',
        'bjet_tag_position',
        #'massTop',
        'massW_SD',
        'tau21W',
        #'tau32Top',
        "btagDisc_b",
        #"qg_Wj1",
        #"qg_Wj2",
        'collectionSize',
        "fatjet_isGenMatched",
        "drT_gen", "drWj1_gen", "drWj2_gen", "drB_gen", "drW_gen",
        "etaWj1_gen", "etaWj2_gen", "etaB_gen",
        "ptWj1_gen", "ptWj2_gen", "ptB_gen",
        "dr_b_wj1", "dr_b_wj2", "dr_wj1_wj2",
        "dr_b_wj1_gen", "dr_b_wj2_gen", "dr_wj1_wj2_gen",
        #'typeTop'
		]

	if cat==3 and train==False :
		return [
        'genTopPt',
        #'bWj1Wj2_isGenMatched',
        'bjet_tag_position',
        #'massTop',
        #'massW_SD',
        #'tau21W',
        #'tau32Top',
        "btagDisc_b",
        "qg_Wj1",
        "qg_Wj2",
        'collectionSize',
        #"fatjet_isGenMatched",
        #'typeTop'
		]

	if cat==1 and train==True :
		variables_train = [
        #'massTop',
        'tau32Top',
        #"btagDisc_b",
        #"btagDisc_Wj1",
        #"btagDisc_Wj2",
        #"qg_Wj1",
        #"qg_Wj2",
        "m_Wj1Wj2_div_m_bWj1Wj2",
        #'m_Wj1Wj2',
        'pT_Wj1Wj2',
        "dR_Wj1Wj2",
        'm_bWj1Wj2',
        'pT_bWj1Wj2',
        #"dR_bW",
        #'m_bWj1',
        #"dR_bWj1",
        'm_bWj2',
        #"dR_bWj2",
        #'pT_Wj1',
        #'mass_Wj1',
        'pT_Wj2',
        #'mass_Wj2',
        'pT_b',
        #"mass_b",
        #'cosThetab_rest',
        #"HTTv2_area", "HTTv2_Ropt", "HTTv2_RoptCalc"
        #'kinFit_pT_b_o_pT_b',
        #'kinFit_pT_Wj1_o_pT_Wj1',
        #'kinFit_pT_Wj2_o_pT_Wj2',
        #'nllKinFit',
	#'m23_div_m123',
	'atan_m13_div_m12',
	'Rmin_square_one_plus_m13_div_m12_square',
	'Rmax_square_one_plus_m13_div_m12_square',
		]

	if cat==2 and train==True :
		variables_train = [
        #'genTopPt',
        #'bWj1Wj2_isGenMatched',
        #'bjet_tag_position',
        #'massTop',
        'massW_SD',
        'tau21W',
        "btagDisc_b",
        #"btagDisc_Wj1",
        #"btagDisc_Wj2",
        #"qg_Wj1",
        #"qg_Wj2",
        "m_Wj1Wj2_div_m_bWj1Wj2",
        #'m_Wj1Wj2',
        #'pT_Wj1Wj2',
        "dR_Wj1Wj2",
        'm_bWj1Wj2',
        'pT_bWj1Wj2',
        #"dR_bW",
        #'m_bWj1',
        #"dR_bWj1",
        'm_bWj2',
        #"dR_bWj2",
        #'pT_Wj1',
        'mass_Wj1',
        'pT_Wj2',
        'mass_Wj2',
        'pT_b',
        "mass_b",
        #'cosThetab_rest',
        #"HTTv2_area", "HTTv2_Ropt", "HTTv2_RoptCalc"
        #'kinFit_pT_b_o_pT_b',
        #'kinFit_pT_Wj1_o_pT_Wj1',
        #'kinFit_pT_Wj2_o_pT_Wj2',
        #'nllKinFit',
		]

	if cat==3 and train==True and CSVsort == True:
		variables_train = [
        "btagDisc_b",
        "btagDisc_Wj1",
        "btagDisc_Wj2",
        "qg_Wj1",
        "qg_Wj2",
        "m_Wj1Wj2_div_m_bWj1Wj2",
        #'m_Wj1Wj2',
        'pT_Wj1Wj2',
        "dR_Wj1Wj2",
        'm_bWj1Wj2',
        #'pT_bWj1Wj2',
        "dR_bW",
        'm_bWj1',
        #"dR_bWj1",
        'm_bWj2',
        #"dR_bWj2",
        #'pT_Wj1',
        'mass_Wj1',
        'pT_Wj2',
        'mass_Wj2',
        'pT_b',
        "mass_b",
        ##'alphaKinFit',
        #'kinFit_pT_b_o_pT_b',
        #'kinFit_pT_Wj1_o_pT_Wj1',
        #'kinFit_pT_Wj2_o_pT_Wj2',
        #'nllKinFit',
        #"kinFit_pT_b",
        #'kinFit_pT_Wj1',
        #'kinFit_pT_Wj2',
		]

	if cat==3 and train==True and CSVsort == False:
		variables_train =  [
        "btagDisc_b",
        "btagDisc_Wj1",
        "btagDisc_Wj2",
        "qg_Wj1",
        "qg_Wj2",
        "m_Wj1Wj2_div_m_bWj1Wj2",
        #'m_Wj1Wj2',
        'pT_Wj1Wj2',
        "dR_Wj1Wj2",
        'm_bWj1Wj2',
        #'pT_bWj1Wj2',
        "dR_bW",
        'm_bWj1',
        #"dR_bWj1",
        'm_bWj2',
        #"dR_bWj2",
        #'pT_Wj1',
        'mass_Wj1',
        'pT_Wj2',
        'mass_Wj2',
        'pT_b',
        "mass_b",
        ##'alphaKinFit',
        #'kinFit_pT_b_o_pT_b',
        #'kinFit_pT_Wj1_o_pT_Wj1',
        #'kinFit_pT_Wj2_o_pT_Wj2',
        #'nllKinFit',
        #"kinFit_pT_b",
        #'kinFit_pT_Wj1',
        #'kinFit_pT_Wj2',
		]
	if withKinFit : variables_train = variables_train + ['kinFit_pT_b_o_pT_b', 'kinFit_pT_Wj2_o_pT_Wj2', 'nllKinFit']
	return variables_train

keystoDraw=['ttHToNonbb','TTToSemilepton','TTWJetsToLNu']
#treetoread="analyze_hadTopTagger/evtntuple/signal/evtTree"
#sourceA="/hdfs/local/acaan/HTT_withBoost/ttHJetToNonbb_M125_amcatnlo_ak"+str(jet)+"_noCleaning_HTTv2loop_R0o3_higestBtag_fixGen_0p4res.root"

#treetoread="analyze_hadTopTagger/evtntuple/TT/evtTree"
#sourceA="/hdfs/local/acaan/HTT_withBoost/TTToHadronic_2.root"

treetoread=[
    #"analyze_hadTopTagger/evtntuple/TTW/evtTree",
    "analyze_hadTopTagger/evtntuple/TTZ/evtTree",
    #"analyze_hadTopTagger/evtntuple/TTZ/evtTree",
    "analyze_hadTopTagger/evtntuple/signal/evtTree",
    #"analyze_hadTopTagger/evtntuple/signal/evtTree",
    #"analyze_hadTopTagger/evtntuple/TT/evtTree",
    "analyze_hadTopTagger/evtntuple/TT/evtTree",
    ]
sources = [
    #"/hdfs/local/acaan/ttHAnalysis/2017/HTT_wBoost_2018Sep13/histograms/hadTopTagger/TTWJets_LO/TTWJets_LO_1.root",
    #"/hdfs/local/acaan/ttHAnalysis/2017/HTT_wBoost_2018Sep13/histograms/hadTopTagger/TTZJets_LO/TTZJets_LO_1.root",
    #"/hdfs/local/acaan/ttHAnalysis/2017/HTT_wBoost_2018Sep13/histograms/hadTopTagger/TTZJets_LO/TTZJets_LO_2.root",
    #"/hdfs/local/acaan/ttHAnalysis/2017/HTT_wBoost_2018Sep13/histograms/hadTopTagger/ttHToNonbb_M125_powheg/ttHToNonbb_M125_powheg_1.root",
    #"/hdfs/local/acaan/ttHAnalysis/2017/HTT_wBoost_2018Sep13/histograms/hadTopTagger/ttHToNonbb_M125_powheg/ttHToNonbb_M125_powheg_3.root",
    #"/hdfs/local/acaan/ttHAnalysis/2017/HTT_wBoost_2018Sep13/histograms/hadTopTagger/TTToHadronic_PSweights/TTToHadronic_PSweights_14.root",
    #"/hdfs/local/acaan/ttHAnalysis/2017/HTT_wBoost_2018Sep13/histograms/hadTopTagger/TTToHadronic/TTToHadronic_5.root",
    #"/hdfs/local/acaan/ttHAnalysis/2017/HTT_wBoost_2018Sep13/histograms/hadTopTagger/TTToHadronic/TTToHadronic_1.root",
    "/hdfs/local/karmakar/ttHAnalysis/2017/HTTv2_Nov22/histograms/hadTopTagger/TTZJets_LO/TTZJets_LO_1.root",
    "/hdfs/local/karmakar/ttHAnalysis/2017/HTTv2_Nov22/histograms/hadTopTagger/ttHToNonbb_M125_powheg/ttHToNonbb_M125_powheg_1.root",
    #"/hdfs/local/karmakar/ttHAnalysis/2017/HTTv2_Nov22/histograms/hadTopTagger/ttHToNonbb_M125_powheg/ttHToNonbb_M125_powheg_3.root",
    "/hdfs/local/karmakar/ttHAnalysis/2017/HTTv2_Nov22/histograms/hadTopTagger/TTToHadronic/TTToHadronic_1.root",
    ]

weights = "weights"
branch = [x for x in trainVars(category,True) if x not in ['kinFit_pT_b_o_pT_b','kinFit_pT_Wj1_o_pT_Wj1','kinFit_pT_Wj2_o_pT_Wj2']]  + [
target, "counter", "fatjet_isGenMatched",  'kinFit_pT_b', 'kinFit_pT_Wj1', 'kinFit_pT_Wj2',
#'m_Wj1Wj2',
'pT_Wj1',
"drT_gen", "drT_genTriplet", "drT_genJ_max",
"drWj1_gen", "drWj2_gen", "drB_gen",
"dr_wj1_wj2_gen", "dr_b_wj1_gen", "dr_b_wj2_gen",
"genTopMassFromW", "genTopMassFromWj", "genWMassFromWj", "genAntiWMass","genWMass",
"genAntiTopMassFromW", "genAntiTopMassFromWj", "genAntiWMassFromWj",
"genFatPtAll", "genTopPt",
"bjet_tag_position"
]
data = pandas.DataFrame(columns=branch+[weights], index=['bWj1Wj2_isGenMatched',"fatjet_isGenMatched"])
data1 = pandas.DataFrame(columns=branch+[weights], index=['bWj1Wj2_isGenMatched',"fatjet_isGenMatched"])
data2 = pandas.DataFrame(columns=branch+[weights], index=['bWj1Wj2_isGenMatched',"fatjet_isGenMatched"])

print ("Date: ", time.asctime( time.localtime(time.time()) ))
base='typeTop == '+str(category) # && passHadGenHadW
selections = [
   base,
   base+' && passJetSel ',
   base+' && passJetSel && m_bWj1Wj2 > 75 && m_bWj1Wj2 < 275',
   base+' && passJetSel && m_bWj1Wj2 > 75 && m_bWj1Wj2 < 275 && m_Wj1Wj2 < 150',
   base+' && passJetSel && m_bWj1Wj2 > 75 && m_bWj1Wj2 < 275 && m_Wj1Wj2 < 150 && bjet_tag_position <= 4  ',
   base+' && passJetSel && m_bWj1Wj2 > 75 && m_bWj1Wj2 < 275 && m_Wj1Wj2 < 150 && bjet_tag_position <= 3 ',
   base+' && passJetSel && m_bWj1Wj2 > 75 && m_bWj1Wj2 < 275 && m_Wj1Wj2 < 150 && btagDisc_Wj1 < btagDisc_b && btagDisc_Wj2 < btagDisc_b',
   base+' && passJetSel && bjet_tag_position <= 3',
   base+' && passJetSel && btagDisc_Wj1 < btagDisc_b && btagDisc_Wj2 < btagDisc_b',
   base+' && passJetSel && bjet_tag_position <= 4',
]

for ss, sel in enumerate(selections) :
    total_evt = 0
    total_sig_evt = 0
    total_bkg_entries = 0
    total_sig_entries = 0
    total_bkg_entries_reduced = 0
    print (ss, category)
    if ss != 1 and category == 1 : continue
    if ss != 9 and category == 2 : continue
    if category == 3 :
        #if ss != 1 and CSVsort == True  : continue
        if ss != 9 and CSVsort == True  : continue
        if ss != 8 and CSVsort == False  : continue
    #nev = 0
    print sel
    for tt, sourceA in enumerate(sources) :
        tfile = ROOT.TFile(sourceA)
        tree = tfile.Get(treetoread[tt])
        chunk_arr = tree2array(tree, selection=sel, branches=branch, cache_size=0)
        print trainVars(category,True)
        print ("Date: ", time.asctime( time.localtime(time.time()) ))
        chunk_df = pandas.DataFrame(data=chunk_arr, columns=branch)
        print "data loaded"
        print ("Date: ", time.asctime( time.localtime(time.time()) ))
        chunk_df[weights] = 1.0
        #nev = nev + len(np.unique(chunk_df["counter"].values))
        total_evt += len(np.unique(chunk_df["counter"].values))
        total_sig_evt += len(np.unique(chunk_df.loc[chunk_df[target]==1]["counter"].values))
        total_bkg_entries += len(chunk_df.loc[chunk_df[target]==0])
        total_sig_entries += len(chunk_df.loc[chunk_df[target]==1])
        print ("nev sig", len(np.unique(chunk_df.loc[chunk_df[target]==1]["counter"].values)), " nev total ", len(np.unique(chunk_df["counter"].values)))
        print ("len sig/BKG : ",len(chunk_df.loc[chunk_df[target]==1]),len(chunk_df.loc[chunk_df[target]==0]))
        #################################################################################
        #chunk_df.loc[chunk_df[target] == 0] = chunk_df.loc[chunk_df[target] == 0].drop(drop_indices, inplace = True)
        ########################################################################################
        print (ss, "Doing more branches: ", time.asctime( time.localtime(time.time()) ))
        #chunk_df['key'] = keystoDraw[0]
        #chunk_df['CSV_b'] = chunk_df["btagDisc_b"]
        #chunk_df["genTopMass_FromW"] = chunk_df["genTopMassFromW"] + chunk_df["genAntiTopMassFromW"]
        #chunk_df["genTopMass_FromWjs"] = chunk_df["genTopMassFromWj"] + chunk_df["genAntiTopMassFromWj"]
        #chunk_df["WMass_FromWj"] = chunk_df["genWMassFromWj"] + chunk_df["genAntiWMassFromWj"]
        #chunk_df["WMass"] =  chunk_df["genAntiWMass"] + chunk_df["genWMass"]
        #chunk_df['m_Wj1Wj2_div_m_bWj1Wj2'] =  chunk_df['m_Wj1Wj2']/chunk_df['m_bWj1Wj2']
        chunk_df['kinFit_pT_b_o_pT_b'] =  chunk_df['kinFit_pT_b']/chunk_df['pT_b']
        chunk_df['kinFit_pT_Wj1_o_pT_Wj1'] =  chunk_df['kinFit_pT_Wj1']/chunk_df['pT_Wj1']
        chunk_df['kinFit_pT_Wj2_o_pT_Wj2'] =  chunk_df['kinFit_pT_Wj2']/chunk_df['pT_Wj2']
        #chunk_df['cosThetaWj1_restW'] = abs(chunk_df['cosThetaWj1_restW'])
        # "m_Wj1Wj2_div_m_bWj1Wj2"
        #chunk_df['bjet_tag_position']= np.where(chunk_df['bjet_tag_position'] > 3, 4, chunk_df['bjet_tag_position'])
        #chunk_df['target'] = 0
        #chunk_df['target'] = np.where((chunk_df["drT_genTriplet"] < 1.5) & (chunk_df["drB_gen"] < 0.5) , 1, chunk_df['target'])
        #dr_match = 0.3
        #chunk_df['target'] = np.where((chunk_df["drWj1_gen"] < dr_match) & (chunk_df["drWj2_gen"] < dr_match) &  (chunk_df["drB_gen"] < dr_match) , 1, chunk_df['target']) #
        #chunk_df['target'] = np.where((chunk_df["drT_gen"] < 0.75) , 1, chunk_df['target'])
        # & (chunk_df["drWj1_gen"] < 0.75) & (chunk_df["drWj2_gen"] < 0.75) & (chunk_df["bjet_tag_position"] == 1)
        #chunk_df.replace([np.inf, -np.inf], np.nan).dropna(subset=branch, inplace = True)
        print (ss, "read one file: ", sourceA)
        if category == 3 : reduction = 0.96
        if category == 2 : reduction = 0.94
        if category == 1 : reduction = 0.5
        print (ss, "Throw away BKG: ", time.asctime( time.localtime(time.time()) ))
        print reduction
        print time.asctime( time.localtime(time.time()) )
        removeN=len( chunk_df.loc[chunk_df[target] == 0] )*reduction
        drop_indices = np.random.choice(chunk_df.loc[chunk_df[target] == 0].index, int(removeN), replace=False)
        #print len(chunk_df.loc[chunk_df[target] == 0].drop(drop_indices, inplace = True))
        #data1 = data1.append(chunk_df.loc[chunk_df[target] == 0].drop(drop_indices))
        #print len(data1)
        #data2 = data2.append(chunk_df.loc[chunk_df[target] == 1])
        data = pandas.concat([chunk_df.loc[chunk_df[target] == 0].drop(drop_indices), chunk_df.loc[chunk_df[target] == 1]])
        #else : data = pandas.concat([chunk_df.loc[chunk_df[target] == 0], chunk_df.loc[chunk_df[target] == 1]])
        #data = data.append(pandas.concat([data1, data2]))
        #data = data.append(chunk_df.loc[chunk_df[target] == 1])
        data.dropna(subset=branch, inplace = True)
        data.replace([np.inf, -np.inf], np.nan).dropna(subset=data.columns,inplace = True, how='any')
        data = data.fillna(data.median()).clip(-1e11,1e11) ## for GridSearchCV not crash
        total_bkg_entries_reduced += len(data.loc[data[target]==0])
        print (ss, "len sig/BKG after reduction: ",len(data.loc[data[target]==1]),len(data.loc[data[target]==0]))
    print ("Date: ", time.asctime( time.localtime(time.time()) ))
    print ("Total" )
    #print ("nev sig", len(np.unique(chunk_df.loc[chunk_df[target]==1]["counter"].values)), " nev total ", nev)
    #print ("len sig/BKG : ",len(chunk_df.loc[chunk_df[target]==1]),len(chunk_df.loc[chunk_df[target]==0]))
    print ("total_evt", total_evt)
    print ("total_sig_evt", total_sig_evt)
    print ("total_bkg_entries", total_bkg_entries)
    print ("total_sig_entries", total_sig_entries)
    print ("total_bkg_entries_reduced", total_bkg_entries_reduced)

    doOld = False
    if doOld :
        oldPKL = "HadTopTagger_sklearnV0o17o1_HypOpt/all_HadTopTagger_sklearnV0o17o1_HypOpt_XGB_ntrees_1000_deph_3_lr_0o01_CSV_sort_withKinFit_withKinFit.pkl"
        from sklearn.externals import joblib
        oldclf = joblib.load(oldPKL)
        oldVars = [
                'CSV_b',
                'qg_Wj2',
                'pT_bWj1Wj2',
                'pT_Wj2',
                'm_Wj1Wj2',
                'nllKinFit',
                'kinFit_pT_b_o_pT_b'#,
        ]
        evaluateFOM(oldclf, keys[0], oldVars ,"oldWithKinfit"+str(True), "train_auc" , "test_auct", 1, 1, 1, "f_score_dict", data)

    nS = len(data.loc[data[target].values == 0])
    nB = len(data.loc[data[target].values == 1])
    print "length of sig, bkg without NaN: ", nS, nB

    print ("Date: ", time.asctime( time.localtime(time.time()) ) , "balance datasets")
    ## Balance datasets
    data.loc[data[target]==0, ['weights']] *= 500000/data.loc[data[target]==0]['weights'].sum()
    data.loc[data[target]==1, ['weights']] *= 500000/data.loc[data[target]==1]['weights'].sum()

    print ("sum weights sig/BKG",data.loc[data[target]==1][weights].sum(),data.loc[data[target]==0][weights].sum())
    #print ("len hadtruth",data.loc[data['hadtruth'] == 1][weights].sum())
    #print ("len negWeight",data.loc[data["genWeight"] < 1][weights].sum())
    #make_plots_genpt(data,"accuracy ttH MG", "g", channel+"/HTT_withBoost_cat"+str(category)+"_ak"+str(jet)+"_effWithGenTpt.pdf")

    print ("len sig (gen pt > 200)",len(data.loc[(data[target]==1) & (data["genFatPtAll"] > 200 )]))
    print ("len sig/BKG (fattag)",
        len(data.loc[(data["fatjet_isGenMatched"]==1) ]), # & (data["b_isGenMatched"]==1)
        len(data.loc[(data["fatjet_isGenMatched"]==0) ]) # | (data["b_isGenMatched"]==0)
        )

    print ("len sig/BKG (target)",
        len(data.loc[(data[target]==1) ]), # & (data["b_isGenMatched"]==1)
        len(data.loc[(data[target]==0) ]) # | (data["b_isGenMatched"]==0)
        )

    print ("sum weights sig/BKG",data.loc[data[target]==1]['weights'].sum(),data.loc[data[target]==0]['weights'].sum())
    ##########################################################################
    # plot correlation matrix
    for ii in [1,2] :
    	if ii == 1 :
    		datad=data.loc[data[target].values == 1]
    		label="signal"
    	else :
    		datad=data.loc[data[target].values == 0]
    		label="BKG"
    	datacorr = datad[trainVars(category,True)] #.loc[:,trainVars(False)] #dataHToNobbCSV[[trainVars(True)]]
    	correlations = datacorr.corr()
    	fig = plt.figure(figsize=(10, 10))
    	ax = fig.add_subplot(111)
    	cax = ax.matshow(correlations, vmin=-1, vmax=1)
    	ticks = np.arange(0,len(trainVars(category,True)),1)
    	plt.rc('axes', labelsize=8)
    	ax.set_xticks(ticks)
    	ax.set_yticks(ticks)
    	ax.set_xticklabels(trainVars(category,True),rotation=-90)
    	ax.set_yticklabels(trainVars(category,True))
    	fig.colorbar(cax)
    	fig.tight_layout()
    	#plt.subplots_adjust(left=0.9, right=0.9, top=0.9, bottom=0.1)
        savecorr = "{}/cat_{}_nvar_{}_ak{}_{}_corr_sel_{}.pdf".format(channel,str(category),str(len(trainVars(category,True))),str(jet),label,str(ss))
    	fig.savefig(savecorr)
    	ax.clear()
        print (savecorr, "saved")

    #df_y_count = data.groupby(labels).size().reset_index().rename(columns={0:'bWj1Wj2_isGenMatched'})
    #print data.index.get_values()
    #print data[['bWj1Wj2_isGenMatched',"fatjet_isGenMatched"]]
    ## make plots
    if makeplots :
        nbins=8
        color1='g'
        color2='b'
        printmin=False
        plotResiduals=False

        make_plots(
            trainVars(category,True), 20,
            data.loc[data[target]==1], "signal", color1,
            data.loc[data[target]==0], "BKG", color2,
            channel+"/HTT_withBoost_cat"+str(category)+"_"+str(bdtType)+"_nvar_"+str(len(trainVars(category,True)))+"_sel_"+str(ss)+".pdf",
            printmin,
            plotResiduals
            )

        make_plots(
            ['bjet_tag_position'], 20,
            data.loc[(data[target]==1)], "signal", color1,
            data.loc[(data[target]==0)], "BKG", color2,
            channel+"/HTT_withBoost_cat"+str(category)+"_"+str(bdtType)+"_sel_"+str(ss)+"_btagPos.pdf",
            printmin,
            plotResiduals
            )

        make_plots(
            [
            "drT_gen", "drT_genTriplet", "drT_genJ_max",
            "drWj1_gen", "drWj2_gen", "drB_gen",
            "dr_wj1_wj2_gen", "dr_b_wj1_gen", "dr_b_wj2_gen",
            ], 20,
            data.loc[data[target]==1], "signal", color1,
            data.loc[data[target]==0], "BKG", color2,
            channel+"/HTT_withBoost_cat"+str(category)+"_"+str(bdtType)+"_sel_"+str(ss)+"_gen_dr.pdf",
            printmin,
            plotResiduals
            )

        make_plots_gen(
            [
            #"genTopMass_FromW", "genTopMass_FromWjs", "WMass_FromWj", "WMass",
            "genTopMassFromW", "genTopMassFromWj", "genWMassFromWj",
            "genAntiTopMassFromW", "genAntiTopMassFromWj", "genAntiWMassFromWj"
            #"dr_wj1_wj2_gen", "dr_b_wj1_gen", "dr_b_wj2_gen",
            #"drWj1_gen", "drWj2_gen", "drB_gen",
            #"drW_gen", "drT_gen", "drT_genTriplet", "drT_genJ_max",
            # "etaWj1_gen", "etaWj2_gen", "etaB_gen",
            # "ptWj1_gen", "ptWj2_gen", "ptB_gen",
            #"genFatPtAll", "genFatEtaAll", #"drB_gen",
            ], 20,
            data.loc[data[target]==1], "all", color1,
            #data.loc[data[target]==0], "BKG", color2,
            channel+"/HTT_withBoost_cat"+str(category)+"_"+str(bdtType)+"_sel_"+str(ss)+"_genvars.pdf",
            printmin
            )

    traindataset, valdataset  = train_test_split(data[trainVars(category,True)+[target,'weights']], test_size=0.3, random_state=7) # ,"fatjet_isGenMatched", "counter"

    if HypOpt==True :
    	# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    	param_grid = {
        			'n_estimators': [ 500, 2500, 3000],
        			'min_child_weight': [10, 100],
        			'max_depth': [ 2, 3, 4, 5],
        			'learning_rate': [0.05, 0.03, 0.1]
                    #'n_estimators': [ 500 ]
    				}
    	scoring = "roc_auc"
    	early_stopping_rounds = 150 # Will train until validation_0-auc hasn't improved in 100 rounds.
    	cv=3
    	cls = xgb.XGBClassifier()
        saveopt = "{}/{}_{}_{}_sel_{}_cat_{}_GSCV.log".format(channel,bdtType,trainvar,str(len(trainVars(category,True))),str(ss) ,str(category))
    	file = open(saveopt,"w")
        print ("opt being saved on ", saveopt)
        #file.write("Date: "+ str(time.asctime( time.localtime(time.time()) ))+"\n")
        file.write(str(trainVars(category,True))+"\n")
        result_grid = val_tune_rf(cls,
            traindataset[trainVars(category,True)].values, traindataset[target].astype(np.bool),
            valdataset[trainVars(category,True)].values, valdataset[target].astype(np.bool), param_grid, file)
    	#file.write(result_grid)
        #file.write("Date: "+ str(time.asctime( time.localtime(time.time()) ))+"\n")
    	file.close()
        print ("opt saved on ", saveopt)
        """
    	fit_params = { "eval_set" : [(valdataset[trainVars(category,True)].astype(np.float64), valdataset[target].astype(np.bool))],
                               "eval_metric" : "auc",
                               "early_stopping_rounds" : early_stopping_rounds,
    						   'sample_weight': valdataset[weights].astype(np.float64) }
    	gs = GridSearchCV(cls, param_grid, scoring, fit_params, cv = cv, verbose = 1, error_score=0.0)
    	gs.fit(traindataset[trainVars(category,True)].values,
    	traindataset[target].astype(np.bool)
    	)
    	for i, param in enumerate(gs.cv_results_["params"]):
    		print("params : {} \n    cv auc = {}  +- {} ".format(param,gs.cv_results_["mean_test_score"][i],gs.cv_results_["std_test_score"][i]))
    	print("best parameters",gs.best_params_)
    	print("best score",gs.best_score_)
    	#print("best iteration",gs.best_iteration_)
    	#print("best ntree limit",gs.best_ntree_limit_)
        saveopt = "{}/{}_{}_{}_GSCV.log".format(channel,bdtType,trainvar,str(len(trainVars(category,True))))
    	file = open(saveopt,"w")
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
        print ("opt saved on ", saveopt)
        """

    cls = xgb.XGBClassifier(
    			n_estimators = options.ntrees,
    			max_depth = options.treeDeph,
    			min_child_weight = options.mcw, # min_samples_leaf
    			learning_rate = options.lr,
                #objective="multi:softmax"
    			)
    cls.fit(
    	traindataset[trainVars(category,True)].values,
    	traindataset[target].astype(np.bool),
    	sample_weight= (traindataset[weights].astype(np.float64)) #,
    	)

    print trainVars(category,True)
    saveopt = "{}/{}_{}_{}_sel_{}_cat_{}_roc.log".format(channel,bdtType,trainvar,str(len(trainVars(category,True))),str(ss), str(category))
    print ("opt being saved on ", saveopt)
    file = open(saveopt,"w")
    #file.write("Date: "+str(time.asctime( time.localtime(time.time()) ))+"\n")
    file.write(str(trainVars(category,True))+"\n")
    print traindataset[trainVars(category,True)].columns.values.tolist()
    print ("XGBoost trained")
    print traindataset[trainVars(category,True)].values
    proba = cls.predict_proba(traindataset[trainVars(category,True)].values )
    fpr, tpr, thresholds = roc_curve(traindataset[target], proba[:,1],
    	sample_weight=(traindataset[weights].astype(np.float64)) )
    train_auc = auc(fpr, tpr, reorder = True)
    print("XGBoost train set auc - {}".format(train_auc))
    file.write("train "+str(train_auc)+"\n")
    proba = cls.predict_proba(valdataset[trainVars(category,True)].values )
    fprt, tprt, thresholds = roc_curve(valdataset[target], proba[:,1], sample_weight=(valdataset[weights].astype(np.float64))  )
    test_auct = auc(fprt, tprt, reorder = True)
    print("XGBoost test set auc - {}".format(test_auct))
    file.write("test "+str(test_auct)+"\n")
    #file.write("Date: "+ str(time.asctime( time.localtime(time.time()) ))+"\n")
    print "nvar = "+str(len(trainVars(category,True)))
    print ("Date: ", time.asctime( time.localtime(time.time()) ))
    file.close()
    ################################################################################
    if doXML==True :
    	bdtpath=channel+"/"+process+"_"+channel+"_XGB_"+trainvar+"_"+bdtType+"_nvar"+str(len(trainVars(category,True)))+"_cat_"+str(category)
    	print ("Output pkl ", time.asctime( time.localtime(time.time()) ))
    	if withKinFit :
    		pickle.dump(cls, open(bdtpath+"_withKinFit_sel_"+str(ss)+".pkl", 'wb'))
    		print ("saved "+bdtpath+"_withKinFit.pkl")
    	else :
    		pickle.dump(cls, open(bdtpath+".pkl", 'wb'))
    		print ("saved "+bdtpath+".pkl")
    	print ("starting xml conversion")
    ###########################################################################
    ## feature importance plot
    fig, ax = plt.subplots()
    f_score_dict =cls.get_booster().get_fscore()
    f_score_dict = {trainVars(category,True)[int(k[1:])] : v for k,v in f_score_dict.items()}
    feat_imp = pandas.Series(f_score_dict).sort_values(ascending=True)
    feat_imp.plot(kind='barh', title='Feature Importances')
    fig.tight_layout()
    fig.savefig("{}/cat_{}_nvar_{}_ak{}_XGB_importance_sel_{}_cat_{}.pdf".format(channel,str(category),str(len(trainVars(category,True))),str(jet),str(ss),str(category)))
    ###########################################################################
    # the bellow takes time: you may want to comment if you are setting up
    if options.evaluateFOM==True :
        evaluateFOM(cls,keys[0],trainVars(category,True),"WithKinfit"+str(True), train_auc , test_auct, 1, 1, 1, f_score_dict, valdataset)
    ###################################################################
    sys.stdout.flush()
