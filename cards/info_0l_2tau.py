hasHTT = False

channelInTree='0l_2tau_OS_forBDTtraining'

#inputPath='/hdfs/local/acaan/ttHAnalysis/2017/0l_2tau_toBDT_2018Oct03_two_loops_noMassCut_isolatedTrue_withAK8/histograms/0l_2tau/forBDTtraining_OS/'
#inputPath='/hdfs/local/acaan/ttHAnalysis/2017/0l_2tau_toBDT_2018Oct03_two_loops_noMassCut_isolated_withAK8/histograms/0l_2tau/forBDTtraining_OS/'

#inputPath='/hdfs/local/acaan/ttHAnalysis/2017/0l_2tau_datacards_2018Oct16_withboost_FR/histograms/0l_2tau/forBDTtraining_OS/'

inputPath='/hdfs/local/acaan/ttHAnalysis/2017/0l_2tau_toBDT_2018Oct22_withBoost_FR_subjetISO/histograms/0l_2tau/forBDTtraining_OS/'

channelInTreeTight='0l_2tau_OS_forBDTtraining'
inputPathTight='/hdfs/local/acaan/ttHAnalysis/2016/0l_2tau_2018Jan26_forBDT_tightLtightT/histograms/0l_2tau/forBDTtraining_OS/'
channelInTreeFS='0l_2tau_OS_Tight'
inputPathTightFS='/hdfs/local/acaan/ttHAnalysis/2016/2018Jan28_BDT_toTrees_FS_looseT/histograms/0l_2tau/Tight_OS/'
criteria=[]
testtruth="bWj1Wj2_isGenMatchedWithKinFit"
FullsimWP="TightLep_MediumTau"
FastsimWP="TightLep_TightTau"
FastsimTWP="TightLep_TightTau"

HTT_var = [
    "res-HTT", "res-HTT_IHEP",
    "minDR_HTTv2_lep", "DR_HTTv2_tau", "minDR_HTTv2_L",
    "minDR_AK12_lep", "DR_AK12_tau", "minDR_AK12_L",
    "res-HTT_CSVsort3rd", "res-HTT_highestCSV",
    "res-HTT_CSVsort3rd_WithKinFit", "res-HTT_highestCSV_WithKinFit",
    "HTTv2_lead_pt", "AK12_lead_pt",
    "HadTop_pt_multilep",
    "HadTop_pt_CSVsort3rd", "HadTop_pt_highestCSV",
    "HadTop_pt_CSVsort3rd_WithKinFit", "HadTop_pt_highestCSV_WithKinFit",
    "genTopPt", "genTopPt_multilep",
    "genTopPt_CSVsort3rd", "genTopPt_highestCSV",
    "genTopPt_CSVsort3rd_WithKinFit", "genTopPt_highestCSV_WithKinFit",
    "HTT_boosted", "genTopPt_HTT_boosted", "HadTop_pt_boosted",
    "HTT_boosted_WithKinFit", "genTopPt_HTT_boosted_WithKinFit", "HadTop_pt_boosted_WithKinFit",
    "HTT_semi_boosted", "genTopPt_semi_boosted", "HadTop_pt_semi_boosted",
    "HTT_semi_boosted_WithKinFit", "genTopPt_HTT_semi_boosted_WithKinFit", "HadTop_pt_HTT_semi_boosted_WithKinFit",
    "nJet", "nBJetLoose", "nBJetMedium", "nHTTv2", "nElectron", "nMuon",
    "N_jetAK12", "N_jetAK8", "HTT_semi_boosted_fromAK8",
    "hadtruth", "hadtruth_boosted", "hadtruth_semi_boosted",
    "bWj1Wj2_isGenMatchedWithKinFit", "bWj1Wj2_isGenMatched_IHEP",
    "bWj1Wj2_isGenMatched_CSVsort3rd", "bWj1Wj2_isGenMatched_highestCSV",
    "bWj1Wj2_isGenMatched_CSVsort3rd_WithKinFit", "bWj1Wj2_isGenMatched_highestCSV_WithKinFit",
    "bWj1Wj2_isGenMatched_boosted", "bWj1Wj2_isGenMatched_boosted_WithKinFit",
    "bWj1Wj2_isGenMatched_semi_boosted", "bWj1Wj2_isGenMatched_semi_boosted_WithKinFit",
    'bWj1Wj2bWj1Wj2_isGenMatched_semi_boosted_fromAK8',
    "genTopPt_semi_boosted_fromAK8", "HadTop_pt_semi_boosted_fromAK8", "minDR_AK8_lep",
    "bWj1Wj2bWj1Wj2_isGenMatched_semi_boosted_noISO", "bWj1Wj2_isGenMatched_semi_boosted_WithKinFit_noISO",
    'bWj1Wj2bWj1Wj2_isGenMatched_semi_boosted_fromAK8_noISO',
    "minDR_HTTv2subjets_lep_noISO", "minDR_AK12subjets_lep_noISO", "minDR_AK8subjets_lep_noISO"
]

if 'evtLevelSUM_TTH' in bdtType :
    fastsimTT=312.88+625.424
    fastsimTTV=4.15035+11.1
    fastsimDY=154.231
    TTdatacard=198.+348.6
    TTVdatacard=0.53+4.33
    DYdatacard=126.6

def trainVars(all):
        if all==True :return [
  "mindr_tau1_jet", "mindr_tau2_jet",
  "avg_dr_jet", "ptmiss", "htmiss", "tau1_mva", "tau2_mva", "tau1_pt", "tau2_pt",
  "tau1_eta", "tau2_eta", "dr_taus", "mT_tau1", "mT_tau2", "mTauTauVis", "mTauTau",
  "minDR_HTTv2_lep", "minDR_HTTv2_lep_2",
  "minDR_AK12_lep",
  "res-HTT", "res-HTT_IHEP",
  "res-HTT_CSVsort3rd", "res-HTT_highestCSV",
  "res-HTT_CSVsort3rd_WithKinFit", "res-HTT_highestCSV_WithKinFit",
  "HTTv2_lead_pt", "AK12_lead_pt",
  "HadTop_pt",  "genTopPt",
  "HadTop_pt_multilep",
  "HadTop_pt_CSVsort3rd", "HadTop_pt_highestCSV",
  "HadTop_pt_CSVsort3rd_WithKinFit", "HadTop_pt_highestCSV_WithKinFit",
  "genTopPt_multilep",
  "genTopPt_CSVsort3rd", "genTopPt_highestCSV",
  "genTopPt_CSVsort3rd_WithKinFit", "genTopPt_highestCSV_WithKinFit",
  "res-HTT_2", "res-HTT_IHEP_2",
  "res-HTT_CSVsort3rd_2", "res-HTT_highestCSV_2",
  "res-HTT_CSVsort3rd_WithKinFit_2", "res-HTT_highestCSV_WithKinFit_2",
  "minDR_AK8_lep",
  "HadTop_pt_2",  "genTopPt_2",
  "HadTop_pt_multilep_2",
  "HadTop_pt_CSVsort3rd_2", "HadTop_pt_highestCSV_2",
  "HadTop_pt_CSVsort3rd_WithKinFit_2", "HadTop_pt_highestCSV_WithKinFit_2",
  "genTopPt_multilep_2", "HTT_semi_boosted_fromAK8",
  "genTopPt_CSVsort3rd_2", "genTopPt_highestCSV_2",
  "genTopPt_CSVsort3rd_WithKinFit_2", "genTopPt_highestCSV_WithKinFit_2",
  "HTT_boosted", "genTopPt_boosted", "HadTop_pt_boosted",
  "HTT_boosted_WithKinFit", "genTopPt_boosted_WithKinFit", "HadTop_pt_boosted_WithKinFit",
  "HTT_boosted_2", "genTopPt_boosted_2", "HadTop_pt_boosted_2",
  "HTT_boosted_WithKinFit_2", "genTopPt_boosted_WithKinFit_2", "HadTop_pt_boosted_WithKinFit_2",
  "HTT_semi_boosted", "genTopPt_semi_boosted", "HadTop_pt_semi_boosted",
  "HTT_semi_boosted_WithKinFit", "genTopPt_semi_boosted_WithKinFit", "HadTop_pt_semi_boosted_WithKinFit",
  "nJet", "nBJetLoose", "nBJetMedium", "nHTTv2",
  "N_jetAK12", "N_jetAK8",
  "hadtruth", "hadtruth_2", "hadtruth_boosted", "hadtruth_boosted_2", "hadtruth_semi_boosted",
  "bWj1Wj2_isGenMatchedWithKinFit", "bWj1Wj2_isGenMatched_IHEP",
  "bWj1Wj2_isGenMatched_CSVsort3rd", "bWj1Wj2_isGenMatched_highestCSV",
  "bWj1Wj2_isGenMatched_CSVsort3rd_WithKinFit", "bWj1Wj2_isGenMatched_highestCSV_WithKinFit",
  "bWj1Wj2_isGenMatchedWithKinFit_2", "bWj1Wj2_isGenMatched_IHEP_2",
  "bWj1Wj2_isGenMatched_CSVsort3rd_2", "bWj1Wj2_isGenMatched_highestCSV_2",
  "bWj1Wj2_isGenMatched_CSVsort3rd_WithKinFit_2", "bWj1Wj2_isGenMatched_highestCSV_WithKinFit_2",
  "bWj1Wj2_isGenMatched_boosted", "bWj1Wj2_isGenMatched_boosted_2",
  "bWj1Wj2_isGenMatched_boosted_WithKinFit", "bWj1Wj2_isGenMatched_boosted_WithKinFit_2",
  "bWj1Wj2_isGenMatched_semi_boosted", "bWj1Wj2_isGenMatched_semi_boosted_WithKinFit", "evtWeight",
  "bWj1Wj2_isGenMatched_semi_boosted_fromAK8",
  "resolved_and_semi",
    "resolved_and_semi_AK8",
    "boosted_and_semi",
    "boosted_and_semi_AK8",
    "resolved_and_boosted",
    "cleanedJets_fromAK12", "cleanedJets_fromAK8",
    "AK12_without_subjets", "AK8_without_subjets", "hadtruth_semi_boosted_fromAK8",
    "genTopPt_semi_boosted_fromAK8", "HadTop_pt_semi_boosted_fromAK8", "max_tau_eta",
    "bWj1Wj2_isGenMatched_semi_boosted_noISO", "bWj1Wj2_isGenMatched_semi_boosted_WithKinFit_noISO",
    'bWj1Wj2_isGenMatched_semi_boosted_fromAK8_noISO', "bWj1Wj2_isGenMatched_boosted_noISO",
    "minDR_HTTv2subjets_lep_noISO", "minDR_AK12subjets_lep_noISO", "minDR_AK8subjets_lep_noISO",
    "minDR_HTTv2subjets_lep", "minDR_AK12subjets_lep", "minDR_AK8subjets_lep",
    "prob_fake_hadTau_lead", "prob_fake_hadTau_sublead"
		]

        if trainvar=="oldVar"  and channel=="0l_2tau" and bdtType=="evtLevelSUM_TTH" and all==False :return [
		"mindr_tau1_jet", "mindr_tau2_jet",
        "avg_dr_jet", "ptmiss",  "tau1_pt", "tau2_pt",
        "tau1_eta", "tau2_eta", "htmiss",
        #"max_tau_eta",
        "dr_taus",
        "mT_tau1", "mT_tau2", "mTauTauVis",
        "mTauTau",
        "nJet", "nBJetLoose", "nBJetMedium",
        "res-HTT_CSVsort3rd_2", "res-HTT_CSVsort3rd",
        "HadTop_pt_CSVsort3rd_2", "HadTop_pt_CSVsort3rd",
		]

        if trainvar=="Updated"  and channel=="0l_2tau" and bdtType=="evtLevelSUM_TTH" and all==False :return [
		"mindr_tau1_jet", "mindr_tau2_jet",
        "avg_dr_jet", "ptmiss",  "tau1_pt", "tau2_pt",
        "tau1_eta", "tau2_eta", #"htmiss",
        #"max_tau_eta",
        "dr_taus",
        "mT_tau1", "mT_tau2", "mTauTauVis",
        "mTauTau",
        "nJet", "nBJetLoose", "nBJetMedium",
        "res-HTT_CSVsort3rd_2", "res-HTT_CSVsort3rd",
        "HadTop_pt_CSVsort3rd_2", "HadTop_pt_CSVsort3rd",
		]

        if trainvar=="Boosted_AK12" and channel=="0l_2tau" and bdtType=="evtLevelSUM_TTH" and all==False :return [
		"mindr_tau1_jet", "mindr_tau2_jet",
        "avg_dr_jet", "ptmiss",  "tau1_pt", "tau2_pt",
        "tau1_eta", "tau2_eta", #"htmiss",
        #"max_tau_eta",
        "dr_taus",
        "mT_tau1", "mT_tau2", "mTauTauVis",
        "mTauTau",
        "nJet", "nBJetLoose", "nBJetMedium",
        "res-HTT_CSVsort3rd_2", "res-HTT_CSVsort3rd",
        #"HadTop_pt_CSVsort3rd_2",
        "HadTop_pt_CSVsort3rd",
		#"cleanedJets_fromAK12",
		"resolved_and_semi", #"boosted_and_semi",
		#"N_jetAK12", #"N_jetAK8", "nHTTv2",
		"minDR_HTTv2_lep", #"HTTv2_lead_pt",
		"minDR_AK12_lep", #"AK12_lead_pt",
		"HTT_boosted", #"HadTop_pt_boosted",
		"HTT_semi_boosted", #"HadTop_pt_semi_boosted",
		]

        if trainvar=="Boosted_AK12_basic" and channel=="0l_2tau" and bdtType=="evtLevelSUM_TTH" and all==False :return [
		"mindr_tau1_jet", "mindr_tau2_jet",
        "avg_dr_jet", "ptmiss",  "tau1_pt", "tau2_pt",
        "tau1_eta", "tau2_eta", #"htmiss",
        #"max_tau_eta",
        "dr_taus",
        "mT_tau1", "mT_tau2", "mTauTauVis",
        "mTauTau",
        "nJet", "nBJetLoose", "nBJetMedium",
        "res-HTT_CSVsort3rd_2", "res-HTT_CSVsort3rd",
        #"HadTop_pt_CSVsort3rd_2",
        "HadTop_pt_CSVsort3rd",
		"N_jetAK12", "nHTTv2", "AK12_lead_pt", #"HTTv2_lead_pt",
		]

        if trainvar=="Boosted_AK8" and channel=="0l_2tau" and bdtType=="evtLevelSUM_TTH" and all==False :return [
        "mindr_tau1_jet", "mindr_tau2_jet",
        "avg_dr_jet", "ptmiss",  "tau1_pt", "tau2_pt",
        "tau1_eta", "tau2_eta", #"htmiss",
        #"max_tau_eta",
        "dr_taus",
        "mT_tau1", "mT_tau2", "mTauTauVis",
        "mTauTau",
        "nJet", "nBJetLoose", "nBJetMedium",
        "res-HTT_CSVsort3rd_2", "res-HTT_CSVsort3rd",
        #"HadTop_pt_CSVsort3rd_2",
        "HadTop_pt_CSVsort3rd",
		#"cleanedJets_fromAK12",
		"resolved_and_semi_AK8", "boosted_and_semi_AK8",
		#"N_jetAK12", #"N_jetAK8", "nHTTv2",
		"minDR_HTTv2_lep", #"HTTv2_lead_pt",
		"minDR_AK8_lep", #"AK12_lead_pt",
		"HTT_boosted", #"HadTop_pt_boosted",
		"HTT_semi_boosted_fromAK8", #"HadTop_pt_semi_boosted_fromAK8",
		]

        if trainvar=="Boosted_AK8_basic" and channel=="0l_2tau" and bdtType=="evtLevelSUM_TTH" and all==False :return [
		"mindr_tau1_jet", "mindr_tau2_jet",
        "avg_dr_jet", "ptmiss",  "tau1_pt", "tau2_pt",
        "tau1_eta", "tau2_eta", #"htmiss",
        #"max_tau_eta",
        "dr_taus",
        "mT_tau1", "mT_tau2", "mTauTauVis",
        "mTauTau",
        "nJet", "nBJetLoose", "nBJetMedium",
        "res-HTT_CSVsort3rd_2", "res-HTT_CSVsort3rd",
        "HadTop_pt_CSVsort3rd_2", "HadTop_pt_CSVsort3rd",
		"N_jetAK8", "nHTTv2", #"AK8_lead_pt", #"HTTv2_lead_pt",
		]

        if trainvar=="Boosted_AK8_noISO" and channel=="0l_2tau" and bdtType=="evtLevelSUM_TTH" and all==False :return [
		"mindr_tau1_jet", "mindr_tau2_jet",
        "avg_dr_jet", "ptmiss",  "tau1_pt", "tau2_pt",
        "tau1_eta", "tau2_eta", #"htmiss",
        #"max_tau_eta",
        "dr_taus",
        "mT_tau1", "mT_tau2", "mTauTauVis",
        "mTauTau",
        "nJet", "nBJetLoose", "nBJetMedium",
        "res-HTT_CSVsort3rd_2", "res-HTT_CSVsort3rd",
        #"HadTop_pt_CSVsort3rd_2",
        "HadTop_pt_CSVsort3rd",
		#"cleanedJets_fromAK12",
		"resolved_and_semi_AK8", "boosted_and_semi_AK8",
		#"N_jetAK12", #"N_jetAK8", "nHTTv2",
		"minDR_HTTv2_lep", #"HTTv2_lead_pt",
		"minDR_AK8_lep", #"AK12_lead_pt",
		"HTT_boosted", #"HadTop_pt_boosted",
		"HTT_semi_boosted_fromAK8", #"HadTop_pt_semi_boosted_fromAK8",
		]

        if trainvar=="Boosted_AK12_noISO" and channel=="0l_2tau" and bdtType=="evtLevelSUM_TTH" and all==False :return [
		"mindr_tau1_jet", "mindr_tau2_jet",
        "avg_dr_jet", "ptmiss",  "tau1_pt", "tau2_pt",
        "tau1_eta", "tau2_eta", #"htmiss",
        #"max_tau_eta",
        "dr_taus",
        "mT_tau1", "mT_tau2", "mTauTauVis",
        "mTauTau",
        "nJet", "nBJetLoose", "nBJetMedium",
        "res-HTT_CSVsort3rd_2", "res-HTT_CSVsort3rd",
        #"HadTop_pt_CSVsort3rd_2",
        "HadTop_pt_CSVsort3rd",
		#"cleanedJets_fromAK12",
		"resolved_and_semi", #"boosted_and_semi",
		#"N_jetAK12", #"N_jetAK8", "nHTTv2",
		"minDR_HTTv2_lep", #"HTTv2_lead_pt",
		"minDR_AK12_lep", #"AK12_lead_pt",
		"HTT_boosted", #"HadTop_pt_boosted",
		"HTT_semi_boosted", #"HadTop_pt_semi_boosted",
		]

"""
python sklearn_Xgboost_csv_evtLevel_ttH.py --channel "0l_2tau" --variables "Boosted_AK8" --bdtType "evtLevelSUM_TTH" --ntrees 1000 --treeDeph 4 --lr 0.01 --mcw 1000 --doXML &

python sklearn_Xgboost_csv_evtLevel_ttH.py --channel "0l_2tau" --variables "Boosted_AK12" --bdtType "evtLevelSUM_TTH" --ntrees 1000 --treeDeph 4 --lr 0.01 --mcw 1000 --doXML   &

python sklearn_Xgboost_csv_evtLevel_ttH.py --channel "0l_2tau" --variables "Boosted_AK8_basic" --bdtType "evtLevelSUM_TTH" --ntrees 1000 --treeDeph 4 --lr 0.01 --mcw 1000 --doXML  &

python sklearn_Xgboost_csv_evtLevel_ttH.py --channel "0l_2tau" --variables "Boosted_AK12_basic" --bdtType "evtLevelSUM_TTH" --ntrees 1000 --treeDeph 4 --lr 0.01 --mcw 1000 --doXML   &

python sklearn_Xgboost_csv_evtLevel_ttH.py --channel "0l_2tau" --variables "Updated" --bdtType "evtLevelSUM_TTH" --ntrees 1000 --treeDeph 4 --lr 0.01 --mcw 1000 --doXML &

python sklearn_Xgboost_csv_evtLevel_ttH.py --channel "0l_2tau" --variables "oldVar" --bdtType "evtLevelSUM_TTH" --ntrees 1000 --treeDeph 4 --lr 0.01 --mcw 1000 --doXML &

python sklearn_Xgboost_csv_evtLevel_ttH.py --channel "0l_2tau" --variables "Boosted_AK8_noISO" --bdtType "evtLevelSUM_TTH" --ntrees 1000 --treeDeph 4 --lr 0.01 --mcw 1000 --doXML &

python sklearn_Xgboost_csv_evtLevel_ttH.py --channel "0l_2tau" --variables "Boosted_AK12_noISO" --bdtType "evtLevelSUM_TTH" --ntrees 1000 --treeDeph 4 --lr 0.01 --mcw 1000 --doXML  &


------------------------------------------------------------------------------------------------------------------------
Job ID 3710534 executable log (/home/acaan/ttHAnalysis/2017/2lss_0tau_datacards_2018Oct07_withBoost_multilepCat/logs/2lss/ClusterHistogramAggregator_histograms_harvested_stage1_5_2lss_Fakeable_wFakeRateWeights_SS_0-37_executable.log.1):
------------------------------------------------------------------------------------------------------------------------
<check_that_histograms_are_valid.py>: input files = '/hdfs/local/acaan/ttHAnalysis/2017/2lss_0tau_datacards_2018Oct07_withBoost_multilepCat/histograms/2lss/addBackgrounds_2lss_TTGJets_Rares_Fakeable_wFakeRateWeights_SS.root /hdfs/local/acaan/ttHAnalysis/2017/2lss_0tau_datacards_2018Oct07_withBoost_multilepCat/histograms/2lss/addBackgrounds_2lss_fakes_TTGJets_Rares_Fakeable_wFakeRateWeights_SS.root /hdfs/local/acaan/ttHAnalysis/2017/2lss_0tau_datacards_2018Oct07_withBoost_multilepCat/histograms/2lss/addBackgrounds_2lss_TTGJets_ext1_Rares_Fakeable_wFakeRateWeights_SS.root /hdfs/local/acaan/ttHAnalysis/2017/2lss_0tau_datacards_2018Oct07_withBoost_multilepCat/histograms/2lss/addBackgrounds_2lss_fakes_TTGJets_ext1_Rares_Fakeable_wFakeRateWeights_SS.root /hdfs/local/acaan/ttHAnalysis/2017/2lss_0tau_datacards_2018Oct07_withBoost_multilepCat/histograms/2lss/addBackgrounds_2lss_TTTT_Rares_Fakeable_wFakeRateWeights_SS.root'
<check_that_histogram_exists>: input file = '/hdfs/local/acaan/ttHAnalysis/2017/2lss_0tau_datacards_2018Oct07_withBoost_multilepCat/histograms/2lss/addBackgrounds_2lss_TTGJets_Rares_Fakeable_wFakeRateWeights_SS.root'
ERROR: Input file '/hdfs/local/acaan/ttHAnalysis/2017/2lss_0tau_datacards_2018Oct07_withBoost_multilepCat/histograms/2lss/addBackgrounds_2lss_TTGJets_Rares_Fakeable_wFakeRateWeights_SS.root' does not exist !!

------------------------------------------------------------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/acaan/ttHAnalysis/2017/2lss_0tau_datacards_2018Oct07_withBoost_multilepCat/scripts/2lss/sbatch_hadd_2lss_stage1_5_Fakeable_wFakeRateWeights_SS_ClusterHistogramAggregator.py", line 25, in <module>
    cluster_histogram_aggregator.create_output_file()
  File "/home/acaan/VHbbNtuples_8_0_x/CMSSW_9_4_6_patch1/python/tthAnalysis/HiggsToTauTau/ClusterHistogramAggregator.py", line 31, in create_output_file
    max_input_files_per_job = self.max_input_files_per_job
  File "/home/acaan/VHbbNtuples_8_0_x/CMSSW_9_4_6_patch1/python/tthAnalysis/HiggsToTauTau/ClusterHistogramAggregator.py", line 87, in aggregate
    self.sbatch_manager.waitForJobs()
  File "/home/acaan/VHbbNtuples_8_0_x/CMSSW_9_4_6_patch1/python/tthAnalysis/HiggsToTauTau/sbatchManager.py", line 589, in waitForJobs
    raise Status.raiseError(completion[failed_job].status)
tthAnalysis.HiggsToTauTau.sbatchManager.sbatchManagerRuntimeError
[acaan@quasar HiggsToTauTau]$ ls /hdfs/local/acaan/ttHAnalysis/2017/2lss_0tau_datacards_2018Oct07_withBoost_multilepCat/histograms/2lss/addBackgrounds_2lss_TTGJets_Rares_Fakeable_wFakeRateWeights_SS.root
ls: cannot access /hdfs/local/acaan/ttHAnalysis/2017/2lss_0tau_datacards_2018Oct07_withBoost_multilepCat/histograms/2lss/addBackgrounds_2lss_TTGJets_Rares_Fakeable_wFakeRateWeights_SS.root: No such file or directory

"""
