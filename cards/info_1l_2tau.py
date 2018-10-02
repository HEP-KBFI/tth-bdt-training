hasHTT = True

channelInTree="1l_2tau_OS_forBDTtraining"
#inputPath="/hdfs/local/acaan/ttHAnalysis/2017/1l_2tau_toBDT_2018Sep24/histograms/1l_2tau/forBDTtraining_OS/" ## to compare boosted
inputPath="/hdfs/local/acaan/ttHAnalysis/2017/1l_2tau_toBDT_2018Sep27_tauLoose/histograms/1l_2tau/forBDTtraining_OS/"
#inputPath="/hdfs/local/acaan/ttHAnalysis/2017/1l_2tau_toBDT_2018Sep27_tauLoose_noMassCut_corrected/histograms/1l_2tau/forBDTtraining_OS/"

channelInTreeTight="1l_2tau_OS_forBDTtraining"
inputPathTight="/hdfs/local/acaan/ttHAnalysis/2016/1l_2tau_2018Jan26_forBDT_tightLtightT/histograms/1l_2tau/forBDTtraining_OS/"
channelInTreeFS="1l_2tau_OS_Tight"
inputPathTightFS="/hdfs/local/acaan/ttHAnalysis/2016/2018Jan28_BDT_toTrees_FS_looseT/histograms/1l_2tau/Tight_OS/"
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
    "HTT_boosted", "genTopPt_HTT_boosted", "HadTop_pt_HTT_boosted",
    "HTT_boosted_WithKinFit", "genTopPt_HTT_boosted_WithKinFit", "HadTop_pt_HTT_boosted_WithKinFit",
    "HTT_semi_boosted", "genTopPt_HTT_semi_boosted", "HadTop_pt_HTT_semi_boosted",
    "HTT_semi_boosted_WithKinFit", "genTopPt_HTT_semi_boosted_WithKinFit", "HadTop_pt_HTT_semi_boosted_WithKinFit",
    "nJet", "nBJetLoose", "nBJetMedium", "nHTTv2", "nElectron", "nMuon",
    "N_jetAK12", "N_jetAK8", "HadTop_pt_semi_boosted_fromAK8", "genTopPt_semi_boosted_fromAK8",
    "hadtruth", "hadtruth_boosted", "hadtruth_semi_boosted",
    "bWj1Wj2_isGenMatchedWithKinFit", "bWj1Wj2_isGenMatched_IHEP",
    "bWj1Wj2_isGenMatched_CSVsort3rd", "bWj1Wj2_isGenMatched_highestCSV",
    "bWj1Wj2_isGenMatched_CSVsort3rd_WithKinFit", "bWj1Wj2_isGenMatched_highestCSV_WithKinFit",
    "bWj1Wj2_isGenMatched_boosted", "bWj1Wj2_isGenMatched_boosted_WithKinFit",
    "bWj1Wj2_isGenMatched_semi_boosted", "bWj1Wj2_isGenMatched_semi_boosted_WithKinFit"
]

if "evtLevelSUM_TTH" in bdtType :
    fastsimTT=490.128+774.698
    fastsimTTV=23.5482+5.0938
    fastsimTTtight=490.128+774.698
    fastsimTTVtight=23.5482+5.0938
    if "_T" in bdtType:
    	TTdatacard=192.06
    	TTVdatacard=0.52+6.11
    	TTfullsim=223.00
    	TTVfullsim=1.46+7.03
    elif "_VT" in bdtType:
    	TTdatacard=91.10
    	TTVdatacard=0.39+4.68
    	TTfullsim=223.00
    	TTVfullsim=1.46+7.03
    else :
        TTdatacard=91.10
        TTVdatacard=0.39+4.68
        TTfullsim=223.00
        TTVfullsim=1.46+7.03


def trainVars(all):
        if channel=="1l_2tau" and all==True :return [
		"lep_conePt", #
		"mindr_lep_jet", "mindr_tau1_jet", "mindr_tau2_jet",
		"avg_dr_jet", "ptmiss",
		"htmiss", "mT_lep",
		"tau1_pt", "tau2_pt",
		"tau1_eta", "tau2_eta",
		"dr_taus",
		"dr_lep_tau_os",
		"dr_lep_tau_ss",
		"dr_lep_tau_lead",
		"dr_lep_tau_sublead",
		"dr_HadTop_tau_lead","dr_HadTop_tau_sublead", "dr_HadTop_tautau",
		"dr_HadTop_lepton","mass_HadTop_lepton",
		"costS_HadTop_tautau",
		"costS_tau",
		"mTauTauVis",
		"mvaOutput_hadTopTagger",
		"mT_lepHadTop", #
		"mT_lepHadTopH",
		"HadTop_pt","HadTop_eta",
		"dr_lep_HadTop",
		"dr_HadTop_tau_OS","dr_HadTop_tau_SS",
		"nJet" ,
		"nBJetLoose" , "nBJetMedium",
		"genWeight", "evtWeight"
		] + HTT_var

        if trainvar=="oldVar"  and channel=="1l_2tau" and bdtType=="evtLevelTT_TTH" and all==False :return [
		"htmiss",
		"mTauTauVis",
		"dr_taus",
		"avg_dr_jet",
		"nJet",
		"nBJetLoose",
		"tau1_pt",
		"tau2_pt"
		]

	if trainvar=="oldVarHTT"  and channel=="1l_2tau"  and bdtType=="evtLevelTT_TTH" and all==False :return [
	"htmiss",
	"mTauTauVis",
	"dr_taus",
	"avg_dr_jet",
	"nJet",
	"nBJetLoose",
	"tau1_pt",
	"tau2_pt",
	"mvaOutput_hadTopTaggerWithKinFit",
	"HadTop_pt"
	]

	if trainvar=="noHTT" and channel=="1l_2tau"  and bdtType=="evtLevelTTV_TTH" and all==False :return [
		"avg_dr_jet",
		"dr_taus",
		"ptmiss",
		"lep_conePt",
		"mT_lep",
		"mTauTauVis",
		"mindr_lep_jet",
		"mindr_tau1_jet",
		"dr_lep_tau_ss",
		"dr_lep_tau_sublead",
		"costS_tau",
		"tau1_pt",
		"tau2_pt"
		]

	if trainvar=="HTTMVAonlyWithKinFit" and channel=="1l_2tau"  and bdtType=="evtLevelTTV_TTH" and all==False :return [
		"lep_conePt", #"lep_eta", #"lep_tth_mva",
		"mindr_lep_jet", #"mindr_tau1_jet",
		"mindr_tau2_jet",
		"avg_dr_jet", #"ptmiss",
		"mT_lep", #"htmiss", #"tau1_mva", "tau2_mva",
		"tau2_pt",
		"dr_taus", #"dr_lep_tau_os",
		"dr_lep_tau_ss", #"dr_lep_tau_lead", #"dr_lep_tau_sublead",
		"mTauTauVis",
		"dr_HadTop_tau_lead", #"dr_HadTop_tau_sublead",
		"mass_HadTop_lepton", #"costS_HadTop_tautau",
		"mvaOutput_hadTopTaggerWithKinFit" #"mvaOutput_hadTopTagger",
		]

	if trainvar=="HTTMVAonlyWithKinFitLepID" and channel=="1l_2tau"  and bdtType=="evtLevelTTV_TTH" and all==False :return [
		"lep_conePt", #"lep_eta",
		"lep_tth_mva",
		"mindr_lep_jet", #"mindr_tau1_jet",
		"mindr_tau2_jet",
		"avg_dr_jet", #"ptmiss",
		"mT_lep", #"htmiss", #
		"tau1_mva", "tau2_mva",
		"tau2_pt",
		"dr_taus", #"dr_lep_tau_os",
		"dr_lep_tau_ss", #"dr_lep_tau_lead", #"dr_lep_tau_sublead",
		"mTauTauVis",
		"dr_HadTop_tau_lead", #"dr_HadTop_tau_sublead",
		"mvaOutput_hadTopTaggerWithKinFit" #"mvaOutput_hadTopTagger",
		]

        if trainvar=="oldVar"  and channel=="1l_2tau" and bdtType=="evtLevelTTV_TTH" and all==False :return [
		"htmiss",
		"mTauTauVis",
		"dr_taus",
		"avg_dr_jet",
		"nJet",
		"nBJetLoose",
		"tau1_pt",
		"tau2_pt"
		]

	if trainvar=="noHTT" and channel=="1l_2tau"  and bdtType=="evtLevelTT_TTH" and all==False :return [
		"avg_dr_jet",
		"dr_taus",
		"ptmiss",
		"mT_lep",
		"nJet",
		"mTauTauVis",
		"mindr_lep_jet",
		"mindr_tau1_jet",
		"mindr_tau2_jet",
		"dr_lep_tau_lead",
		"costS_tau",
		"nBJetLoose",
		"tau1_pt",
		"tau2_pt"
		]

	if trainvar=="HTTMVAonlyWithKinFit" and channel=="1l_2tau"  and bdtType=="evtLevelTT_TTH" and all==False :return [
				"avg_dr_jet",
				"dr_taus",
				"ptmiss",
				"lep_conePt",
				"mT_lep",
				"mTauTauVis",
				"mindr_lep_jet",
				"mindr_tau1_jet",
				"nJet",
				"dr_lep_tau_ss",
				"dr_lep_tau_lead",
				"costS_tau",
				"mvaOutput_hadTopTaggerWithKinFit",
				"mT_lepHadTopH"
		]

	if trainvar=="HTTMVAonlyNoKinFitLepID" and channel=="1l_2tau"  and bdtType=="evtLevelTT_TTH" and all==False :return [
				"avg_dr_jet",
				"htmiss",
				"mT_lep",
				"mTauTauVis",
				"mindr_lep_jet",
				"mindr_tau1_jet",
				"nJet",
				"dr_lep_tau_ss",
				"costS_tau",
				"mvaOutput_hadTopTaggerWithKinFit",
				"lep_tth_mva",
				"tau1_mva",
				"tau2_mva",
				"tau1_pt",
				"tau2_pt",
		]

	if trainvar=="HTTMVAonlyWithKinFit" and channel=="1l_2tau"  and bdtType=="evtLevelTT_TTH"  and all==False :return [
				"avg_dr_jet",
				"dr_taus",
				"htmiss",
				"mT_lep",
				"mTauTauVis",
				"mindr_lep_jet",
				"mindr_tau1_jet",
				"nJet",
				"dr_lep_tau_ss",
				"dr_lep_tau_lead",
				"costS_tau",
				"mvaOutput_hadTopTaggerWithKinFit",
				"nBJetLoose",
				"tau1_pt",
				"tau2_pt"
		]

	if trainvar=="HTT" and channel=="1l_2tau"  and bdtType=="evtLevelTTV_TTH" and all==False :return [
						"avg_dr_jet",
						"dr_taus",
						"ptmiss",
						"lep_conePt",
						"mT_lep",
						"mTauTauVis",
						"mindr_lep_jet",
						"mindr_tau1_jet",
						"dr_lep_tau_ss",
						"dr_lep_tau_sublead",
						"costS_tau",
						"tau1_pt",
						"tau2_pt",
						"mvaOutput_hadTopTaggerWithKinFit",
		]

	if trainvar=="HTT" and channel=="1l_2tau"  and bdtType=="evtLevelTT_TTH" and all==False :return [
						"avg_dr_jet",
						"dr_taus",
						"ptmiss",
						"mT_lep",
						"nJet",
						"mTauTauVis",
						"mindr_lep_jet",
						"mindr_tau1_jet",
						"mindr_tau2_jet",
						"dr_lep_tau_lead",
						"costS_tau",
						"nBJetLoose",
						"tau1_pt",
						"tau2_pt",
						"mvaOutput_hadTopTaggerWithKinFit",
						"HadTop_pt",
						"mvaOutput_Hj_tagger",
		]

	if trainvar=="HTT" and channel=="1l_2tau"  and "evtLevelSUM_TTH" in bdtType and all==False :return [
						"avg_dr_jet",
						"dr_taus",
						"ptmiss",
						"lep_conePt",
						"mT_lep",
						"mTauTauVis",
						"mindr_lep_jet",
						"mindr_tau1_jet",
						"mindr_tau2_jet",
						"dr_lep_tau_ss",
						"dr_lep_tau_lead",
						"costS_tau",
						"nBJetLoose",
						"tau1_pt",
						"tau2_pt",
						"mvaOutput_hadTopTaggerWithKinFit",
						"HadTop_pt",
	]

	if trainvar=="noHTT" and channel=="1l_2tau" and "evtLevelSUM_TTH" in bdtType and all==False :return [
						"avg_dr_jet",
						"dr_taus",
						"ptmiss",
						"lep_conePt",
						"mT_lep",
						"mTauTauVis",
						"mindr_lep_jet",
						"mindr_tau1_jet",
						"mindr_tau2_jet",
						"nJet",
						"dr_lep_tau_ss",
						"dr_lep_tau_lead",
						"costS_tau",
						"nBJetLoose",
						"tau1_pt",
						"tau2_pt",
	]
