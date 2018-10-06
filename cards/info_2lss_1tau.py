hasHTT = True

#channelInTree='2lss_1tau_lepSS_sumOS_Tight'
channelInTree='2lss_1tau_lepSS_sumOS_forBDTtraining'
inputPath='/hdfs/local/acaan/ttHAnalysis/2017/2lss_1tau_2018Sep27_tauLoose_noMassCut_corrected/histograms/2lss_1tau/forBDTtraining_SS_OS/'
FastsimWP= "LooseLep_TightTau"
#channelInTree='2lss_1tau_lepSS_sumOS_Tight'
#inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2lss_1tau_2018Feb27_BDT_TLepVVLTau/histograms/2lss_1tau/forBDTtraining_SS_OS/'
criteria=[]
testtruth="bWj1Wj2_isGenMatchedWithKinFit"
channelInTreeTight='2lss_1tau_lepSS_sumOS_Tight'
#channelInTreeTight='2lss_1tau_lepSS_sumOS_Loose'
inputPathTight='/hdfs/local/acaan/ttHAnalysis/2016/2lss_1tau_2018Feb26_BDT_TLepTTau/histograms/2lss_1tau/forBDTtraining_SS_OS/'
FastsimTWP="TightLep_MediumTau"
if bdtType=="evtLevelSUM_TTH_M" :
	channelInTreeFS='2lss_1tau_lepSS_sumOS_Tight'
	inputPathTightFS='/hdfs/local/acaan/ttHAnalysis/2016/2lss_1tau_2018Feb26_VHbb_trees_TLepMTau/histograms/2lss_1tau/forBDTtraining_SS_OS/'
	FullsimWP= "TightLep_MediumTau"
if bdtType=="evtLevelSUM_TTH_T" :
	channelInTreeFS='2lss_1tau_lepSS_sumOS_Tight'
	inputPathTightFS='/hdfs/local/acaan/ttHAnalysis/2016/2lss_1tau_2018Feb26_VHbb_trees_TLepTTau/histograms/2lss_1tau/forBDTtraining_SS_OS/'
	FullsimWP= "TightLep_TightTau"
else :
	channelInTreeFS='2lss_1tau_lepSS_sumOS_Tight'
	inputPathTightFS='/hdfs/local/acaan/ttHAnalysis/2016/2lss_1tau_2018Feb26_VHbb_trees_TLepMTau/histograms/2lss_1tau/forBDTtraining_SS_OS/'
	FullsimWP= "TightLep_MediumTau"

HTT_var = []

### TT-sample is usually much more than fakes
if 'evtLevelSUM_TTH' in bdtType  :
	# VTightTau
	fastsimTT=22.14+20.22
	fastsimTTtight=1.06791+1.17204
	fastsimTTV=21.73+13.27
	fastsimTTVtight=17.2712+18.3258
	if bdtType=="evtLevelSUM_TTH_M" :
		TTdatacard=9.82
		TTVdatacard=7.75+10.49
		TTfullsim=1.53
		TTVfullsim=7.56+9.39
	if bdtType=="evtLevelSUM_TTH_T" :
		TTdatacard=7.52
		TTVdatacard=5.95+9.19
		TTfullsim=1.21
		TTVfullsim=5.81+8.30


def trainVars(all):
        if channel=="2lss_1tau" and all==True  :return [
		"lep1_genLepPt", "lep2_genLepPt",
  "tau_genTauPt",
  "lep1_fake_prob", "lep2_fake_prob",  "tau_fake_prob", "tau_fake_test",
  "Hj_tagger", "Hjj_tagger", "mvaOutput_Hjj_tagger",
  "fittedHadTop_pt", "fittedHadTop_eta", "unfittedHadTop_pt",
  "unfittedHadTop_eta","fitHTptoHTpt" , "fitHTptoHTmass",
  "dr_lep1_HTfitted", "dr_lep2_HTfitted", "dr_tau_HTfitted", "mass_lep1_HTfitted", "mass_lep2_HTfitted",
  "dr_lep1_HTunfitted", "dr_lep2_HTunfitted", "dr_tau_HTunfitted",
  "genWeight", "evtWeight",
  "HadTop_pt", "HadTop_eta", "genTopPt","min(met_pt,400)",
  "mbb","ptbb", "mbb_loose","ptbb_loose",
  "res-HTT", "res-HTT_IHEP",
  "minDR_HTTv2_lep", "minDR_HTTv2_tau", "minDR_HTTv2_L",
  "minDR_AK12_lep", "DR_AK12_tau", "minDR_AK12_L",
  "res-HTT_CSVsort3rd", "res-HTT_highestCSV",
  "res-HTT_CSVsort3rd_WithKinFit", "res-HTT_highestCSV_WithKinFit",
  "HTTv2_lead_pt", "AK12_lead_pt",
  "HTTv2_lead_mass", "AK12_lead_mass",
  "HadTop_pt_multilep",
  "HadTop_pt_CSVsort3rd", "HadTop_pt_highestCSV",
  "HadTop_pt_CSVsort3rd_WithKinFit", "HadTop_pt_highestCSV_WithKinFit",
  "genTopPt_multilep",
  "genTopPt_CSVsort3rd", "genTopPt_highestCSV",
  "genTopPt_CSVsort3rd_WithKinFit", "genTopPt_highestCSV_WithKinFit",
  "HTT_boosted", "genTopPt_boosted", "HadTop_pt_HTT_boosted",
  "HTT_boosted_WithKinFit", "genTopPt_boosted_WithKinFit", "HadTop_pt_HTT_boosted_WithKinFit",
  "HTT_semi_boosted", "genTopPt_semi_boosted", "HadTop_pt_HTT_semi_boosted",
  "HTT_semi_boosted_WithKinFit", "genTopPt_semi_boosted_WithKinFit", "HadTop_pt_HTT_semi_boosted_WithKinFit",
  "nJet", "nBJetLoose", "nBJetMedium", "nLep","nTau",
  "lep1_isTight", "lep2_isTight", "tau_isTight","failsTightChargeCut", "nHTTv2", "nElectron", "nMuon",
  "N_jetAK12", "N_jetAK8",
  "hadtruth", "hadtruth_boosted", "hadtruth_semi_boosted",
  "bWj1Wj2_isGenMatchedWithKinFit", "bWj1Wj2_isGenMatched_IHEP",
  "bWj1Wj2_isGenMatched_CSVsort3rd", "bWj1Wj2_isGenMatched_highestCSV",
  "bWj1Wj2_isGenMatched_CSVsort3rd_WithKinFit", "bWj1Wj2_isGenMatched_highestCSV_WithKinFit",
  "bWj1Wj2_isGenMatched_boosted", "bWj1Wj2_isGenMatched_boosted_WithKinFit",
  "bWj1Wj2_isGenMatched_semi_boosted", "bWj1Wj2_isGenMatched_semi_boosted_WithKinFit"
		]


        if trainvar=="noHTT" and channel=="2lss_1tau" and bdtType=="evtLevelTT_TTH" and all==False :
			return [
			'avg_dr_jet',
			'dr_lep1_tau',
			'dr_lep2_tau',
			'dr_leps',
			'lep1_conePt',
			'lep2_conePt',
			'mT_lep2',
			'mTauTauVis1',
			'mTauTauVis2',
			'mbb',
			'mindr_lep1_jet',
			'mindr_lep2_jet',
			'mindr_tau_jet',
			'nJet',
			'ptmiss',
			'tau_pt',
                        'tau_mva',
                        'nBJetMedium',
			'max_eta_Lep',
			'lep1_eta', 'lep2_eta', 'tau_eta',
                        'lep1_pt', 'lep2_pt',
                #'HadTop_eta', 'HadTop_pt', 'MT_met_lep1', 'avg_dr_jet',
                #'bWj1Wj2_isGenMatched', 'bWj1Wj2_isGenMatchedWithKinFit',
                #'evtWeight',
                #'fitHTptoHTpt', 'fittedHadTop_eta', 'fittedHadTop_pt', #'genTopPt', 'genWeight', 'hadtruth',
                #'htmiss', 'lep1_conePt',  #'lep1_frWeight',
                #'lep1_genLepPt',
                #'mT_lep1', 'mT_lep2', 'max_lep_eta',
                #'mbb', 'ptbb', 'mbb_loose', 'ptbb_loose',
                #'memOutput_LR', 'memOutput_errorFlag', 'memOutput_isValid', 'memOutput_ttZ_LR', 'memOutput_ttZ_Zll_LR', 'memOutput_tt_LR',
                #'mindr_lep1_jet',
                #'mindr_lep2_jet',
                #'mindr_tau_jet',


			]

        if trainvar=="noHTT" and channel=="2lss_1tau" and bdtType=="evtLevelTTV_TTH" and all==False :
			return [
			'avg_dr_jet',
			'dr_lep1_tau',
			'dr_leps',
			'lep1_conePt',
			'lep2_conePt',
			'mT_lep1',
			'mT_lep2',
			'mTauTauVis1',
			'mTauTauVis2',
			'mindr_lep1_jet',
			'mindr_lep2_jet',
			'mindr_tau_jet',
			'ptmiss',
			'max_lep_eta',
			'tau_pt'
			]

        if trainvar=="noHTT" and channel=="2lss_1tau" and "evtLevelSUM_TTH" in bdtType and all==False :
			return [
			'avg_dr_jet',
			'dr_lep1_tau',
			'dr_lep2_tau',
			'dr_leps',
			'lep1_conePt',
			'lep2_conePt',
			'mT_lep1',
			'mT_lep2',
			'mTauTauVis1',
			'mTauTauVis2',
			'max_lep_eta',
			'mbb',
			'mindr_lep1_jet',
			'mindr_lep2_jet',
			'mindr_tau_jet',
			'nJet',
			'ptmiss',
			'tau_pt',
			]

        if trainvar=="HTT" and channel=="2lss_1tau" and bdtType=="evtLevelTT_TTH" and all==False :
			return [
			"avg_dr_jet",
			"dr_lep1_tau",
			"dr_lep2_tau",
			"dr_leps",
			"lep2_conePt",
			"mT_lep1",
			"mT_lep2",
			"mTauTauVis2",
			"max_lep_eta",
			"mbb",
			"mindr_lep1_jet",
			"mindr_lep2_jet",
			"mindr_tau_jet",
			"nJet",
			"ptmiss",
			"tau_pt",
			'mvaOutput_hadTopTaggerWithKinFit',
			'unfittedHadTop_pt'
			]
			"""
			'avg_dr_jet',
			'dr_lep1_tau',
			'dr_lep2_tau',
			'dr_leps',
			'lep1_conePt',
			'lep2_conePt',
			'mT_lep2',
			'mTauTauVis2',
			'mbb',
			'mindr_lep1_jet',
			'mindr_lep2_jet',
			'mindr_tau_jet',
			'ptmiss',
			'tau_pt',
			'mvaOutput_hadTopTaggerWithKinFit',
			'mvaOutput_Hj_tagger',
			'unfittedHadTop_pt',
			"""

        if trainvar=="HTT_LepID" and channel=="2lss_1tau" and bdtType=="evtLevelTT_TTH" and all==False :
			return [
			'mTauTauVis1', 'mTauTauVis2', 'tau_pt',
			'mvaOutput_hadTopTaggerWithKinFit',
			'lep1_tth_mva', 'lep2_tth_mva'
			]

        if trainvar=="oldVar"  and channel=="2lss_1tau"  and bdtType=="evtLevelTT_TTH" and all==False :return [
		"max_lep_eta",
		"nJet25_Recl",
		"mindr_lep1_jet",
		"mindr_lep2_jet",
		"min(met_pt,400)",
		"avg_dr_jet",
		"MT_met_lep1"
		]

        if trainvar=="oldVar"  and channel=="2lss_1tau"  and bdtType=="evtLevelTTV_TTH" and all==False :return [
		"max_lep_eta",
		"MT_met_lep1",
		"nJet25_Recl",
		"mindr_lep1_jet",
		"mindr_lep2_jet",
		"lep1_conePt",
		"lep2_conePt"
		]

        if trainvar=="oldVarA"  and channel=="2lss_1tau"  and bdtType=="evtLevelTT_TTH" and all==False :return [
		"nJet","mindr_lep1_jet","avg_dr_jet",
		"max_lep_eta",
		"lep2_conePt","dr_leps","tau_pt","dr_lep1_tau"
		]

        if trainvar=="oldVarA"  and channel=="2lss_1tau"  and bdtType=="evtLevelTTV_TTH" and all==False :return [
		"mindr_lep1_jet","mindr_lep2_jet", "avg_dr_jet", "max_lep_eta",
		"lep1_conePt", "lep2_conePt", "mT_lep1", "dr_leps", "mTauTauVis1", "mTauTauVis2"
		]

        if trainvar=="HTT_LepID" and channel=="2lss_1tau" and bdtType=="evtLevelTTV_TTH" and all==False :
			return [
			'dr_lep1_tau', 'dr_lep2_tau', 'dr_leps',
			'mT_lep1', 'mT_lep2', 'mTauTauVis1',
			'mTauTauVis2', 'mindr_lep1_jet', 'mindr_lep2_jet',
			'mvaOutput_hadTopTaggerWithKinFit', 'nJet25_Recl', 'ptmiss', 'tau_pt',
			'unfittedHadTop_pt', 'lep1_pt',
			'lep1_tth_mva', 'lep2_tth_mva', 'tau_mva'
			]

        if trainvar=="HTT" and channel=="2lss_1tau" and bdtType=="evtLevelTTV_TTH" and all==False :
			return [
			'avg_dr_jet',
			'dr_lep1_tau',
			'dr_lep2_tau',
			'dr_leps',
			'lep1_conePt',
			'lep2_conePt',
			'mT_lep1',
			'mT_lep2',
			'mTauTauVis1',
			'mTauTauVis2',
			'max_lep_eta',
			'mindr_lep1_jet',
			'mindr_lep2_jet',
			'mindr_tau_jet',
			'nJet',
			'ptmiss',
			'tau_pt',
			'mvaOutput_hadTopTaggerWithKinFit',
			]

        if trainvar=="HTTMEM" and channel=="2lss_1tau" and bdtType=="evtLevelTT_TTH" and all==False :
			return [
			'avg_dr_jet',
			'dr_lep1_tau',
			'dr_lep2_tau',
			'dr_leps',
			'lep2_conePt',
			'mT_lep1',
			'mT_lep2',
			'mTauTauVis2',
			'max_lep_eta',
			'mbb',
			'mindr_lep1_jet',
			'mindr_lep2_jet',
			'mindr_tau_jet',
			'nJet',
			'ptmiss',
			'tau_pt',
			"memOutput_LR",
			'mvaOutput_hadTopTaggerWithKinFit',
			'unfittedHadTop_pt',
			]

        if trainvar=="HTTMEM" and channel=="2lss_1tau" and bdtType=="evtLevelTTV_TTH" and all==False :
			return [
			'avg_dr_jet',
			'dr_lep1_tau',
			'dr_lep2_tau',
			'dr_leps',
			'lep2_conePt',
			'mT_lep1',
			'mT_lep2',
			'mTauTauVis2',
			'max_lep_eta',
			'mbb',
			'mindr_lep1_jet',
			'mindr_lep2_jet',
			'mindr_tau_jet',
			'nJet',
			'ptmiss',
			'tau_pt',
			"memOutput_LR",
			'mvaOutput_hadTopTaggerWithKinFit',
			'unfittedHadTop_pt',
			]

        if trainvar=="HTTMEM" and channel=="2lss_1tau" and bdtType=="evtLevelSUM_TTH_M" and all==False :
			return [
			'avg_dr_jet',
			'dr_lep1_tau',
			'dr_lep2_tau',
			'dr_leps',
			'lep2_conePt',
			'mT_lep1',
			'mT_lep2',
			'mTauTauVis2',
			'max_lep_eta',
			'mbb',
			'mindr_lep1_jet',
			'mindr_lep2_jet',
			'mindr_tau_jet',
			'nJet',
			'ptmiss',
			'tau_pt',
			#"memOutput_LR",
			'mvaOutput_hadTopTaggerWithKinFit',
			#'mvaOutput_Hj_tagger',
			'HadTop_pt', #'unfittedHadTop_pt',
			]

        if trainvar=="HTT" and channel=="2lss_1tau" and "evtLevelSUM_TTH_M" in bdtType and all==False :
			return [
			'avg_dr_jet',
			'dr_lep1_tau',
			'dr_lep2_tau',
			'dr_leps',
			'lep2_conePt',
			'mT_lep1',
			'mT_lep2',
			'mTauTauVis2',
			'max_lep_eta',
			'mbb',
			'mindr_lep1_jet',
			'mindr_lep2_jet',
			'mindr_tau_jet',
			'nJet',
			'ptmiss',
			'tau_pt',
			'mvaOutput_hadTopTaggerWithKinFit',
			#'mvaOutput_Hj_tagger',
			'unfittedHadTop_pt',
			]

        if trainvar=="HTTMEM" and channel=="2lss_1tau" and bdtType=="evtLevelSUM_TTH_T" and all==False :
			return [
			'avg_dr_jet',
			'dr_lep1_tau',
			'dr_lep2_tau',
			'dr_leps',
			'lep2_conePt',
			'mT_lep1',
			'mT_lep2',
			'mTauTauVis2',
			'max_lep_eta',
			'mbb',
			'mindr_lep1_jet',
			'mindr_lep2_jet',
			'mindr_tau_jet',
			'nJet',
			'ptmiss',
			'tau_pt',
			"memOutput_LR",
			'mvaOutput_hadTopTaggerWithKinFit',
			'mvaOutput_Hj_tagger',
			'unfittedHadTop_pt',
			]
