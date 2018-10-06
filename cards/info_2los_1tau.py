hasHTT = True

channelInTree='2los_1tau_forBDTtraining' #'2los_1tau_Loose'
inputPath='/hdfs/local/acaan/ttHAnalysis/2017/2los_1tau_toBDT_2018Sep27_noMassCut/histograms/2los_1tau/forBDTtraining/'
FastsimWP= "LooseLep_TightTau"
criteria=[]
testtruth="bWj1Wj2_isGenMatchedWithKinFit"

def trainVars(all):
        if channel=="2los_1tau" and all==True  :return [
"lep1_pt", "lep1_conePt", "lep1_eta", "lep1_tth_mva", "mindr_lep1_jet", "mT_lep1", "dr_lep1_tau_os",
"lep2_pt", "lep2_conePt", "lep2_eta", "lep2_tth_mva", "mindr_lep2_jet", "mT_lep2", "dr_lep2_tau_ss",
"mindr_tau_jet", "avg_dr_jet", "ptmiss",  "htmiss", "tau_mva", "tau_pt", "tau_eta",
"dr_leps", "mTauTauVis", "genWeight", "evtWeight",
"lep1_genLepPt", "lep2_genLepPt",
"tau_genTauPt",
"lep1_fake_prob", "lep2_fake_prob", "tau_fake_prob","dr_leps",
"max_lep_eta", "min_lep_eta",
"HadTop_eta",
"mbb", "ptbb", "b1_pt", "b2_pt", "drbb", "detabb",
"mbb_loose", "ptbb_loose", "b1_loose_pt", "b2_loose_pt", "drbb_loose", "detabb_loose",
"fittedHadTop_pt", "fittedHadTop_eta", "fitHTptoHTpt", "fitHTptoHTmass",
"minDR_HTTv2_lep", "minDR_HTTv2_tau", "minDR_HTTv2_L",
"minDR_AK12_lep", "DR_AK12_tau", "minDR_AK12_L",
"HTTv2_lead_mass", "AK12_lead_mass",
"HTTv2_lead_pt", "AK12_lead_pt",
"res-HTT", "res-HTT_IHEP",
"res-HTT_CSVsort3rd", "res-HTT_highestCSV",
"res-HTT_CSVsort3rd_WithKinFit", "res-HTT_highestCSV_WithKinFit",
"HadTop_pt",  "genTopPt",
"HadTop_pt_multilep",
"HadTop_pt_CSVsort3rd", "HadTop_pt_highestCSV",
"HadTop_pt_CSVsort3rd_WithKinFit", "HadTop_pt_highestCSV_WithKinFit",
"genTopPt_multilep",
"genTopPt_CSVsort3rd", "genTopPt_highestCSV",
"genTopPt_CSVsort3rd_WithKinFit", "genTopPt_highestCSV_WithKinFit",
"HTT_boosted", "genTopPt_boosted", "HadTop_pt_boosted",
"HTT_boosted_WithKinFit", "genTopPt_boosted_WithKinFit", "HadTop_pt_boosted_WithKinFit",
"HTT_semi_boosted", "genTopPt_semi_boosted", "HadTop_pt_semi_boosted",
"HTT_semi_boosted_WithKinFit", "genTopPt_semi_boosted_WithKinFit", "HadTop_pt_semi_boosted_WithKinFit",
"nJet", "nBJetLoose", "nBJetMedium", "lep1_isTight", "lep1_charge",
"lep2_isTight", "lep2_charge", "tau_isTight",
"tau_charge", "lep1_tau_charge", "nLep",  "nTau",
"nHTTv2", "nElectron", "nMuon",
"N_jetAK12", "N_jetAK8",
"hadtruth",  "hadtruth_boosted", "hadtruth_semi_boosted",
"bWj1Wj2_isGenMatchedWithKinFit", "bWj1Wj2_isGenMatched_IHEP",
"bWj1Wj2_isGenMatched_CSVsort3rd", "bWj1Wj2_isGenMatched_highestCSV",
"bWj1Wj2_isGenMatched_CSVsort3rd_WithKinFit", "bWj1Wj2_isGenMatched_highestCSV_WithKinFit",
"bWj1Wj2_isGenMatched_boosted",
"bWj1Wj2_isGenMatched_boosted_WithKinFit",
"bWj1Wj2_isGenMatched_semi_boosted", "bWj1Wj2_isGenMatched_semi_boosted_WithKinFit"
		]

        if trainvar=="noHTT" and channel=="2los_1tau" and bdtType=="evtLevelTT_TTH" and all==False :
			return [
			'avg_dr_jet', 'dr_lep1_tau_os',
			'dr_lep2_tau_ss',
			'dr_leps',
			'lep1_conePt',
			'lep2_conePt',
			'mT_lep1',
			'mT_lep2',
			'mTauTauVis',
			'tau_pt', 'tau_eta',
			'max_lep_eta', 'min_lep_eta', 'lep2_eta','lep1_eta',
			'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet',
			'mbb', 'ptbb',  #(medium b)
			'mbb_loose', 'ptbb_loose', 'b1_loose_pt', 'b2_loose_pt', 'drbb_loose', 'detabb_loose',
			'ptmiss', 'htmiss',
			'nBJetLoose',
			'nBJetMedium',
			'nJet',
			]

        if trainvar=="noHTT" and channel=="2los_1tau" and bdtType=="evtLevelTTV_TTH" and all==False :
			return [
			'avg_dr_jet', #'dr_lep1_tau_os',
			'dr_lep2_tau_ss',
			'dr_leps',
			#'lep1_conePt',
			#'lep2_conePt',
			#'mT_lep1',
			'mT_lep2',
			'mTauTauVis',
			'tau_pt', 'tau_eta',
			#'max_lep_eta', #'min_lep_eta', #'lep2_eta','lep1_eta',
			'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet',
			#'mbb', 'ptbb', # (medium b)
			'mbb_loose', #'ptbb_loose',
			'ptmiss', #'htmiss',
			#'nBJetLoose',
			#'nBJetMedium',
			'nJet',
			]

        if trainvar=="HTT" and channel=="2los_1tau" and bdtType=="evtLevelTT_TTH" and all==False :
			return [
			'avg_dr_jet', #'dr_lep1_tau_os',
			'dr_lep2_tau_ss',
			'dr_leps',
			#'lep1_conePt',
			#'lep2_conePt',
			#'mT_lep1',
			'mT_lep2',
			'mTauTauVis',
			'tau_pt', 'tau_eta',
			#'max_lep_eta', 'min_lep_eta', 'lep2_eta','lep1_eta',
			'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet',
			#'mbb', 'ptbb', # (medium b)
			'mbb_loose', #'ptbb_loose',
			'ptmiss', #'htmiss',
			#'nBJetLoose',
			#'nBJetMedium',
			'nJet',
			'mvaOutput_hadTopTaggerWithKinFit',
			'unfittedHadTop_pt',
			#'dr_lepOS_HTfitted', 'dr_lepOS_HTunfitted',
			#'dr_lepSS_HTfitted', 'dr_lepSS_HTunfitted',
			'dr_tau_HTfitted', #'dr_tau_HTunfitted',
			'fitHTptoHTmass', #'fitHTptoHTpt',
			#'HadTop_eta', 'HadTop_pt',
			#'mass_lepOS_HTfitted', 'mass_lepSS_HTfitted',
			]
