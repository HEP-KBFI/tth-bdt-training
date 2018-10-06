hasHTT = True

channelInTree='2lss_SS_forBDTtraining'
#inputPath='/hdfs/local/acaan/ttHAnalysis/2017/2lss_0tau_toBDT_2018Sep24_two_loops/histograms/2lss/forBDTtraining_SS/'
#inputPath='/hdfs/local/acaan/ttHAnalysis/2017/2lss_0tau_toBDT_2018Sep27_two_loops_lepLoose/histograms/2lss/forBDTtraining_SS/' ## no mass cut in training or aplication
#inputPath='/hdfs/local/acaan/ttHAnalysis/2017/2lss_0tau_toBDT_2018Sep27_two_loops_lepLoose_noMassCut/histograms/2lss/forBDTtraining_SS/' ### this one has no mass cut on aplication
#inputPath='/hdfs/local/acaan/ttHAnalysis/2017/2lss_0tau_toBDT_2018Sep27_two_loops_lepLoose_noMassCut_2/histograms/2lss/forBDTtraining_SS/'
#inputPath='/hdfs/local/acaan/ttHAnalysis/2017/2lss_0tau_toBDT_2018Sep29_noMassCut_Notisolated/histograms/2lss/forBDTtraining_SS/'
inputPath='/hdfs/local/acaan/ttHAnalysis/2017/2lss_0tau_toBDT_2018Sep29_noMassCut_isolated/histograms/2lss/forBDTtraining_SS/'


channelInTreeTight=''
inputPathTight=''
channelInTreeFS=''
inputPathTightFS=''
criteria=[]
testtruth="bWj1Wj2_isGenMatchedWithKinFit"
FullsimWP=""
FastsimWP=""
FastsimTWP=""

HTT_var = []

def trainVars(all):
	#if all==False :
	return [
	"lep1_pt", "lep1_conePt", "lep1_eta", "lep1_tth_mva", "mindr_lep1_jet",
	"mindr_lep2_jet",
	"mT_lep1",  "MT_met_lep1",
	"lep2_pt", "lep2_conePt", "lep2_eta", "max_lep_eta", "avg_dr_lep",
	"lep2_tth_mva", "mT_lep2",
	"avg_dr_jet",  "nJet25_Recl", "ptmiss", "htmiss",
	"dr_leps",
	"lep1_genLepPt", "lep2_genLepPt",
	"lep1_frWeight", "lep2_frWeight",
	"mvaOutput_2lss_ttV",  "mvaOutput_2lss_ttbar", "mvaDiscr_2lss",
	"Hj_tagger", "Hjj_tagger",
	"lumiScale", "genWeight", "evtWeight",
	"min(met_pt,400)",
	"mbb","ptbb", "mbb_loose","ptbb_loose",
	"minDR_HTTv2_lep", "minDR_HTTv2_2_lep",
	"minDR_AK12_lep", "minDR_AK12_2_lep",
	#///
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
	"HTTv2_lead_pt", "AK12_lead_pt",
	"HadTop_pt_2",  "genTopPt_2",
	"HadTop_pt_multilep_2",
	"HadTop_pt_CSVsort3rd_2", "HadTop_pt_highestCSV_2",
	"HadTop_pt_CSVsort3rd_WithKinFit_2", "HadTop_pt_highestCSV_WithKinFit_2",
	"genTopPt_multilep_2",
	"genTopPt_CSVsort3rd_2", "genTopPt_highestCSV_2",
	"genTopPt_CSVsort3rd_WithKinFit_2", "genTopPt_highestCSV_WithKinFit_2",
	#////
	"HTT_boosted", "genTopPt_boosted", "HadTop_pt_boosted",
	"HTT_boosted_WithKinFit", "genTopPt_boosted_WithKinFit", "HadTop_pt_boosted_WithKinFit",
	"HTT_semi_boosted", "genTopPt_semi_boosted", "HadTop_pt_semi_boosted",
	"HTT_semi_boosted_WithKinFit", "genTopPt_semi_boosted_WithKinFit", "HadTop_pt_semi_boosted_WithKinFit",
	#///
	"nJet", "nBJetLoose", "nBJetMedium", "nLep",
	"lep1_isTight", "lep2_isTight", "failsTightChargeCut",
	"nHTTv2", "nElectron", "nMuon",
	"N_jetAK12", "N_jetAK8",
	"hadtruth", "hadtruth_boosted", "hadtruth_semi_boosted",
	#////
	"bWj1Wj2_isGenMatchedWithKinFit", "bWj1Wj2_isGenMatched_IHEP",
	"bWj1Wj2_isGenMatched_CSVsort3rd", "bWj1Wj2_isGenMatched_highestCSV",
	"bWj1Wj2_isGenMatched_CSVsort3rd_WithKinFit", "bWj1Wj2_isGenMatched_highestCSV_WithKinFit",
	"bWj1Wj2_isGenMatchedWithKinFit_2", "bWj1Wj2_isGenMatched_IHEP_2",
	"bWj1Wj2_isGenMatched_CSVsort3rd_2", "bWj1Wj2_isGenMatched_highestCSV_2",
	"bWj1Wj2_isGenMatched_CSVsort3rd_WithKinFit_2", "bWj1Wj2_isGenMatched_highestCSV_WithKinFit_2",
	#/////
	"bWj1Wj2_isGenMatched_boosted", "bWj1Wj2_isGenMatched_boosted_WithKinFit",
	"bWj1Wj2_isGenMatched_semi_boosted", "bWj1Wj2_isGenMatched_semi_boosted_WithKinFit",
	"resolved_and_semi",
	"boosted_and_semi",
	"resolved_and_boosted", "cleanedJets_fromAK12"
	]

	if trainvar=="noHTT"  and bdtType=="evtLevelTTV_TTH" and all==False :return [
		"lep1_pt", "lep1_conePt", "lep1_eta", "lep1_tth_mva", "mindr_lep1_jet",
		"mindr_lep2_jet",
		"mT_lep1",  "MT_met_lep1",
		"lep2_pt", "lep2_conePt", "lep2_eta", "max_lep_eta", "avg_dr_lep",
		"lep2_tth_mva", "mT_lep2",
		"avg_dr_jet",  "nJet25_Recl", "ptmiss", "htmiss",
		]

	if trainvar=="noHTT"   and bdtType=="evtLevelTT_TTH" and all==False :return [
		"lep1_pt", "lep1_conePt", "lep1_eta", "lep1_tth_mva", "mindr_lep1_jet",
		"mindr_lep2_jet",
		"mT_lep1",  "MT_met_lep1",
		"lep2_pt", "lep2_conePt", "lep2_eta", "max_lep_eta", "avg_dr_lep",
		"lep2_tth_mva", "mT_lep2",
		"avg_dr_jet",  "nJet25_Recl", "ptmiss", "htmiss",
		]

	if trainvar=="noHTT" and "evtLevelSUM_TTH" in bdtType and all==False :return [
		"lep1_pt", "lep1_conePt", "lep1_eta", "lep1_tth_mva", "mindr_lep1_jet",
		"mindr_lep2_jet",
		"mT_lep1",  "MT_met_lep1",
		"lep2_pt", "lep2_conePt", "lep2_eta", "max_lep_eta", "avg_dr_lep",
		"lep2_tth_mva", "mT_lep2",
		"avg_dr_jet",  "nJet25_Recl", "ptmiss", "htmiss",
		]
