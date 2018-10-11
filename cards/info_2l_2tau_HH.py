hasHTT = False

channelInTree='hh_2l_2tau_sumOS_forBDTtraining'
inputPath='/hdfs/local/snandan/hhAnalysis/2017/2018Oct02_forBDTtraining/histograms/hh_2l_2tau/forBDTtraining_sumOS/'

## to make balancing of BKGs
fastsimTT=207.466+1.53457 # taken from a first run on the training samples
fastsimTTV=0.757482+1.51967
fastsimDY=1.0 # not using for this channel
## missing diboson
TTdatacard=15.9438
DYdatacard=15.9438
TTVdatacard=0.272943+0.0531975
## missing diboson

def trainVars(all):
	if all==True :return [ ## add all variavles to be read from the tree
	"lep1_pt", "lep1_conePt", "lep1_eta", "lep1_tth_mva", "mT_lep1",
      "lep2_pt", "lep2_conePt", "lep2_eta", "lep2_tth_mva", "mT_lep2",
      "tau1_pt", "tau1_eta", "tau1_mva",
      "tau2_pt", "tau2_eta", "tau2_mva",
      "dr_lep1_tau1", "dr_lep1_tau2", "dr_lep2_tau1", "dr_lep2_tau2",
      "dr_leps", "dr_taus", "avg_dr_jet",
      "met", "mht", "met_LD", "HT", "STMET",
      "Smin_llMEt", "m_ll", "pT_ll", "pT_llMEt", "Smin_lltautau",
      "mTauTauVis", "ptTauTauVis", "diHiggsVisMass", "diHiggsMass",
      "logTopness_publishedChi2", "logTopness_fixedChi2",
      "genWeight", "evtWeight",
      "nJet", "nBJet_loose", "nBJet_medium",
      "lep1_isElectron", "lep1_charge", "lep2_isElectron", "lep2_charge",
      "nElectron", "nMuon", "gen_mHH"
	]

	if trainvar=="noTopness"  and bdtType=="evtLevelSUM_HH_res" and all==False :return [
		"lep1_conePt", "mT_lep1",
		"lep2_conePt", "mT_lep2",
		"tau1_pt", "tau1_eta",
		"tau2_pt", "tau2_eta",
		"dr_lep1_tau1", "dr_lep1_tau2", "dr_lep2_tau1", "dr_lep2_tau2",
		"dr_leps", "dr_taus", "avg_dr_jet",
		"met", "mht", "met_LD",
		"mTauTauVis", "ptTauTauVis", "diHiggsVisMass", "diHiggsMass",
		"nJet", "nBJet_loose", "nBJet_medium",
		"lep1_charge", "lep2_charge",
		"nElectron", "gen_mHH"
		]

	if trainvar=="Updated"   and bdtType=="evtLevelSUM_TTH" and all==False :return [
		"lep1_conePt", "lep2_conePt", "lep3_conePt",
		"mindr_lep1_jet", "mindr_lep2_jet", "mindr_lep3_jet",
		"mT_lep1", "mT_lep2", "mT_lep3",
		#"max_lep_mT", "min_lep_mT",
		"max_lep_eta", "nJet",
		"res-HTT_CSVsort3rd", "HadTop_pt_CSVsort3rd", "mvaOutput_Hj_tagger",
		#"max_lep_dr_os",
		"min_lep_dr_os",
		"dr_lss",#"dr_los1","dr_los2",
		#"dr_leps",
		#"avg_dr_jet",
		#"nBJetLoose",
		"nBJetMedium",
		"nElectron", #"nMuon",
		#"mbb_medium",
		"ptmiss",
		]
