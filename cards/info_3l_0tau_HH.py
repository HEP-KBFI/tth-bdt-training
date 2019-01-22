hasHTT = True

channelInTree='hh_3l_OS_forBDTtraining'
inputPath='/hdfs/local/ssawant/hhAnalysis/2017/20181108/histograms/hh_3l/forBDTtraining_OS/'



# to make balancing of BKGs 
fastsimTT   = 61.3864 + 0.0189422 + 0.  # taken from a first run on the training samples 
fastsimTTV  = 9.4032+7.75227
fastsimDY   = 36.745 
fastsimWZ   = 227.257
#
TTdatacard  = 34.4078
DYdatacard  = 41.6669    # 23.81 + 4.45982 + 13.3971
TTVdatacard = 9.0017 + 7.71022 
WZdatacard  = 265.006    # 264.693 + 0.0691472 + 0.243493


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
	if all==True :return [
		"lep1_pt", "lep1_conePt", "lep1_eta", "lep1_tth_mva", "mindr_lep1_jet", "mT_lep1", 
		"lep2_pt", "lep2_conePt", "lep2_eta", "lep2_tth_mva", "mindr_lep2_jet", "mT_lep2",
		"lep3_pt", "lep3_conePt", "lep3_eta", "lep3_tth_mva", "mindr_lep3_jet", "mT_lep3", 
		"avg_dr_jet", "ptmiss",  "htmiss", "dr_leps",
		#"lumiScale", 
		"genWeight", "evtWeight",
		"lep1_genLepPt", "lep2_genLepPt", "lep3_genLepPt",
		"lep1_fake_prob", "lep2_fake_prob", "lep3_fake_prob",
		"lep1_frWeight", "lep2_frWeight", "lep3_frWeight",  
		#"mvaOutput_3l_ttV", "mvaOutput_3l_ttbar", "mvaDiscr_3l",
		"mbb_loose", "mbb_medium",
		"dr_lss", "dr_los1", "dr_los2",
		"met", "mht", "met_LD", "HT", "STMET",
		#"mSFOS2l", 
		"m_jj", "diHiggsVisMass", "diHiggsMass",
		"mTMetLepton1", "mTMetLepton2",
		"vbf_m_jj", "vbf_dEta_jj", "numSelJets_nonVBF",
		#
		"nJet", "nBJetLoose", "nBJetMedium", "nElectron", "nMuon",
		"lep1_isTight", "lep2_isTight", "lep3_isTight",
		"sumLeptonCharge", "numSameFlavor_OS", "isVBF"
		]

	#if trainvar=="noTopness"  and bdtType=="evtLevelSUM_HH_res" and all==False :return [
	if (trainvar=="noTopness"  and bdtType=="evtLevelSUM_HH_res" and all==False) or \
	(trainvar=="noTopness"  and bdtType=="evtLevelWZ_HH_res" and all==False) or \
	(trainvar=="noTopness"  and bdtType=="evtLevelDY_HH_res" and all==False) or \
	(trainvar=="noTopness"  and bdtType=="evtLevelTT_HH_res" and all==False) :
		return [
		"lep1_conePt", "lep1_eta", #"lep1_tth_mva", 
		"mindr_lep1_jet", "mT_lep1",
                "lep2_conePt", "lep2_eta", #"lep2_tth_mva", 
		"mindr_lep2_jet", "mT_lep2",
                "lep3_conePt", "lep3_eta", #"lep3_tth_mva", 
		"mindr_lep3_jet", "mT_lep3",
                "avg_dr_jet", #"ptmiss",  "htmiss", 
		"dr_leps",
                #"lumiScale", "genWeight", "evtWeight",
                #"lep1_genLepPt", "lep2_genLepPt", "lep3_genLepPt",
                #"lep1_fake_prob", "lep2_fake_prob", "lep3_fake_prob",
                #"lep1_frWeight", "lep2_frWeight", "lep3_frWeight",
                #"mvaOutput_3l_ttV", "mvaOutput_3l_ttbar", "mvaDiscr_3l",
                #"mbb_loose", "mbb_medium",
                "dr_lss", "dr_los1", "dr_los2",
                #"met", "mht", 
		"met_LD", #"HT", "STMET",
                #"mSFOS2l", 
		"m_jj", "diHiggsMass", #"diHiggsVisMass",
                "mTMetLepton1", "mTMetLepton2",
                #"vbf_m_jj", "vbf_dEta_jj", "numSelJets_nonVBF",
                #
                "nJet", #"nBJetLoose", "nBJetMedium", 
		"nElectron", #"nMuon",
                #"lep1_isTight", "lep2_isTight", "lep3_isTight",
                "sumLeptonCharge", "numSameFlavor_OS", #"isVBF"
		]

	if trainvar=="Updated"   and bdtType=="evtLevelSUM_TTH" and all==False :return [
                "lep1_pt", "lep1_conePt", "lep1_eta", "lep1_tth_mva", "mindr_lep1_jet", "mT_lep1",
                "lep2_pt", "lep2_conePt", "lep2_eta", "lep2_tth_mva", "mindr_lep2_jet", "mT_lep2",
                "lep3_pt", "lep3_conePt", "lep3_eta", "lep3_tth_mva", "mindr_lep3_jet", "mT_lep3",
                "avg_dr_jet", "ptmiss",  "htmiss", "dr_leps",
                #"lumiScale", "genWeight", "evtWeight",
                "lep1_genLepPt", "lep2_genLepPt", "lep3_genLepPt",
                "lep1_fake_prob", "lep2_fake_prob", "lep3_fake_prob",
                "lep1_frWeight", "lep2_frWeight", "lep3_frWeight",
                #"mvaOutput_3l_ttV", "mvaOutput_3l_ttbar", "mvaDiscr_3l",
                "mbb_loose", "mbb_medium",
                "dr_lss", "dr_los1", "dr_los2",
                "met", "mht", "met_LD", "HT", "STMET",
                #"mSFOS2l", 
		"m_jj", "diHiggsVisMass", "diHiggsMass",
                "mTMetLepton1", "mTMetLepton2",
                "vbf_m_jj", "vbf_dEta_jj", "numSelJets_nonVBF",
                #
                "nJet", "nBJetLoose", "nBJetMedium", "nElectron", "nMuon",
                "lep1_isTight", "lep2_isTight", "lep3_isTight",
                "sumLeptonCharge", "numSameFlavor_OS", "isVBF"
		]

