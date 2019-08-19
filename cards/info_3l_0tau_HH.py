hasHTT = True


def read_from(
        Bkg_mass_rand,
	tauID_training,
	tauID_application = "dR03mvaMedium"
    	):

        mass_rand_algo="default"
        if(Bkg_mass_rand == "oversampling"):
		mass_rand_algo= "oversampling"
        else:
		mass_rand_algo= "default"

	channelInTree = 'hh_3l_OS_forBDTtraining'
	inputPath = '/hdfs/local/ssawant/hhAnalysis/2017/20181108/histograms/hh_3l/forBDTtraining_OS/' ## dR03mvaLoose Ntuples with DYJets_HT samples and mTauTau variable

	# you only calculate limits in one tauID WP,
	# so this should be the same independent of the tauID wp you are using for define the training samples
	# I am guessing one, fix with the suitable case
	# to make balancing of BKGs
	TTdatacard  = 34.4078
	DYdatacard  = 41.6669    # 23.81 + 4.45982 + 13.3971
	TTZdatacard = 9.0017
	TTWdatacard = 7.71022
	WZdatacard  = 265.006    # 264.693 + 0.0691472 + 0.243493

	keys=[
	##Ram: as we discussed before think about removing spurious BKGs from training
	## as eg 'ttHToNonbb_M125_powheg_ext1', VHToNonbb_M125_v14-v2...
	## --> they serve only to make the training stable
                'TTTo2L2Nu', 'TTToSemiLeptonic', ## It does have a * on the loading, it will load the PSWeights
		#'TTToHadronic', -- there is zero events in this sample
                'DY', ## It does have a * on the loading, it will load all DY
                #'ZZ', ## ZZ inclusive samples + ZZZ
                #'WW',  ## WW inclusive samples -1
		#'WpWpJJ_EWK_QCD', ## WW inclusive samples -- there is only one event on this samples -- you should not use it for training
                'WZ', ## WZ inclusive sample + WZZ
                #'TTZJets',  ## TTV inclusive samples
		#'TTWJets', there is no representative yield neither stats
                ##'TTWW', ## TTWW inclusive sample -- the yield is negligible, it will just add noise
                #'VHToNonbb_M125',  ## VH inclusive samples
                #'ttH', ## TTH inclusive samples
		## 2v2t and 4v not a major contribution to signal in the 2l_2tau channel
		'signal_ggf_spin0_250_hh_4t', #'signal_ggf_spin0_250_hh_2v2t',  'signal_ggf_spin0_250_hh_4v',
		'signal_ggf_spin0_260_hh_4t', #'signal_ggf_spin0_260_hh_2v2t',  'signal_ggf_spin0_260_hh_4v',
		'signal_ggf_spin0_270_hh_4t', #'signal_ggf_spin0_270_hh_2v2t',  'signal_ggf_spin0_270_hh_4v',
		'signal_ggf_spin0_280_hh_4t', #'signal_ggf_spin0_280_hh_2v2t',  'signal_ggf_spin0_280_hh_4v',
		'signal_ggf_spin0_300_hh_4t', #'signal_ggf_spin0_300_hh_2v2t',  'signal_ggf_spin0_300_hh_4v',
		'signal_ggf_spin0_350_hh_4t', #'signal_ggf_spin0_350_hh_2v2t',  'signal_ggf_spin0_350_hh_4v',
		'signal_ggf_spin0_400_hh_4t', #'signal_ggf_spin0_400_hh_2v2t',  'signal_ggf_spin0_400_hh_4v',
		'signal_ggf_spin0_450_hh_4t', #'signal_ggf_spin0_450_hh_2v2t',  'signal_ggf_spin0_450_hh_4v',
		'signal_ggf_spin0_500_hh_4t', #'signal_ggf_spin0_500_hh_2v2t',  'signal_ggf_spin0_500_hh_4v',
		'signal_ggf_spin0_550_hh_4t', #'signal_ggf_spin0_550_hh_2v2t',  'signal_ggf_spin0_550_hh_4v',
		'signal_ggf_spin0_600_hh_4t', #'signal_ggf_spin0_600_hh_2v2t',  'signal_ggf_spin0_600_hh_4v',
		'signal_ggf_spin0_650_hh_4t', #'signal_ggf_spin0_650_hh_2v2t',  'signal_ggf_spin0_650_hh_4v',
		'signal_ggf_spin0_700_hh_4t', #'signal_ggf_spin0_700_hh_2v2t',  'signal_ggf_spin0_700_hh_4v',
		'signal_ggf_spin0_750_hh_4t', #'signal_ggf_spin0_750_hh_2v2t',  'signal_ggf_spin0_750_hh_4v',
		'signal_ggf_spin0_800_hh_4t', #'signal_ggf_spin0_800_hh_2v2t',  'signal_ggf_spin0_800_hh_4v',
		'signal_ggf_spin0_850_hh_4t', #'signal_ggf_spin0_850_hh_2v2t',  'signal_ggf_spin0_850_hh_4v',
		'signal_ggf_spin0_900_hh_4t', #'signal_ggf_spin0_900_hh_2v2t',  'signal_ggf_spin0_900_hh_4v',
		'signal_ggf_spin0_1000_hh_4t', #'signal_ggf_spin0_1000_hh_2v2t',  'signal_ggf_spin0_1000_hh_4v',
	]
	masses = [250,260,270,280,300,350,400,450,500,550,600,650,700,750,800,850,900,1000]
	masses_test = [300,500,800]

	masses_low = [250,260,270,280,300,350,400]
	masses_test_low = [300]

	masses_high = [450,500,550,600,650,700,750,800,850,900,1000]
	masses_test_high = [500,800]

	output = {
		"channelInTree" : channelInTree,
		"inputPath" : inputPath,
		"TTdatacard" : TTdatacard,
		"DYdatacard" : DYdatacard,
		"TTZdatacard" : TTZdatacard,
		"TTWdatacard" : TTWdatacard,
		"WZdatacard" : WZdatacard,
		"keys" : keys,
		"masses" : masses,
		"masses_test": masses_test,
		"masses_low" : masses_low,
		"masses_test_low": masses_test_low,
		"masses_high" : masses_high,
		"masses_test_high": masses_test_high,
		"mass_randomization" : mass_rand_algo,
		}

	return output


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
