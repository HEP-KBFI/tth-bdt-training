# this needs to be a function to we also use on the NN fw
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

	channelInTree='hh_2l_2tau_sumOS_forBDTtraining'
	if ("dR03mvaLoose" in tauID_training):
		inputPath='/hdfs/local/ram/hhAnalysis/2017/2019Feb4_2l_2tau_forBDTtraining_dR03mvaLoose_w_mTauTau/histograms/hh_2l_2tau/forBDTtraining_sumOS/' ## dR03mvaLoose Ntuples with DYJets_HT samples and mTauTau variable
	elif ("dR03mvaVLoose" in tauID_training):
		inputPath='/hdfs/local/ram/hhAnalysis/2017/2019Feb4_2l_2tau_forBDTtraining_dR03mvaVLoose_w_mTauTau/histograms/hh_2l_2tau/forBDTtraining_sumOS/' ## dR03mvaVLoose Ntuples with DYJets_HT samples and mTauTau variable
	elif ("dR03mvaVVLoose" in tauID_training):
		inputPath='/hdfs/local/ram/hhAnalysis/2017/2019Feb4_2l_2tau_forBDTtraining_dR03mvaVVLoose_w_mTauTau/histograms/hh_2l_2tau/forBDTtraining_sumOS/' ## dR03mvaVVLoose Ntuples with DYJets_HT samples and mTauTau variable
	else : print ("There are no ntuples for this training WP: ", tauID_training)

	# you only calculate limits in one tauID WP,
	# so this should be the same independent of the tauID wp you are using for define the training samples
	# I am guessing one, fix with the suitable case
	TTdatacard = 1.0
	DYdatacard = 1.0
	TTZdatacard = 1.0
	TTWdatacard = 1.0
	WWdatacard = 1.0
	WZdatacard = 1.0
	ZZdatacard = 1.0
	VHdatacard = 1.0
	TTHdatacard = 1.0
	if tauID_application == "dR03mvaLoose" :
		TTdatacard=43.65
		DYdatacard=31.11
		TTZdatacard=0.38 ## =TTZ
		TTWdatacard=(0.14+0.01) ## =TTW+TTWW
		WWdatacard=0.30 ##  = WW
		WZdatacard=1.94 ##  = WZ 
		ZZdatacard=6.05  ## = ZZ
		VHdatacard=1.23
		TTHdatacard=0.47
	elif tauID_application == "dR03mvaMedium" :
		TTdatacard=15.94
		DYdatacard=15.73
		TTZdatacard=0.27 ## = TTZ
		TTWdatacard=0.05 ## = TTW+TTWW
		WWdatacard=0.23 ##  = WW
		WZdatacard=1.08 ##  = WZ
		ZZdatacard=4.78 ##  = ZZ
		VHdatacard=1.19
		TTHdatacard=0.28
	else : print ("Warning: you did not defined datacard normalization")

	keys=[
	##Ram: as we discussed before think about removing spurious BKGs from training
	## as eg 'ttHToNonbb_M125_powheg_ext1', VHToNonbb_M125_v14-v2...
	## --> they serve only to make the training stable
                'TTTo2L2Nu', 'TTToSemiLeptonic', ## It does have a * on the loading, it will load the PSWeights
		#'TTToHadronic', -- there is zero events in this sample
                'DY', ## It does have a * on the loading, it will load all DY
                'ZZ', ## ZZ inclusive samples + ZZZ
                'WW',  ## WW inclusive samples -1
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
		"WWdatacard" : WWdatacard,
		"WZdatacard" : WZdatacard,
		"ZZdatacard" : ZZdatacard,
		"VHdatacard" : VHdatacard,
		"TTHdatacard" : TTHdatacard,
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

def trainVars(all, trainvar = None, bdtType="evtLevelSUM_HH_2l_2tau_res"):
	if all==True :return [ ## add all variavles to be read from the tree
	"lep1_pt", "lep1_conePt", "lep1_eta", "lep1_tth_mva", "mT_lep1",
	"lep2_pt", "lep2_conePt", "lep2_eta", "lep2_tth_mva", "mT_lep2",
	"tau1_pt", "tau1_eta", "tau1_mva",
	"tau2_pt", "tau2_eta", "tau2_mva",
	"dr_lep1_tau1", "dr_lep1_tau2", "dr_lep2_tau1", "dr_lep2_tau2",
	"min_dr_lep_tau", "max_dr_lep_tau",
	"dr_leps", "dr_taus", "avg_dr_jet",
	"met", "mht", "met_LD", "HT", "STMET",
	"Smin_llMEt", "m_ll", "pT_ll", "pT_llMEt", "Smin_lltautau",
	"mTauTauVis", "ptTauTauVis", "diHiggsVisMass", 	"diHiggsMass",
	"logTopness_publishedChi2", "logTopness_fixedChi2",
	"genWeight", "evtWeight",
	"nJet", "nBJet_loose", "nBJet_medium",
	"lep1_isElectron", "lep1_charge", "lep2_isElectron", "lep2_charge", "sum_lep_charge",
	"nElectron", "nMuon", "gen_mHH",
	"lep1_phi", "lep2_phi", "tau1_phi", "tau2_phi",
	"m_lep1_tau1", "m_lep1_tau2", "m_lep2_tau1", "m_lep2_tau2", "mTauTau",
	"deltaEta_lep1_tau1", "deltaEta_lep1_tau2", "deltaEta_lep2_tau1", "deltaEta_lep2_tau2", "deltaEta_lep1_lep2", "deltaEta_tau1_tau2",
	"deltaPhi_lep1_tau1", "deltaPhi_lep1_tau2", "deltaPhi_lep2_tau1", "deltaPhi_lep2_tau2", "deltaPhi_lep1_lep2", "deltaPhi_tau1_tau2",
	"dr_lep1_tau1_tau2_min", "dr_lep1_tau1_tau2_max", "dr_lep2_tau1_tau2_min", "dr_lep2_tau1_tau2_max",
	"dr_lep_tau_min_OS", "dr_lep_tau_min_SS",
	"max_tau_eta", "max_lep_eta",
	"event"
	]

	if trainvar=="noTopness"  and bdtType=="evtLevelSUM_HH_2l_2tau_res" and all==False :return [
		#"lep1_conePt",
		"mT_lep1",
		#"lep2_conePt",
		"mT_lep2",
		"tau1_pt", #"tau1_eta",
		"tau2_pt", #"tau2_eta",
		#"dr_lep1_tau1", "dr_lep1_tau2", "dr_lep2_tau1", "dr_lep2_tau2",
		#"max_dr_lep_tau",
		#"min_dr_lep_tau",
		#"dr_taus", #"dr_leps", "avg_dr_jet",
		"met", #"mht", #"met_LD",
		#"mTauTauVis", #"ptTauTauVis",
		"m_ll", "mTauTau",
		"diHiggsVisMass", "diHiggsMass",
		"nBJet_medium", #"nJet",  "nBJet_loose",
		#"lep1_charge", "lep2_charge",
		"nElectron",
		"max_tau_eta", "max_lep_eta",
		"sum_lep_charge",
		"gen_mHH"
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

        if trainvar=="moreVars"  and bdtType=="evtLevelSUM_HH_2l_2tau_res" and all==False :return [
                "lep1_conePt", "lep1_eta", "lep1_phi",  "mT_lep1",
                "lep2_conePt",  "lep2_eta", "lep2_phi", "mT_lep2",
                "tau1_pt", "tau1_eta", "tau1_phi",
                "tau2_pt", "tau2_eta", "tau2_phi",
                "m_lep1_tau1", "m_lep1_tau2", "m_lep2_tau1", "m_lep2_tau2",
                "dr_lep1_tau1", "dr_lep1_tau2", "dr_lep2_tau1", "dr_lep2_tau2",
                "dr_leps", "dr_taus", #"avg_dr_jet",
                "met_LD", "met", "mht",
                "diHiggsVisMass", "diHiggsMass", "mTauTauVis", "mTauTau", #"ptTauTauVis",
                "nBJet_medium", #"nJet", "nBJet_loose",
                #"lep1_charge", "lep2_charge",
                "nElectron", "gen_mHH",
                "m_ll",
                "deltaEta_lep1_tau1", "deltaEta_lep1_tau2", "deltaEta_lep2_tau1", "deltaEta_lep2_tau2", "deltaEta_lep1_lep2", "deltaEta_tau1_tau2",
                "deltaPhi_lep1_tau1", "deltaPhi_lep1_tau2", "deltaPhi_lep2_tau1", "deltaPhi_lep2_tau2", "deltaPhi_lep1_lep2", "deltaPhi_tau1_tau2",
                "dr_lep1_tau1_tau2_min", "dr_lep1_tau1_tau2_max", "dr_lep2_tau1_tau2_min", "dr_lep2_tau1_tau2_max",
                "dr_lep_tau_min_OS", "dr_lep_tau_min_SS",
		"max_tau_eta", "max_lep_eta",
		"sum_lep_charge"
                ]

        if trainvar=="testVars"  and bdtType=="evtLevelSUM_HH_2l_2tau_res" and all==False :return [
                "diHiggsMass", "m_ll", "met_LD", "gen_mHH",
                "diHiggsVisMass", "mTauTau",
                "mTauTauVis", "tau1_eta", "dr_leps", "tau1_pt", "nElectron", "nBJet_medium",
                "dr_taus", "dr_lep_tau_min_SS", "dr_lep1_tau1_tau2_min", "tau2_pt", #"tau2_eta",
                "dr_lep_tau_min_OS", "lep1_conePt", #"lep2_eta", "lep1_eta",
                "dr_lep_tau_min_SS"
                ]

        if trainvar=="testVars2"  and bdtType=="evtLevelSUM_HH_2l_2tau_res" and all==False :return [ ## Use this for all Tau ID.s: Loose, VLoose, VVLoose 
                "diHiggsMass", "nBJet_medium", "tau1_pt", "dr_lep_tau_min_SS", "met_LD", "diHiggsVisMass", "m_ll", "tau2_pt", "dr_taus",
                "mTauTau", "gen_mHH", "mT_lep1", "mT_lep2", "deltaEta_lep1_tau2", "dr_lep1_tau1_tau2_min", "max_tau_eta", "max_lep_eta", "nElectron",
		"dr_lep_tau_min_OS", "mht", "met", "dr_lep1_tau1_tau2_max", "dr_leps", "tau1_eta", "deltaEta_lep1_tau1", "m_lep1_tau2"
		]

        if trainvar=="finalVars_allMasses"  and bdtType=="evtLevelSUM_HH_2l_2tau_res" and all==False :return [ ## Use this for VLoose Tau ID only for training with all masses
                "diHiggsMass", "diHiggsVisMass", "tau1_pt", "nBJet_medium", "gen_mHH", "nElectron", "dr_lep_tau_min_SS", "met_LD", "tau2_pt", "dr_lep_tau_min_OS"
		]

        if trainvar=="finalVars_LowMasses"  and bdtType=="evtLevelSUM_HH_2l_2tau_res" and all==False :return [ ## Use this for VLoose Tau ID only for training with Low masses (<= 400 GeV)
                "diHiggsVisMass", "nBJet_medium", "nElectron", "tau1_pt", "dr_lep_tau_min_SS", "mT_lep1", "tau2_pt", "met", "mT_lep2",  "diHiggsMass", "gen_mHH"
		]

        if trainvar=="finalVars_HighMasses"  and bdtType=="evtLevelSUM_HH_2l_2tau_res" and all==False :return [ ## Use this for VLoose Tau ID only for training with High masses (> 400 GeV)
		"diHiggsMass", "tau1_pt", "met_LD", "nBJet_medium", "dr_lep_tau_min_SS", "gen_mHH", "dr_lep1_tau1_tau2_min", "nElectron", "met", "dr_lep_tau_min_OS"
		]


