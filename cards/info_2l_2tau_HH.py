## --- Choose Tau ID ----
#tauID = "dR03mvaLoose"
#tauID = "dR03mvaVLoose"
#tauID = "dR03mvaVVLoose"
## ---------------------

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
	TTVdatacard = 1.0
	VVdatacard = 1.0
	VHdatacard = 1.0
	TTHdatacard = 1.0
	if tauID_application == "dR03mvaLoose" :
		TTdatacard=43.65
		DYdatacard=31.11
		TTVdatacard=0.38+0.14+0.01 ## =TTZ+TTW+TTWW
		VVdatacard=0.30+1.94+6.05  ## = WW+WZ+ZZ
		VHdatacard=1.23
		TTHdatacard=0.47
	elif tauID_application == "dR03mvaMedium" :
		TTdatacard=15.94
		DYdatacard=15.73
		TTVdatacard=0.27+0.05+0.0 ## =TTZ+TTW+TTWW
		VVdatacard=0.23+1.08+4.78 ## = WW+WZ+ZZ
		VHdatacard=1.19
		TTHdatacard=0.28
	else : print ("Warning: you did not defined datacard normalization")

	keys=[
	##Ram: as we discussed before think about removing spurious BKGs from training
	## as eg 'ttHToNonbb_M125_powheg_ext1', VHToNonbb_M125_v14-v2...
	## --> they serve only to make the training stable
                'TTTo2L2Nu', 'TTToSemiLeptonic', 'TTToHadronic', ## TTBar inclusive samples
		'TTTo2L2Nu_PSweights', 'TTToSemiLeptonic_PSweights', 'TTToHadronic_PSweights', ## TTBar PSWeights samples
                'DYJetsToLL_M-50_LO', 'DY1JetsToLL_M-50', 'DY2JetsToLL_M-50', 'DY3JetsToLL_M-50', 'DY4JetsToLL_M-50', ## DYJets inclusive samples  
		'DYJetsToLL_M-50_LO_ext1', 'DY1JetsToLL_M-50_ext1', 'DY2JetsToLL_M-50_ext1', 'DY3JetsToLL_M-50_ext1', ## DYJets ext samples
                'DYJetsToLL_M-4to50_HT-100to200', 'DYJetsToLL_M-4to50_HT-200to400', 'DYJetsToLL_M-4to50_HT-400to600', 'DYJetsToLL_M-4to50_HT-600toInf', ## DYJets M-4to50 (HT binned) inclusive samples 
		'DYJetsToLL_M-4to50_HT-100to200_ext1', 'DYJetsToLL_M-4to50_HT-200to400_ext1', 'DYJetsToLL_M-4to50_HT-400to600_ext1',  ## DYJets M-4to50 (HT binned) ext samples
                'DYJetsToLL_M50_HT100to200', 'DYJetsToLL_M50_HT200to400', 'DYJetsToLL_M50_HT400to600',  'DYJetsToLL_M50_HT600to800',  ## DYJets M50 (HT binned) inclusive samples -1 
		'DYJetsToLL_M50_HT800to1200', 'DYJetsToLL_M50_HT1200to2500', 'DYJetsToLL_M50_HT2500toInf',                            ## DYJets M50 (HT binned) inclusive samples -2
		'DYJetsToLL_M50_HT100to200_ext1', 'DYJetsToLL_M50_HT200to400_ext1', 'DYJetsToLL_M50_HT400to600_ext1',                 ## DYJets M50 (HT binned) ext samples 
                'ZZTo4L', 'ZZTo2L2Q', 'ZZTo2L2Nu', ## ZZ inclusive samples
		'ZZTo4L_ext1',                     ## ZZ ext sample
                'WWTo2L2Nu', 'WWToLNuQQ', 'WWTo1L1Nu2Q', 'WWTo4Q', ## WW inclusive samples -1
		'WWTo2L2Nu_DoubleScattering', 'WpWpJJ_EWK_QCD', 'WpWpJJ_EWK_QCD_v14-v1', ## WW inclusive samples -2
		'WWTo2L2Nu_PSweights', 'WWToLNuQQ_PSweights', 'WWToLNuQQ_ext1', 'WWTo4Q_PSweights', ## WW PSWeights and ext samples
                'WZTo3LNu', ## WZ inclusive sample
		'WZTo3LNu_1Jets_MLL-50', 'WZTo3LNu_2Jets_MLL-50', 'WZTo3LNu_3Jets_MLL-50', 'WZTo3LNu_0Jets_MLL-50', ## WZ (MLL-50) Jet binned samples
		'WZTo3LNu_1Jets_MLL-4to50', 'WZTo3LNu_2Jets_MLL-4to50', 'WZTo3LNu_3Jets_MLL-4to50', 'WZTo3LNu_0Jets_MLL-4to50', ## WZ (MLL-4to50) Jet binned samples
                'ZZZ', 'WWW_4F', 'WWZ_4F', 'WZZ', ## Tri-boson inclusive samples 
                'TTZJets', 'TTWJets', ## TTV inclusive samples
		'TTZJets_LO_ext1', 'TTWJets_LO_ext1', ## TTV ext samples
                'TTWW', ## TTWW inclusive sample
                'VHToNonbb_M125', 'VHToNonbb_M125_v14-v2', ## VH inclusive samples
                'ttHToNonbb', 'ttHJetToNonbb_M125_amcatnlo', ## TTH inclusive samples
		'ttHToNonbb_M125_powheg_ext1',               ## TTH ext samples
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

	output = {
		"channelInTree" : channelInTree,
		"inputPath" : inputPath,
		"TTdatacard" : TTdatacard,
		"DYdatacard" : DYdatacard,
		"TTVdatacard" : TTVdatacard,
		"VVdatacard" : VVdatacard,
		"VHdatacard" : VHdatacard,
		"TTHdatacard" : TTHdatacard,
		"keys" : keys,
		"masses" : masses,
		"masses_test": masses_test,
		"mass_randomization" : mass_rand_algo,                                                                                                                                                                                                              
		}

	return output

def trainVars(all):
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

        if trainvar=="testVars2"  and bdtType=="evtLevelSUM_HH_2l_2tau_res" and all==False :return [
                "diHiggsMass", "nBJet_medium", "tau1_pt", "dr_lep_tau_min_SS", "met_LD", "diHiggsVisMass", "m_ll", "tau2_pt", "dr_taus", 
                "mTauTau", "gen_mHH", "mT_lep1", "mT_lep2", "deltaEta_lep1_tau2", "dr_lep1_tau1_tau2_min", "max_tau_eta", "max_lep_eta", "nElectron",
		"dr_lep_tau_min_OS", "mht", "met", "dr_lep1_tau1_tau2_max"
		#"lep2_eta", "dr_leps",
                # "diHiggsMass", "m_ll", "met_LD", "gen_mHH", "tau1_pt", "dr_lep_tau_min_SS", "dr_leps", "dr_taus", "lep1_conePt", "mTauTau"
                #"diHiggsVisMass",
                #"mTauTauVis", "tau1_eta", "dr_leps", "tau1_pt", "nElectron", "nBJet_medium",
                #"tau2_pt", "tau2_eta", "dr_taus", "dr_lep_tau_min_SS", "dr_lep1_tau1_tau2_min",
                #"dr_lep_tau_min_OS", "lep1_conePt", "lep2_eta", "lep1_eta",
                #"dr_lep_tau_min_SS"
                ]
