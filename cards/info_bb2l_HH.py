hasHTT = False

channelInTree='hh_bb2l_OS_Tight'
inputPath='/hdfs/local/snandan/hhAnalysis/2017/2018Nov26_forBDTtrainning/histograms/hh_bb2l/Tight_OS/'
#inputPath='/hdfs/local/snandan/hhAnalysis/2017/2018Oct17_forBDTtraining/histograms/hh_bb2l/Tight_OS/'

## to make balancing of BKGs
#fastsimTT=407.608+0.138336+96139.7 # taken from a first run on the training samples
#fastsimTT=407.608+0.138336+145865.0
fastsimTT = 225.449+0.10744+139094.0
#fastsimTTV=281.601+161.272
#fastsimDY=34061.5 # not using for this channel
fastsimDY = 15539.7
## missing diboson
TTdatacard=415075
DYdatacard=34363.3
TTVdatacard=322.353+ 281.601
## missing diboson

def trainVars(all):
	if all==True :return [ ## add all variavles to be read from the tree
		"lep1_pt", "lep1_conePt", "lep1_eta",
      "lep2_pt", "lep2_conePt", "lep2_eta",
      "bjet1_pt", "bjet1_eta",
      "bjet2_pt", "bjet2_eta",
      "met", "mht", "met_LD",
      "HT", "STMET",
      "m_Hbb", "dR_Hbb", "dPhi_Hbb", "pT_Hbb",
      "m_ll", "dR_ll", "dPhi_ll", "pT_ll",
      "min_dPhi_lepMEt", "max_dPhi_lepMEt",
      "m_Hww", "pT_Hww", "Smin_Hww",
      "dR_b1lep1", "dR_b1lep2", "dR_b2lep1", "dR_b2lep2",
      "m_HHvis", "pT_HHvis", "dPhi_HHvis",
      "m_HH", "pT_HH", "dPhi_HH", "Smin_HH",
      "mT2_W", "mT2_top_2particle", "mT2_top_3particle",
      "m_HH_hme",
      "logTopness_publishedChi2", "logHiggsness_publishedChi2", "logTopness_fixedChi2", "logHiggsness_fixedChi2",
      "vbf_m_jj", "vbf_dEta_jj",
       "genWeight", "evtWeight",
       "nJet", "nBJetLoose", "nBJetMedium",
      "isHbb_boosted",
      "nJet_vbf", "isVBF", "event"
		]

	if trainvar=="noTopness"  and bdtType=="evtLevelSUM_HH_bb2l_res" and all==False :return [


			   "mht",
      "HT",
      "m_Hbb",# "dR_Hbb",
      "m_ll", #"dR_ll",
      #"max_dPhi_lepMEt",
      "Smin_Hww",
      #"dPhi_HHvis",
      #"pT_HH",#'max_lep_pt','max_bjet_pt',#"max_dR_b_lep",
      #"mT2_top_2particle",
		#	   "logTopness_fixedChi2", "logHiggsness_fixedChi2",
       "nJet", "nBJetLoose",
			   "gen_mHH"]

	'''		'm_ll',#'dR_ll',
		'm_Hbb',#'pT_Hbb',
		'nBJetMedium',
		'm_Hww',
		'logTopness_fixedChi2','logHiggsness_fixedChi2',
		'mT2_top_3particle',
		'pT_HH',
		'dPhi_HH',
		'min_dPhi_lepMEt',
		'max_dR_b_lep',
		'met',
                'max_lep_pt','max_bjet_pt','gen_mHH'
		]'''

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

'''             "lep1_pt", "lep1_conePt", "lep1_eta",
      "lep2_pt", "lep2_conePt", "lep2_eta",
      "bjet1_pt", "bjet1_eta",
      "bjet2_pt", "bjet2_eta",
      "met", "mht", "met_LD",
      "HT", "STMET",
      "m_Hbb", "dR_Hbb", "dPhi_Hbb", "pT_Hbb",
      "m_ll", "dR_ll", "dPhi_ll", "pT_ll",
      "min_dPhi_lepMEt", "max_dPhi_lepMEt",
      "m_Hww", "pT_Hww", "Smin_Hww",
      "dR_b1lep1", "dR_b1lep2", "dR_b2lep1", "dR_b2lep2",
      "m_HHvis", "pT_HHvis", "dPhi_HHvis",
      "m_HH", "pT_HH", "dPhi_HH", "Smin_HH",
      "mT2_W", "mT2_top_2particle", "mT2_top_3particle",
      "m_HH_hme",
      "logTopness_publishedChi2", "logHiggsness_publishedChi2", "logTopness_fixedChi2", "logHiggsness_fixedChi2",
        "nJet", "nBJetLoose", "nBJetMedium"'''
