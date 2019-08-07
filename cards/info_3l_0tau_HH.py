


def read_from(Bkg_mass_rand, tauID=""):
        mass_rand_algo="default"
        if(Bkg_mass_rand == "oversampling"):
                mass_rand_algo= "oversampling"
        else:
                mass_rand_algo= "default"


        #channelInTree='hh_3l_OS_forBDTtraining'
        #inputPath='/hdfs/local/ssawant/hhAnalysis/2017/20181108/histograms/hh_3l/forBDTtraining_OS/'

        '''
        v201811xx: vDAE woAK8
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
        '''

        #v20190508: vDAE woAK8, BDT training includeing parametrize gen_mHH
        # to make balancing of BKGs
        '''
        fastsimTT   = 61.3864 + 0.0189422 + 0.  # taken from a first run on the training samples 
        fastsimTTV  = 9.4032+7.75227
        fastsimDY   = 36.745 
        fastsimWZ   = 227.257
        '''

        '''
        # hhAnalysis/2017/20181127 :  hh_3l_OS_Tight/sel/evt
        TTdatacard  = 33.7377 
        DYdatacard  = 42.1208         
        WWdatacard  = 15.8726
        WZdatacard  = 265.414
        ZZdatacard  = 19.9733
        TTWdatacard = 9.0017 # + 0.31488 # TTW + TTWW; no TTWW BDT file present currently 20190508
        TTZdatacard = 7.71022
        TTHdatacard = 4.34615
        THdatacard  = 9.26705
        VHdatacard  = 8.09768

        signal_ggf_spin0_400_hh_2v2tdatacard = 2.40653
        signal_ggf_spin0_400_hh_4tdatacard   = 0.18858
        signal_ggf_spin0_400_hh_4vdatacard   = 7.28739

        signal_ggf_spin0_700_hh_2v2tdatacard = 5.15843
        signal_ggf_spin0_700_hh_4tdatacard   = 0.477015
        signal_ggf_spin0_700_hh_4vdatacard   = 13.5689
        '''

        '''
        # vDAE_woAK8
        channelInTree='hh_3l_OS_forBDTtraining'
        inputPath='/hdfs/local/ssawant/hhAnalysis/2017/20181108/histograms/hh_3l/forBDTtraining_OS/'

        # /hdfs/local/ssawant/hhAnalysis/2017/20190615_wAK8woLS/histograms/hh_3l/Tight_OS/hadd/hadd_stage2_Tight_OS.root : hh_3l_OS_Tight/sel/evt
        TTdatacard  = 10.3342
        DYdatacard  = 12.1582
        WWdatacard  = 8.97866
        WZdatacard  = 111.955
        ZZdatacard  = 5.20875
        TTWdatacard = 4.89453 # + 0.211243 # TTW + TTWW; no TTWW BDT file present currently 20190508
        TTZdatacard = 4.5804
        TTHdatacard = 2.5738
        THdatacard  = 5.38349
        VHdatacard  = 2.12483

        signal_ggf_spin0_400_hh_2v2tdatacard = 0.72928
        signal_ggf_spin0_400_hh_4tdatacard   = 0.0617587
        signal_ggf_spin0_400_hh_4vdatacard   = 2.64265

        signal_ggf_spin0_700_hh_2v2tdatacard = 4.06975
        signal_ggf_spin0_700_hh_4tdatacard   = 0.313392
        signal_ggf_spin0_700_hh_4vdatacard   = 12.0124
        '''


        # 20190712_wAK8woLS
        channelInTree='hh_3l_OS_forBDTtraining'
        inputPath='/hdfs/local/ssawant/hhAnalysis/2017/20190712_wAK8woLS_forBDTtraining/histograms/hh_3l/forBDTtraining_OS'

        # /hdfs/local/ssawant/hhAnalysis/2017/20190712_wAK8woLS/histograms/hh_3l/Tight_OS/hadd/hadd_stage2_Tight_OS.root : hh_3l_OS_Tight/sel/evt
        TTdatacard  = 33.843
        DYdatacard  = 42.5519
        WWdatacard  = 15.8726
        WZdatacard  = 265.697
        ZZdatacard  = 20.0088
        TTWdatacard = 9.00436 # + 0.31488   # TTW + TTWW; no TTWW BDT file present currently 20190508
        TTZdatacard = 7.71189
        TTHdatacard = 4.35592
        THdatacard  = 9.26705
        VHdatacard  = 8.23902

        signal_ggf_spin0_400_hh_2v2tdatacard = 2.40653
        signal_ggf_spin0_400_hh_4tdatacard   = 0.18858
        signal_ggf_spin0_400_hh_4vdatacard   = 7.29628

        signal_ggf_spin0_700_hh_2v2tdatacard = 5.16843
        signal_ggf_spin0_700_hh_4tdatacard   = 0.477723
        signal_ggf_spin0_700_hh_4vdatacard   = 13.5731

        signal_ggf_spin0_250_hh_4tdatacard  = 0.0785402
        signal_ggf_spin0_260_hh_4tdatacard  = 0.0869949
        signal_ggf_spin0_270_hh_4tdatacard  = 0.0850966
        signal_ggf_spin0_280_hh_4tdatacard  = 0.095667
        signal_ggf_spin0_300_hh_4tdatacard  = 0.113509
        signal_ggf_spin0_350_hh_4tdatacard  = 0.151533
        signal_ggf_spin0_400_hh_4tdatacard  = 0.18858
        signal_ggf_spin0_450_hh_4tdatacard  = 0.222347
        signal_ggf_spin0_500_hh_4tdatacard  = 0.270874
        signal_ggf_spin0_550_hh_4tdatacard  = 0.34707
        signal_ggf_spin0_600_hh_4tdatacard  = 0.403076
        signal_ggf_spin0_650_hh_4tdatacard  = 0.430994
        signal_ggf_spin0_700_hh_4tdatacard  = 0.477723
        signal_ggf_spin0_750_hh_4tdatacard  = 0.519853
        signal_ggf_spin0_800_hh_4tdatacard  = 0.593098
        signal_ggf_spin0_850_hh_4tdatacard  = 0.621889
        signal_ggf_spin0_900_hh_4tdatacard  = 0.678992
        signal_ggf_spin0_1000_hh_4tdatacard  = 0.673031




        '''
        keys=[
                'TTTo2L2Nu','TTToSemiLeptonic', #'TTToHadronic', ## It does have a * on the loading, it will load the PSWeights
                'DY', ## It does have a * on the loading, it will load all DY
                'WZ', ## WZ inclusive sample + WZZ
                #'ZZ', ## ZZ inclusive samples + ZZZ
                #'WW',  ## WW inclusive samples -1
                'TTWJets', # TTW + TTWW; no TTWW BDT file present currently 20190508
                'TTZJets',
                'signal_ggf_spin0_400_hh_2v2t', 'signal_ggf_spin0_400_hh_4t',  'signal_ggf_spin0_400_hh_4v',
                'signal_ggf_spin0_700_hh_2v2t', 'signal_ggf_spin0_700_hh_4t',  'signal_ggf_spin0_700_hh_4v'                               
                ## missing: diboson/ singleH
        ] 
        '''
        keys=[
                'TTTo2L2Nu','TTToSemiLeptonic', #'TTToHadronic', ## It does have a * on the loading, it will load the PSWeights
                'DY', ## It does have a * on the loading, it will load all DY
                'WZ', ## WZ inclusive sample + WZZ
                'ZZ', ## ZZ inclusive samples + ZZZ
                'WW',  ## WW inclusive samples -1
                'TTWJets', # TTW + TTWW; no TTWW BDT file present currently 20190508
                'TTZJets',
                #'signal_ggf_spin0_400_hh_2v2t', 'signal_ggf_spin0_400_hh_4t',  'signal_ggf_spin0_400_hh_4v',
                #'signal_ggf_spin0_700_hh_2v2t', 'signal_ggf_spin0_700_hh_4t',  'signal_ggf_spin0_700_hh_4v'
                #
                'signal_ggf_spin0_250_hh_4t',
                'signal_ggf_spin0_260_hh_4t',
                'signal_ggf_spin0_270_hh_4t',
                'signal_ggf_spin0_280_hh_4t',
                'signal_ggf_spin0_300_hh_4t',
                'signal_ggf_spin0_350_hh_4t',
                'signal_ggf_spin0_400_hh_4t',
                'signal_ggf_spin0_450_hh_4t',
                'signal_ggf_spin0_500_hh_4t',
                'signal_ggf_spin0_550_hh_4t',
                'signal_ggf_spin0_600_hh_4t',
                'signal_ggf_spin0_650_hh_4t',
                'signal_ggf_spin0_700_hh_4t',
                'signal_ggf_spin0_750_hh_4t',
                'signal_ggf_spin0_800_hh_4t',
                'signal_ggf_spin0_850_hh_4t',
                'signal_ggf_spin0_900_hh_4t',
                'signal_ggf_spin0_1000_hh_4t',
                ## missing: diboson/ singleH
        ]

        #masses = [400, 700]
        #masses_test = [400, 700]
        #masses      = [250, 260, 270, 280, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 1000]
        #masses_test = [250, 260, 270, 280, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 1000]
        masses      = [250, 260, 270, 280, 300, 350, 400, 450,     550, 600, 650, 700, 750, 800, 850, 900, 1000]
        masses_test = [500]


        
        output = {
                "channelInTree" : channelInTree,
                "inputPath" : inputPath,
                "TTdatacard"  : TTdatacard,
                "DYdatacard"  : DYdatacard,
                "WWdatacard"  : WWdatacard,
                "WZdatacard"  : WZdatacard,
                "ZZdatacard"  : ZZdatacard,
                "TTWdatacard" : TTWdatacard,
                "TTZdatacard" : TTZdatacard,
                "TTHdatacard" : TTHdatacard,
                "THdatacard"  : THdatacard,
                "VHdatacard"  : VHdatacard,
                "keys" : keys,
                "masses" : masses,
                "masses_test": masses_test,
                "mass_randomization" : mass_rand_algo,
                }

        # Add nEvetns in datacards for signal also in 'output' dictionary
        for key in keys:
            if not 'signal' in key: continue
            sDatacard = key+'datacard'
            #print('key: {}, datacard: {}'.format(key,sDatacard))
            exec("output.update({\"%s\" : %s})" % (sDatacard,sDatacard)) # e.g. output.update({"signal_ggf_spin0_400_hh_4vdatacard" : signal_ggf_spin0_400_hh_4vdatacard })


        return output



        
def trainVars(all, trainvar = None, bdtType="evtLevelSUM_HH_3l_0tau_res"):
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
		"sumLeptonCharge", "numSameFlavor_OS", "isVBF",
                "event",
                # Boosted W reconstruction w/ AK8        
                "isWjjBoosted","jet1_pt","jet2_pt","jet1_m","jet2_m",
                "dr_j1j2",        
                "dr_WjetsLep3","dr_Wjet1Lep3","dr_Wjet2Lep3",
                #
                "gen_mHH"        
		]

	#if trainvar=="noTopness"  and bdtType=="evtLevelSUM_HH_res" and all==False :return [
	if (trainvar=="noTopness"  and bdtType=="evtLevelSUM_HH_3l_0tau_res" and all==False) or \
	(trainvar=="noTopness"  and bdtType=="evtLevelWZ_HH_3l_0tau_res" and all==False) or \
	(trainvar=="noTopness"  and bdtType=="evtLevelDY_HH_3l_0tau_res" and all==False) or \
	(trainvar=="noTopness"  and bdtType=="evtLevelTT_HH_3l_0tau_res" and all==False) :
                '''
                # 24 variables
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
		"m_jj", #"diHiggsMass", #"diHiggsVisMass",
                #"mTMetLepton1", "mTMetLepton2",
                #"vbf_m_jj", "vbf_dEta_jj", "numSelJets_nonVBF",
                #
                "nJet", #"nBJetLoose", "nBJetMedium", 
		"nElectron", #"nMuon",
                #"lep1_isTight", "lep2_isTight", "lep3_isTight",
                "sumLeptonCharge", "numSameFlavor_OS", #"isVBF"
                "gen_mHH"          
		]
                ''' 

                # 21 variables 
		return [
		"lep1_conePt", "lep1_eta", #"lep1_tth_mva", 
		"mindr_lep1_jet", "mT_lep1",
                "lep2_conePt", "lep2_eta", #"lep2_tth_mva", 
		"mindr_lep2_jet", "mT_lep2",
                "lep3_conePt", "lep3_eta", #"lep3_tth_mva", 
		"mindr_lep3_jet", "mT_lep3",
                #"avg_dr_jet", #"ptmiss",  "htmiss", 
		#"dr_leps",
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
		"m_jj", #"diHiggsMass", #"diHiggsVisMass",
                #"mTMetLepton1", "mTMetLepton2",
                #"vbf_m_jj", "vbf_dEta_jj", "numSelJets_nonVBF",
                #  
                #"nJet", #"nBJetLoose", "nBJetMedium", 
		"nElectron", #"nMuon",
                #"lep1_isTight", "lep2_isTight", "lep3_isTight",
                "sumLeptonCharge", "numSameFlavor_OS", #"isVBF"
                # AK8 variables
                #"isWjjBoosted",
                "jet1_pt","jet2_pt",#"jet1_m","jet2_m",
                #"dr_j1j2",    
                #"dr_WjetsLep3","dr_Wjet1Lep3",
                "dr_Wjet2Lep3",
                #        
                "gen_mHH"          
		]

                '''
                # take top 10 variables out
                return [
                #"gen_mHH",           
                "dr_lss", "dr_los1", "dr_los2",
                "mindr_lep1_jet", "mindr_lep2_jet",
                "lep1_conePt", "lep2_conePt",
                "mT_lep1", "mT_lep2",
                "met_LD", #10
                "m_jj" # 11+1        
                #"dr_leps",
                #"mT_lep3",
                #"lep1_conePt", "lep2_conePt", "lep2_conePt"
                ]
                '''

                

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

