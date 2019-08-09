def num_to_str(num):
    temp_str = str(num)
    final_str = temp_str.replace('.', 'o')
    return final_str

def numpyarrayTProfileFill(data_X, data_Y, data_wt, hprof):
    for x,y,w in np.nditer([data_X, data_Y, data_wt]):
        #print("x: {}, y: {}, w: {}".format(x, y, w))                                                                                                                                                                                   
        hprof.Fill(x,y,w)


def numpyarrayHisto1DFill(data_X, data_wt, histo1D):
    for x,w in zip(data_X, data_wt):
        #print("x: {},  w: {}".format(x, w))                                                                                                                                                                                             
        histo1D.Fill(x,w)


def AddHistToStack(data_copy, var_name,  hstack, nbins, X_min, X_max, FillColor, processName):
    histo1D = TH1D( 'histo1D', processName, nbins, X_min, X_max)
    data_X_array  = np.array(data_copy[var_name].values, dtype=np.float)
    data_wt_array = np.array(data_copy['totalWeight'].values, dtype=np.float)
    numpyarrayHisto1DFill(data_X_array, data_wt_array, histo1D)
    histo1D.SetFillColor(FillColor)
    #histo1D.Draw()                                                                                                                                                                                                                             
    hstack.Add(histo1D)

def BuildTHstack_2l_2tau(hstack, data_copy, var_name, nbins, X_min, X_max):
    ttbar_samples = ['TTTo2L2Nu','TTToSemiLeptonic']
    vv_samples = ['ZZ', 'WZ', 'WW']
    ttv_samples = ['TTZJets', 'TTWJets']
    data_copy_TT  =  data_copy.loc[(data_copy['key'].isin(ttbar_samples))] ## TTbar                                                                                                                                                              
    data_copy_DY  =  data_copy.loc[(data_copy['key']=='DY')] ## DY                                                                                                                                                                              
    data_copy_VV  =  data_copy.loc[(data_copy['key'].isin(vv_samples))] ## VV                                                                                                                                                                   
    data_copy_TTV =  data_copy.loc[(data_copy['key'].isin(ttv_samples))] ## TTV                                                                                                                                                                  
    data_copy_TTH =  data_copy.loc[(data_copy['key']=='TTH')] ## TTH                                                                                                                                                                             
    data_copy_VH  =  data_copy.loc[(data_copy['key']=='VH')] ## VH                                                                                                                                                                                
 
    if(data_copy_TTH.empty != True): AddHistToStack(data_copy_TTH, var_name, hstack, nbins, X_min, X_max, 5, 'TTH') ## Yellow                                                                                                                   
    if(data_copy_TTV.empty != True): AddHistToStack(data_copy_TTV, var_name, hstack, nbins, X_min, X_max, 1, 'TTV')  ## Black                                                                                                                   
    if(data_copy_VH.empty != True): AddHistToStack(data_copy_VH, var_name, hstack, nbins, X_min, X_max, 6, 'VH')  ## Magenta                                                                                                                      
    if(data_copy_VV.empty != True): AddHistToStack(data_copy_VV, var_name, hstack, nbins, X_min, X_max, 3, 'VV')    ## Green                                                                                                                     
    if(data_copy_DY.empty != True): AddHistToStack(data_copy_DY, var_name, hstack, nbins, X_min, X_max, 2, 'DY')      ## Red                                                                                                                      
    if(data_copy_TT.empty != True): AddHistToStack(data_copy_TT, var_name, hstack, nbins, X_min, X_max, 4, 'TTbar')  ##



def MakeTHStack_New(channel, data, var_name_list, label):
    data_copy = data.copy(deep=True) ## Making a deep copy of data ( => separate data and index from data)                                                                                                                                          
    data_copy =  data_copy.loc[(data_copy['target']==0)] ## Only take backgrounds                                                                                                                                                               

    Histo_Dict = {
        "gen_mHH": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "diHiggsMass": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "diHiggsVisMass": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "met": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "mht": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "met_LD": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "tau1_pt": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "tau2_pt": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "mT_lep1": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "mT_lep2": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "m_ll": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "mTauTau": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "m_lep1_tau2": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "dr_lep_tau_min_SS": {'X_min': 0., 'X_max': 1.0, 'nbins': 10},
        "dr_lep_tau_min_OS": {'X_min': 0., 'X_max': 1.0, 'nbins': 10},
        "dr_taus": {'X_min': 0., 'X_max': 1.0, 'nbins': 10},
        "dr_lep1_tau1_tau2_min": {'X_min': 0., 'X_max': 1.0, 'nbins': 10},
        "dr_lep1_tau1_tau2_max": {'X_min': 0., 'X_max': 1.0, 'nbins': 10},
        "max_tau_eta": {'X_min': 0., 'X_max': 3.0, 'nbins': 30},
        "max_lep_eta": {'X_min': 0., 'X_max': 3.0, 'nbins': 30},
        "nElectron": {'X_min': 0., 'X_max': 3.0, 'nbins': 3},
        "nBJet_medium": {'X_min': 0., 'X_max': 3.0, 'nbins': 3},
        "dr_leps": {'X_min': 0., 'X_max': 1.0, 'nbins': 10},
        "tau1_eta": {'X_min': -3.0, 'X_max': 3.0, 'nbins': 60},
        "deltaEta_lep1_tau1": {'X_min': -5.0, 'X_max': 5.0, 'nbins': 100},
        "deltaEta_lep1_tau2": {'X_min': -5.0, 'X_max': 5.0, 'nbins': 100},
        }


    for var_name in var_name_list:
        data_X  = np.array(data_copy[var_name].values, dtype=np.float)
        data_wt = np.array(data_copy['totalWeight'].values, dtype=np.float)

        N_x  = len(data_X)
        N_wt = len(data_wt)

        if(N_x == N_wt):
            # Create a new canvas, and customize it.                                                                                                                                                                                            
            c1 = TCanvas( 'c1', 'Dynamic Filling Example', 200, 10, 700, 500)
            #c1.SetFillColor(42)                                                                                                                                                                                                                  
            #c1.GetFrame().SetFillColor(21)                                                                                                                                                                                                      
            c1.GetFrame().SetBorderSize(6)
            c1.GetFrame().SetBorderMode(-1)

            #print("N_x: {}, N_wt: {}".format(N_x, N_wt))                                                                                                                                                                                            
            print("Variable Name: {}".format(var_name))
            PlotTitle = var_name
            hstack  = THStack('hstack', PlotTitle)
            BuildTHstack_2l_2tau(hstack, data_copy, var_name, Histo_Dict[var_name]['nbins'], Histo_Dict[var_name]['X_min'], Histo_Dict[var_name]['X_max'])
            hstack.Draw("hist")
            hstack.GetYaxis().SetTitle("Events")
            hstack.GetXaxis().SetTitle(var_name)
            c1.Modified()
            c1.Update()
            gPad.BuildLegend(0.75,0.75,0.95,0.95,"")
            FileName = "{}/{}_{}_{}.root".format(channel, "THStack", var_name, label)
            c1.SaveAs(FileName)
        else:
            print('Arrays not of same length')
            print("N_x: {}, N_wt: {}".format(N_x, N_wt))



def MakeHisto1D_New(channel, data, var_name_list, label):
    Histo_Dict = {
        "gen_mHH": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "diHiggsMass": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "diHiggsVisMass": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "met": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "mht": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "met_LD": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "tau1_pt": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "tau2_pt": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "mT_lep1": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "mT_lep2": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "m_ll": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "mTauTau": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "m_lep1_tau2": {'X_min': 0., 'X_max': 1100., 'nbins': 55},
        "dr_lep_tau_min_SS": {'X_min': 0., 'X_max': 1.0, 'nbins': 10},
        "dr_lep_tau_min_OS": {'X_min': 0., 'X_max': 1.0, 'nbins': 10},
        "dr_taus": {'X_min': 0., 'X_max': 1.0, 'nbins': 10},
        "dr_lep1_tau1_tau2_min": {'X_min': 0., 'X_max': 1.0, 'nbins': 10},
        "dr_lep1_tau1_tau2_max": {'X_min': 0., 'X_max': 1.0, 'nbins': 10},
        "max_tau_eta": {'X_min': 0., 'X_max': 3.0, 'nbins': 30},
        "max_lep_eta": {'X_min': 0., 'X_max': 3.0, 'nbins': 30},
        "nElectron": {'X_min': 0., 'X_max': 3.0, 'nbins': 3},
        "nBJet_medium": {'X_min': 0., 'X_max': 3.0, 'nbins': 3},
        "dr_leps": {'X_min': 0., 'X_max': 1.0, 'nbins': 10},
        "tau1_eta": {'X_min': -3.0, 'X_max': 3.0, 'nbins': 60},
        "deltaEta_lep1_tau1": {'X_min': -5.0, 'X_max': 5.0, 'nbins': 100},
        "deltaEta_lep1_tau2": {'X_min': -5.0, 'X_max': 5.0, 'nbins': 100},
        }

    data_copy = data.copy(deep=True) ## Making a deep copy of data ( => separate data and index from data)                                                                                                                                     
    data_copy =  data_copy.loc[(data_copy['target']==0)] ## Only take backgrounds                                                                                                                                                                   
                      

    for var_name in var_name_list:
        print("Variable Name: {}".format(var_name))
        data_X  = np.array(data_copy[var_name].values, dtype=np.float)
        data_wt = np.array(data_copy['totalWeight'].values, dtype=np.float)

        N_x  = len(data_X)
        N_wt = len(data_wt)

        # Create a new canvas, and customize it.                                                                                                                                                                                        
        c1 = TCanvas( 'c1', "Canvas", 200, 10, 700, 500)
        #c1.SetFillColor(42)                                                                                                                                                                                                                     
        #c1.GetFrame().SetFillColor(21)                                                                                                                                                                                                          
        c1.GetFrame().SetBorderSize(6)
        c1.GetFrame().SetBorderMode(-1)

        if(N_x == N_wt):
            #print("N_x: {}, N_wt: {}".format(N_x, N_wt))                                                                                                                                                                                          
            PlotTitle = var_name
            histo1D  = TH1D( 'histo1D', PlotTitle, Histo_Dict[var_name]['nbins'], Histo_Dict[var_name]['X_min'], Histo_Dict[var_name]['X_max'])
            histo1D.GetYaxis().SetTitle("Events")
            histo1D.GetXaxis().SetTitle(var_name)
            numpyarrayHisto1DFill(data_X, data_wt, histo1D)
            histo1D.Draw()
            c1.Modified()
            c1.Update()
            FileName = "{}/{}_{}_{}.root".format(channel, "Histo1D", var_name, label)
            c1.SaveAs(FileName)
        else:
            print('Arrays not of same length')
            print("N_x: {}, N_wt: {}".format(N_x, N_wt))


def MakeTProfile_New(channel, data, var_name_list, Target, doFit, label, TrainMode, mass_list):
    data_copy = data.copy(deep=True) ## Making a deep copy of data ( => separate data and index from data)                                                                                                                                       
    data_copy =  data_copy.loc[(data_copy['target']==Target)] ## Only take 1 for signal                                                                                                                                                      

    # Create a new canvas, and customize it.                                                                                                                                                                                                     
    c1 = TCanvas( 'c1', 'Dynamic Filling Example', 200, 10, 700, 500)
    #c1.SetFillColor(42)                                                                                                                                                                                                                            
    #c1.GetFrame().SetFillColor(21)                                                                                                                                                                                                                 
    c1.GetFrame().SetBorderSize(6)
    c1.GetFrame().SetBorderMode(-1)

    
    Func_Dict = {
       "diHiggsMass": {'y_min': 0., 'y_max': 1100., 'fitFunc_AllMassTraining': "pol6", 'fitFunc_LowMassTraining': "pol1", 'fitFunc_HighMassTraining': "pol6"},
       "diHiggsVisMass": {'y_min': 0., 'y_max': 1100., 'fitFunc_AllMassTraining': "pol1", 'fitFunc_LowMassTraining': "pol4", 'fitFunc_HighMassTraining': "pol4"},
       "met": {'y_min': 0., 'y_max': 1100., 'fitFunc_AllMassTraining': "pol1", 'fitFunc_LowMassTraining': "pol4", 'fitFunc_HighMassTraining': "pol1"},
       "mht": {'y_min': 0., 'y_max': 1100., 'fitFunc_AllMassTraining': "pol3", 'fitFunc_LowMassTraining': "pol4", 'fitFunc_HighMassTraining': "pol1"},
       "met_LD": {'y_min': 0., 'y_max': 1100., 'fitFunc_AllMassTraining': "pol1", 'fitFunc_LowMassTraining': "pol1", 'fitFunc_HighMassTraining': "pol1"},
       "tau1_pt": {'y_min': 0., 'y_max': 1100., 'fitFunc_AllMassTraining': "pol1", 'fitFunc_LowMassTraining': "pol1", 'fitFunc_HighMassTraining': "pol1"},
       "tau2_pt": {'y_min': 0., 'y_max': 1100., 'fitFunc_AllMassTraining': "pol4", 'fitFunc_LowMassTraining': "pol4", 'fitFunc_HighMassTraining': "pol3"},
       "mT_lep1": {'y_min': 0., 'y_max': 1100., 'fitFunc_AllMassTraining': "pol3", 'fitFunc_LowMassTraining': "pol4", 'fitFunc_HighMassTraining': "pol1"},
       "mT_lep2": {'y_min': 0., 'y_max': 1100., 'fitFunc_AllMassTraining': "pol1", 'fitFunc_LowMassTraining': "pol1", 'fitFunc_HighMassTraining': "pol1"},
       "m_ll": {'y_min': 0., 'y_max': 1100., 'fitFunc_AllMassTraining': "pol1", 'fitFunc_LowMassTraining': "pol1", 'fitFunc_HighMassTraining': "pol1"},
       "mTauTau": {'y_min': 0., 'y_max': 1100., 'fitFunc_AllMassTraining': "pol1", 'fitFunc_LowMassTraining': "pol4", 'fitFunc_HighMassTraining': "pol1"},
       "m_lep1_tau2": {'y_min': 0., 'y_max': 1100., 'fitFunc_AllMassTraining': "pol3", 'fitFunc_LowMassTraining': "pol1", 'fitFunc_HighMassTraining': "pol6"},
       "dr_lep_tau_min_SS": {'y_min': 0., 'y_max': 1.0, 'fitFunc_AllMassTraining': "pol6", 'fitFunc_LowMassTraining': "pol5", 'fitFunc_HighMassTraining': "pol4"},
       "dr_lep_tau_min_OS": {'y_min': 0., 'y_max': 1.0, 'fitFunc_AllMassTraining': "pol6", 'fitFunc_LowMassTraining': "pol5", 'fitFunc_HighMassTraining': "pol4"},
       "dr_taus": {'y_min': 0., 'y_max': 1.0, 'fitFunc_AllMassTraining': "pol6", 'fitFunc_LowMassTraining': "pol4", 'fitFunc_HighMassTraining': "pol4"},
       "dr_lep1_tau1_tau2_max": {'y_min': 0., 'y_max': 1.0, 'fitFunc_AllMassTraining': "pol1", 'fitFunc_LowMassTraining': "pol4", 'fitFunc_HighMassTraining': "pol1"},
       "dr_lep1_tau1_tau2_min": {'y_min': 0., 'y_max': 1.0, 'fitFunc_AllMassTraining': "pol6", 'fitFunc_LowMassTraining': "pol4", 'fitFunc_HighMassTraining': "pol4"},
       "max_tau_eta": {'y_min': 0., 'y_max': 3.0, 'fitFunc_AllMassTraining': "pol3", 'fitFunc_LowMassTraining': "pol1", 'fitFunc_HighMassTraining': "pol1"},
       "max_lep_eta": {'y_min': 0., 'y_max': 3.0, 'fitFunc_AllMassTraining': "pol3", 'fitFunc_LowMassTraining': "pol3", 'fitFunc_HighMassTraining': "pol6"},
       "nElectron": {'y_min': 0., 'y_max': 3.0, 'fitFunc_AllMassTraining': "pol2", 'fitFunc_LowMassTraining': "pol4", 'fitFunc_HighMassTraining': "pol6"},
       "nBJet_medium": {'y_min': 0., 'y_max': 3.0, 'fitFunc_AllMassTraining': "pol2", 'fitFunc_LowMassTraining': "pol5", 'fitFunc_HighMassTraining': "pol4"},
       "dr_leps": {'y_min': 0., 'y_max': 1.0, 'fitFunc_AllMassTraining': "pol6", 'fitFunc_LowMassTraining': "pol3", 'fitFunc_HighMassTraining': "pol6"},
       "tau1_eta": {'y_min': -3.0, 'y_max': 3.0, 'fitFunc_AllMassTraining': "pol2", 'fitFunc_LowMassTraining': "pol3", 'fitFunc_HighMassTraining': "pol1"},
       "deltaEta_lep1_tau1": {'y_min': -5.0, 'y_max': 5.0, 'fitFunc_AllMassTraining': "pol1", 'fitFunc_LowMassTraining': "pol2", 'fitFunc_HighMassTraining': "pol1"},
       "deltaEta_lep1_tau2": {'y_min': -5.0, 'y_max': 5.0, 'fitFunc_AllMassTraining': "pol1", 'fitFunc_LowMassTraining': "pol4", 'fitFunc_HighMassTraining': "pol1"},
       }

    for var_name in var_name_list:
        print("Variable Name: {}".format(var_name))
        if(Target == 1):
            FileName = "{}/{}_{}_{}.root".format(channel, "TProfile_signal", var_name, label)
            #TProfileFile = TFile(FileName, "RECREATE")                                                                                                                                                                                              
            if(doFit): Fit_Func_FileName = "{}/{}_{}.root".format(channel, "TProfile_signal_fit_func", var_name)
        elif(Target == 0):
            FileName = "{}/{}_{}_{}.root".format(channel, "TProfile", var_name, label)
            #TProfileFile = TFile(FileName, "RECREATE")                                                                                                                                                                                           
        else:
            FileName = "{}/{}_{}.root".format(channel, "TProfile", var_name, label)
            #TProfileFile = TFile(FileName, "RECREATE")                                                                                                                                                                                            
        #c1.SaveAs(FileName)                                                                                                                                                                                                                        

        data_X  = np.array(data_copy['gen_mHH'].values, dtype=np.float)
        data_Y  = np.array(data_copy[var_name].values, dtype=np.float)
        data_wt = np.array(data_copy['totalWeight'].values, dtype=np.float)

        N_x  = len(data_X)
        N_y  = len(data_Y)
        N_wt = len(data_wt)

        if((N_x == N_y) and (N_y == N_wt)):
            #print("N_x: {}, N_y: {}, N_wt: {}".format(N_x, N_y, N_wt))                                                                                                                                                                             
            PlotTitle = 'Profile of '+var_name+' vs gen_mHH'
            #hprof  = TProfile( 'hprof', PlotTitle, 17, 250., 1100., y_min, y_max)                                                                                                                                                                 
            #xbins = array.array('d', [250., 260., 270., 280., 300., 350., 400., 450., 500., 550., 600., 650., 700., 750., 800., 850., 900., 1000.])                                                                                              
            hprof  = TProfile( 'hprof', PlotTitle, (len(mass_list) - 1), mass_list[0], (mass_list[(len(mass_list) - 1)] + 100.0), Func_Dict[var_name]['y_min'], Func_Dict[var_name]['y_max'])
            xbins = array.array('d', mass_list)
            hprof.SetBins((len(xbins) - 1), xbins)
            hprof.GetXaxis().SetTitle("gen_mHH (GeV)")
            hprof.GetYaxis().SetTitle(var_name)
            numpyarrayTProfileFill(data_X, data_Y, data_wt, hprof)
            hprof.Draw()
            c1.Modified()
            c1.Update()
            #c1.SaveAs(FileName)
          
            fitFuncName = "fitFunction_" + var_name
            print("fitFuncName", fitFuncName)


            if(doFit and (Target == 1)): ## do the fit for signal only                                                                                                                                                                             
                if(TrainMode == 0): ## All masses used in the training
                    f_old = TF1(fitFuncName, Func_Dict[var_name]['fitFunc_AllMassTraining'], 250., 1000.)
                elif(TrainMode == 1): ## Only Low masses used in the training
                    f_old = TF1(fitFuncName, Func_Dict[var_name]['fitFunc_LowMassTraining'], 250., 400.)
                elif(TrainMode == 2): ## Only High masses used in the training
                    f_old = TF1(fitFuncName, Func_Dict[var_name]['fitFunc_HighMassTraining'], 450., 1000.)                                                                                                                                                              
                r_old = TFitResultPtr()
                r_old = hprof.Fit(f_old, "SF") ## Fitting using Minuit instead of the linear fitter                                                                                                                                                
                f_old.Draw("same")
                c1.Modified()
                c1.Update()
                c1.SaveAs(FileName)
                FuncFile = TFile(Fit_Func_FileName, "RECREATE")
                f_old.Write()
                FuncFile.Close()
            else:
                print("No fit will be performed")
                c1.SaveAs(FileName)
        else:
            print('Arrays not of same length')
            print("N_x: {}, N_y: {}, N_wt: {}".format(N_x, N_y, N_wt)) 



def ReweightDataframe_New(data, channel, var_name_list, masses):
    for var_name in var_name_list:
        print("Variable Name: {}".format(var_name))
        Fit_Func_FileName = "{}/{}_{}.root".format(channel, "TProfile_signal_fit_func", var_name)
        file = TFile.Open(Fit_Func_FileName)
        func = TF1()
        fitFuncName = "fitFunction_" + var_name
        file.GetObject(fitFuncName, func)
        print("Number of parameters", func.GetNpar())
        Npar = func.GetNpar()

        corr_factor_Dict = {}
        for x in masses:
            corr_factor_Dict[x] = func.Eval(x)
            #print("Corr. factor: %f , gen_mHH: %f" % (corr_factor_Dict[x], x))                                                                                                                                                          
        print("Started the scaling of ", var_name)

        process = psutil.Process(os.getpid())
        print(process.memory_info().rss)
        print(datetime.now() - startTime)

        for x in masses:
            print("gen_mHH %f" % x)
            data.loc[(data['gen_mHH']==x), [var_name]] /= corr_factor_Dict[x]

        print("Finished the scaling of ", var_name)
        process = psutil.Process(os.getpid())
        print(process.memory_info().rss)
        print(datetime.now() - startTime)
        file.Close()       




