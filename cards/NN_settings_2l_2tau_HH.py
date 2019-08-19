
###########################################                                                                                                                                                                                                                          
## Settings for NN training in jupyter .nb                                                                                                                                                                                                                            
##########################################                                                                                                                                                                                                                               
  
#channel='2l_2tau_HH'                                                                                                                                                                                                                                                 
bdtType = "evtLevelSUM_HH_2l_2tau_res"

## --- Choose Tau ID ----                                                                                                                                                                                                                                            
#tauID = "dR03mvaLoose"                                                                                                                                                                                                                                              
tauID = "dR03mvaVLoose"
#tauID = "dR03mvaVVLoose"                                                                                                                                                                                                                                            
## ---------------------                                                                                                                                                                                                                                                
   
##-------Choose your background gen_mHH randomiz. method ----##                                                                                                                                                                                                        
#Bkg_mass_rand="default"                                                                                                                                                                                                                                              
Bkg_mass_rand="oversampling"
##----------------------------------------------------------##                                                                                                                                                                                                        


##-------- Choose Input variables and mass range for NN training --------------##                                                                                                                                                                                    
variables="finalVars_allMasses"  ## To include all masses use these (for tauID = "dR03mvaVLoose")                                                                                                                                                                    
TrainMode=0                      ## To include all masses use these (for tauID = "dR03mvaVLoose")                                                                                                                                                                    

#variables="finalVars_LowMasses" ## To include Low masses (=< 400 GeV) only use these (for tauID = "dR03mvaVLoose")                                                                                                                                                
#TrainMode=1                     ## To include Low masses (=< 400 GeV) only use these (for tauID = "dR03mvaVLoose")                                                                                                                                                    

#variables="finalVars_HighMasses" ## To include High masses (> 400 GeV) only use these (for tauID = "dR03mvaVLoose")                                                                                                                                               
#TrainMode=2                      ## To include High masses (> 400 GeV) only use these (for tauID = "dR03mvaVLoose")                                                                                                                                                 
##----------------------------------------------------------##                                                                                                                                                                                                           
  

## --- To reweight Input Variables set this to true ---- #                                                                                                                                                                                                            
do_ReweightVars=True
##----------------------------------------------------------##                                                                                                                                                                                                           
  

## --- Set this to true to get Histograms and THStack plots ---- ###                                                                                                                                                                                                  
do_2l_2tau_diagnostics=True
##----------------------------------------------------------##                                                                                                                                                                                                        

#weights="totalWeight" ## Was used in the BDT code                                                                                                                                                                                                                    
#target='target'       ## Was used in the BDT code          
