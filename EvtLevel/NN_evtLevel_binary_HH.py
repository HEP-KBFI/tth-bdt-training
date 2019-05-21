import ROOT
import pandas
import math
#matplotlib.use('agg')
#matplotlib inline
import matplotlib #as matplot
print(matplotlib.__version__)
#print(matplotlib.path)
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import numpy as np
import time

from sklearn.metrics import roc_curve, auc
import pylab
import sklearn as sk
print(sk.__version__)
from sklearn.model_selection import train_test_split

import tensorflow as tf
print(tf.__version__)
######################
import keras as kr
from keras.models import Sequential
from keras.layers import InputLayer, Input
from keras.layers import Reshape, MaxPooling2D
from keras.layers import Conv2D, Dense, Flatten, Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, ELU, PReLU 
from keras.optimizers import Adamax
from keras.optimizers import Nadam
print(kr.__version__)
#from keras import backend as K ## Uncommenting this line along with "K.set_session(session)" gives Attribute error 
import tensorflow as tf
print(tf.__version__)
config = tf.ConfigProto(intra_op_parallelism_threads=32, \
                        inter_op_parallelism_threads=32, \
                        allow_soft_placement=True, \
                        device_count = {'CPU': 32}
                       )
session = tf.Session(config=config)
#K.set_session(session)  ## Uncommenting this line along with "from keras import backend as K" gives Attribute error 

# test that matplotlib imported ok
#x1, y1 = [-1, 12], [1, 4]
#x2, y2 = [1, 10], [3, 2]
#plt.plot(x1, y1, x2, y2, marker = 'o')
#plt.show()


################
## load the data
################
channel='2l_2tau_HH'
bdtType = "evtLevelSUM_HH_2l_2tau_res"
variables="testVars2"
tauID = "dR03mvaVLoose"  ## Christian's choice from previous param. BDT studies
Bkg_mass_rand="default"
#Bkg_mass_rand="oversampling"



#startTime = datetime.now()
execfile("../python/data_manager.py")
#run("../python/data_manager.py")
if channel=='2l_2tau_HH' :
    execfile("../cards/info_2l_2tau_HH.py")  
    #run("../cards/info_2l_2tau_HH.py")

output = read_from( Bkg_mass_rand, tauID)


print ("read from:", output["inputPath"])
print ("Date: ", time.asctime( time.localtime(time.time()) ))
data=load_data_2017(
        output["inputPath"],
        output["channelInTree"],
        trainVars(True),
        [],
        bdtType,
        channel,
        output["keys"],
        output["masses"],
        output["mass_randomization"]
        )

print (len(data))
print ("Date: ", time.asctime( time.localtime(time.time()) ))


weights="evtWeight"

print ("Date: ", time.asctime( time.localtime(time.time()) ))
data["weight_cx"] = data[weights]
data["weight_train"] = data[weights]
data["NN_output"] = 1.0

data_even = data.loc[(data["event"].values % 2 == 0) ]
data_odd = data.loc[~(data["event"].values % 2 == 0) ]

order_train = [data_odd, data_even]
order_train_name = ["odd","even"]

print("data[weight_cx]", data["weight_cx"])
print("data[weight_train]", data["weight_train"])

print ("balance datasets by even/odd chunck")
for data_do in order_train :
    #### Normalization by cross section
    for wei in ["weight_cx", "weight_train"] :
        if 'SUM_HH' in bdtType :
            data_do.loc[(data_do['key'].isin(['TTTo2L2Nu','TTToSemiLeptonic'])), [wei]]              *= output["TTdatacard"]/data_do.loc[(data_do['key'].isin(['TTTo2L2Nu','TTToSemiLeptonic'])), weights].sum()
            data_do.loc[(data_do['key']=='DY'), [wei]]                            *= output["DYdatacard"]/data_do.loc[(data_do['key']=='DY'), weights].sum()
            if "evtLevelSUM_HH_bb1l_res" in bdtType :
                data_do.loc[(data_do['key']=='W'), [wei]]                         *= Wdatacard/data_do.loc[(data_do['key']=='W')].sum() ## Saswati check please !!!                                               
        if "evtLevelSUM_HH_2l_2tau_res" in bdtType :
               data_do.loc[(data_do['key']=='TTZJets'), [wei]]                       *= output["TTZdatacard"]/data_do.loc[(data_do['key']=='TTZJets'), weights].sum() ## TTZJets                               
               data_do.loc[(data_do['key']=='TTWJets'), [wei]]                       *= output["TTWdatacard"]/data_do.loc[(data_do['key']=='TTWJets'), weights].sum() ## TTWJets + TTWW                        
               data_do.loc[(data_do['key']=='ZZ'), [wei]]                            *= output["ZZdatacard"]/data_do.loc[(data_do['key']=='ZZ'), weights].sum() ## ZZ +ZZZ                                     
               data_do.loc[(data_do['key']=='WZ'), [wei]]                            *= output["WZdatacard"]/data_do.loc[(data_do['key']=='WZ'), weights].sum() ## WZ + WZZ_4F                                 
               data_do.loc[(data_do['key']=='WW'), [wei]]                            *= output["WWdatacard"]/data_do.loc[(data_do['key']=='WW'), weights].sum() ## WW + WWZ + WWW_4F    
               #data_do.loc[(data_do['key']=='VH'), [wei]]                        *= output["VHdatacard"]/data_do.loc[(data_do['key']=='VH'), weights].sum() # consider removing                               
               #data_do.loc[(data_do['key']=='TTH'), [wei]]                       *= output["TTHdatacard"]/data_do.loc[(data_do['key']=='TTH'), weights].sum() # consider removing  

        ### Normalize sig/BKG and do table of nevents/mass
        for mass in output["masses"] :
            data_do.loc[(data_do['target']==1) & (data_do["gen_mHH"] == mass),"weight_train"] *= 1000./data_do.loc[(data_do['target']==1) & (data_do["gen_mHH"]== mass), "weight_train"].sum()
            data_do.loc[(data_do['target']==0) & (data_do["gen_mHH"] == mass),"weight_train"] *= 1000./data_do.loc[(data_do['target']==0) & (data_do["gen_mHH"]== mass), "weight_train"].sum()
        print ("Date: ", time.asctime( time.localtime(time.time()) ))


        print("data_odd[weight_cx]", data_odd["weight_cx"])
        print("data_odd[weight_train]", data_odd["weight_train"])


        print ("training statistics by mass")
        for mass in output["masses"] :
            print(
                  str(mass)+": sig = "+\
                  str(len(data_do.loc[(data['target']==1) & (data_do["gen_mHH"] == mass),["weight_train"]]))+\
                  " BKG = "+str(len(data_do.loc[(data['target']==0) & (data_do["gen_mHH"] == mass),["weight_train"]]))
                  )

        print ("\n norm by mass - test")
        for mass in output["masses"] :
            print(
                  str(mass)+": sig = "+\
                  str(data_do.loc[(data_do['target']==1) & (data_do["gen_mHH"] == mass),"weight_train"].sum())+\
                  " BKG = "+str(data_do.loc[(data_do['target']==0) & (data_do["gen_mHH"] == mass),"weight_train"].sum())
                  )


"""
Check of the resulting weights - the sizes of the training weight
"""
fig, ax = plt.subplots(figsize=(4, 4))
keysToBKG = ['WW', 'WZ', 'ZZ', 'DY', 'TTTo2L2Nu', 'TTToSemiLeptonic','TTZJets', 'TTWJets'] # 'VH', 'TTH', 'TTToHadronic'
if "evtLevelSUM_HH_bb1l_res" in bdtType : keysToBKG.append('W')
#colors = ['cyan','orange','k','r','green','magenta','b',]
vars = ["weight_train"]#"multitarget"]

for kk, key in enumerate(keysToBKG) :
  for vv, var in enumerate(vars) : 
    ax.hist(
        np.array(data.loc[(data['key']==key), var].values,dtype='float64'), # 
        weights=data.loc[(data['key']==key), "evtWeight"], # "weight_train_cat"
        range=(-1.0,10.),bins=40, histtype='step', normed=True, lw=2, 
        label=key
    )
    ax.set_xlabel(var)
ax.legend(loc="best", title= channel)


## load the variables
BDTvariables=trainVars(False, variables, bdtType)
trainvar = BDTvariables
features = trainvar
print trainvar


"""
Draw some plots on lists of variables for BKG
"""
## i try to do 3 X 3 plots (= enter up to nine entries in each sublist)
## try to add strictly decreasing variables as first in each sublist, better for the legend positioning


listdraw = [
    ['diHiggsMass', 'nBJet_medium', 'gen_mHH', 'tau1_pt', 'nElectron', 'dr_lep_tau_min_SS', 'tau2_pt', 'met', 'diHiggsVisMass'],
    ['max_tau_eta', 'dr_lep_tau_min_OS', 'mT_lep2', 'dr_lep1_tau1_tau2_min', 'm_ll', 'met_LD', 'max_lep_eta', 'dr_leps', 'mT_lep1'],
    ['tau1_eta', 'mTauTau', 'm_lep1_tau2', 'deltaEta_lep1_tau1', 'dr_lep1_tau1_tau2_max', 'dr_taus', 'deltaEta_lep1_tau2']
]


for featuresDraw in listdraw:
    sizeArray=int(math.sqrt(len(featuresDraw))) if math.sqrt(len(featuresDraw)) % int(math.sqrt(len(featuresDraw))) == 0 else int(math.sqrt(len(featuresDraw)))+1
    plt.figure(figsize=(4*sizeArray,4*sizeArray))
    for n, feature in enumerate(featuresDraw) :
        min_value, max_value = np.percentile(data[feature], [0.0, 99])
        # fig, ax = plt.subplots(figsize=(4, 4))
        plt.subplot(sizeArray, sizeArray, n+1)
        for kk, key in enumerate(keysToBKG) :
            if 'TTZJets' in key or 'TTWJets' in key : linestyle = "--"
            else :linestyle = "-"
            plt.hist(
            np.array(data.loc[(data['key']==key), feature].values,dtype='float64'), 
            weights=data.loc[(data['key']==key) , "evtWeight"], 
            range=(min_value, max_value), 
            bins=12, histtype='step', ls=linestyle, 
            normed=True, lw=2, #color=colors[kk],
            label=key
            )
            plt.xlabel(feature)
        if n == 0 : plt.legend(loc="upper right", title= channel)


for featuresDraw in listdraw:
    sizeArray=int(math.sqrt(len(featuresDraw))) if math.sqrt(len(featuresDraw)) % int(math.sqrt(len(featuresDraw))) == 0 else int(math.sqrt(len(featuresDraw)))+1
    plt.figure(figsize=(4*sizeArray,4*sizeArray))
    for n, feature in enumerate(featuresDraw) :
        min_value, max_value = np.percentile(data[feature], [0.0, 99])
        # fig, ax = plt.subplots(figsize=(4, 4))
        plt.subplot(sizeArray, sizeArray, n+1)
        for mass in [300,400,700] :
            plt.hist(
            np.array(data.loc[(data["gen_mHH"] == mass), feature].values,dtype='float64'), 
            weights=data.loc[(data["gen_mHH"] == mass) , "evtWeight"], 
            range=(min_value, max_value), 
            bins=10, histtype='step', ls=linestyle, 
            normed=True, lw=2, #color=colors[kk],
            label="mass = "+str(mass)
            )
        plt.hist(
        np.array(data.loc[(data['target']==0), feature].values,dtype='float64'), 
        weights=data.loc[(data['target']==0) , "evtWeight"], 
        range=(min_value, max_value), 
        bins=10, histtype='step', ls='--', 
        normed=True, lw=3, color='k',
        label="BKG"
        )
        plt.xlabel(feature)
        if n == 0 : plt.legend(loc="upper right", title= channel)


# create model -- for binary activation='sigmoid'
nclasses = 1

def nn_model_binary():
    "create a model."
    model = Sequential()
    model.add(Dense(2*len(features), input_dim=len(features), kernel_initializer='he_uniform')) 
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.1))
    for Nnodes in [8,8] :
        model.add(Dense(Nnodes, kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(0.1))
    model.add(Dense(nclasses, activation='sigmoid'))
    model.compile(
    loss='binary_crossentropy', 
    optimizer=Nadam(lr=0.0005, schedule_decay=0.00005), # , beta_1 = 0.95, beta_2 = 0.999
    metrics=['accuracy'], 
    )
    return model


nn_model_binary().summary()

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler

features = trainvar
# Port Keras Framework into SK-Learn
# https://stackoverflow.com/questions/39467496/error-when-using-keras-sk-learn-api
k_model_binary  = KerasClassifier(
    build_fn=nn_model_binary, 
    epochs=10, 
    batch_size=64, 
    verbose=2
)

print("data_odd[features].values", data_odd[features].values)
print("data_odd[target].values", data_odd['target'].values)
print("sample_weight=data_odd[weight_train].values", data_odd["weight_train"].values)
print("sample_weight=data_odd[weight_train].astype(float32)", data_odd["weight_train"].astype('float32'))


print("data_odd[weight_cx]", data_odd["weight_cx"])
print("data_odd[weight_train]", data_odd["weight_train"])


history = k_model_binary.fit(
    data_odd[features].values, 
    data_odd['target'].values,
    sample_weight=data_odd["weight_train"].values,
    #sample_weight=data_odd["weight_train"],
    validation_data=(
        data_even[features].values, 
        data_even['target'].values, 
        data_even["weight_train"].values
        #data_even["weight_train"].astype('float32')
    )
)




"""
overtraining test
"""
# Extract number of run epochs from the training history
epochs = range(1, len(history.history["loss"])+1)
plt.figure(figsize=(9, 4))
#fig = plt.figure(figsize=(4, 4))
plt.subplot(1, 2, 1)
# Extract loss on training and validation dataset and plot them together
plt.plot(epochs, history.history["loss"], "o-", label="Training")
plt.plot(epochs, history.history["val_loss"], "o-", label="Validation")
plt.xlabel("Epochs"), plt.ylabel("Loss")
#plt.yscale("log")
#plt.xlim(0,40)
plt.ylim(0.0,1.0)
plt.grid()
plt.legend();

plt.subplot(1, 2, 2)
#fig = plt.figure(figsize=(4, 4))
# Extract loss on training and validation dataset and plot them together
plt.plot(epochs, history.history["acc"], "o-", label="Training")
plt.plot(epochs, history.history["val_acc"], "o-", label="Validation")
plt.xlabel("Epochs"), plt.ylabel("Accuracy")
#plt.yscale("log")
plt.ylim(0.0,1.5)
plt.grid()
plt.legend(loc="best");

plt.show()

## ---- Needed since "pip install eli5 --user" installs ----------------####
##----- eli5 inside "/home/ram/.local/lib/python2.7/site-packages" by default ---###
import sys
sys.path.append('/home/ram/.local/lib/python2.7/site-packages')

import eli5
from eli5.sklearn import PermutationImportance
"""
to calculate variables importance, it takes time and it is not completelly 'enlightant',
do not do all the time.
"""

print ("Date: ", time.asctime( time.localtime(time.time()) ))
perm = PermutationImportance(k_model_binary, random_state=1).fit( # , scoring="f1_samples"
    data_odd[features].values, 
    data_odd['target'].values,
    sample_weight=data_odd["weight_train"].values
)
print ("Date: ", time.asctime( time.localtime(time.time()) ))
eli5.show_weights(perm, feature_names = data_odd[features].columns.tolist(), top=len(features))


"""
Calculate the output in all dataset 
-- to pass to the training/test
"""

data_odd["NN_output"]  = k_model_binary.predict_proba(data_odd[features].values, verbose=1)[:, 1]
data_even["NN_output"] = k_model_binary.predict_proba(data_even[features].values, verbose=1)[:, 1]


hist_params = {'normed': True, 'bins': 10 , 'histtype':'step'}
target = 'target'
plt.clf()

plt.figure('XGB',figsize=(6, 6))

values, bins, _ = plt.hist(
    data_odd.loc[data_odd.target.values == 1, "NN_output"].values , 
    weights=data_odd.loc[data_odd.target.values == 1, "weight_cx"].values,
    label="sig odd", color='r', range=(0,1), **hist_params
    )
values, bins, _ = plt.hist(
    data_odd.loc[data_odd.target.values == 0, "NN_output"].values , 
    weights=data_odd.loc[data_odd.target.values == 0, "weight_cx"].values,
    label="BKG odd", color='g', range=(0,1), **hist_params
    )

values, bins, _ = plt.hist(
    data_even.loc[data_even.target.values == 1, "NN_output"].values , 
    weights=(data_even.loc[data_even.target.values == 1, "weight_cx"].values),
    label="sig even", color='r', ls='--', range=(0,1), **hist_params)
values, bins, _ = plt.hist(
    data_even.loc[data_even.target.values == 0, "NN_output"].values , 
    weights=(data_even.loc[data_even.target.values == 0, "weight_cx"].values),
    label="BKG even", color='g', ls='--', range=(0,1), **hist_params)

#plt.xscale('log')
#plt.yscale('log')
plt.legend(loc='best')
plt.show()


###############################
## classifier plot by mass
hist_params = {'normed': True, 'bins': 8 , 'histtype':'step', "lw": 2}
plt.clf()
colorcold = ['g', 'r', 'y']
colorhot = ['b', 'magenta', 'orange']

fig, ax = plt.subplots(figsize=(6, 6))
for mm, mass in enumerate(output["masses_test"]) :
    y_pred = data_even.loc[(data_even.target.values == 0) & (data_even["gen_mHH"] == mass), "NN_output"].values
    y_predS = data_even.loc[(data_even.target.values == 1) & (data_even["gen_mHH"] == mass), "NN_output"].values
    y_pred_train = data_odd.loc[(data_odd.target.values == 0) & (data_odd["gen_mHH"] == mass), "NN_output"].values
    y_predS_train = data_odd.loc[(data_odd.target.values == 1) & (data_odd["gen_mHH"] == mass), "NN_output"].values
    dict_plot = [
       [y_pred, "-", colorhot[mm],  str(mass)+" GeV test BKG"],
       [y_predS, "-", colorcold[mm], str(mass)+" GeV test signal"],
       [y_pred_train, "--", colorhot[mm], str(mass)+" GeV train BKG" ],
       [y_predS_train, "--", colorcold[mm],      str(mass)+" GeV train signal"]
    ]
    for item in dict_plot :
        values1, bins, _ = ax.hist(
            item[0],
            ls=item[1], color = item[2],
            label=item[3],
            range=(0,1),
            **hist_params
            )
        normed = sum(y_pred)
        mid = 0.5*(bins[1:] + bins[:-1])
        err=np.sqrt(values1*normed)/normed # denominator is because plot is normalized
        plt.errorbar(mid, values1, yerr=err, fmt='none', color= item[2], ecolor= item[2], edgecolor=item[2], lw=2)
#plt.xscale('log')
#plt.yscale('log')
ax.legend(loc='upper center', title="by mass ", fontsize = 'small')
#nameout = channel+'/'+bdtType+'_'+trainvar+'_'+str(len(trainVars(False)))+'_'+hyppar+'_mass_'+ str(mass)+'_XGBclassifier.pdf'


###############################
# by mass ROC
styleline = ['-', '--', '-.', ':']
colors_mass = ['m', 'b', 'k', 'r', 'g',  'y', 'c', ]
fig, ax = plt.subplots(figsize=(6, 6))
sl = 0


for mm, mass in enumerate(output["masses_test"]) :
    for dd, data_do in  enumerate(order_train) :
        if dd == 0 : val_data = 1
        else : val_data = 0
        fpr, tpr, thresholds = roc_curve(
            data_do.loc[(data_do["gen_mHH"] == mass), "target"].astype(np.bool),
            data_do.loc[(data_do["gen_mHH"] == mass), "NN_output"].values,
            sample_weight=(data_do.loc[(data_do["gen_mHH"].astype(np.int) == int(mass)), "weight_cx"].astype(np.float64))
        )
        train_auc = auc(fpr, tpr, reorder = True)
        print("train set auc " + str(train_auc) + " (mass = " + str(mass) + ")")
        fprt, tprt, thresholds = roc_curve(
            order_train[val_data].loc[(order_train[val_data]["gen_mHH"].astype(np.int) == int(mass)), target].astype(np.bool), 
            order_train[val_data].loc[(order_train[val_data]["gen_mHH"] == mass), "NN_output"].values, #proba[:,1],
            sample_weight=(order_train[val_data].loc[(order_train[val_data]["gen_mHH"].astype(np.int) == int(mass)), "weight_cx"].astype(np.float64))
        )
        test_auct = auc(fprt, tprt, reorder = True)
        print("test set auc " + str(test_auct) + " (mass = " + str(mass) + ")")
        ax.plot(
            fpr, tpr,
            lw = 2, linestyle = styleline[dd + dd*1], color = colors_mass[mm],
            label = order_train_name[dd] + ' train (area = %0.3f)'%(train_auc) + " (mass = " + str(mass) + ")"
            )
        sl += 1
        ax.plot(
            fprt, tprt,
            lw = 2, linestyle = styleline[dd + 1 + + dd*1], color = colors_mass[mm],
            label = order_train_name[dd] + ' test (area = %0.3f)'%(test_auct) + " (mass = " + str(mass) + ")"
            )
        sl += 1
ax.set_ylim([0.0,1.0])
ax.set_xlim([0.0,1.0])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc="lower right", fontsize = 'small')
ax.grid()


"""
output a training and export to .pb (to be used on cpp)
"""
print ("Date: ", time.asctime( time.localtime(time.time()) ))
nameout = "model_erase"

out = k_model_binary.model.save("test_"+nameout+".hdf5")
file = open(nameout+"_variables.log","w")
file.write(str(features)+"\n")
file.close()




""" 
If you want to load a model to reconpute anything or check loading just substitute k_model --> k_model_loaded
It only loads hdf5 format [COMMENTED OUT FOR THE TIME BEING]
"""
#from keras.models import load_model
#k_model_loaded = load_model("test_model_2lss_ttH_3cat_no4mom_noSemi_v6.hdf5")


## the next do correlation matrices with variables
import seaborn 

for target in [0,1] :
    corr_mat = data.loc[(data['target']==0), features].astype(float).corr() #
    fig, ax = plt.subplots(figsize=(20, 12)) 
    seaborn.heatmap(corr_mat, square=True, ax=ax, vmin=-1., vmax=1.);



