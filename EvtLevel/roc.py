import matplotlib.pyplot as plt
import sys
arg=sys.argv[1]
execfile('roc_1lTT.log')
print 'arg====================== ',arg
if arg =='':
    xval = xval
    yval=yval
    xtrain =xtrain
    ytrain=ytrain
    test_auc = test_auc
    train_auc =train_auc

elif arg == 'all' :
    xval = xval_all
    yval=yval_all
    xtrain =xtrain_all
    ytrain=ytrain_all
    test_auc = testall_auc
    train_auc =trainall_auc

else :
    xval = xval_tt
    yval=yval_tt
    xtrain =xtrain_tt
    ytrain=ytrain_tt
    test_auc = testtt_auc
    train_auc =traintt_auc

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(xval,yval, lw=1,ls='--', color = 'b',label='XGB val (area = %0.3f), 1l'%(test_auc))
ax.plot(xtrain,ytrain, lw=1, color = 'b',label='XGB train (area = %0.3f), 1l'%(train_auc))
execfile('roc_2lTT.log')
if arg =='':
    xval = xval
    yval=yval
    xtrain =xtrain
    ytrain=ytrain
    test_auc = test_auc
    train_auc =train_auc
    plot ='Singlemass_allbkg.png'
elif arg == 'all' :
    xval = xval_all
    yval=yval_all
    xtrain =xtrain_all
    ytrain=ytrain_all
    test_auc = testall_auc
    train_auc =trainall_auc
    plot = 'all.png'
else :
    xval = xval_tt
    yval=yval_tt
    xtrain =xtrain_tt
    ytrain=ytrain_tt
    test_auc = testtt_auc
    train_auc = traintt_auc
    plot ='Singlemass_TTbkg.png'
ax.plot(xval,yval, lw=1,ls='--', color = 'g',label='XGB val (area = %0.3f), 2l'%(test_auc))
ax.plot(xtrain,ytrain, lw=1, color = 'g',label='XGB train (area = %0.3f), 2l'%(train_auc))
ax.set_ylim([0.0,1.0])
ax.set_xlim([0.0,1.0])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc="lower right")
ax.grid()

fig.savefig(plot)
