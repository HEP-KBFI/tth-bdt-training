import matplotlib.pyplot as plt
import sys
arg=sys.argv[1]

execfile('roc_2l_2tau_800GeVOnly_dR03mvaLoose_TEST.log')
print 'arg====================== ',arg

if arg =='cc': # Only the test mass included in training
   xval = xval
   yval = yval
   xtrain = xtrain
   ytrain = ytrain
   test_auc = test_auc
   train_auc = train_auc
   print 'xval: ',  xval
elif arg == 'all' :
    xval = xval_all
    yval=yval_all
    xtrain =xtrain_all
    ytrain=ytrain_all
    test_auc = testall_auc
    train_auc =trainall_auc
elif arg == 'NEW' : # All masses except the test mass included in training
    xval=xvalNEW
    yval=yvalNEW
    xtrain=xtrainNEW
    ytrain=ytrainNEW
    test_auc=test_aucNEW
    train_auc=train_aucNEW
    print 'xvalNEW: ',  xval
elif arg == 'TT' :
    xval = xval_tt
    yval=yval_tt
    xtrain =xtrain_tt
    ytrain=ytrain_tt
    test_auc = testtt_auc
    train_auc =traintt_auc

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(xval,yval, lw=1,ls='--', color = 'b',label='XGB val (area = %0.3f), Loose'%(test_auc))
ax.plot(xtrain,ytrain, lw=1, color = 'b',label='XGB train (area = %0.3f), Loose'%(train_auc))


execfile('roc_2l_2tau_800GeVOnly_dR03mvaVLoose_TEST.log')
if arg =='cc':
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
elif arg == 'NEW' :
    xval=xvalNEW
    yval=yvalNEW
    xtrain=xtrainNEW
    ytrain=ytrainNEW
    test_auc=test_aucNEW
    train_auc=train_aucNEW
elif arg == 'TT' :
    xval = xval_tt
    yval=yval_tt
    xtrain =xtrain_tt
    ytrain=ytrain_tt
    test_auc = testtt_auc
    train_auc = traintt_auc
    plot ='Singlemass_TTbkg.png'
ax.plot(xval,yval, lw=1,ls='--', color = 'g',label='XGB val (area = %0.3f), VLoose'%(test_auc))
ax.plot(xtrain,ytrain, lw=1, color = 'g',label='XGB train (area = %0.3f), VLoose'%(train_auc))

execfile('roc_2l_2tau_800GeVOnly_dR03mvaVVLoose_TEST.log')
if arg =='cc':
    xval=xval
    yval=yval
    xtrain =xtrain
    ytrain=ytrain
    test_auc=test_auc
    train_auc=train_auc
    plot ='Singlemass_allbkg.png'
elif arg == 'all' :
    xval = xval_all
    yval=yval_all
    xtrain =xtrain_all
    ytrain=ytrain_all
    test_auc = testall_auc
    train_auc =trainall_auc
    plot = 'all.png'
elif arg == 'NEW' :
    xval=xvalNEW
    yval=yvalNEW
    xtrain=xtrainNEW
    ytrain=ytrainNEW
    test_auc=test_aucNEW
    train_auc=train_aucNEW
elif arg == 'TT' :
    xval = xval_tt
    yval=yval_tt
    xtrain =xtrain_tt
    ytrain=ytrain_tt
    test_auc = testtt_auc
    train_auc = traintt_auc
    plot ='Singlemass_TTbkg.png'
ax.plot(xval,yval, lw=1,ls='--', color = 'r',label='XGB val (area = %0.3f), VVLoose'%(test_auc))
ax.plot(xtrain,ytrain, lw=1, color = 'r',label='XGB train (area = %0.3f), VVLoose'%(train_auc))


ax.set_ylim([0.0,1.0])
ax.set_xlim([0.0,1.0])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc="lower right")
ax.grid()

fig.savefig('roc_800GeVOnly_TEST_allOtherMassesPresent.pdf') ## Trained on all masses except the one being tested => NEW
#fig.savefig('roc_800GeVOnly_TEST_allOtherMassesAbsent.pdf') ## Trained on only the mass being tested => cc
