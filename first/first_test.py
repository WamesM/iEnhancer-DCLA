import sys
sys.path.append( 'D:/pycharm_pro/My-Enhancer-classification/' )
from model import get_model, get_model_onehot
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score,accuracy_score,recall_score,matthews_corrcoef,confusion_matrix,roc_curve, precision_recall_curve
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import matplotlib.pyplot as plt
# import umap
# import umap.plot
# from matplotlib.backends.backend_pdf import PdfPages

def plotROC(test,score):
    fpr,tpr,threshold = roc_curve(test, score)
    auc_roc = roc_auc_score(test, score)
    plt.figure()
    font = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 22,
         }
    lw = 3
    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='iEnhancer-DCLA (auRoc = %f)' %auc_roc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.tick_params(labelsize=20)
    plt.xlabel('False Positive Rate',font)
    plt.ylabel('True Positive Rate',font)
    plt.title('Receiver operating characteristic curve',font)
    plt.legend(loc="lower right")
    plt.savefig('x.jpg',dpi=350)
    plt.show()


names = ['first']
for name in names:
    for i in [0]:
        model = get_model()
        model.load_weights("./model/our_model_7/%sModel%s.tf" % (name, i))
        Data_dir = 'D:/pycharm_pro/My-Enhancer-classification/%s/first_index/' % name
        test = np.load(Data_dir+'%s_test_enhancers7.npz' % name)
        X_en_tes,  y_tes = test['X_en_tes'], test['y_tes']

        print("****************Testing %s cell line specific model on %s cell line****************" % (name, name))
        # model.fit(X_en_tes, y_tes)
        y_pred1 = model.predict([X_en_tes])
        y_pred = np.where(y_pred1 > 0.5, 1, 0)

        acc = accuracy_score(y_tes, y_pred)
        sn = recall_score(y_tes, y_pred)
        mcc = matthews_corrcoef(y_tes, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_tes, y_pred).ravel()
        sp = tn / (tn + fp)
        auc = roc_auc_score(y_tes, y_pred1)
        aupr = average_precision_score(y_tes, y_pred1)
        f1 = f1_score(y_tes, np.round(y_pred1.reshape(-1)))
        print("ACC : ", acc)
        print("SN : ", sn)
        print("SP : ", sp)
        print("MCC : ", mcc)
        print("AUC : ", auc)
        print("AUPR : ", aupr)
        print("f1_score : ", f1)

        plotROC(y_tes, y_pred1)
