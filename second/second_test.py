import sys
sys.path.append( 'D:/pycharm_pro/My-Enhancer-classification/second/' )
from model import get_model, get_model_onehot
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score,accuracy_score,recall_score,matthews_corrcoef,confusion_matrix, roc_curve
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import umap.plot


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
#    if aucVal is None:
#        plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve')
#    else:
#        plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' %aucVal)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.tick_params(labelsize=20)
    plt.xlabel('False Positive Rate',font)
    plt.ylabel('True Positive Rate',font)
    plt.title('Receiver operating characteristic curve',font)
    plt.legend(loc="lower right")
    plt.savefig('x.jpg', dpi=350)
    plt.show()


names = ['second']
for name in names:
    for i in [2]:
        model = get_model()
        model.load_weights("./model/our_model_7/%sModel%s.tf" % (name, i))
        Data_dir = 'D:/pycharm_pro/My-Enhancer-classification/%s/second_index/' % name
        test = np.load(Data_dir+'%s_test_enhancers7.npz' % name)
        X_en_tes,  y_tes = test['X_en_tes'], test['y_tes']

        print("****************Testing %s cell line specific model on %s cell line****************" % (name, name))
        # model.fit(X_en_tes, y_tes)
        y_pred1 = model.predict([X_en_tes])
        y_pred = np.where(y_pred1 > 0.5, 1, 0)
        # for i in y_pred1:
        #     if i >= 0.5:
        #         i = 1
        #         y_pred.append(i)
        #     else:
        #         y_pred.append(0)

        # print(y_pred)
        # print(y_tes)
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

        # layer_model_1 = Model(inputs=model.input, outputs=model.layers[7].output)
        # layer_model_2 = Model(inputs=model.input, outputs=model.layers[1].output)
        # # X_train = layer_model.predict(X_en_tra)
        # # print(X_train.shape)
        # X_test_1 = layer_model_1.predict(X_en_tes)
        # X_test_2 = layer_model_2.predict(X_en_tes)
        # # print(X_test)
        # # print(X_test.shape)
        # y_tes_1 = y_tes.reshape(-1)

        # print(y_tes.shape)
        # modell = get_model_1()
        # modell.summary()
        # modell.fit(X_test,y_tes)
        # import shap
        # shap.initjs()  # notebook环境下，加载用于可视化的JS代码
        # # background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
        # explainer = shap.DeepExplainer(modell, X_test)
        # shap_values = explainer.shap_values(X_test)
        # shap.summary_plot(shap_values, X_test)

        # featureDict = {
        #     'n_neighbors': 15,
        #     'min_dist': 0.15,
        #     'metric': 'chebyshev',
        # }
        # mapper = umap.UMAP(**featureDict).fit(X_test_1)
        # plotObj = umap.plot.points(mapper, labels=y_tes_1, color_key=['#0000FF', '#FF0000'], width=600, height=600, )
        # plt.show()

        # featureDict = {
        #     'n_neighbors': 4,
        #     'min_dist': 0.4,
        #     'metric': 'chebyshev',
        # }
        # for n_neighbors in np.arange(5, 30, 5).astype(int):
        #     featureDict['n_neighbors'] = n_neighbors
        #     for min_dist in np.arange(0.03, 0.2, 0.03):
        #         featureDict['min_dist'] = min_dist
        #         mapper = umap.UMAP(**featureDict).fit(X_test_1)
        #         plotObj = umap.plot.points(mapper, labels=y_tes_1, color_key=['#0000FF', '#FF0000'], width=600,
        #                                    height=600)
        #         plt.show()
        #         plt.close()  # close the plotted figure
        #
        # featureDict = {
        #     'n_neighbors': 4,
        #     'min_dist': 0.4,
        #     'metric': 'chebyshev',
        # }
        # mapper = umap.UMAP(**featureDict).fit(np.mean(X_test_2, axis=2))
        # plotObj = umap.plot.points(mapper, labels=y_tes_1, color_key=['#0000FF', '#FF0000'], width=550, height=550)
        # plt.show()



