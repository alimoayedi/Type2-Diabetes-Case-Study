import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


#import scikitplot as skplt
from sklearn import svm
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics as metrics

minMaxScaler = preprocessing.MinMaxScaler()
stdScaler = preprocessing.StandardScaler()

# Import some data to play with
train_url = r"D:\Synced Folder\Pattern Recognition-Term Project\Datasets-01\baseline_train_missingRemoved.csv"
test_url = r"D:\Synced Folder\Pattern Recognition-Term Project\Datasets-01\00-test_baseline_not imputed.csv"

train_data = pd.read_csv(train_url)
test_data = pd.read_csv(test_url)

train_labels = train_data.iloc[:,0].copy()
test_labels = test_data.iloc[:,0].copy()

train_data.drop('Classlabel', axis=1, inplace=True)
test_data.drop('Classlabel', axis=1, inplace=True)


# Binarize labels
for index,label in enumerate(train_labels):
    if label == 'Positive':
        train_labels[index] = 1
    else:
        train_labels[index] = 0

for index,label in enumerate(test_labels):
    if label == 'Positive':
        test_labels[index] = 1
    else:
        test_labels[index] = 0

#impute missing values of **test** set
for cName in test_data:
    if False not in test_data[cName].isin({0, 1, 2, 3, 4, float(0), float(1), float(2), float(3), float(4), np.NaN}).values:
        test_data[cName] = test_data[cName].fillna(train_data[cName].mode().values[0])
        test_data[cName] = test_data[cName].astype(int)
    elif test_data[cName].isnull().any().any():
        avg = test_data[cName].mean()
        test_data[cName].fillna(avg, inplace=True)

#interpolate missing data
# descrete
for cName in train_data:
    if False not in train_data[cName].isin({0, 1, 2, 3, 4, float(0), float(1), float(2), float(3), float(4), np.NaN}).values:
        mode = train_data[cName].mode().values[0]
        train_data[cName].fillna(mode, inplace=True)
        train_data[cName] = train_data[cName].astype(int)
# continous
for cName in train_data:
    if train_data[cName].isnull().any().any():
        avg = train_data[cName].mean()
        train_data[cName].fillna(avg, inplace=True)

# train data
for cName in train_data:
    if (train_data[cName]>5).any() or train_data[cName].dtype == np.float:
        train_data[cName] = round(pd.DataFrame(minMaxScaler.fit_transform(train_data[cName].values.reshape(-1,1))),5)
        #train_data[cName] = pd.DataFrame(stdScaler.fit_transform(train_data[cName].values.reshape(-1,1)))

#for cName in train_data:
#    if (train_data[cName]>5).any() or train_data[cName].dtype == np.float:
#        disc = preprocessing.KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='kmeans')
#        disc.fit(train_data[cName].values.reshape(-1,1))
#        train_data[cName] = disc.transform(train_data[cName].values.reshape(-1,1))

new_train_data=[]
new_train_data=pd.DataFrame(new_train_data)
for cName in train_data:
    if train_data[cName].isin(range(0,2)).all():
        new_train_data = pd.concat([new_train_data, train_data[cName]],axis=1)
    elif train_data[cName].isin(range(1,3)).all(): 
        new_train_data = pd.concat([new_train_data, train_data[cName] - 1],axis=1)
    elif (train_data[cName]>=0).all() and (train_data[cName]<=1).all():
        print(cName)
        new_train_data = pd.concat([new_train_data, train_data[cName]],axis=1)
    else:
        dummies = pd.get_dummies(train_data[cName])
        names=[]
        for count in range(0, dummies.shape[1]):
            names.extend([cName+str(count)])
        dummies.columns = names
        new_train_data = pd.concat([new_train_data, dummies],axis=1)

pd.concat([train_labels, train_data], axis=1).to_csv(r"D:\Synced Folder\Pattern Recognition-Term Project\Datasets-01\check.csv", sep=',')

# test data
for cName in test_data:
    if (test_data[cName]>5).any() or test_data[cName].dtype == np.float:
        test_data[cName] = round(pd.DataFrame(minMaxScaler.fit_transform(test_data[cName].values.reshape(-1,1))),6)
        #test_data[cName] = pd.DataFrame(stdScaler.fit_transform(test_data[cName].values.reshape(-1,1)))

#for cName in test_data:
#    if (test_data[cName]>5).any() or test_data[cName].dtype == np.float:
#        disc = preprocessing.KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='kmeans')
#        disc.fit(test_data[cName].values.reshape(-1,1))
#        sum(test_data[cName]) = disc.transform(test_data[cName].values.reshape(-1,1))

new_test_data=[]
new_test_data=pd.DataFrame(new_test_data)
for cName in test_data:
    if test_data[cName].isin(range(0,2)).all():
        new_test_data = pd.concat([new_test_data, test_data[cName]],axis=1)
    elif test_data[cName].isin(range(1,3)).all(): 
        new_test_data = pd.concat([new_test_data, test_data[cName] - 1],axis=1)
    elif (test_data[cName]>=0).all() and (test_data[cName]<=1).all():
        print(cName)
        new_test_data = pd.concat([new_test_data, test_data[cName]],axis=1)
    else:
        dummies = pd.get_dummies(test_data[cName])
        names=[]
        for count in range(0, dummies.shape[1]):
            names.extend([cName+str(count)])
        dummies.columns = names
        new_test_data = pd.concat([new_test_data, dummies],axis=1)

pd.concat([test_labels, new_test_data], axis=1).to_csv(r"D:\Synced Folder\Pattern Recognition-Term Project\Datasets-01\check.csv", sep=',')

train_data = train_data.values.astype(float)
test_data = test_data.values.astype(float)
train_labels = train_labels.values.astype(float)
test_labels = test_labels.values.astype(float)

#pca = PCA(n_components=10, copy=True)
#A=pca.fit_transform(new_train_data)


###############################################################################

########## SVM
classifier = svm.SVC(kernel='linear',probability=True)
fittedModel = classifier.fit(train_data, train_labels)
SVM_predicted_labels = fittedModel.predict(test_data)

SVMaccuracy = metrics.accuracy_score(test_labels, SVM_predicted_labels)
SVMPRF = metrics.precision_recall_fscore_support(test_labels, SVM_predicted_labels)
SVMPrecision = np.mean(SVMPRF[0])
SVMRecall = np.mean(SVMPRF[1])
SVMFScore = (2*SVMRecall*SVMPrecision)/(SVMRecall+SVMPrecision)
SVM_pred_prob = classifier.predict_proba(test_data)
SVMfpr, SVMtpr, _ = metrics.roc_curve(test_labels, SVM_pred_prob[:,1], pos_label=1)
SVM_AUC = round(metrics.roc_auc_score(test_labels, SVM_predicted_labels),4)    
print("SVM Done!")

########## KNN
KNNModel = KNeighborsClassifier(n_neighbors=9)
KNNModel.fit(train_data, train_labels) 
KNN_predicted_labels = KNNModel.predict(test_data)

KNNaccuracy = metrics.accuracy_score(test_labels, KNN_predicted_labels)
KNNPRF = metrics.precision_recall_fscore_support(test_labels, KNN_predicted_labels)
KNNPrecision = np.mean(KNNPRF[0])
KNNRecall = np.mean(KNNPRF[1])
KNNFScore = (2*KNNRecall*KNNPrecision)/(KNNRecall+KNNPrecision)
KNN_pred_prob = KNNModel.predict_proba(test_data)
KNNfpr, KNNtpr, _ = metrics.roc_curve(test_labels, KNN_pred_prob[:,1], pos_label=1)
KNN_AUC = round(metrics.roc_auc_score(test_labels, KNN_predicted_labels),4)    
print("KNN Done!")

########## Logistic Regression
logistic = LogisticRegression(solver='lbfgs',multi_class='ovr')
logistic.fit(train_data, train_labels)
LG_predicted_labels = logistic.predict(test_data)

LGaccuracy = metrics.accuracy_score(test_labels, LG_predicted_labels)
LGPRF = metrics.precision_recall_fscore_support(test_labels, LG_predicted_labels)
LGPrecision = np.mean(LGPRF[0])
LGRecall = np.mean(LGPRF[1])
LGFScore = (2*LGRecall*LGPrecision)/(LGRecall+LGPrecision)
LG_pred_prob = logistic.predict_proba(test_data)
LGfpr, LGtpr, _ = metrics.roc_curve(test_labels, LG_pred_prob[:,1], pos_label=1)
LG_AUC = round(metrics.roc_auc_score(test_labels, LG_predicted_labels),4)
print("Logistic Done!")

########## Naive Bayes
naiveBayes = GaussianNB()
naiveBayes.fit(train_data, train_labels)
NB_predicted_labels = naiveBayes.predict(test_data)

NBaccuracy = metrics.accuracy_score(test_labels, NB_predicted_labels)
NBPRF = metrics.precision_recall_fscore_support(test_labels, NB_predicted_labels)
NBPrecision = np.mean(NBPRF[0])
NBRecall = np.mean(NBPRF[1])
NBFScore = (2*NBRecall*NBPrecision)/(NBRecall+NBPrecision)
NB_pred_prob = naiveBayes.predict_proba(test_data)
NBfpr, NBtpr, _ = metrics.roc_curve(test_labels, NB_pred_prob[:,1], pos_label=1)
NB_AUC = round(metrics.roc_auc_score(test_labels, NB_predicted_labels),4)
print("Naive Bayes Done!")

###############################################################################
# SVM PLOT
# to draw x=y line
center=[0,1]
one = [0,1]
# define legends
blue_patch = mpatches.Patch(color='blue', label='ROC curve (area = %.4f)'%SVM_AUC)
plt.plot(SVMfpr, SVMtpr,color='blue')
plt.plot(center,one,'--',color='k')
plt.xlim([0,1])
plt.ylim([0,1.01])
plt.legend(handles=[blue_patch], loc=4)
plt.title('ROC Curve, SVM Classifier')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.savefig('(Reversed)ROC SVM-baseline.png', format='png', bbox_inches = 'tight', dpi=1200)
plt.show()


# KNN PLOT
# define legends
blue_patch = mpatches.Patch(color='blue', label='ROC curve (area = %.4f)'%KNN_AUC)
plt.plot(KNNfpr, KNNtpr,color='blue')
plt.plot(center,one,'--',color='k')
plt.xlim([0,1])
plt.ylim([0,1.01])
plt.legend(handles=[blue_patch], loc=4)
plt.title('ROC Curve, KNN Classifier (k=9)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.savefig('(Reversed)ROC KNN-baseline.png', format='png', bbox_inches = 'tight', dpi=1200)
plt.show()


# Logistic Reg. PLOT
# define legends
blue_patch = mpatches.Patch(color='blue', label='ROC curve (area = %.4f)'%LG_AUC)
plt.plot(LGfpr, LGtpr,color='blue')
plt.plot(center,one,'--',color='k')
plt.xlim([0,1])
plt.ylim([0,1.01])
plt.legend(handles=[blue_patch], loc=4)
plt.title('ROC Curve, Logistic Regression Classifier')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.savefig('(Reversed)ROC LG-baseline.png', format='png', bbox_inches = 'tight', dpi=1200)
plt.show()


# Naive Bayes PLOT
# define legends
blue_patch = mpatches.Patch(color='blue', label='ROC curve (area = %.4f)'%NB_AUC)
plt.plot(NBfpr, NBtpr,color='blue')
plt.plot(center,one,'--',color='k')
plt.xlim([0,1])
plt.ylim([0,1.01])
plt.legend(handles=[blue_patch], loc=4)
plt.title('ROC Curve, Naive Bayes Classifier')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.savefig('(Reversed)ROC NB-baseline.png', format='png', bbox_inches = 'tight', dpi=1200)
plt.show()


print("###### SVM #######")
print("Accuracy: %.4f" %(SVMaccuracy*100))
print("Precision: %.4f" %SVMPrecision)
print("Recall: %.4f" %SVMRecall)
print("f_score: %.4f" %SVMFScore)
print("AUC: %.4f" %SVM_AUC)

print("###### KNN #######")
print("Accuracy: %.4f" %(KNNaccuracy*100))
print("Precision: %.4f" %KNNPrecision)
print("Recall: %.4f" %KNNRecall)
print("f_score: %.4f" %KNNFScore)
print("AUC: %.4f" %KNN_AUC)

print("###### LG #######")
print("Accuracy: %.4f" %(LGaccuracy*100))
print("Precision: %.4f" %LGPrecision)
print("Recall: %.4f" %LGRecall)
print("f_score: %.4f" %LGFScore)
print("AUC: %.4f" %LG_AUC)

print("###### NB #######")
print("Accuracy: %.4f" %(NBaccuracy*100))
print("Precision: %.4f" %NBPrecision)
print("Recall: %.4f" %NBRecall)
print("f_score: %.4f" %NBFScore)
print("AUC: %.4f" %NB_AUC)


#skplt.metrics.plot_roc(test_labels, SVM_pred_prob, title="SVM classifier", plot_micro=False)
#plt.savefig('ROC SVM-baseline.png', format='png', bbox_inches = 'tight', dpi=1200)
