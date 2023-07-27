import numpy as np
import pandas as pd

dataset_url = r"D:\Synced Folder\Pattern Recognition-Term Project\NHANES0910_Raw_Extended_WekaModel.csv"
dataset = pd.read_csv(dataset_url, dtype=str)
train_dataset = dataset.copy()
test_dataset = dataset.copy()

train_indexes = np.random.choice(4322, 2161, replace=False)
test_indexes = np.array([index for index in range(0,4322) if index not in train_indexes])

train_dataset.drop(train_dataset.index[train_indexes], axis=0, inplace=True)
test_dataset.drop(test_dataset.index[test_indexes], axis=0, inplace=True)
False in train_dataset == test_dataset
train_dataset.to_csv(r"D:\Synced Folder\Pattern Recognition-Term Project\train_Data.csv", sep=',')
test_dataset.to_csv(r"D:\Synced Folder\Pattern Recognition-Term Project\test_Data.csv", sep=',')


###############################################################################
###############################################################################
###############################################################################
import numpy as np
import pandas as pd

train_url = r"D:\Synced Folder\Pattern Recognition-Term Project\Datasets-01\00-train_UR_IM_CS.csv"
test_url = r"D:\Synced Folder\Pattern Recognition-Term Project\Datasets-01\00-test_UR.csv"

train_data = pd.read_csv(train_url, dtype=str)
test_data = pd.read_csv(test_url, dtype=str)

train_features = train_data.columns

new_test_data=[]
new_test_data=pd.DataFrame(new_test_data)

for feature in train_features:
        new_test_data = pd.concat([new_test_data, test_data[feature]],axis=1)
        
new_test_data.to_csv(r"D:\Synced Folder\Pattern Recognition-Term Project\Datasets-01\00-test_UR_CS.csv", sep=',')

###############################################################################
###############################################################################
###############################################################################

import numpy as np
import pandas as pd

train_url = r"D:\Synced Folder\Pattern Recognition-Term Project\Datasets-01\01-test_UR_IM_PC.csv"
test_url = r"D:\Synced Folder\Pattern Recognition-Term Project\Datasets-01\00-train_UR.csv"

train_data = pd.read_csv(train_url, dtype=str)
test = pd.read_csv(test_url, dtype=str)

attr=[]
attr.append(round((-0.261*test['Medicine'].astype(float)-0.162*test['PFQ051'].astype(float)-0.161*test['MCQ160A'].astype(float)-0.157*test['CDQ010'].astype(float)-0.156*test['HUQ050'].astype(float)),4))
attr.append(round((-0.206*test['DBQ750'].astype(float)-0.201*test['CBQ685'].astype(float)-0.188*test['DBQ770'].astype(float)-0.185*test['CBD715'].astype(float)-0.184*test['CBD735'].astype(float)),4))
attr.append(round((0.165*test['DBQ770'].astype(float)+0.164*test['CBQ685'].astype(float)+0.157*test['CBD720'].astype(float)+0.154*test['CBD725'].astype(float)-0.15*test['DPQ020'].astype(float)),4))
attr.append(round((0.321*test['d00867'].astype(float)+0.321*test['d00608'].astype(float)+0.321*test['d00275'].astype(float)+0.258*test['d00255'].astype(float)+0.214*test['d00254'].astype(float)),4))
attr.append(round((0.399*test['d07048'].astype(float)+0.399*test['d05352'].astype(float)+0.399*test['d03984'].astype(float)+0.399*test['d05825'].astype(float)+0.399*test['d07076'].astype(float)),4))
attr.append(round((0.246*test['d04513'].astype(float)-0.246*test['d00086'].astype(float)-0.214*test['d04011'].astype(float)-0.185*test['d04378'].astype(float)-0.163*test['d00265'].astype(float)),4))
attr.append(round((-0.224*test['d04513'].astype(float)-0.224*test['d00086'].astype(float)-0.221*test['Gender'].astype(float)-0.216*test['d04011'].astype(float)-0.204*test['RHQ420'].astype(float)),4))
attr.append(round((0.372*test['d01290'].astype(float)+0.372*test['d04784'].astype(float)+0.372*test['d04284'].astype(float)+0.264*test['d00033'].astype(float)+0.231*test['d00350'].astype(float)),4))
attr.append(round((0.424*test['d05612'].astype(float)+0.424*test['d05691'].astype(float)+0.424*test['d04235'].astype(float)+0.344*test['d04141'].astype(float)+0.263*test['d04145'].astype(float)),4))
attr.append(round((-0.165*test['DBQ235B'].astype(float)-0.164*test['DBQ235C'].astype(float)+0.162*test['a51865'].astype(float)+0.162*test['d00206'].astype(float)-0.161*test['DBQ235A'].astype(float)),4))

newTest = pd.DataFrame(attr).transpose()
newTest = pd.concat([test['Classlabel'], newTest], axis=1)
newTest.to_csv(r"D:\Synced Folder\Pattern Recognition-Term Project\Datasets-01\01-train_UR_PC.csv", sep=',')


#test ----> train

attr.append(round((-0.253*test['Medicine'].astype(float)-0.188*test['PFQ051'].astype(float)-0.17*test['Age'].astype(float)-0.159*test['CDQ010'].astype(float)-0.158*test['PFQ049'].astype(float)),4))
attr.append(round((-0.24*test['DBQ750'].astype(float)-0.231*test['CBQ685'].astype(float)-0.227*test['DBQ760'].astype(float)-0.226*test['DBQ770'].astype(float)-0.215*test['DBQ780'].astype(float)),4))
attr.append(round((0.18*test['DPQ020'].astype(float)+0.172*test['DPQ060'].astype(float)+0.155*test['DPQ100'].astype(float)+0.151*test['DPQ010'].astype(float)-0.147*test['Age'].astype(float)),4))
attr.append(round((0.324*test['Gender'].astype(float)+0.303*test['RHQ131'].astype(float)+0.29*test['RHQ420'].astype(float)-0.213*test['OCQ510'].astype(float)-0.199*test['OCQ550'].astype(float)),4))
attr.append(round((0.221*test['d00749'].astype(float)+0.189*test['RDQ070'].astype(float)+0.177*test['d04611'].astype(float)+0.175*test['MCQ010'].astype(float)+0.151*test['RDQ090'].astype(float)),4))
attr.append(round((0.448*test['d04785'].astype(float)+0.448*test['d00363'].astype(float)+0.448*test['d04899'].astype(float)+0.243*test['d03434'].astype(float)+0.147*test['d04812'].astype(float)),4))
attr.append(round((0.223*test['DBQ235B'].astype(float)+0.22*test['DBQ235C'].astype(float)+0.216*test['DBQ235A'].astype(float)+0.161*test['DBQ197'].astype(float)-0.147*test['d00749'].astype(float)),4))
attr.append(round((0.266*test['DBQ235B'].astype(float)+0.261*test['DBQ235A'].astype(float)+0.254*test['DBQ235C'].astype(float)-0.188*test['d04109'].astype(float)-0.178*test['d04750'].astype(float)),4))
attr.append(round((0.362*test['d05217'].astype(float)+0.362*test['d00119'].astype(float)+0.362*test['d03752'].astype(float)+0.197*test['KIQ022'].astype(float)+0.192*test['d00051'].astype(float)),4))
attr.append(round((-0.218*test['d06842'].astype(float)-0.218*test['d00123'].astype(float)+0.191*test['d00119'].astype(float)+0.191*test['d03752'].astype(float)+0.191*test['d05217'].astype(float)),4))


###############################################################################
###############################################################################
###############################################################################
import numpy as np
import pandas as pd

train_url = r"D:\Synced Folder\Pattern Recognition-Term Project\Datasets\test_Data.csv"
train_data = pd.read_csv(train_url, dtype=str)

train_data = train_data.sample(frac=1).reset_index(drop=True)

train_data.to_csv(r"D:\Synced Folder\Pattern Recognition-Term Project\test_Data.csv", sep=',')
