import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

# 定义新的标签转换函数
def convert_labels(label):
    label_mapping = {
        'NL': 0,
        'Dementia': 1,
        'MCI': 0,
        'NL to MCI': 1,
        'MCI to Dementia': 1,
        'MCI to NL': 0,
        'Dementia to MCI': 0,
        'NL to Dementia': 1
    }
    if label in label_mapping:
        return label_mapping[label]
    else:
        print(f"Unexpected label: {label}")
        return -1  

# 处理 DX 列缺失值的函数
def fill_missing_values(row):
    if row['DX'] == 0:
        if row['DX_bl'] == 'CN':
            return 'NL'
        elif row['DX_bl'] == 'AD':
            return'Dementia'
        elif row['DX_bl'] in ['LMCI', 'EMCI', 'SMC']:
            return 'MCI'
        elif pd.isnull(row['DX_bl']):
            return '无诊断'
        else:
            print(f"Unrecognized value in DX_bl: {row['DX_bl']}")
            return -1
    else:
        return row['DX']

# DX_bl列标签转换函数
def convert2_labels(label):
    label_mapping = {
        'CN': 2,
        'SMC': 3,
        'LMCI': 4,
        'EMCI': 5,
    }
    if label in label_mapping:
        return label_mapping[label]
    else:
        print(f"Unexpected label: {label}")
        return -1

##首先计算每类患者各种数据的均值，用来填充缺失值##
# 读取数据文件
data1 = pd.read_csv("TADPOLE_D1_D2.csv")
# 转换特征列字符为数值型
data1['FDG_bl'] = pd.to_numeric(data1['FDG_bl'], errors='coerce')
data1['CDRSB_bl'] = pd.to_numeric(data1['CDRSB_bl'], errors='coerce')
data1['ADAS11_bl'] = pd.to_numeric(data1['ADAS11_bl'], errors='coerce')
data1['MMSE_bl'] = pd.to_numeric(data1['MMSE_bl'], errors='coerce')
data1['APOE4'] = pd.to_numeric(data1['APOE4'], errors='coerce')
data1['AGE'] = pd.to_numeric(data1['AGE'], errors='coerce')
data1['RAVLT_immediate_bl'] = pd.to_numeric(data1['RAVLT_immediate_bl'], errors='coerce')
data1['RAVLT_learning_bl'] = pd.to_numeric(data1['RAVLT_learning_bl'], errors='coerce')
data1['RAVLT_forgetting_bl'] = pd.to_numeric(data1['RAVLT_forgetting_bl'], errors='coerce')
data1['RAVLT_perc_forgetting_bl'] = pd.to_numeric(data1['RAVLT_perc_forgetting_bl'], errors='coerce')
data1['FAQ_bl'] = pd.to_numeric(data1['FAQ_bl'], errors='coerce')
data1['Hippocampus_bl'] = pd.to_numeric(data1['Hippocampus_bl'], errors='coerce')
data1['WholeBrain_bl'] = pd.to_numeric(data1['WholeBrain_bl'], errors='coerce')
data1['Entorhinal_bl'] = pd.to_numeric(data1['Entorhinal_bl'], errors='coerce')
data1['Fusiform_bl'] = pd.to_numeric(data1['Fusiform_bl'], errors='coerce')
data1['MidTemp_bl'] = pd.to_numeric(data1['MidTemp_bl'], errors='coerce')
data1['PTAU_UPENNBIOMK9_04_19_17'] = pd.to_numeric(data1['PTAU_UPENNBIOMK9_04_19_17'], errors='coerce')
data1['TAU_UPENNBIOMK9_04_19_17'] = pd.to_numeric(data1['TAU_UPENNBIOMK9_04_19_17'], errors='coerce')
data1['ABETA_UPENNBIOMK9_04_19_17'] = pd.to_numeric(data1['ABETA_UPENNBIOMK9_04_19_17'], errors='coerce')
data1['MOCA_bl'] = pd.to_numeric(data1['MOCA_bl'], errors='coerce')
data1['ICV_bl'] = pd.to_numeric(data1['ICV_bl'], errors='coerce')
# 转换 DX 列的标签为数字
data1['DX_bl'] = data1['DX_bl'].apply(convert2_labels)
labels = data1['DX_bl']
#pd.set_option('display.max_rows', None)
# 提取需要的列
columns = ['AGE', 'APOE4', 'FDG_bl', 'CDRSB_bl', 'ADAS11_bl', 'MMSE_bl', 'RAVLT_immediate_bl', 'RAVLT_learning_bl', 'RAVLT_forgetting_bl', 'RAVLT_perc_forgetting_bl', 'FAQ_bl', 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl', 'ICV_bl', 'MOCA_bl', 'ABETA_UPENNBIOMK9_04_19_17', 'TAU_UPENNBIOMK9_04_19_17', 'PTAU_UPENNBIOMK9_04_19_17']
data1 = data1[columns + ['DX_bl']]
# 剔除缺失值
data1 = data1.dropna(subset=columns)
# 按照 DX 类别分别计算均值
mean_values_CN = data1[data1['DX_bl'] == 2].mean()
mean_values_SMC = data1[data1['DX_bl'] == 3].mean()
mean_values_LMCI = data1[data1['DX_bl'] == 4].mean()
mean_values_EMCI = data1[data1['DX_bl'] == 5].mean()
print("CN 类的均值：")
print(mean_values_CN)
print("SMC 类的均值：")
print(mean_values_SMC)
print("LMCI 类的均值：")
print(mean_values_LMCI)
print("EMCI 类的均值：")
print(mean_values_EMCI)
##首先计算每类患者各种数据的均值，用来填充缺失值##



# 重新读取 CSV 文件
data = pd.read_csv("TADPOLE_D1_D2.csv")

# 提取特定列和特定行的数据
selected_data = data[data['DX_bl'].isin(['CN', 'EMCI', 'LMCI', 'SMC'])][['AGE', 'APOE4', 'FDG_bl', 'CDRSB_bl', 'ADAS11_bl', 'MMSE_bl', 'RAVLT_immediate_bl', 'RAVLT_learning_bl', 'RAVLT_forgetting_bl', 'RAVLT_perc_forgetting_bl', 'FAQ_bl', 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl', 'ICV_bl', 'MOCA_bl', 'ABETA_UPENNBIOMK9_04_19_17', 'TAU_UPENNBIOMK9_04_19_17', 'PTAU_UPENNBIOMK9_04_19_17', 'RID', 'DX_bl', 'DX']]
# 打印提取后的数据框
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(selected_data)

# 转换特征列字符为数值型
selected_data['FDG_bl'] = pd.to_numeric(selected_data['FDG_bl'], errors='coerce')
selected_data['CDRSB_bl'] = pd.to_numeric(selected_data['CDRSB_bl'], errors='coerce')
selected_data['ADAS11_bl'] = pd.to_numeric(selected_data['ADAS11_bl'], errors='coerce')
selected_data['MMSE_bl'] = pd.to_numeric(selected_data['MMSE_bl'], errors='coerce')
selected_data['APOE4'] = pd.to_numeric(selected_data['APOE4'], errors='coerce')
selected_data['AGE'] = pd.to_numeric(selected_data['AGE'], errors='coerce')
selected_data['RAVLT_immediate_bl'] = pd.to_numeric(selected_data['RAVLT_immediate_bl'], errors='coerce')
selected_data['RAVLT_learning_bl'] = pd.to_numeric(selected_data['RAVLT_learning_bl'], errors='coerce')
selected_data['RAVLT_forgetting_bl'] = pd.to_numeric(selected_data['RAVLT_forgetting_bl'], errors='coerce')
selected_data['RAVLT_perc_forgetting_bl'] = pd.to_numeric(selected_data['RAVLT_perc_forgetting_bl'], errors='coerce')
selected_data['FAQ_bl'] = pd.to_numeric(selected_data['FAQ_bl'], errors='coerce')
selected_data['Hippocampus_bl'] = pd.to_numeric(selected_data['Hippocampus_bl'], errors='coerce')
selected_data['WholeBrain_bl'] = pd.to_numeric(selected_data['WholeBrain_bl'], errors='coerce')
selected_data['Entorhinal_bl'] = pd.to_numeric(selected_data['Entorhinal_bl'], errors='coerce')
selected_data['Fusiform_bl'] = pd.to_numeric(selected_data['Fusiform_bl'], errors='coerce')
selected_data['MidTemp_bl'] = pd.to_numeric(selected_data['MidTemp_bl'], errors='coerce')
selected_data['PTAU_UPENNBIOMK9_04_19_17'] = pd.to_numeric(selected_data['PTAU_UPENNBIOMK9_04_19_17'], errors='coerce')
selected_data['TAU_UPENNBIOMK9_04_19_17'] = pd.to_numeric(selected_data['TAU_UPENNBIOMK9_04_19_17'], errors='coerce')
selected_data['ABETA_UPENNBIOMK9_04_19_17'] = pd.to_numeric(selected_data['ABETA_UPENNBIOMK9_04_19_17'], errors='coerce')
selected_data['MOCA_bl'] = pd.to_numeric(selected_data['MOCA_bl'], errors='coerce')
selected_data['ICV_bl'] = pd.to_numeric(selected_data['ICV_bl'], errors='coerce')
selected_data['RID'] = pd.to_numeric(selected_data['RID'], errors='coerce')

#标签列处理
#DX列标签缺失的替换
selected_data['DX'].fillna(0, inplace=True)
#pd.set_option('display.max_rows', None)
#print(selected_data['DX'].head(3104))
selected_data['DX'] = selected_data.apply(fill_missing_values, axis=1)
#pd.set_option('display.max_rows', None)
#print(selected_data['DX'].head(3104))
# 转换 DX 列的标签为数字
selected_data['DX'] = selected_data['DX'].apply(convert_labels)
labels = selected_data['DX']
pd.set_option('display.max_rows', None)
print(labels.head(3104))


# 提取特征并填充特征列的缺失
# 转换 DX_bl 列的类别为数字
selected_data['DX_bl'] = selected_data['DX_bl'].apply(convert2_labels)
features = selected_data[['AGE', 'APOE4', 'FDG_bl', 'CDRSB_bl', 'ADAS11_bl', 'MMSE_bl', 'RAVLT_immediate_bl', 'RAVLT_learning_bl', 'RAVLT_forgetting_bl', 'RAVLT_perc_forgetting_bl', 'FAQ_bl', 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl', 'ICV_bl', 'MOCA_bl', 'ABETA_UPENNBIOMK9_04_19_17', 'TAU_UPENNBIOMK9_04_19_17', 'PTAU_UPENNBIOMK9_04_19_17']]
for column in ['AGE', 'APOE4', 'FDG_bl', 'CDRSB_bl', 'ADAS11_bl', 'MMSE_bl', 'RAVLT_immediate_bl', 'RAVLT_learning_bl', 'RAVLT_forgetting_bl', 'RAVLT_perc_forgetting_bl', 'FAQ_bl', 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl', 'ICV_bl', 'MOCA_bl', 'ABETA_UPENNBIOMK9_04_19_17', 'TAU_UPENNBIOMK9_04_19_17', 'PTAU_UPENNBIOMK9_04_19_17']:
    mask1 = selected_data['DX_bl'] == 2
    mask2 = selected_data['DX_bl'] == 3
    mask3 = selected_data['DX_bl'] == 4
    mask4 = selected_data['DX_bl'] == 5
    features.loc[mask1, column] = features.loc[mask1, column].fillna(mean_values_CN[column])
    features.loc[mask2, column] = features.loc[mask2, column].fillna(mean_values_SMC[column])
    features.loc[mask3, column] = features.loc[mask3, column].fillna(mean_values_LMCI[column])
    features.loc[mask4, column] = features.loc[mask4, column].fillna(mean_values_EMCI[column])
features['APOE4'].fillna(0, inplace=True)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# print(features.head(1000))
print(features)

# 划分训练集和测试集
X_train = features.iloc[:10000]
y_train = labels.iloc[:10000]
X_test = features.iloc[10001:]
y_test = labels.iloc[10001:]

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义参数网格
param_grid = {
    'C': [0.1, 1, 1.5, 2, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'degree': [2, 3, 4]
}

# 创建并训练 SVM 模型，使用网格搜索调参
clf = svm.SVC()
grid_search = GridSearchCV(clf, param_grid, scoring='accuracy', cv=5, verbose=2)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)
# 获取最佳参数和模型
best_params = grid_search.best_params_
best_clf = grid_search.best_estimator_

# 在测试集上进行预测
y_pred = best_clf.predict(X_test)

# 定义预测结果转换函数
def convert_prediction(prediction):
    prediction_mapping = {
        0: 'keep stable',
        1: 'may convert to AD or be progressive',
    }
    if prediction == -1:
        return '未知'
    return prediction_mapping[prediction]

# 定义DX_bl分类转换函数
def convert2_prediction(DX_bl):
    prediction_mapping = {
        2: 'CN',
        3: 'SMC',
        4: 'LMCI',
        5: 'EMCI',
    }
    if DX_bl == -1:
        return '未知'
    return prediction_mapping[DX_bl]

# 转换预测结果
y_pred_converted = [convert_prediction(p) for p in y_pred]

# 打印转换后的预测结果
RID = selected_data['RID'].iloc[10001:]
selected_data['DX_bl'] = [convert2_prediction(q) for q in selected_data['DX_bl']]
DX_bl = selected_data['DX_bl'].iloc[10001:]
results = pd.DataFrame({
    'RID': RID,
    'DX_bl': DX_bl,
    'prediction': y_pred_converted
})
print(results)



# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率:", accuracy)

from sklearn.metrics import roc_curve, auc

# 计算 ROC 曲线和 AUC 值
fpr, tpr, thresholds = roc_curve(y_test, best_clf.decision_function(X_test))
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# 计算特异性
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
print("特异性:", specificity)
