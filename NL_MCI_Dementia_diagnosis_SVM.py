import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix

# 定义新的标签转换函数
def convert_labels(label):
    label_mapping = {
        'NL': 1,
        'Dementia': 3,
        'MCI': 2,
        'NL to MCI': 2,
        'MCI to Dementia': 3,
        'MCI to NL': 1,
        'Dementia to MCI': 2,
        'NL to Dementia': 3
    }
    if label in label_mapping:
        return label_mapping[label]
    else:
        print(f"Unexpected label: {label}")
        return -1

# 处理 DX 列缺失值的新函数
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



##首先计算每类患者各种数据的均值，用来填充缺失值##
# 读取数据文件
data1 = pd.read_csv("TADPOLE_D1_D2.csv")
# 转换特征列字符为数值型
data1['ABETA_UPENNBIOMK9_04_19_17'] = pd.to_numeric(data1['ABETA_UPENNBIOMK9_04_19_17'], errors='coerce')
data1['TAU_UPENNBIOMK9_04_19_17'] = pd.to_numeric(data1['TAU_UPENNBIOMK9_04_19_17'], errors='coerce')
data1['PTAU_UPENNBIOMK9_04_19_17'] = pd.to_numeric(data1['PTAU_UPENNBIOMK9_04_19_17'], errors='coerce')
data1['CDRSB'] = pd.to_numeric(data1['CDRSB'], errors='coerce')
data1['APOE4'] = pd.to_numeric(data1['APOE4'], errors='coerce')
data1['AGE'] = pd.to_numeric(data1['AGE'], errors='coerce')
data1['Hippocampus'] = pd.to_numeric(data1['Hippocampus'], errors='coerce')
data1['RAVLT_immediate'] = pd.to_numeric(data1['RAVLT_immediate'], errors='coerce')
data1['ADAS11'] = pd.to_numeric(data1['ADAS11'], errors='coerce')
data1['MidTemp'] = pd.to_numeric(data1['MidTemp'], errors='coerce')
data1['Entorhinal'] = pd.to_numeric(data1['Entorhinal'], errors='coerce')
data1['WholeBrain'] = pd.to_numeric(data1['WholeBrain'], errors='coerce')
data1['FDG'] = pd.to_numeric(data1['FDG'], errors='coerce')
#标签列处理和提取
#DX列标签缺失的替换
data1['DX'].fillna(0, inplace=True)
pd.set_option('display.max_rows', None)
#print(data['DX'].head(3104))
data1['DX'] = data1.apply(fill_missing_values, axis=1)
pd.set_option('display.max_rows', None)
#print(data['DX'].head(3104))
# 转换 DX 列的标签为数字
data1['DX'] = data1['DX'].apply(convert_labels)
labels = data1['DX']
pd.set_option('display.max_rows', None)
# 提取需要的列
columns = ['CDRSB', 'ADAS11', 'RAVLT_immediate', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'MidTemp', 'FDG', 'ABETA_UPENNBIOMK9_04_19_17', 'TAU_UPENNBIOMK9_04_19_17', 'PTAU_UPENNBIOMK9_04_19_17']
data1 = data1[columns + ['DX']]
# 剔除缺失值
data1 = data1.dropna(subset=columns)
# 按照 DX 类别分别计算均值
mean_values_nl = data1[data1['DX'] == 1].mean()
mean_values_dementia = data1[data1['DX'] == 3].mean()
mean_values_mci = data1[data1['DX'] == 2].mean()
print("NL 类的均值：")
print(mean_values_nl)
print("Dementia 类的均值：")
print(mean_values_dementia)
print("MCI 类的均值：")
print(mean_values_mci)
##首先计算每类患者各种数据的均值，用来填充缺失值##





# 重新读取数据
data = pd.read_csv("TADPOLE_D1_D2.csv")

# 转换特征列字符为数值型
data['ABETA_UPENNBIOMK9_04_19_17'] = pd.to_numeric(data['ABETA_UPENNBIOMK9_04_19_17'], errors='coerce')
data['TAU_UPENNBIOMK9_04_19_17'] = pd.to_numeric(data['TAU_UPENNBIOMK9_04_19_17'], errors='coerce')
data['PTAU_UPENNBIOMK9_04_19_17'] = pd.to_numeric(data['PTAU_UPENNBIOMK9_04_19_17'], errors='coerce')
data['CDRSB'] = pd.to_numeric(data['CDRSB'], errors='coerce')
data['APOE4'] = pd.to_numeric(data['APOE4'], errors='coerce')
data['AGE'] = pd.to_numeric(data['AGE'], errors='coerce')
data['Hippocampus'] = pd.to_numeric(data['Hippocampus'], errors='coerce')
data['RAVLT_immediate'] = pd.to_numeric(data['RAVLT_immediate'], errors='coerce')
data['ADAS11'] = pd.to_numeric(data['ADAS11'], errors='coerce')
data['MidTemp'] = pd.to_numeric(data['MidTemp'], errors='coerce')
data['Entorhinal'] = pd.to_numeric(data['Entorhinal'], errors='coerce')
data['WholeBrain'] = pd.to_numeric(data['WholeBrain'], errors='coerce')
data['FDG'] = pd.to_numeric(data['FDG'], errors='coerce')

#标签列处理和提取
#DX列标签缺失的替换
data['DX'].fillna(0, inplace=True)
pd.set_option('display.max_rows', None)
#print(data['DX'].head(3104))
data['DX'] = data.apply(fill_missing_values, axis=1)
pd.set_option('display.max_rows', None)
#print(data['DX'].head(3104))
# 转换 DX 列的标签为数字
data['DX'] = data['DX'].apply(convert_labels)
labels = data['DX']
pd.set_option('display.max_rows', None)
#print(labels.head(3104))

# 提取特征并填充特征列的缺失
#features = data[['WholeBrain', 'Hippocampus', 'Entorhinal', 'MidTemp']]
features = data[['AGE', 'APOE4', 'CDRSB', 'ADAS11', 'RAVLT_immediate', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'MidTemp', 'FDG', 'ABETA_UPENNBIOMK9_04_19_17', 'TAU_UPENNBIOMK9_04_19_17', 'PTAU_UPENNBIOMK9_04_19_17']]
for column in ['CDRSB', 'ADAS11', 'RAVLT_immediate', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'MidTemp', 'FDG', 'ABETA_UPENNBIOMK9_04_19_17', 'TAU_UPENNBIOMK9_04_19_17', 'PTAU_UPENNBIOMK9_04_19_17']:
    mask1 = data['DX'] == 1
    mask2 = data['DX'] == 2
    mask3 = data['DX'] == 3
    features.loc[mask1, column] = features.loc[mask1, column].fillna(mean_values_nl[column])
    features.loc[mask2, column] = features.loc[mask2, column].fillna(mean_values_mci[column])
    features.loc[mask3, column] = features.loc[mask3, column].fillna(mean_values_dementia[column])
features['APOE4'].fillna(0, inplace=True)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(features.head(1000))


# 划分训练集和测试集
X_train = features.iloc[:10000]
y_train = labels.iloc[:10000]
X_test = features.iloc[10001:]
y_test = labels.iloc[10001:]

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建并训练 SVM 模型
clf = svm.SVC(kernel='rbf',C=2.1, gamma=0.1, degree=2)
clf.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = clf.predict(X_test)


# 定义预测结果转换函数
def convert_prediction(prediction):
    prediction_mapping = {
        1: 'NL',
        2: 'MCI',
        3: 'Dementia',
    }
    if prediction == -1:
        return '未知'  # 或者其他您认为合适的处理方式
    return prediction_mapping[prediction]

# 转换预测结果
y_pred_converted = [convert_prediction(p) for p in y_pred]

# 打印转换后的预测结果
RID = data['RID'].iloc[10001:]
results = pd.DataFrame({
    'RID': RID,
    'diagnosis prediction': y_pred_converted
})
print(results)


# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率:", accuracy)




# 将标签二值化（多分类问题需要对每个类别单独计算ROC和AUC）
y_test_binarized = label_binarize(y_test, classes=[1, 2, 3])  # 对应 'NL', 'MCI', 'Dementia'
# 提取决策函数的值（每一列对应一个类别的决策值）
y_score = clf.decision_function(X_test)

# 绘制ROC曲线
plt.figure(figsize=(10, 8))
for i, label in enumerate(['NL', 'MCI', 'Dementia']):  # 分别计算每个类别的ROC
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {label} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')  # 添加对角线
plt.title('ROC Curves for SVM Classifier', fontsize=16)
plt.xlabel('False Positive Rate (FPR)', fontsize=14)
plt.ylabel('True Positive Rate (TPR)', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(alpha=0.3)
plt.show()



# 计算特异性
def calculate_specificity(y_true, y_pred, class_label):
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])
    # 提取真负例和假正例
    tn = cm.sum() - (cm[class_label-1, :].sum() + cm[:, class_label-1].sum() - cm[class_label-1, class_label-1])
    fp = cm[:, class_label-1].sum() - cm[class_label-1, class_label-1]
    specificity = tn / (tn + fp)
    return specificity
# 逐类别计算特异性
specificity_results = {}
for class_label, class_name in enumerate(['NL', 'MCI', 'Dementia'], start=1):
    specificity = calculate_specificity(y_test, y_pred, class_label)
    specificity_results[class_name] = specificity
    print(f"Class {class_name} Specificity: {specificity:.2f}")