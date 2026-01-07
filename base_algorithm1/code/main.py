# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 11:19:02 2025

@author: tomorrow
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

data0 = pd.read_csv(r'DR_USA_Roundabout_SR/vehicle_tracks_000.csv')
data1 = pd.read_csv(r'DR_USA_Roundabout_SR/vehicle_tracks_001.csv')
data2 = pd.read_csv(r'DR_USA_Roundabout_SR/vehicle_tracks_002.csv')
data3 = pd.read_csv(r'DR_USA_Roundabout_SR/vehicle_tracks_003.csv')
data4 = pd.read_csv(r'DR_USA_Roundabout_SR/vehicle_tracks_004.csv')
data5 = pd.read_csv(r'DR_USA_Roundabout_SR/vehicle_tracks_005.csv')
data6 = pd.read_csv(r'DR_USA_Roundabout_SR/vehicle_tracks_006.csv')
data7 = pd.read_csv(r'DR_USA_Roundabout_SR/vehicle_tracks_007.csv')
data8 = pd.read_csv(r'DR_USA_Roundabout_SR/vehicle_tracks_008.csv')
data9 = pd.read_csv(r'DR_USA_Roundabout_SR/vehicle_tracks_009.csv')

case0 = pd.read_csv(r'0.csv', sep='\t', header=None).values
case1 = pd.read_csv(r'1.csv', sep='\t', header=None).values
case2 = pd.read_csv(r'2.csv', sep='\t', header=None).values
case3 = pd.read_csv(r'3.csv', sep='\t', header=None).values
case4 = pd.read_csv(r'4.csv', sep='\t', header=None).values
case5 = pd.read_csv(r'5.csv', sep='\t', header=None).values
case6 = pd.read_csv(r'6.csv', sep='\t', header=None).values
case7 = pd.read_csv(r'7.csv', sep='\t', header=None).values
case8 = pd.read_csv(r'8.csv', sep='\t', header=None).values
case9 = pd.read_csv(r'9.csv', sep='\t', header=None).values

# 一共有两种，环岛内先行，环岛外先行
center = np.array([[990.0,1020.0]])
# 1.提取一个片段
allcase = [case0, case1, case2, case3, case4, case5, case6, case7, case8, case9]
alldata = [data0, data1, data2, data3, data4, data5, data6, data7, data8, data9]
insidefirst=[]
outsidefirst=[]
for k in range(len(allcase)):
    case_used = allcase[k]
    data_uesd = alldata[k]
    for case in case_used:
        id_x, id_h, time_start, time_end = case  
        df_x = data_uesd[(data_uesd['track_id'] == id_x) & 
                     (data_uesd['timestamp_ms'] >= time_start) & 
                     (data_uesd['timestamp_ms'] <= time_end)]
        data_x = df_x.loc[:,['x', 'y', 'vx', 'vy', 'psi_rad']].to_numpy()
        data_x = np.column_stack((data_x[:,0:2], np.sqrt(np.sum(data_x[:,2:4]**2, axis=1)), data_x[:,4]))
        
        df_h = data_uesd[(data_uesd['track_id'] == id_h) & 
                     (data_uesd['timestamp_ms'] >= time_start) & 
                     (data_uesd['timestamp_ms'] <= time_end)]
        data_h = df_h.loc[:,['x', 'y', 'vx', 'vy', 'psi_rad']].to_numpy()
        data_h = np.column_stack((data_h[:,0:2], np.sqrt(np.sum(data_h[:,2:4]**2, axis=1)), data_h[:,4]))
        d2c_x = np.linalg.norm(data_x[0,0:2]-center)
        d2c_h = np.linalg.norm(data_h[0,0:2]-center)
        
        pos_h = data_h[-1,0:2]
        lastdist_h2x = np.linalg.norm(pos_h-data_x[:,0:2],axis=1)
        if min(lastdist_h2x)<3:
            # 此时最后一点正好是最近点，就用它
            near_point_x = data_x[np.argmin(lastdist_h2x),0:2]
            near_point_h = pos_h
            dx=0
            dh =0
            for i in range(np.argmin(lastdist_h2x)+1,len(data_h)):
                dh+=np.linalg.norm(data_h[i,0:2]-data_h[i-1,0:2])
            realdata = [[dx, data_x[np.argmin(lastdist_h2x),2], dh, data_h[np.argmin(lastdist_h2x),2]]]
            for i in range(np.argmin(lastdist_h2x)-1,-1,-1):
                dx += np.linalg.norm(data_x[i,0:2]-data_x[i+1,0:2])
                dh += np.linalg.norm(data_h[i,0:2]-data_h[i+1,0:2])
                realdata.append([dx, data_x[i,2], dh, data_h[i,2]])
        else:
            mindist = 100
            px, ph = -1, -1
            for i in range(len(data_h)):
                lastdist_h2x = np.linalg.norm(data_h[i,0:2]-data_x[:,0:2], axis=1)
                if min(lastdist_h2x)< mindist:
                    mindist = min(lastdist_h2x)
                    px = np.argmin(lastdist_h2x)
                    ph = i
                    if mindist < 1:
                        break
            if mindist<3:
                # 是中间两个点
                near_point_x = data_x[px, 0:2]
                near_point_h = data_h[ph, 0:2]
                dx = 0
                dh = 0
                if px<ph:
                    for i in range(px+1,ph+1):
                        dh+=np.linalg.norm(data_h[i,0:2]-data_h[i-1,0:2])
                    realdata = [[dx, data_x[px ,2], dh, data_h[px,2]]]
                    for i in range(px-1, -1, -1):
                        dx += np.linalg.norm(data_x[i,0:2]-data_x[i+1,0:2])
                        dh += np.linalg.norm(data_h[i,0:2]-data_h[i+1,0:2])
                        realdata.append([dx, data_x[i,2], dh, data_h[i,2]])
                else:
                    for i in range(ph,px+1):
                        dx+=np.linalg.norm(data_x[i,0:2]-data_x[i-1,0:2])
                    realdata = [[dx, data_x[ph ,2], dh, data_h[ph,2]]]
                    for i in range(ph-1, -1, -1):
                        dx += np.linalg.norm(data_x[i,0:2]-data_x[i+1,0:2])
                        dh += np.linalg.norm(data_h[i,0:2]-data_h[i+1,0:2])
                        realdata.append([dx, data_x[i,2], dh, data_h[i,2]])
            else:
                delta = 0.5*np.array([np.cos(data_h[-1, 3]), np.sin(data_h[-1, 3])])
                d=0
                newpoint = data_h[-1,0:2]
                flag=-1
                mindist=100
                while mindist>3:
                    d+=0.5
                    newpoint += delta
                    newdist = np.linalg.norm(newpoint - data_x[:,0:2],axis=1)
                    if min(newdist)>=mindist:  # 延伸之后也就那样
                        flag=1
                        break
                    elif min(newdist)<=3:  # 延申之后找到了冲突点
                        pp = np.argmin(newdist)
                        flag=0
                        break
                    else:
                        pp = np.argmin(newdist)
                        mindist = min(newdist)
                if flag==0:
                    dx=0
                    dh=d
                    for i in range(pp+1, len(data_x)):
                        dx+=np.linalg.norm(data_x[i,0:2]-data_x[i-1,0:2])
                    realdata = [[dx, data_x[pp,2], dh, data_h[pp,2]]]
                    for i in range(pp-1,-1,-1):
                        dx += np.linalg.norm(data_x[i,0:2]-data_x[i+1,0:2])
                        dh += np.linalg.norm(data_h[i,0:2]-data_h[i+1,0:2])
                        realdata.append([dx, data_x[i,2], dh, data_h[i,2]])
                elif flag == 1:
                    dx=0
                    dh=d-0.5
                    for i in range(pp+1, len(data_x)):
                        dx+=np.linalg.norm(data_x[i,0:2]-data_x[i-1,0:2])
                    realdata = [[dx, data_x[pp,2], dh, data_h[pp,2]]]
                    for i in range(pp-1,-1,-1):
                        dx += np.linalg.norm(data_x[i,0:2]-data_x[i+1,0:2])
                        dh += np.linalg.norm(data_h[i,0:2]-data_h[i+1,0:2])
                        realdata.append([dx, data_x[i,2], dh, data_h[i,2]])
        if realdata[0][0]>realdata[0][2]:
            continue
        if d2c_x<d2c_h:
            insidefirst.append(np.flipud(realdata))
        else:
            outsidefirst.append(np.flipud(realdata))



## 应该计算合作加速度了

svmdata = []
time1, time2 = [], []
# 对于环岛内部先行来说
for case in insidefirst:
    for t in range(min(100, len(case))):
        tinside = case[t, 0]/(case[t,1]+0.01)  # 内部车到达冲突点的时间
        toutside = case[t, 2]/(case[t,3]+0.01)  # 外部车到达冲突点的时间
        a_in_out = 2*(case[t,2]-case[t,3]*tinside)/(tinside**2+0.01)  # 内部车看外部车
        a_out_in = 2*(case[t,0]-case[t,1]*toutside)/(toutside**2+0.01)  # 外部车看内部车
        if abs(tinside - toutside)<50 and abs(tinside)<40 and abs(toutside)<40:
            svmdata.append([tinside**2,toutside**2, (case[t,0]-case[t,1]*toutside),(case[t,2]-case[t,3]*tinside), 0])
            # svmdata.append([tinside, toutside, case[t,1], case[t,3], 0])
        if t > min(len(case), 100)-10 and abs(tinside)<13 and abs(toutside)<13:
            time1.append(-tinside+toutside)

for case in outsidefirst:
    for t in range(min(100, len(case))):
        tinside = case[t, 2]/(case[t,3]+0.01)  # 内部车到达冲突点的时间
        toutside = case[t, 0]/(case[t,1]+0.01)  # 外部车到达冲突点的时间
        a_in_out = 2*(case[t,0]-case[t,1]*tinside)/(tinside**2+0.01)  # 内部车看外部车
        a_out_in = 2*(case[t,2]-case[t,3]*toutside)/(toutside**2+0.01)  # 外部车看内部车
        if abs(tinside - toutside)<50 and abs(tinside)<40 and abs(toutside)<40:
            svmdata.append([tinside**2, toutside**2, (case[t,2]-case[t,3]*toutside), (case[t,0]-case[t,1]*tinside), 1])
            # svmdata.append([tinside, toutside, case[t,3], case[t,1], 1])
        if t > min(len(case), 100)-10 and abs(tinside)<13 and abs(toutside)<13:
            time2.append(tinside-toutside)
svmdata = np.array(svmdata)



plt.hist(time1, bins=20, density=True, edgecolor='black', alpha=0.5)
plt.title('Histogram of Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

plt.hist(time2, bins=20, density=True, edgecolor='black', alpha=0.5)
plt.title('Histogram of Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

import seaborn as sns
sns.kdeplot(time1, shade=True)  # 'shade=True' 可以填充曲线下方区域
plt.title('Kernel Density Estimate of Data')
plt.xlabel('Value')
plt.ylabel('Density')
sns.kdeplot(time2, fill=True)  # 'shade=True' 可以填充曲线下方区域
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()

'''
x = svmdata[:, 0]
y = svmdata[:, 1]
labels = svmdata[:, -1] # 最后一列作为标签

# 创建散点图
plt.figure(figsize=(8, 6))

for label in np.unique(labels):
    mask = labels == label
    plt.scatter(x[mask], y[mask], label='Priority for {} Vehicle'.format('Inner' if label == 0 else 'Outer'))

# 添加标题和坐标轴标签
plt.title('(A) Scatter Plot of TTCP')
plt.xlabel(r'$TTCP_{in}^2$')  # 使用LaTeX格式设置坐标轴标签
plt.ylabel(r'$TTCP_{out}^2$')

# 显示图例
plt.legend()

# 显示网格
plt.grid(True)

# 显示图形
plt.show()

x = svmdata[:, 2]  
y = svmdata[:, 3] 
labels = svmdata[:, -1] # 最后一列作为标签

# 创建散点图
plt.figure(figsize=(8, 6))

# 绘制两类不同的点，假设最后一列只有0和1两种值
for label in np.unique(labels):
    mask = labels == label
    plt.scatter(x[mask], y[mask], label='Priority for {} Vehicle'.format('Inner' if label == 0 else 'Outer'), alpha=0.7)

# 添加标题和坐标轴标签
plt.title('(B) Scatter Plot of Remaining Space')
plt.xlabel('Space for inside vehicle '+ r'$(m)$')  # 使用LaTeX格式设置坐标轴标签
plt.ylabel('Space for outside vehicle '+ r'$(m)$')

# 显示图例
plt.legend()

# 显示网格
plt.grid(True, alpha=0.5)

# 显示图形
plt.show()


'''




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 分离特征和标签
X = svmdata[:, :-1]  # 自变量/特征
y = svmdata[:, -1]   # 因变量/标签

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建SVM分类器，默认使用RBF核函数
svm_classifier = SVC(kernel='linear')

# 训练SVM模型
svm_classifier.fit(X_train_scaled, y_train)

# 使用模型进行预测
y_pred = svm_classifier.predict(X_test_scaled)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# 打印详细的分类报告
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 打印混淆矩阵
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))



'''

from sklearn.ensemble import RandomForestClassifier
# 分离特征和标签m
X = svmdata[:, :-1]  # 自变量/特征
y = svmdata[:, -1].astype(int)   # 因变量/标签


# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)

# 创建并训练随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=1, random_state=42, max_depth=3)
rf_classifier.fit(X_train, y_train)

# 预测测试集
y_pred = rf_classifier.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


# 获取特征的重要性
importances = rf_classifier.feature_importances_

# 绘制条形图显示每个特征的重要性
indices = np.argsort(importances)[::-1]
plt.figure()
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'], rotation=90)
plt.tight_layout()
plt.show()


from sklearn.tree import plot_tree

# 选择随机森林中的第一个决策树
tree = rf_classifier.estimators_[0]

plt.figure(figsize=(40,20))
plot_tree(tree, feature_names=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'], filled=True, rounded=True, class_names=['Class 0', 'Class 1'])
plt.show()
'''



