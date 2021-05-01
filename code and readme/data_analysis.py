# -*- utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings(action='once')
pd.set_option('mode.chained_assignment', None)

##reading data
credit_data = pd.read_csv("./creditcard.csv", encoding = "utf8")

# print(credit_data)

X= credit_data.drop(labels='Class',axis = 1) #Features
y=credit_data.loc[:,'Class'] #Response


#fraud vs. normal transactions
counts = credit_data.Class.value_counts()
normal = counts[0]
fraudulent = counts[1]
normal_percent = (normal/(normal+fraudulent))*100
fraudulent_percent = (fraudulent/(normal+fraudulent))*100
#print('There were {} non-fraudulent transactions ({:.3f}%) and {} fraudulent transactions ({:.3f}%).'.format(normal, normal_percent, fraudulent, fraudulent_percent))



##normal_fraud_pie_chart
labels = 'normal', 'fraudulent'
sizes = [284315, 492]

fig1, ax1 = plt.subplots()
ax1.pie(sizes,  labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Normal_Fraud_Pie_Chart')
plt.savefig('./run_picture/normal_fraud_pie_chart.jpg')
# plt.show()


#visualizations of time and amount
plt.figure(figsize=(12,8))
plt.title('Distribution of Time Feature')
sns.distplot(credit_data.Time)
plt.savefig('./run_picture/Distribution of Time Feature.jpg')
# plt.show()

plt.figure(figsize=(12,8))
plt.title('Distribution of Monetary Value Feature')
sns.distplot(credit_data["Amount"])
plt.savefig('./run_picture/Distribution of Monetary Value Feature.jpg')
# plt.show()
#heatmap
corr = credit_data.corr()
plt.figure(figsize=(12,10))
heat = sns.heatmap(data=corr)
plt.title('Heatmap of Correlation')
plt.savefig('./run_picture/Heatmap of Correlation.jpg')
plt.show()


scaler_time = StandardScaler()
scaler_amount = StandardScaler()
#scaling time
scaled_time = scaler_time.fit_transform(credit_data[['Time']])
flat_list1 = [item for sublist in scaled_time.tolist() for item in sublist]
scaled_time = pd.Series(flat_list1)

#scaling amount
scaled_amount = scaler_amount.fit_transform(credit_data[['Amount']])
flat_list1 = [item for sublist in scaled_amount.tolist() for item in sublist]
scaled_amount = pd.Series(flat_list1)

#concatenating newly created columns w original credit_data
credit_data = pd.concat([credit_data, scaled_amount.rename('scaled_amount'), scaled_time.rename('scaled_time')], axis=1)


#dropping old amount and time columns
credit_data.drop(['Amount', 'Time'], axis=1, inplace=True)
# print(credit_data)
