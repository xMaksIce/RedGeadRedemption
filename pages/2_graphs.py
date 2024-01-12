import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('./data/card_transdata_preprocessed.csv')
st.title("Мошенничество с кредитными картами")
st.header("Тепловая карта")
plt.figure(figsize=(7, 5))
sns.heatmap(data.corr().round(3), annot=True, cmap='coolwarm')
st.pyplot(plt)

outlier = data[['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']]
Q1 = outlier.quantile(0.25)
Q3 = outlier.quantile(0.75)
IQR = Q3-Q1
data_filtered = outlier[~((outlier < (Q1 - 1.5 * IQR)) | (outlier > (Q3 + 1.5 * IQR))).any(axis=1)]

st.header("Гистограммы")
columns = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']
for column in columns:
    plt.figure(figsize=(7, 5))
    sns.histplot(data_filtered[column], bins=50, kde=True)
    plt.title(column)
    st.pyplot(plt)

st.header("Ящики с усами")
for column in columns:
    plt.figure(figsize=(6, 4))
    data_filtered.boxplot(figsize=(10, 6), column=column, grid=True)
    plt.title(column)
    st.pyplot(plt)

st.header("Круговая диаграмма")
plt.figure(figsize=(5, 5))
data['fraud'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Мошенничество')
st.pyplot(plt)
