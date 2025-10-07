import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title("📊 مشروع الانحدار الخطي المطورة روعة- آيسكريم")

uploaded_file = st.file_uploader("ارفع ملف IceCreamData.csv", type="csv")
if uploaded_file:
    Icecream = pd.read_csv(uploaded_file)
    st.write("📄 البيانات الأصلية:", Icecream)

    st.subheader("📈 تحليل البيانات")
    st.write(Icecream.describe())

    st.subheader("📊 الرسوم البيانية")
    fig1 = sns.jointplot(x='Revenue', y='Temperature', data=Icecream, color='purple')
    st.pyplot(fig1.figure)

    fig2 = sns.pairplot(Icecream)
    st.pyplot(fig2)

    fig3 = sns.lmplot(x='Temperature', y='Revenue', data=Icecream)
    st.pyplot(fig3.figure)

    X = Icecream[['Temperature']]
    y = Icecream[['Revenue']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    regressor = LinearRegression(fit_intercept=True)
    regressor.fit(X_train, y_train)

    st.subheader("📐 نتائج النموذج")
    st.write("المعامل (m):", regressor.coef_)
    st.write("الانحراف (b):", regressor.intercept_)

    st.subheader("📉 الرسم البياني التدريبي")
    fig4, ax = plt.subplots()
    ax.scatter(X_train, y_train, color='green')
    ax.plot(X_train, regressor.predict(X_train), color='red')
    ax.set_xlabel('Temperature [°C]')
    ax.set_ylabel('Revenue [$]')
    ax.set_title('Revenue vs Temperature (Training)')
    st.pyplot(fig4)

    st.subheader("🔮 توقع الإيرادات")
    temp_input = st.number_input("أدخل درجة الحرارة للتوقع:", min_value=0.0, max_value=50.0)
    Temp = pd.DataFrame([[temp_input]], columns=["Temperature"])
    Revenue = regressor.predict(Temp)
    st.write(f"الإيراد المتوقع عند درجة حرارة {temp_input}°C هو: ${Revenue[0][0]:.2f}")
