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
    X = Icecream[['Temperature']]
    y = Icecream[['Revenue']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    regressor = LinearRegression(fit_intercept=True)
    regressor.fit(X_train, y_train)


    st.subheader("🔮 توقع الإيرادات")
    temp_input = st.number_input("أدخل درجة الحرارة للتوقع:", min_value=0.0, max_value=50.0)
    Temp = pd.DataFrame([[temp_input]], columns=["Temperature"])
    Revenue = regressor.predict(Temp)
    st.write(f"الإيراد المتوقع عند درجة حرارة {temp_input}°C هو: ${Revenue[0][0]:.2f}")
