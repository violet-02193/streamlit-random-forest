import streamlit as st
import joblib
import pandas as pd

# 加载训练好的模型
model = joblib.load("random_forest_model.joblib")

# 打印模型的特征名称（调试用途）
print("Model feature names:", model.feature_names_in_)

# 设置页面标题
st.title("Random Forest Model Prediction")

# 输入特征
st.write("Enter the feature values below:")

# 根据模型的特征名称动态生成输入框
feature_inputs = {}
for feature_name in model.feature_names_in_:
    feature_inputs[feature_name] = st.number_input(f"Feature {feature_name}", value=0.0)

# 当用户点击按钮时进行预测
if st.button("Predict"):
    # 创建特征 DataFrame，确保列名与训练时一致
    features = pd.DataFrame([list(feature_inputs.values())], columns=model.feature_names_in_)

    # 进行预测
    prediction = model.predict(features)

    # 显示预测结果
    st.write(f"Prediction: {prediction[0]}")
