import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型
model = joblib.load('C:/Users/Administrator/Desktop/streamlit/Rf_model.pkl')

# 定义特征选项
sex_options = {
    0: '女性 (0)', 
    1: '男性 (1)'
}

adl_options = {
    0: '能力完好 (0)', 
    1: '1-2项有困难 (1)', 
    2: '2-3项有困难 (2)', 
    3: '5-6项有困难 (3)'
}

fall_history_options = {
    0: '无跌倒史 (0)', 
    1: '过去一年有跌倒情况发生 (1)'
}

# 定义特征名称与数据一致
feature_names = [
   "Age", "Gender", "ADL", "Fall history", "Depression", "Cognition"
]

# Streamlit 用户界面
st.title("老年感觉障碍患者跌倒风险预测器")

# 年龄：数值输入
age = st.number_input("年龄 (60-120)", min_value=60, max_value=120, value=60)

# 性别：类别选择
sex = st.selectbox("性别 (0=女性, 1=男性):", options=list(sex_options.keys()), format_func=lambda x: sex_options[x])

# 日常生活自理能力：类别选择
adl = st.selectbox("日常生活自理能力 (0=能力完好, 1=1-2项有困难, 2=2-3项有困难, 3=5-6项有困难):", options=list(adl_options.keys()), format_func=lambda x: adl_options[x])

# 跌倒史：类别选择
fall_history = st.selectbox("跌倒史 (0=无跌倒史, 1=过去一年有跌倒情况发生):", options=list(fall_history_options.keys()), format_func=lambda x: fall_history_options[x])

# 抑郁：数值输入
depression = st.number_input("抑郁得分 (0-30):", min_value=0, max_value=30, value=10)

# 认知功能：数值输入
cognitive_function = st.number_input("认知功能得分 (0-21):", min_value=0, max_value=21, value=10)

# 处理输入并进行预测
feature_values = [age, sex, adl, fall_history, depression, cognitive_function]
features = np.array([feature_values])

if st.button("预测"):  # 点击按钮预测
    # 预测类别和概率
    predicted_proba = model.predict_proba(features)[0]
    
    # 获取预测类别
    predicted_class = np.argmax(predicted_proba)  # 获取最大概率的类别
    
    # 显示预测结果，使用中文描述预测类别
    risk_category = "有跌倒风险" if predicted_class == 1 else "无跌倒风险"
    st.write(f"**预测类别:** {risk_category}")
    #st.write(f"预测为无跌倒风险的概率: {predicted_proba[1] * 100:.1f}%")
    
    # 根据预测结果给出建议
    probability = predicted_proba[predicted_class] * 100
    
    # 根据预测类别提供建议
    if predicted_class == 1:
        advice = (
            f"根据我们的模型，您的健康风险较高。"
            f"模型预测您面临较高健康风险的概率为 {probability:.1f}%。"
            "虽然这只是一个估算，但它表明您可能处于较高的风险中。"
            "我们建议您尽快咨询医生，进行进一步评估，并确保及时得到准确诊断和治疗。"
        )
    else:
        advice = (
            f"根据我们的模型，您的健康风险较低。"
            f"模型预测您没有健康问题的概率为 {probability:.1f}%。"
            "尽管如此，保持健康的生活方式仍然非常重要。"
            "我们建议您定期检查，出现任何症状应及时就医。"
        )
    st.write(advice)

    # 计算 SHAP 值并显示 force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    # 对于二分类模型，shap_values 是一个长度为 2 的列表。选择索引 1 表示类别 1
    shap.plots.force(explainer.expected_value[1], shap_values[1][0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")

