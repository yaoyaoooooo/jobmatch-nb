import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# 把项目根目录加入 Python 搜索路径
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobmatch_nb.model.predict import load_model
from jobmatch_nb.matching.matcher import match_jobs
from jobmatch_nb.paths import MODELS_DIR, PROCESSED_DIR

st.set_page_config(page_title="岗位需求匹配系统", layout="wide")

st.title("基于朴素贝叶斯的岗位需求匹配系统")

model_path = MODELS_DIR / "nb_job_classifier.joblib"
data_path = PROCESSED_DIR / "jobs_labeled.csv"

if not model_path.exists():
    st.error(f"模型文件不存在：{model_path}")
    st.stop()

if not data_path.exists():
    st.error(f"处理后的岗位数据不存在：{data_path}")
    st.stop()

model = load_model(model_path)
jobs_df = pd.read_csv(data_path)

resume_text = st.text_area(
    "请输入求职者简历文本 / 技能描述",
    height=220,
    value="熟悉Python、SQL、数据分析、机器学习，能够使用Pandas、Scikit-learn完成数据处理与建模。"
)

topk = st.slider("返回岗位数量", min_value=5, max_value=50, value=20, step=5)

if st.button("开始匹配"):
    pred_label, prob_map, result_df = match_jobs(model, jobs_df, resume_text, topk=topk)

    st.subheader("预测岗位类别")
    st.write(pred_label)

    st.subheader("类别概率")
    prob_df = pd.DataFrame({
        "类别": list(prob_map.keys()),
        "概率": list(prob_map.values())
    }).sort_values("概率", ascending=False)
    st.dataframe(prob_df, use_container_width=True)

    st.subheader("Top 匹配岗位")
    show_cols = [
        "job_title", "company", "city", "salary", "label",
        "category_posterior", "similarity", "match_score", "job_text"
    ]
    existing_cols = [c for c in show_cols if c in result_df.columns]
    st.dataframe(result_df[existing_cols], use_container_width=True)
