#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install -r requirements.txt


# In[19]:


# app.py - 完整的APP壳子，带模拟数据和精美UI
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io

# ========== 页面配置 ==========
st.set_page_config(
    page_title="ICI治疗响应预测系统",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== 自定义CSS样式 ==========
st.markdown("""
<style>
    /* 主标题样式 */
    .main-title {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    /* 副标题样式 */
    .sub-title {
        font-size: 1.5rem;
        color: #5D737E;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    /* 卡片样式 */
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        border-left: 5px solid #2E86AB;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* 指标卡样式 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    
    /* 进度条样式 */
    .stProgress > div > div > div > div {
        background-color: #2E86AB;
    }
    
    /* 按钮样式 */
    .stButton > button {
        background-color: #2E86AB;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #236A8E;
        color: white;
    }
    
    /* 选项卡样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #F1F3F4;
        border-radius: 5px 5px 0px 0px;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2E86AB;
        color: white;
    }
    
    /* 文件上传区域样式 */
    .uploadedFile {
        background-color: #f0f7ff;
        padding: 15px;
        border-radius: 10px;
        border: 2px dashed #2E86AB;
    }
</style>
""", unsafe_allow_html=True)

# ========== 模拟数据生成函数 ==========
def generate_mock_data():
    """生成模拟的单细胞数据"""
    # 模拟细胞类型和比例
    cell_types = ['MAIT激活型', 'MAIT非激活型', '初始T细胞', '效应记忆T细胞', 
                  '细胞毒性T细胞', '耗竭T细胞', '调节性T细胞']
    
    # 模拟患者数据
    patients = [f"患者_{i+1:03d}" for i in range(20)]
    
    # 生成随机比例数据
    np.random.seed(42)
    data = []
    for patient in patients:
        proportions = np.random.dirichlet(np.ones(len(cell_types)) * 0.5)
        response = np.random.choice(['R', 'NR'], p=[0.3, 0.7])
        data.append([patient] + list(proportions) + [response])
    
    # 创建DataFrame
    columns = ['患者ID'] + cell_types + ['响应标签']
    df = pd.DataFrame(data, columns=columns)
    
    return df, cell_types

def generate_mock_umap():
    """生成模拟的UMAP数据"""
    np.random.seed(123)
    n_cells = 500
    
    # 生成聚类数据
    clusters = np.random.choice(['MAIT', 'Naive', 'Cytotox', 'Exhausted', 'Treg'], 
                                n_cells, p=[0.15, 0.3, 0.25, 0.2, 0.1])
    
    umap_df = pd.DataFrame({
        'UMAP1': np.random.normal(0, 1, n_cells),
        'UMAP2': np.random.normal(0, 1, n_cells),
        '细胞类型': clusters,
        '患者': np.random.choice([f'P{i}' for i in range(10)], n_cells)
    })
    
    # 添加一些聚类结构
    for i, cluster in enumerate(['MAIT', 'Naive', 'Cytotox', 'Exhausted', 'Treg']):
        mask = umap_df['细胞类型'] == cluster
        umap_df.loc[mask, 'UMAP1'] += i * 2
        umap_df.loc[mask, 'UMAP2'] += np.random.normal(0, 0.5, sum(mask))
    
    return umap_df

def generate_mock_performance():
    """生成模拟的性能指标"""
    metrics = {
        '准确率': 0.87,
        '精确率': 0.85,
        '召回率': 0.88,
        'F1分数': 0.86,
        'AUC': 0.91,
        '特异性': 0.90
    }
    return metrics

def generate_mock_roc():
    """生成模拟的ROC曲线数据"""
    np.random.seed(42)
    fpr = np.linspace(0, 1, 100)
    tpr = 1 - np.exp(-5 * fpr)  # 模拟ROC曲线形状
    tpr += np.random.normal(0, 0.02, len(tpr))  # 添加噪声
    tpr = np.clip(tpr, 0, 1)
    
    return fpr, tpr

# ========== 侧边栏导航 ==========
st.sidebar.markdown("""
<div style="text-align: center;">
    <h2 style="color: #2E86AB;">🩺 ICI预测系统</h2>
    <p style="color: #666;">v1.0.0</p>
</div>
<hr>
""", unsafe_allow_html=True)

# 导航菜单
menu_options = ["🏠 项目主页", "📤 数据上传", "🔬 分析预测", 
                "📊 结果可视化", "🧪 模型验证", "⚙️ 系统设置"]
menu = st.sidebar.radio("导航菜单", menu_options)

# ========== 主页 ==========
if menu == "🏠 项目主页":
    st.markdown('<h1 class="main-title">基于外周血T细胞的ICI治疗响应预测系统</h1>', unsafe_allow_html=True)
    
    # 项目介绍卡片
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h4>🎯 项目目标</h4>
                <p>通过外周血单细胞转录组数据，预测患者对免疫检查点抑制剂（ICI）的治疗响应。</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h4>🧬 核心技术</h4>
                <p>scFoundation + Geneformer + 注意力机制模型，构建三层预测系统。</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="card">
                <h4>🏥 临床价值</h4>
                <p>避免无效治疗，减少副作用，实现个体化精准医疗。</p>
            </div>
            """, unsafe_allow_html=True)
    
    # 技术路线图
    st.markdown('<h3 class="sub-title">🔧 技术路线图</h3>', unsafe_allow_html=True)
    
    # 使用HTML创建流程图
    st.markdown("""
    <div style="background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #ddd;">
        <div style="display: flex; justify-content: center; align-items: center; margin: 20px 0;">
            <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; width: 180px; margin: 0 10px;">
                <h4 style="margin: 0;">📊 原始数据</h4>
                <p style="margin: 5px 0 0 0; font-size: 0.9em;">外周血单细胞测序</p>
            </div>
            <div style="font-size: 24px;">→</div>
            <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; border-radius: 10px; width: 180px; margin: 0 10px;">
                <h4 style="margin: 0;">🧬 scFoundation</h4>
                <p style="margin: 5px 0 0 0; font-size: 0.9em;">细胞嵌入提取</p>
            </div>
            <div style="font-size: 24px;">→</div>
            <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; border-radius: 10px; width: 180px; margin: 0 10px;">
                <h4 style="margin: 0;">🤖 Geneformer</h4>
                <p style="margin: 5px 0 0 0; font-size: 0.9em;">细胞亚群分类</p>
            </div>
            <div style="font-size: 24px;">→</div>
            <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; border-radius: 10px; width: 180px; margin: 0 10px;">
                <h4 style="margin: 0;">🎯 注意力模型</h4>
                <p style="margin: 5px 0 0 0; font-size: 0.9em;">治疗响应预测</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 数据集信息
    st.markdown('<h3 class="sub-title">📁 可用数据集</h3>', unsafe_allow_html=True)
    
    dataset_info = pd.DataFrame({
        '数据集': ['GSE166181', 'GSE145281', 'GSE153098', 'GSE120575', 'GSE123813'],
        '癌症类型': ['黑色素瘤', '膀胱癌', '黑色素瘤', '黑色素瘤', '皮肤癌'],
        '样本数': [66, 10, 4, 19, 15],
        '响应者(R)': [35, 5, 0, 9, 8],
        '非响应者(NR)': [31, 5, 4, 10, 7],
        'CD8+T细胞数': ['16,885', '14,475', '712', '2,709', '15,672']
    })
    
    st.dataframe(dataset_info, use_container_width=True)

# ========== 数据上传页面 ==========
elif menu == "📤 数据上传":
    st.markdown('<h1 class="main-title">数据上传与预处理</h1>', unsafe_allow_html=True)
    
    # 文件上传区域
    st.markdown("""
    <div class="uploadedFile">
        <h4 style="color: #2E86AB; margin-top: 0;">📤 上传单细胞数据文件</h4>
        <p>支持格式：.h5ad (AnnData), .csv, .tsv, .txt</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "选择文件",
            type=['h5ad', 'csv', 'tsv', 'txt'],
            label_visibility="collapsed"
        )
    
    with col2:
        use_example = st.checkbox("使用示例数据", value=True)
    
    # 数据预览
    if uploaded_file or use_example:
        st.success("✅ 数据加载成功！")
        
        # 显示模拟的数据预览
        with st.expander("📋 数据预览", expanded=True):
            tab1, tab2, tab3 = st.tabs(["元数据", "基因表达矩阵", "质量控制"])
            
            with tab1:
                # 模拟元数据
                metadata = pd.DataFrame({
                    '样本ID': [f'Sample_{i}' for i in range(1, 11)],
                    '患者ID': [f'P{100+i}' for i in range(10)],
                    '癌症类型': ['黑色素瘤']*5 + ['肺癌']*5,
                    '治疗前响应': ['NR', 'R', 'NR', 'NR', 'R', 'R', 'NR', 'R', 'NR', 'R'],
                    '细胞数': np.random.randint(1000, 5000, 10),
                    '基因数': [18000]*10
                })
                st.dataframe(metadata, use_container_width=True)
            
            with tab2:
                # 模拟基因表达矩阵
                genes = [f'Gene_{i}' for i in range(1, 21)]
                cells = [f'Cell_{i}' for i in range(1, 11)]
                expression_data = np.random.randn(20, 10)
                expression_df = pd.DataFrame(expression_data, index=genes, columns=cells)
                st.dataframe(expression_df.style.background_gradient(cmap='Blues'), use_container_width=True)
            
            with tab3:
                # 模拟QC指标
                qc_data = pd.DataFrame({
                    '指标': ['细胞总数', '平均基因数/细胞', '中位数UMI', '线粒体基因比例', '核糖体基因比例'],
                    '数值': ['10,245', '2,348', '5,672', '8.5%', '15.2%'],
                    '状态': ['✅ 通过', '✅ 通过', '✅ 通过', '⚠️ 警告', '✅ 通过']
                })
                st.dataframe(qc_data, use_container_width=False)
        
        # 预处理选项
        st.markdown('<h3 class="sub-title">⚙️ 预处理设置</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_genes = st.slider("最小基因数", 200, 1000, 200)
            max_genes = st.slider("最大基因数", 2500, 10000, 5000)
        
        with col2:
            mt_cutoff = st.slider("线粒体基因阈值%", 0.0, 20.0, 10.0, 0.5)
            rb_cutoff = st.slider("核糖体基因阈值%", 0.0, 50.0, 50.0, 1.0)
        
        with col3:
            norm_method = st.selectbox("归一化方法", ["LogNormalize", "SCTransform", "CLR"])
            n_hvg = st.slider("高变基因数", 1000, 5000, 2000)
        
        if st.button("🚀 开始预处理", type="primary", use_container_width=True):
            # 显示预处理进度
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            steps = ["加载数据", "质量控制", "归一化", "特征选择", "降维"]
            for i, step in enumerate(steps):
                progress = (i + 1) / len(steps)
                progress_bar.progress(progress)
                status_text.text(f"正在进行: {step}...")
                st.session_state[f'preprocess_step_{i}'] = True
            
            progress_bar.progress(1.0)
            status_text.text("✅ 预处理完成！")
            st.success("数据已准备好进行分析！")
            st.session_state['data_preprocessed'] = True

# ========== 分析预测页面 ==========
elif menu == "🔬 分析预测":
    st.markdown('<h1 class="main-title">模型预测分析</h1>', unsafe_allow_html=True)
    
    # 模型选择
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_type = st.selectbox(
            "选择预测模型",
            ["scFoundation + Geneformer + 注意力模型", 
             "随机森林模型", 
             "深度学习混合模型"]
        )
    
    with col2:
        threshold = st.slider("预测阈值", 0.0, 1.0, 0.5, 0.05)
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_prediction = st.button("🔍 开始预测", type="primary", use_container_width=True)
    
    if run_prediction:
        # 模拟模型加载和预测过程
        with st.spinner("正在加载模型..."):
            import time
            time.sleep(1)
        
        # 创建进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 模拟预测步骤
        steps = [
            ("加载scFoundation模型", 0.1),
            ("提取细胞嵌入特征", 0.3),
            ("Geneformer细胞分类", 0.5),
            ("计算亚群比例", 0.7),
            ("注意力模型预测", 0.9),
            ("生成结果", 1.0)
        ]
        
        for step_name, progress in steps:
            time.sleep(0.5)
            progress_bar.progress(progress)
            status_text.text(f"正在进行: {step_name}")
        
        # 显示预测结果
        st.success("✅ 预测完成！")
        
        # 模拟预测结果
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>0.78</h3>
                <p>响应概率</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background-color: #FF6B6B; color: white; padding: 1.5rem; border-radius: 10px; text-align: center;">
                <h3>非响应者 (NR)</h3>
                <p>预测分类</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background-color: #4ECDC4; color: white; padding: 1.5rem; border-radius: 10px; text-align: center;">
                <h3>高风险</h3>
                <p>治疗风险</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 详细预测信息
        st.markdown('<h3 class="sub-title">📋 详细预测信息</h3>', unsafe_allow_html=True)
        
        prediction_details = pd.DataFrame({
            '患者ID': ['P001', 'P002', 'P003', 'P004', 'P005'],
            '响应概率': [0.78, 0.35, 0.92, 0.45, 0.67],
            '预测分类': ['NR', 'NR', 'R', 'NR', 'R'],
            '置信度': [0.89, 0.76, 0.94, 0.81, 0.87],
            '推荐治疗': ['不推荐ICI', '不推荐ICI', '推荐ICI', '不推荐ICI', '推荐ICI']
        })
        
        st.dataframe(prediction_details.style.applymap(
            lambda x: 'background-color: #FFEBEE' if x == 'NR' else 'background-color: #E8F5E9', 
            subset=['预测分类']
        ), use_container_width=True)

# ========== 结果可视化页面 ==========
elif menu == "📊 结果可视化":
    st.markdown('<h1 class="main-title">分析结果可视化</h1>', unsafe_allow_html=True)
    
    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 细胞亚群分析", 
        "📈 预测性能", 
        "👥 患者分类", 
        "🧬 生物标志物"
    ])
    
    with tab1:
        st.markdown('<h3 class="sub-title">细胞亚群UMAP可视化</h3>', unsafe_allow_html=True)
        
        # 生成模拟UMAP数据
        umap_df = generate_mock_umap()
        
        # 创建交互式UMAP图
        fig = px.scatter(
            umap_df, 
            x='UMAP1', 
            y='UMAP2', 
            color='细胞类型',
            hover_data=['患者'],
            title="CD8+T细胞亚群UMAP降维",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # 亚群比例饼图
        st.markdown('<h3 class="sub-title">细胞亚群比例分布</h3>', unsafe_allow_html=True)
        
        cell_proportions = umap_df['细胞类型'].value_counts().reset_index()
        cell_proportions.columns = ['细胞类型', '数量']
        
        fig2 = px.pie(
            cell_proportions, 
            values='数量', 
            names='细胞类型',
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            st.dataframe(cell_proportions, use_container_width=True)
    
    with tab2:
        st.markdown('<h3 class="sub-title">模型性能评估</h3>', unsafe_allow_html=True)
        
        # ROC曲线
        fpr, tpr = generate_mock_roc()
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name='ROC曲线',
            line=dict(color='#2E86AB', width=3)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='随机分类',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig_roc.update_layout(
            title=f'ROC曲线 (AUC = 0.91)',
            xaxis_title='假阳性率',
            yaxis_title='真阳性率',
            width=800,
            height=500
        )
        
        st.plotly_chart(fig_roc, use_container_width=True)
        
        # 性能指标雷达图
        metrics = generate_mock_performance()
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=list(metrics.values()),
            theta=list(metrics.keys()),
            fill='toself',
            line_color='#2E86AB'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab3:
        st.markdown('<h3 class="sub-title">患者响应分类</h3>', unsafe_allow_html=True)
        
        # 生成模拟患者数据
        np.random.seed(123)
        n_patients = 50
        response_probs = np.random.beta(2, 5, n_patients)
        actual_response = (response_probs > 0.5).astype(int)
        predicted_response = (response_probs + np.random.normal(0, 0.1, n_patients) > 0.5).astype(int)
        
        patients_df = pd.DataFrame({
            '患者ID': [f'PAT_{i:03d}' for i in range(n_patients)],
            '实际响应': ['R' if x == 1 else 'NR' for x in actual_response],
            '预测响应': ['R' if x == 1 else 'NR' for x in predicted_response],
            '预测概率': response_probs,
            '癌症类型': np.random.choice(['黑色素瘤', '肺癌', '膀胱癌', '肾癌'], n_patients)
        })
        
        # 混淆矩阵热图
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        cm = confusion_matrix(patients_df['实际响应'], patients_df['预测响应'], labels=['NR', 'R'])
        
        fig_cm, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['预测NR', '预测R'],
                    yticklabels=['实际NR', '实际R'],
                    ax=ax)
        ax.set_title('混淆矩阵')
        st.pyplot(fig_cm)
        
        # 患者分类表
        st.dataframe(patients_df.head(10), use_container_width=True)
    
    with tab4:
        st.markdown('<h3 class="sub-title">关键生物标志物分析</h3>', unsafe_allow_html=True)
        
        # 模拟标志基因数据
        marker_genes = pd.DataFrame({
            '基因符号': ['PDCD1', 'CTLA4', 'LAG3', 'TIGIT', 'TIM3', 'GZMB', 'PRF1', 'IFNG', 
                       'CXCL13', 'CCL5', 'TNF', 'IL2', 'FOXP3', 'CD274', 'CD8A'],
            'log2FC': np.random.uniform(-3, 5, 15),
            'p_value': 10**(-np.random.uniform(1, 10, 15)),
            '细胞类型': np.random.choice(['MAIT', '耗竭T', '效应T', '记忆T', '调节T'], 15),
            '功能': ['免疫检查点', '免疫检查点', '免疫检查点', '抑制受体', '抑制受体',
                   '细胞毒性', '细胞毒性', '细胞因子', '趋化因子', '趋化因子',
                   '细胞因子', '细胞因子', '转录因子', '配体', '标志物']
        })
        
        marker_genes['-log10(p)'] = -np.log10(marker_genes['p_value'])
        
        # 火山图
        fig_volcano = px.scatter(
            marker_genes,
            x='log2FC',
            y='-log10(p)',
            color='细胞类型',
            hover_data=['基因符号', '功能'],
            title='差异表达基因火山图',
            size='-log10(p)',
            size_max=15
        )
        
        # 添加阈值线
        fig_volcano.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="red")
        fig_volcano.add_vline(x=1, line_dash="dash", line_color="red")
        fig_volcano.add_vline(x=-1, line_dash="dash", line_color="red")
        
        st.plotly_chart(fig_volcano, use_container_width=True)

# ========== 模型验证页面 ==========
elif menu == "🧪 模型验证":
    st.markdown('<h1 class="main-title">模型验证与比较</h1>', unsafe_allow_html=True)
    
    # 交叉验证结果
    st.markdown('<h3 class="sub-title">五折交叉验证结果</h3>', unsafe_allow_html=True)
    
    cv_results = pd.DataFrame({
        '折数': [1, 2, 3, 4, 5, '平均'],
        '准确率': [0.85, 0.88, 0.86, 0.87, 0.89, 0.87],
        'AUC': [0.90, 0.92, 0.91, 0.89, 0.93, 0.91],
        'F1分数': [0.84, 0.87, 0.85, 0.86, 0.88, 0.86],
        '召回率': [0.83, 0.86, 0.85, 0.84, 0.87, 0.85]
    })
    
    st.dataframe(cv_results.style.highlight_max(subset=['准确率', 'AUC', 'F1分数', '召回率'], color='lightgreen'), 
                 use_container_width=True)
    
    # 模型比较
    st.markdown('<h3 class="sub-title">不同模型性能比较</h3>', unsafe_allow_html=True)
    
    model_comparison = pd.DataFrame({
        '模型': ['scFoundation+Geneformer', '随机森林', '支持向量机', '逻辑回归', 'XGBoost', '传统标志物(PD-L1)'],
        '准确率': [0.87, 0.82, 0.79, 0.76, 0.84, 0.65],
        'AUC': [0.91, 0.86, 0.83, 0.80, 0.88, 0.70],
        'F1分数': [0.86, 0.81, 0.78, 0.75, 0.83, 0.64],
        '计算时间(秒)': [45, 12, 8, 5, 20, 2]
    })
    
    fig_compare = px.bar(
        model_comparison,
        x='模型',
        y=['准确率', 'AUC', 'F1分数'],
        barmode='group',
        title='不同模型性能比较',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    st.plotly_chart(fig_compare, use_container_width=True)
    
    # 外部验证结果
    st.markdown('<h3 class="sub-title">外部数据集验证</h3>', unsafe_allow_html=True)
    
    external_val = pd.DataFrame({
        '数据集': ['GSE166181', 'GSE145281', 'GSE153098', 'GSE120575', 'GSE123813'],
        '癌症类型': ['黑色素瘤', '膀胱癌', '黑色素瘤', '黑色素瘤', '皮肤癌'],
        '样本数': [66, 10, 4, 19, 15],
        '准确率': [0.87, 0.80, 0.75, 0.84, 0.82],
        'AUC': [0.91, 0.85, 0.78, 0.88, 0.86],
        '泛化能力': ['优秀', '良好', '一般', '良好', '良好']
    })
    
    st.dataframe(external_val, use_container_width=True)

# ========== 系统设置页面 ==========
elif menu == "⚙️ 系统设置":
    st.markdown('<h1 class="main-title">系统设置与配置</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="sub-title">📊 显示设置</h3>', unsafe_allow_html=True)
        
        theme = st.selectbox("界面主题", ["浅色", "深色", "自动"])
        chart_style = st.selectbox("图表风格", ["Plotly", "Matplotlib", "Seaborn"])
        page_layout = st.selectbox("页面布局", ["宽屏", "窄屏", "自适应"])
        
        st.markdown('<h3 class="sub-title">💾 数据设置</h3>', unsafe_allow_html=True)
        
        cache_size = st.slider("缓存大小(MB)", 100, 1000, 500)
        auto_save = st.checkbox("自动保存结果", value=True)
        export_format = st.multiselect(
            "导出格式",
            ["CSV", "Excel", "PDF", "HTML", "PNG"],
            default=["CSV", "PNG"]
        )
    
    with col2:
        st.markdown('<h3 class="sub-title">🔧 模型设置</h3>', unsafe_allow_html=True)
        
        default_model = st.selectbox(
            "默认预测模型",
            ["三层深度学习模型", "随机森林", "XGBoost", "集成模型"]
        )
        
        confidence_threshold = st.slider("置信度阈值", 0.5, 0.99, 0.8, 0.01)
        
        st.markdown('<h3 class="sub-title">🛠️ 高级设置</h3>', unsafe_allow_html=True)
        
        debug_mode = st.checkbox("调试模式")
        log_level = st.selectbox("日志级别", ["INFO", "DEBUG", "WARNING", "ERROR"])
        
        if st.button("重置所有设置", type="secondary"):
            st.warning("这将重置所有系统设置，确定吗？")
            if st.button("确认重置", type="primary"):
                st.success("设置已重置为默认值")
    
    # 系统信息
    st.markdown('<h3 class="sub-title">ℹ️ 系统信息</h3>', unsafe_allow_html=True)
    
    sys_info = pd.DataFrame({
        '项目': ['版本', 'Python版本', 'Streamlit版本', '最后更新', '内存使用', 'CPU使用率'],
        '数值': ['v1.0.0', '3.9.0', '1.28.0', '2024-03-20', '256 MB / 1 GB', '15%']
    })
    
    st.dataframe(sys_info, use_container_width=False, hide_index=True)
    
    if st.button("💾 保存设置", type="primary"):
        st.success("系统设置已保存！")

# ========== 页脚 ==========
st.markdown("""
<hr>
<div style="text-align: center; color: #666; padding: 20px; font-size: 0.9em;">
    <p>📧 联系方式: bioinfo@fmu.edu.cn | 📞 技术支持: 0591-22862000</p>
    <p>© 2024 福建医科大学生物信息学系 | ICI治疗响应预测系统 v1.0.0</p>
    <p style="font-size: 0.8em;">注意：本系统为学术研究原型，临床使用需进一步验证</p>
</div>
""", unsafe_allow_html=True)


# In[ ]:





# In[ ]:





# In[ ]:




