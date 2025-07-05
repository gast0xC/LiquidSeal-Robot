import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from evaluation_utils import calculate_detailed_errors
from sklearn.metrics import r2_score

# --- STREAMLIT DASHBOARD SETUP ---
st.set_page_config(page_title="Godlike AI Model Evaluation", layout="wide")
st.title("🚀 **Godlike AI Model Evaluation Dashboard**")
st.sidebar.title("🔧 Visualization Controls")

# --- SIDEBAR CONTROLS ---
st.sidebar.write("### Select Visualizations to Display")
show_training_history = st.sidebar.checkbox("Training History", True)
show_comparison = st.sidebar.checkbox("Actual vs Predicted Comparison", True)
show_residuals = st.sidebar.checkbox("Residuals", True)
show_error_metrics = st.sidebar.checkbox("Error Metrics Heatmap", True)
show_residual_histogram = st.sidebar.checkbox("Residuals Histogram", True)

# Parameter Selection for Residuals
param_names = ['X', 'Y', 'Z', 'Velocity']
selected_param = st.sidebar.selectbox("Select Parameter for Residual Analysis", param_names)

# --- MOCK DATA (Replace with real data from your model pipeline) ---
actual_data = np.random.rand(50, 4) * 100  # Replace with actual outputs
predicted_data = actual_data + np.random.randn(50, 4) * 5  # Replace with model predictions

# --- METRICS CALCULATION ---
mae, rmse, mape = calculate_detailed_errors(actual_data, predicted_data)
r2 = r2_score(actual_data, predicted_data, multioutput='raw_values')  # R² for each parameter

# --- LAYOUT CONFIGURATION ---
st.markdown("---")

if show_training_history:
    st.subheader("📈 Training Metrics Over Epochs")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.write("**Loss Over Epochs**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(10), np.random.rand(10), label='Train Loss')
        ax.plot(range(10), np.random.rand(10), label='Val Loss')
        ax.set_title("Loss Over Epochs")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.write("**Accuracy Over Epochs**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(10), np.random.rand(10), label='Train Accuracy')
        ax.plot(range(10), np.random.rand(10), label='Val Accuracy')
        ax.set_title("Accuracy Over Epochs")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.legend()
        st.pyplot(fig)

    with col3:
        st.write("**Mean Absolute Error (MAE) Over Epochs**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(10), np.random.rand(10), label='Train MAE')
        ax.plot(range(10), np.random.rand(10), label='Val MAE')
        ax.set_title("MAE Over Epochs")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("MAE")
        ax.legend()
        st.pyplot(fig)

st.markdown("---")

if show_comparison:
    st.subheader("🔍 Actual vs Predicted Comparison")
    fig = go.Figure()
    for i, param in enumerate(param_names):
        fig.add_trace(go.Scatter(y=actual_data[:, i], mode='lines', name=f"Actual {param}"))
        fig.add_trace(go.Scatter(y=predicted_data[:, i], mode='lines', name=f"Predicted {param}"))
    fig.update_layout(
        title="Actual vs Predicted Parameters",
        xaxis_title="Time Steps",
        yaxis_title="Parameter Values",
        legend_title="Legend",
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

if show_residuals:
    st.subheader(f"📊 Residuals for {selected_param}")
    param_index = param_names.index(selected_param)
    residuals = actual_data[:, param_index] - predicted_data[:, param_index]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(residuals, label=f"Residuals ({selected_param})", color='red')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_title(f"Residuals for {selected_param}")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Error")
    ax.legend()
    st.pyplot(fig)

st.markdown("---")

if show_residual_histogram:
    st.subheader(f"📊 Histogram of Residuals for {selected_param}")
    param_index = param_names.index(selected_param)
    residuals = actual_data[:, param_index] - predicted_data[:, param_index]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(residuals, bins=15, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', label='Zero Error')
    ax.set_title(f"Residuals Distribution for {selected_param}")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

st.markdown("---")

if show_error_metrics:
    st.subheader("📉 Error Metrics Heatmap")
    metrics = np.array([mae, rmse, [mape] * len(param_names), r2])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        metrics,
        annot=True,
        fmt=".2f",
        xticklabels=param_names,
        yticklabels=["MAE", "RMSE", "MAPE", "R²"],
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax
    )
    ax.set_title("Error Metrics Heatmap")
    st.pyplot(fig)

st.markdown("---")


# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.info("Interactive dashboard powered by gast0xC")

