import streamlit as st
import numpy as np
import pandas as pd
import yaml
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pipeline import DataPipeline
from model import CustomLSTMCell
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
st.set_page_config(page_title="IA Microclima Agrícola", layout="wide")
sns.set_theme(style="whitegrid")
with open("config.yaml", "r") as f:
    config_base = yaml.safe_load(f)
st.title("🌱 Plataforma de Previsão Microclimática Avançada")
st.markdown("---")
st.sidebar.header("🎛️ Painel de Controlo do Utilizador")
st.sidebar.subheader("📋 Engenharia de Dados")
window_size = st.sidebar.slider("Tamanho da Janela Histórica (Dias passados)", min_value=5, max_value=20, value=config_base['data_pipeline']['window_ingestion']['window_size'])
horizon = st.sidebar.slider("Horizonte de Previsão (τ - Dias à frente)", min_value=1, max_value=7, value=config_base['data_pipeline']['window_ingestion']['horizon'])
st.sidebar.subheader("🧠 Arquitetura e Treino da IA")
hidden_dim = st.sidebar.number_input("Dimensão Oculta da LSTM ($d$)", min_value=8, max_value=128, step=8, value=config_base['model_architecture']['hidden_dim'])
epochs = st.sidebar.slider("Número Máximo de Épocas", min_value=2, max_value=100, value=config_base['training']['epochs'])
lambda_val = st.sidebar.slider("Penalização de Tikhonov ($\lambda$)", min_value=0.001, max_value=0.100, step=0.005, value=config_base['training']['regularization']['lambda_value'], format="%.3f")
patience = st.sidebar.number_input("Paciência do Early Stopping", min_value=2, max_value=15, value=config_base['training']['early_stopping']['patience'])
np.random.seed(42)
n_samples = 150
hoje = datetime(2026, 7, 23)
start_date = hoje - timedelta(days=n_samples - 1)
time_axis = pd.date_range(start=start_date, end=hoje, freq="D")
precipitacao = np.abs(np.sin(np.linspace(0, 10, n_samples)) * 25 + np.random.normal(0, 5, n_samples))
temp_solo = 22 + np.cos(np.linspace(0, 6, n_samples)) * 8 + np.random.normal(0, 1.5, n_samples)
humidade_relativa = 75 - np.cos(np.linspace(0, 6, n_samples)) * 15 + np.random.normal(0, 3, n_samples)
raw_matrix = np.column_stack((precipitacao, temp_solo, humidade_relativa))
nan_indices = np.random.choice(n_samples, size=15, replace=False)
raw_matrix[nan_indices, 0] = np.nan
raw_matrix[np.random.choice(n_samples, size=10, replace=False), 2] = np.nan
pipeline = DataPipeline(window_size=window_size, horizon=horizon)
clean_matrix = pipeline.apply_cubic_spline(raw_matrix)
norm_matrix = pipeline.fit_transform_minmax(clean_matrix)
X_windows, y_windows = pipeline.construct_sliding_windows(norm_matrix)
split_idx = int(len(X_windows) * 0.8)
X_train, X_val = X_windows[:split_idx], X_windows[split_idx:]
y_train, y_val = y_windows[:split_idx], y_windows[split_idx:]
model = CustomLSTMCell(input_dim=3, hidden_dim=hidden_dim, output_dim=3)
st.subheader("🏋️ Processo Numérico de Otimização")
progress_bar = st.progress(0)
status_text = st.empty()
train_losses, val_losses = [], []
best_val_loss = float('inf')
patience_counter = 0
early_stop_epoch = epochs
for epoch in range(epochs):
    train_loss = model.compute_loss_with_tikhonov(X_train, y_train, lambda_val)
    val_loss = model.compute_loss_with_tikhonov(X_val, y_val, lambda_val)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    status_text.text(f"Época {epoch+1}/{epochs} | Perda Treino: {train_loss:.4f} | Perda Validação: {val_loss:.4f}")
    progress_bar.progress((epoch + 1) / epochs)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            early_stop_epoch = epoch + 1
            break
    time.sleep(0.002)
lstm_predictions = []
for seq in X_val:
    lstm_predictions.append(model.forward_sequence(seq))
lstm_predictions = np.array(lstm_predictions)
y_val_real = pipeline.inverse_transform_minmax(y_val)
lstm_real_preds = pipeline.inverse_transform_minmax(lstm_predictions)
future_preds_norm = []
current_window = norm_matrix[-window_size:].copy()
for d in range(horizon):
    pred = model.forward_sequence(current_window)
    future_preds_norm.append(pred)
    current_window = np.vstack((current_window[1:], pred))
future_preds_real = pipeline.inverse_transform_minmax(np.array(future_preds_norm))
future_dates = pd.date_range(start=hoje + timedelta(days=1), periods=horizon, freq="D")
st.markdown("---")
st.subheader("📊 Painel de Visualização Gráfica")
tab1, tab2 = st.tabs(["📈 Projeção Operacional (Futuro)", "🔬 Histórico"])
with tab1:
    st.markdown(f"#### **Tendência Estimada para os Próximos {horizon} Dias**")
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        fig_fut, ax_fut = plt.subplots(figsize=(7, 3.5))
        ax_fut.plot(future_dates, future_preds_real[:, 2], label='Previsão da Humidade (IA)', color='#b91c1c', marker='o', lw=2, linestyle='--')
        ax_fut.set_ylabel('Humidade Relativa (%)')
        ax_fut.set_xlabel('Cronograma Futuro')
        plt.xticks(future_dates, [d.strftime('%Y-%m-%d') for d in future_dates], rotation=35)
        ax_fut.legend()
        st.pyplot(fig_fut)
    with col_f2:
        st.markdown("**Valores Numéricos Projetados:**")
        df_futuro = pd.DataFrame({
            'Data': [d.strftime('%Y-%m-%d') for d in future_dates],
            'Precipitação (mm)': future_preds_real[:, 0],
            'Temp. Solo (°C)': future_preds_real[:, 1],
            'Humidade (%)': future_preds_real[:, 2]
        })
        st.dataframe(df_futuro.style.format({'Precipitação (mm)': '{:.2f}', 'Temp. Solo (°C)': '{:.2f}', 'Humidade (%)': '{:.2f}'}))
with tab2:
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("#### **Curva de Aprendizagem**")
        fig1, ax1 = plt.subplots(figsize=(6, 3.8))
        ax1.plot(range(1, len(train_losses)+1), train_losses, label='Erro de Treino', color='#1e3a8a', lw=2)
        ax1.plot(range(1, len(val_losses)+1), val_losses, label='Erro de Validação', color='#f59e0b', lw=2, linestyle='--')
        ax1.set_xlabel('Épocas')
        ax1.set_ylabel('Perda Total $\mathcal{L}(\Theta)$')
        ax1.legend()
        st.pyplot(fig1)
    with col_g2:
        st.markdown("#### **Aderência nos Dados de Teste (Passado)**")
        fig2, ax2 = plt.subplots(figsize=(6, 3.8))
        val_time_axis = time_axis[-len(y_val_real):]
        ax2.plot(val_time_axis, y_val_real[:, 2], label='Dados Reais', color='#2563eb', lw=2)
        ax2.plot(val_time_axis, lstm_real_preds[:, 2], label='Nossa LSTM', color='#dc2626', lw=2, linestyle='--')
        plt.xticks(rotation=45)
        ax2.legend()
        st.pyplot(fig2)
st.markdown("---")
st.subheader("📋 Diretrizes de Ação Agrícola por Dia")
for idx, data_dia in enumerate(future_dates):
    data_str = data_dia.strftime('%d/%m/%Y')
    hum_dia = future_preds_real[idx, 2]
    prec_dia = future_preds_real[idx, 0]    
    col_day, col_status = st.columns([1, 4])
    with col_day:
        st.markdown(f"**{data_str}**")
    with col_status:
        if hum_dia < 65.0 and prec_dia < 2.0:
            st.error(f"⚠️ Humidade Crítica ({hum_dia:.1f}%). **Ação:** Ativar sistema de irrigação por um ciclo completo.")
        elif hum_dia < 70.0:
            st.warning(f"⚡ Humidade em declínio ({hum_dia:.1f}%). **Ação:** Monitorizar e aplicar irrigação leve se não houver cobertura de nuvens.")
        else:
            st.success(f"✅ Balanço Hídrico Ideal ({hum_dia:.1f}%). **Ação:** Manter sistemas desligados. Economia de água e energia.")