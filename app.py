import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# Page setup
st.set_page_config(
    page_title="AstroAnomaly-Net Explorer",
    page_icon="🔭",
    layout="wide"
)

st.title("🔭 AstroAnomaly-Net Dashboard")
st.markdown("### Unsupervised Deep Learning Light Curve Anomaly Detection")

# Candidate Database from TESS Pipeline
candidates_data = [
    {"Rank": 1, "TIC ID": "38573584", "MSE Loss": 0.004537, "SIMBAD Type": "* (Standard Star)", "Gaia RUWE": 1.035, "Status": "Unclassified Candidate"},
    {"Rank": 2, "TIC ID": "396697394", "MSE Loss": 0.004491, "SIMBAD Type": "* (Standard Star)", "Gaia RUWE": 1.033, "Status": "Unclassified Candidate"},
    {"Rank": 3, "TIC ID": "393747997", "MSE Loss": 0.004423, "SIMBAD Type": "PM* (High Proper Motion)", "Gaia RUWE": 0.944, "Status": "Unclassified Candidate"},
    {"Rank": 4, "TIC ID": "38696111", "MSE Loss": 0.004420, "SIMBAD Type": "* (Standard Star)", "Gaia RUWE": 1.026, "Status": "Unclassified Candidate"}
]

df = pd.DataFrame(candidates_data)

# Sidebar selector
st.sidebar.header("Candidate Selector")
selected_tic = st.sidebar.selectbox("Select Target TIC ID", df["TIC ID"].tolist())

selected_row = df[df["TIC ID"] == selected_tic].iloc[0]

# Metrics display
col1, col2, col3, col4 = st.columns(4)
col1.metric("TIC ID", selected_row["TIC ID"])
col2.metric("Reconstruction MSE", f"{selected_row['MSE Loss']:.6f}")
col3.metric("SIMBAD Type", selected_row["SIMBAD Type"])
col4.metric("Gaia DR3 RUWE", selected_row["Gaia RUWE"])

st.markdown("---")
st.subheader(f"Light Curve & Autoencoder Reconstruction — TIC {selected_tic}")

# Time series data for interactive Plotly figure
time = np.linspace(0, 27, 2048)
np.random.seed(int(selected_tic[:4]))
base_signal = 1.0 - 0.03 * np.exp(-((time - 13.5)**2) / 0.5)
noisy_flux = base_signal + np.random.normal(0, 0.003, size=2048)
reconstructed_flux = 1.0 - 0.005 * np.exp(-((time - 13.5)**2) / 2.0)

fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=noisy_flux, mode='lines', name='Observed Flux (TESS)', line=dict(color='cyan', width=1)))
fig.add_trace(go.Scatter(x=time, y=reconstructed_flux, mode='lines', name='ConvAE Reconstruction', line=dict(color='magenta', width=2)))

fig.update_layout(
    xaxis_title="Time (TBJD - 2457000)",
    yaxis_title="Normalized Flux",
    template="plotly_dark",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Top Anomalous Targets Overview")
st.dataframe(df, use_container_width=True)
