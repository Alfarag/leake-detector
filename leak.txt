import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hybrid AI Leak Detection Dashboard", layout="wide")
st.title("🚨 Hybrid AI Leak Detection and Emission Monitoring System")

st.markdown("This dashboard simulates predictions from a hybrid AI model using CNN, LSTM, and Autoencoder components.")

# Sidebar Inputs
st.sidebar.header("Sensor Inputs (Simulated)")
thermal_input = st.sidebar.slider("Thermal Anomaly Score", 0.0, 1.0, 0.7)
vibration_input = st.sidebar.slider("Vibration/Acoustic Score", 0.0, 1.0, 0.6)
gas_recon_error = st.sidebar.slider("Gas Sensor AE Loss", 0.0, 1.0, 0.5)

# Compute model outputs
P_CNN = thermal_input
P_LSTM = vibration_input
P_AE = 1 - gas_recon_error  # AE loss is inversely proportional to confidence

# Fusion Equation (Eq. 1)
P_final = 0.4 * P_CNN + 0.3 * P_LSTM + 0.3 * P_AE

# Display results
st.subheader("📊 Model Confidence Outputs")
col1, col2, col3 = st.columns(3)
col1.metric("CNN (Thermal)", f"{P_CNN:.2f}")
col2.metric("LSTM (Vibration)", f"{P_LSTM:.2f}")
col3.metric("Autoencoder (Gas)", f"{P_AE:.2f}")

st.markdown("---")
st.subheader("🧠 Final Leak Probability (Fusion)")
st.success(f"P_final = 0.4 × {P_CNN:.2f} + 0.3 × {P_LSTM:.2f} + 0.3 × {P_AE:.2f} = **{P_final:.3f}**")

# Visualization
fig, ax = plt.subplots(figsize=(8, 4))
models = ["CNN", "LSTM", "Autoencoder", "Final"]
scores = [P_CNN, P_LSTM, P_AE, P_final]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
ax.bar(models, scores, color=colors)
ax.set_ylim(0, 1)
ax.set_ylabel("Confidence / Probability")
st.pyplot(fig)

# Alert logic
st.markdown("---")
if P_final > 0.75:
    st.error("⚠️ Potential Leak Detected!")
elif P_final > 0.5:
    st.warning("🟡 Moderate Leak Risk")
else:
    st.success("✅ No Leak Detected")
