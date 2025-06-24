
import streamlit as st
import pandas as pd
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Prediksi Togel AI - Per Digit", layout="centered")
st.title("🎰 Prediksi Togel AI - LSTM Per-Digit")
st.markdown("Model LSTM mempelajari pola per-digit dari angka 4D historis.")

# Input angka histori
teks_angka = st.text_area("Masukkan histori angka 4 digit (satu per baris):", height=200,
value="7123\n4012\n6321\n1980\n3124\n8945\n1098\n7632\n5412\n1093\n8842\n3381\n2764\n0012\n5678")

if teks_angka:
    angka_list = [x.strip().zfill(4) for x in teks_angka.splitlines() if x.strip().isdigit()]
else:
    angka_list = []

def angka_to_digit_array(angka_list):
    return np.array([[int(d) for d in list(a)] for a in angka_list])

if len(angka_list) >= 10:
    X = angka_to_digit_array(angka_list)
    y = X[1:]
    X = X[:-1]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.transform(y)

    # Siapkan data sequence
    n_input = 5
    generator = TimeseriesGenerator(X_scaled, y_scaled, length=n_input, batch_size=1)

    # Model LSTM
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(n_input, 4)))
    model.add(Dense(4))  # output 4 digit
    model.compile(optimizer='adam', loss='mse')
    model.fit(generator, epochs=20, verbose=0)

    # Prediksi
    last_seq = X_scaled[-n_input:]
    pred = model.predict(np.expand_dims(last_seq, axis=0), verbose=0)
    pred_digit = np.round(scaler.inverse_transform(pred)).astype(int).flatten()
    pred_str = ''.join([str(min(max(0, d), 9)) for d in pred_digit])  # pastikan digit valid

    st.success(f"🎯 Prediksi angka selanjutnya: {pred_str}")

else:
    st.warning("Masukkan minimal 10 angka untuk memulai pelatihan.")


# 🔍 UJI AKURASI MULTI-PREDIKSI
st.markdown("---")
st.subheader("🔍 Uji Akurasi Model Multi-Output (Top-1 / Top-3 / Top-5)")

jumlah_uji = st.slider("Jumlah data terakhir untuk uji akurasi", min_value=5, max_value=min(50, len(angka_list)-6), value=10)

def generate_multi_predictions(model, scaler, last_seq, n=5):
    preds = []
    input_seq = last_seq.copy()
    for _ in range(n):
        pred = model.predict(np.expand_dims(input_seq, axis=0), verbose=0)
        pred_digit = np.round(scaler.inverse_transform(pred)).astype(int).flatten()
        pred_str = ''.join([str(min(max(0, d), 9)) for d in pred_digit])
        preds.append(pred_str)
    return preds

def hitung_akurasi_multi_digit(data_list, jumlah_uji):
    benar = {'top1': 0, 'top3': 0, 'top5': 0}
    total = 0
    for i in range(len(data_list) - jumlah_uji - 1, len(data_list) - 1):
        try:
            seq = data_list[:i+1]
            X = angka_to_digit_array(seq)
            y = X[1:]
            X = X[:-1]

            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            y_scaled = scaler.transform(y)

            gen = TimeseriesGenerator(X_scaled, y_scaled, length=5, batch_size=1)
            model = Sequential()
            model.add(LSTM(64, activation='relu', input_shape=(5, 4)))
            model.add(Dense(4))
            model.compile(optimizer='adam', loss='mse')
            model.fit(gen, epochs=10, verbose=0)

            last_seq = X_scaled[-5:]
            top_preds = generate_multi_predictions(model, scaler, last_seq, n=5)
            target = data_list[i+1]

            if target == top_preds[0]:
                benar['top1'] += 1
            if target in top_preds[:3]:
                benar['top3'] += 1
            if target in top_preds:
                benar['top5'] += 1
            total += 1
        except Exception as e:
            st.warning(f"⚠️ Error prediksi multi di index {i}: {e}")
            continue

    if total == 0:
        return {'top1': 0.0, 'top3': 0.0, 'top5': 0.0}
    return {k: round(v / total * 100, 2) for k, v in benar.items()}

if st.button("🔍 Jalankan Uji Akurasi Multi"):
    acc = hitung_akurasi_multi_digit(angka_list, jumlah_uji)
    st.markdown("### 📊 Hasil Akurasi LSTM Multi-Output:")
    st.info(f"🎯 Top-1: {acc['top1']}%")
    st.info(f"🎯 Top-3: {acc['top3']}%")
    st.info(f"🎯 Top-5: {acc['top5']}%")
