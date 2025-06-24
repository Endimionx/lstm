
import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Prediksi Togel AI", layout="centered")
st.title("🎰 Prediksi Togel AI - Markov & LSTM Per-Digit")

# ============================
# Input histori angka
# ============================
st.subheader("Masukkan histori angka 4 digit")
teks_angka = st.text_area("Satu angka per baris:", height=200,
value="7123\n4012\n6321\n1980\n3124\n8945\n1098\n7632\n5412\n1093\n8842\n3381\n2764\n0012\n5678")

angka_list = [x.strip().zfill(4) for x in teks_angka.splitlines() if x.strip().isdigit()]
if len(angka_list) < 10:
    st.warning("Masukkan minimal 10 angka histori.")
    st.stop()

# ============================
# Fungsi umum
# ============================
def angka_to_digit_array(angka_list):
    return np.array([[int(d) for d in list(a)] for a in angka_list])

# ============================
# MARKOV
# ============================
transition = defaultdict(list)
for i in range(len(angka_list) - 1):
    transition[angka_list[i]].append(angka_list[i+1])

def prediksi_markov(current, n=5):
    candidates = transition.get(current, [])
    if not candidates:
        return [str(random.randint(0, 9999)).zfill(4) for _ in range(n)]
    return random.choices(candidates, k=n)

def hitung_akurasi_markov(data, jumlah_uji=10):
    benar = 0
    for i in range(len(data) - jumlah_uji - 1, len(data) - 1):
        pred = prediksi_markov(data[i], n=5)
        if data[i+1] in pred:
            benar += 1
    return round(benar / jumlah_uji * 100, 2)

# ============================
# LSTM Digit
# ============================
def train_lstm_model(X, y):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.transform(y)
    gen = TimeseriesGenerator(X_scaled, y_scaled, length=5, batch_size=1)
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(5, 4)))
    model.add(Dense(4))
    model.compile(optimizer='adam', loss='mse')
    model.fit(gen, epochs=10, verbose=0)
    return model, scaler, X_scaled

def prediksi_multi(model, scaler, last_seq, n=5):
    preds = []
    input_seq = last_seq.copy()
    for _ in range(n):
        pred = model.predict(np.expand_dims(input_seq, axis=0), verbose=0)
        digits = np.round(scaler.inverse_transform(pred)).astype(int).flatten()
        pred_str = ''.join([str(min(max(0, d), 9)) for d in digits])
        preds.append(pred_str)
    return preds

def hitung_akurasi_lstm_digit(data_list, jumlah_uji=10):
    benar = {'top1': 0, 'top3': 0, 'top5': 0}
    total = 0
    for i in range(len(data_list) - jumlah_uji - 1, len(data_list) - 1):
        try:
            potongan = data_list[:i+1]
            X = angka_to_digit_array(potongan)
            y = X[1:]
            X = X[:-1]
            model, scaler, X_scaled = train_lstm_model(X, y)
            last_seq = X_scaled[-5:]
            pred_all = prediksi_multi(model, scaler, last_seq, n=5)
            target = data_list[i+1]
            if target == pred_all[0]: benar['top1'] += 1
            if target in pred_all[:3]: benar['top3'] += 1
            if target in pred_all: benar['top5'] += 1
            total += 1
        except Exception as e:
            st.warning(f"⚠️ Error di index {i}: {e}")
            continue
    return {k: round(v / total * 100, 2) for k, v in benar.items()}

# ============================
# UI: Prediksi dan Evaluasi
# ============================
st.subheader("Pilih Model")
model_choice = st.selectbox("Model", ["Markov", "LSTM Per-Digit", "GRU Per-Digit"])
input_angka = st.text_input("Masukkan angka terakhir:", value=angka_list[-1])

if model_choice == "Markov":
    prediksi = prediksi_markov(input_angka, n=5)
    st.success(f"🎯 Prediksi Markov: {', '.join(prediksi)}")
elif model_choice == "LSTM Per-Digit":
elif model_choice == "GRU Per-Digit":
    benar = {'top1': 0, 'top3': 0, 'top5': 0}
    total = 0
    for i in range(len(angka_list) - jumlah_uji - 1, len(angka_list) - 1):
        try:
            potongan = angka_list[:i+1]
            X = angka_to_digit_array(potongan)
            y = X[1:]
            X = X[:-1]
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            y_scaled = scaler.transform(y)
            gen = TimeseriesGenerator(X_scaled, y_scaled, length=5, batch_size=1)
            model = Sequential()
            model.add(GRU(64, activation='relu', input_shape=(5, 4)))
            model.add(Dense(4))
            model.compile(optimizer='adam', loss='mse')
            model.fit(gen, epochs=10, verbose=0)
            last_seq = X_scaled[-5:]
            pred_all = prediksi_multi(model, scaler, last_seq, n=5)
            target = angka_list[i+1]
            if target == pred_all[0]: benar['top1'] += 1
            if target in pred_all[:3]: benar['top3'] += 1
            if target in pred_all: benar['top5'] += 1
            total += 1
        except Exception as e:
            st.warning(f"GRU error di index {i}: {e}")
            continue
    acc = {k: round(v/total*100,2) for k,v in benar.items()}
    st.markdown("### 📊 Hasil Akurasi GRU:")
    st.info(f"🎯 Top-1: {acc['top1']}%")
    st.info(f"🎯 Top-3: {acc['top3']}%")
    st.info(f"🎯 Top-5: {acc['top5']}%")

elif model_choice == "GRU Per-Digit":
    try:
        X = angka_to_digit_array(angka_list)
        y = X[1:]
        X = X[:-1]
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        y_scaled = scaler.transform(y)
        gen = TimeseriesGenerator(X_scaled, y_scaled, length=5, batch_size=1)
        model = Sequential()
        model.add(GRU(64, activation='relu', input_shape=(5, 4)))
        model.add(Dense(4))
        model.compile(optimizer='adam', loss='mse')
        model.fit(gen, epochs=10, verbose=0)
        last_seq = X_scaled[-5:]
        prediksi = prediksi_multi(model, scaler, last_seq, n=5)
        st.success(f"🎯 Prediksi GRU: {', '.join(prediksi)}")
    except Exception as e:
        st.error(f"❌ Gagal prediksi GRU: {e}")

    try:
        X = angka_to_digit_array(angka_list)
        y = X[1:]
        X = X[:-1]
        model, scaler, X_scaled = train_lstm_model(X, y)
        last_seq = X_scaled[-5:]
        prediksi = prediksi_multi(model, scaler, last_seq, n=5)
        st.success(f"🎯 Prediksi LSTM: {', '.join(prediksi)}")
    except Exception as e:
        st.error(f"❌ Gagal prediksi: {e}")

# ============================
# Uji Akurasi
# ============================
st.markdown("---")
st.subheader("🔍 Uji Akurasi Model")
jumlah_uji = st.slider("Jumlah data terakhir untuk uji akurasi", min_value=5, max_value=min(50, len(angka_list)-6), value=10)

if st.button("🔍 Jalankan Uji Akurasi"):
    if model_choice == "Markov":
        acc = hitung_akurasi_markov(angka_list, jumlah_uji)
        st.info(f"🎯 Akurasi Markov (Top-5): {acc}%")
    elif model_choice == "LSTM Per-Digit":
elif model_choice == "GRU Per-Digit":
    benar = {'top1': 0, 'top3': 0, 'top5': 0}
    total = 0
    for i in range(len(angka_list) - jumlah_uji - 1, len(angka_list) - 1):
        try:
            potongan = angka_list[:i+1]
            X = angka_to_digit_array(potongan)
            y = X[1:]
            X = X[:-1]
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            y_scaled = scaler.transform(y)
            gen = TimeseriesGenerator(X_scaled, y_scaled, length=5, batch_size=1)
            model = Sequential()
            model.add(GRU(64, activation='relu', input_shape=(5, 4)))
            model.add(Dense(4))
            model.compile(optimizer='adam', loss='mse')
            model.fit(gen, epochs=10, verbose=0)
            last_seq = X_scaled[-5:]
            pred_all = prediksi_multi(model, scaler, last_seq, n=5)
            target = angka_list[i+1]
            if target == pred_all[0]: benar['top1'] += 1
            if target in pred_all[:3]: benar['top3'] += 1
            if target in pred_all: benar['top5'] += 1
            total += 1
        except Exception as e:
            st.warning(f"GRU error di index {i}: {e}")
            continue
    acc = {k: round(v/total*100,2) for k,v in benar.items()}
    st.markdown("### 📊 Hasil Akurasi GRU:")
    st.info(f"🎯 Top-1: {acc['top1']}%")
    st.info(f"🎯 Top-3: {acc['top3']}%")
    st.info(f"🎯 Top-5: {acc['top5']}%")

elif model_choice == "GRU Per-Digit":
    try:
        X = angka_to_digit_array(angka_list)
        y = X[1:]
        X = X[:-1]
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        y_scaled = scaler.transform(y)
        gen = TimeseriesGenerator(X_scaled, y_scaled, length=5, batch_size=1)
        model = Sequential()
        model.add(GRU(64, activation='relu', input_shape=(5, 4)))
        model.add(Dense(4))
        model.compile(optimizer='adam', loss='mse')
        model.fit(gen, epochs=10, verbose=0)
        last_seq = X_scaled[-5:]
        prediksi = prediksi_multi(model, scaler, last_seq, n=5)
        st.success(f"🎯 Prediksi GRU: {', '.join(prediksi)}")
    except Exception as e:
        st.error(f"❌ Gagal prediksi GRU: {e}")

        acc = hitung_akurasi_lstm_digit(angka_list, jumlah_uji)
        st.markdown("### 📊 Hasil Akurasi LSTM:")
        st.info(f"🎯 Top-1: {acc['top1']}%")
        st.info(f"🎯 Top-3: {acc['top3']}%")
        st.info(f"🎯 Top-5: {acc['top5']}%")

st.markdown("---")
st.caption("Aplikasi ini bersifat simulasi & edukatif.")
