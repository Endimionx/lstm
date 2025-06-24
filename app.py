
import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Prediksi Togel AI", layout="centered")
st.title("🎰 Prediksi Togel AI (Markov & LSTM)")
st.markdown("Prediksi angka togel berdasarkan data histori menggunakan dua model: Markov dan LSTM.")

# Upload data histori
st.subheader("Masukkan histori angka 4 digit")
teks_angka = st.text_area("Satu angka per baris", height=200, value="5712\n9701\n1098\n1445\n4431")

if teks_angka:
    angka = [baris.strip().zfill(4) for baris in teks_angka.splitlines() if baris.strip().isdigit()]
else:
    angka = []
if angka:

    data = pd.read_csv(uploaded_file, header=None)
    st.write("Contoh data:")
    st.dataframe(data.head())

    angka = data[0].astype(str).str.zfill(4).tolist()

    # =======================
    # MARKOV MODEL
    # =======================
    transition = defaultdict(list)
    for i in range(len(angka) - 1):
        transition[angka[i]].append(angka[i+1])

    def prediksi_markov(current, n=5):
        candidates = transition.get(current, [])
        if not candidates:
            return [str(random.randint(0, 9999)).zfill(4) for _ in range(n)]
        return random.choices(candidates, k=n)

    # =======================
    # LSTM MODEL
    # =======================
    def train_lstm_model(series):
        series = [int(x) for x in series]
        series = np.array(series).reshape(-1, 1)

        scaler = MinMaxScaler()
        scaled_series = scaler.fit_transform(series)

        n_input = 5
        generator = TimeseriesGenerator(scaled_series, scaled_series, length=n_input, batch_size=1)

        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(n_input, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(generator, epochs=10, verbose=0)

        return model, scaler

    def prediksi_lstm(model, scaler, last_sequence, n=5):
        output = []
        seq = np.array(last_sequence).reshape(-1, 1)
        seq = scaler.transform(seq).flatten().tolist()

        for _ in range(n):
            input_seq = np.array(seq[-5:]).reshape((1, 5, 1))
            pred = model.predict(input_seq, verbose=0)
            seq.append(pred[0][0])
            output.append(int(scaler.inverse_transform(pred)[0][0]))
        return [str(x).zfill(4)[-4:] for x in output]

    # =======================
    # Simulasi Prediksi 100x
    # =======================
    def simulasi_prediksi(model_type, current_input, jumlah=100):
        hasil = []
        if model_type == "Markov":
            for _ in range(jumlah):
                hasil += prediksi_markov(current_input, n=1)
        else:
            try:
                angka_int = [int(a) for a in angka]
                model, scaler = train_lstm_model(angka_int)
                last_seq = angka_int[-5:]
                hasil = prediksi_lstm(model, scaler, last_seq, n=jumlah)
            except Exception as e:
                st.error(f"Error simulasi LSTM: {e}")
        return hasil

    # =======================
    # Interface
    # =======================
    st.subheader("Pilih Model Prediksi")
    model_choice = st.selectbox("Model", ["Markov", "LSTM"])

    input_angka = st.text_input("Masukkan angka terakhir (4 digit):", value=angka[-1])
    if len(input_angka) == 4 and input_angka.isdigit():
        if model_choice == "Markov":
            prediksi = prediksi_markov(input_angka)
        else:
            try:
                angka_int = [int(a) for a in angka]
                model, scaler = train_lstm_model(angka_int)
                last_seq = angka_int[-5:]
                prediksi = prediksi_lstm(model, scaler, last_seq)
            except Exception as e:
                st.error(f"Error melatih model LSTM: {e}")
                prediksi = []

        st.success(f"Hasil prediksi ({model_choice}): {', '.join(prediksi)}")

        with st.expander("🔁 Simulasi 100x Prediksi"):
            hasil_simulasi = simulasi_prediksi(model_choice, input_angka, jumlah=100)
            freq = pd.Series(hasil_simulasi).value_counts().sort_values(ascending=False)
            st.write("Frekuensi angka hasil simulasi:")
            st.dataframe(freq.head(20))
    else:
        st.warning("Masukkan angka 4 digit yang valid.")


st.markdown("---")
st.caption("⚠️ Aplikasi ini hanya bersifat edukatif dan simulasi. Tidak menjamin hasil prediksi.")
