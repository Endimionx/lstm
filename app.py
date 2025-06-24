
import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
import altair as alt

st.set_page_config(page_title="Prediksi Togel AI", layout="centered")
st.title("🎰 Prediksi Togel AI (Markov & LSTM)")
st.markdown("Prediksi angka togel berdasarkan data histori menggunakan dua model: Markov dan LSTM.")

# Textarea input
st.subheader("Masukkan histori angka 4 digit")
teks_angka = st.text_area("Satu angka per baris", height=200, value="5712\n9701\n1098\n1445\n4431\n5706\n1092\n1826\n1415\n3784\n2585\n7460\n5061\n0482\n8870\n4736\n6775\n3748\n4403\n2846\n7523\n7981\n2001\n1655\n7002\n9321\n1102\n0874\n9911\n5566")

if teks_angka:
    angka = [baris.strip().zfill(4) for baris in teks_angka.splitlines() if baris.strip().isdigit()]
else:
    angka = []

if len(angka) >= 20:

    transition = defaultdict(list)
    for i in range(len(angka) - 1):
        transition[angka[i]].append(angka[i+1])

    def prediksi_markov(current, n=5):
        candidates = transition.get(current, [])
        if not candidates:
            return [str(random.randint(0, 9999)).zfill(4) for _ in range(n)]
        return random.choices(candidates, k=n)

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

    # 🔍 UJI AKURASI
    st.markdown("---")
    st.subheader("🔍 Uji Akurasi Model")
    jumlah_uji = st.slider("Jumlah data terakhir untuk uji akurasi", min_value=10, max_value=min(50, len(angka)-6), value=20)

    def hitung_akurasi_topN(model_type, data_histori, jumlah_uji=20):
        hasil = {'top1': 0, 'top3': 0, 'top5': 0}
        total = 0
        for i in range(len(data_histori) - jumlah_uji - 1, len(data_histori) - 1):
            input_angka = data_histori[i]
            target = data_histori[i+1]
            if model_type == "Markov":
                prediksi = prediksi_markov(input_angka, n=5)
            else:
                try:
                    potongan = data_histori[:i+1]
                    angka_int = [int(a) for a in potongan]
                    model, scaler = train_lstm_model(angka_int)
                    last_seq = angka_int[-5:]
                    prediksi = prediksi_lstm(model, scaler, last_seq, n=5)
                except Exception as e:
                    st.warning(f"LSTM error @index {i}: {e}")
                    continue
            if target == prediksi[0]:
                hasil['top1'] += 1
            if target in prediksi[:3]:
                hasil['top3'] += 1
            if target in prediksi:
                hasil['top5'] += 1
            total += 1
        if total == 0:
            return {'top1': 0.0, 'top3': 0.0, 'top5': 0.0}
        return {k: round(v / total * 100, 2) for k, v in hasil.items()}

    if st.button("🔍 Jalankan Uji Akurasi"):
        acc_markov = hitung_akurasi_topN("Markov", angka, jumlah_uji)
        acc_lstm = hitung_akurasi_topN("LSTM", angka, jumlah_uji)

        df_chart = pd.DataFrame({
            'Model': ['Markov'] * 3 + ['LSTM'] * 3,
            'Top-N': ['Top-1', 'Top-3', 'Top-5'] * 2,
            'Akurasi (%)': [
                acc_markov['top1'], acc_markov['top3'], acc_markov['top5'],
                acc_lstm['top1'], acc_lstm['top3'], acc_lstm['top5']
            ]
        })

        st.markdown("### 📊 Grafik Akurasi")
        chart = alt.Chart(df_chart).mark_bar().encode(
            x=alt.X('Top-N:N', title='Top-N'),
            y=alt.Y('Akurasi (%):Q'),
            color='Model:N',
            column='Model:N'
        ).properties(height=300)
        st.altair_chart(chart)

        st.markdown("### 📋 Detail Akurasi")
        st.write(df_chart.pivot(index='Top-N', columns='Model', values='Akurasi (%)'))

else:
    st.warning("Masukkan minimal 20 angka histori.")
