import streamlit as st
import numpy as np
import random
from collections import defaultdict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Prediksi Togel AI", layout="centered")
st.title("🎰 Prediksi Togel AI - Markov / LSTM / GRU")

# ==== INPUT ====
st.subheader("Masukkan histori angka 4 digit")
teks_angka = st.text_area("Satu angka per baris:", height=250, value="""7123
4012
6321
1980
3124
8945
1098
7632
5412
1093
8842
3381
2764
0012
5678
4839
7021
1593
2493
3192
9911
8822
1763
4091
2631
7028
1832
3840
1193
8092
1930
3984""")

angka_list = [x.strip().zfill(4) for x in teks_angka.splitlines() if x.strip().isdigit()]
if len(angka_list) < 10:
    st.warning("Masukkan minimal 10 angka histori.")
    st.stop()

def angka_to_digit_array(data):
    return np.array([[int(d) for d in list(a)] for a in data])

# ==== PILIH MODEL ====
st.subheader("Pilih Model")
model_choice = st.selectbox("Model", ["Markov", "LSTM Digit", "GRU Digit"])
input_angka = st.text_input("Masukkan angka terakhir:", value=angka_list[-1])

# ==== MARKOV ====
transition = defaultdict(list)
for i in range(len(angka_list) - 1):
    transition[angka_list[i]].append(angka_list[i+1])

def prediksi_markov(current, n=5):
    return random.choices(transition.get(current, []) or [str(random.randint(0, 9999)).zfill(4)], k=n)

# ==== LSTM / GRU ====
def train_model(data, use_gru=False):
    X = angka_to_digit_array(data[:-1])
    y = angka_to_digit_array(data[1:])
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)
    ys = scaler.transform(y)
    gen = TimeseriesGenerator(Xs, ys, length=5, batch_size=1)
    model = Sequential()
    model.add((GRU if use_gru else LSTM)(64, activation='relu', input_shape=(5, 4)))
    model.add(Dense(4))
    model.compile(optimizer='adam', loss='mse')
    model.fit(gen, epochs=10, verbose=0)
    return model, scaler, Xs

def prediksi_multi(model, scaler, last_seq, n=5):
    out = []
    for _ in range(n):
        p = model.predict(np.expand_dims(last_seq, axis=0), verbose=0)
        d = np.round(scaler.inverse_transform(p)).astype(int).flatten()
        out.append(''.join([str(min(max(0, x), 9)) for x in d]))
    return out

# ==== PREDIKSI ====
st.subheader("Prediksi")
try:
    if model_choice == "Markov":
        pred = prediksi_markov(input_angka)
    else:
        model, scaler, Xs = train_model(angka_list, use_gru=(model_choice == "GRU Digit"))
        pred = prediksi_multi(model, scaler, Xs[-5:], n=5)
    st.success(f"🎯 Prediksi: {', '.join(pred)}")
except Exception as e:
    st.error(f"❌ Gagal prediksi: {e}")

# ==== AKURASI ====
st.subheader("🔍 Uji Akurasi")
jumlah_uji = st.slider("Jumlah data untuk uji akurasi", 5, min(50, len(angka_list)-6), 10)

def hitung_akurasi(data, model_type="LSTM"):
    if len(data) < jumlah_uji + 6:
        st.error("❌ Data tidak cukup untuk uji akurasi. Masukkan setidaknya " + str(jumlah_uji + 6) + " angka.")
        return {'top1': 0.0, 'top3': 0.0, 'top5': 0.0}

    benar = {'top1': 0, 'top3': 0, 'top5': 0}
    total = 0
    for i in range(len(data) - jumlah_uji - 1, len(data) - 1):
        try:
            input_val = data[i]
            target = data[i + 1]
            if model_type == "Markov":
                hasil = prediksi_markov(input_val, n=5)
            else:
                potong = data[:i + 1]
                model, scaler, Xs = train_model(potong, use_gru=(model_type == "GRU"))
                hasil = prediksi_multi(model, scaler, Xs[-5:], n=5)

            if target == hasil[0]:
                benar['top1'] += 1
            if target in hasil[:3]:
                benar['top3'] += 1
            if target in hasil:
                benar['top5'] += 1
            total += 1
        except Exception as e:
            st.warning(f"Gagal prediksi @index {i}: {e}")
    if total == 0:
        st.error("Tidak cukup data untuk menghitung akurasi.")
        return {'top1': 0.0, 'top3': 0.0, 'top5': 0.0}
    return {k: round(v / total * 100, 2) for k, v in benar.items()}

if st.button("🔍 Jalankan Uji Akurasi"):
    mode = "Markov" if model_choice == "Markov" else ("GRU" if model_choice == "GRU Digit" else "LSTM")
    acc = hitung_akurasi(angka_list, model_type=mode)
    st.markdown("### 📊 Akurasi:")
    st.info(f"Top-1: {acc['top1']}%")
    st.info(f"Top-3: {acc['top3']}%")
    st.info(f"Top-5: {acc['top5']}%")

st.caption("Aplikasi ini hanya untuk simulasi & edukasi.")
