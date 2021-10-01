# Deskripsi: Ini adalah program untuk mendeteksi apakah seseorang menderita diabetes atau tidak menggunakan ML dan Python

# Import Library yang digunakan
import pandas as pd
import numpy as np

from PIL import Image
import streamlit as st
import pickle


# Menampilkan Gambar
# image =Image.open('D:/My Activity/Python Notebook/Streamlit/[4] Klasifikasi KIP/header KIP.png')
# st.image(image, caption='ML', use_column_width=True)

# Judul & Sub Judul Halaman Web App
st.header("""
Model Klasifikasi untuk Pendaftaran Program Beasiswa KIP.
""")

st.write("""
Ini merupakan program untuk melakukan prediksi penerimaan siswa calon peserta program KIP.
""")

st.write("""
**Author**: Nursyiva Irsalinda, M.Sc
""")

# load the model from disk
filename = 'model_klasifikasi_KIP.pkl'
lm = pickle.load(open(filename, 'rb'))

ranking_sem_4=st.number_input("Rangking Semester 4")
nilai_rata2_sem_4=st.number_input("Nilai Rata2 Semester 4")
ranking_sem_5=st.number_input("Rangking Semester 5")
nilai_rata2_sem_5=st.number_input("Nilai Rata2 Semester 5")
ranking_sem_6=st.number_input("Rangking Semester 6")
nilai_rata2_sem_6=st.number_input("Nilai Rata2 Semester 6")
hutang_kpd_pihak_lain=st.number_input("Hutang Kepada Pihak Lain")
cicilan_hutang_bulanan= st.number_input("Hutang Bulanan")
total_piutang=st.number_input("Total Piutang ")
cicilan_piutang_dr_pihak_lain = st.number_input("Cicilan Piutang")


dict_input = {'Rangking Semester 4':ranking_sem_4,
 'Nilai Rata2 Semester 4':nilai_rata2_sem_4,
 'Rangking Semester 5':ranking_sem_5,
 'Nilai Rata2 Semester 5':nilai_rata2_sem_5,
 'Rangking Semester 6':ranking_sem_6,
 'Nilai Rata2 Semester 6':nilai_rata2_sem_6,
 'Hutang Kepada Pihak Lain':hutang_kpd_pihak_lain,
 'Hutang Bulanan':cicilan_hutang_bulanan,
 'Total Piutang':total_piutang,
 'Cicilan Piutang':cicilan_piutang_dr_pihak_lain}

X_input = pd.DataFrame(dict_input, index=[0])

pred = lm.predict(np.array(X_input)) 

hasil_klasifikasi = {0: 'Tidak Diterima', 1:'Diterima'}

# Menambahkan subheader dan menampilkan hasil klasifikasi data pengguna
st.subheader('Hasil Klasifikasi:')
st.write(hasil_klasifikasi[int(pred)])
