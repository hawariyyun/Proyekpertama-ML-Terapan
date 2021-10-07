# Proyek-ML-Terapan
# Data Diri
  Rifki Ramadani
  Machine Learning Terapan
  Kabupaten Pasaman, Sumatera Barat
  
# Domain Permasalahan
  **Prediksi Harga Saham Menggunakan Long-Sort Term Memory Recurrent Neural Network : Studi Kasus Saham Aneka Tambang Tbk**
  PT Aneka Tambang Tbk adalah anak perusahaan BUMN pertambangan Inalum. Aneka Tambang Tbk memiliki aset yang diperjual-belikan pada pasar investasi berupa saham, dalam dunia ekonomi dan bisnis menjadi sebuah bidang yang tidak bisa ditinggalkan dalam kehidupan. Ekonomi dan bisnis berhubungan erat dengan sebuah proses yang dinamakan jual-beli. Pada masa sekarang terdapat sebuah proses jual-beli yang sangat populer dan menjanjikan bagi pelaku jual-belinya. Proses tersebut adalah jual-beli bukti kepemilikan sebuah perusahaan (saham).
  Jual-beli saham dengan istilahnya *trading* atau investasi saham, menjadi hal yang dapat memberikan keuntungan atau kerugian yang besar bagi para pelakunya. Jika seorang salah dalam menganalisis kemungkinan naik dan turunnya harga saham, maka dapat mengakibatkan kerugian. Naik-turunnya harga ini dapat dipengaruhi berbagai hal, seperti adanya fluktuasi kurs rupiah terhadap mata uang asing, kebijakan pemerintah, faktor fundamental perusahaan, dan sebagainya.
  (Shunrong dkk., 2012) menyatakan bahwa analisis dan prediksi menggunakan Machine Learning terhadap saham sudah menjadi permasalahan yang terjadi sejak lama, dan mengemukakan algoritma Support Vector Machine dalam pemecahan masalahnya dengan akurasi melebihi 74% pada prediksi tiga saham.
Menggunakan algoritma machine learning untuk permasalahan prediksi harga saham dapat membantu para pelaku bisnis ini dalam mengambil keputusan prediksinya, beberapa algoritma yang dapat diandalkan adalah SVM dan Radial Basis Function (RBF) (Kranthi, 2018).
Penggunaan Machine Learning seperti SVM, RBF, Recurrent Neural Network : Long-Sort Term Memory dan sebagainya dalam prediksi merupakan sebuah hal yang dapat diterapkan untuk menghindarkan pelaku bisnis dari pengambilan keputusan pembelian saham yang tidak beralasan atau asal-asalan sehingga dapat meminimalkan kerugian.
**REFERENSI :**

1.   Shen, Shunrong, Dkk. 2012. Stock Market Forecasting Using Machine Learning Algorithms
2.   Kranthi, V S.R. 2018. Stock Market Prediction Using Machine Learning. *International Research Journal of Engineering and Technology (IRJET)*, 5(10), pp. 1032-1035 

# Memahami Segi Bisnis
  ## Permasalahan
1.   Bagaimana algoritma LSTM dapat melakukan prediksi terhadap harga saham terkini?
2.   Bagaimana akurasi prediksi algoritma RNN:LSTM terhadap harga saham?
  ## Tujuan
1.   Membuat model machine learning yang dapat melakukan prediksi harga saham dengan nilai kesalahan atau error sekecil mungkin
2.   Mengetahui perbandingan error antara harga saham sebenarnya dengan harga prediksi model LSTM
  ## Solusi
1.   Menggunakan algoritma Recurrent Neural Network : LSTM (Long-Short Term Memory)
2.   Menggunakan metriks evaluasi MSE, MAE, dan RMSE untuk mengetahui besarnya selisih antara harga saham sebenarnya dengan harga saham hasil prediksi model LSTM

# Memahami Data
  Dataset diberi nama 'anekatambangtbk.csv', merupakan dataset harga saham harian yang saat itu diperdagangkan ANTM.JK bahkan sampai saat ini. Dataset berupa dataset time series dengan rentang waktu antar 29 September 2005 sampai 3 Februari 2021.Dataset yang digunakan berasal dari website penyedia dataset kaggle.com dengan detail url : https://www.kaggle.com/muhardianabasandi/antam-stock-market-by-kitto
  ## Penjelasan Fitur
  Fitur-fitur yang tersedia pada dataset :

*   Date : Merupakan fitur/variabel penyimpan berupa timeseries tahun 2005 sampai 2021
*   Open : Harga pembuka harian
*   High : Harga tertinggi harian
*   Low : Harga terendah harian
*   Close : Harga penutupan harian
*   Adj. Close : Penyesuaian harga penutupan harian
*   Volume : Jumlah saham yang diperjualbelikan
  ## Import Library
  Library yang dibutuhkan selama pembangunan model prediksi
  ## Membaca Data
  Dataset dibaca menggunakan library pandas dan melakukan pemisahan antara fitur tanggal dengan fitur lainnya. Setelah membaca dataset, dataset tersebut dideskripsikan menggunakan fungsi describe() yang telah disediakan library pandas
  ## Visualisasi awal dataset kolom 'Close'
  Visualisasi data menggunakan nilai pada kolom 'Close', karena variabel/fitur ini akan kita gunakan sebagai acuan dalam prediksi. Kolom 'Close' digunakan karena prediksi/forecasting yang akan dilakukan pada dataset ini adalah prediksi terhadap harga saham penutupan harian, hal ini berguna sebagai acuan bagi investor dalam melakukan transaksi (*trading*) maupun investasi dalam jangka waktu harian. Jadi, prediksi yang akan dilakukan berupa prediksi saham harian, harga penutup menjadi penentu apakah investor maupun trader untung atau rugi pada hari itu
  
# Persiapan Data
  ## Membersihkan data
  Melakukan pengecekan pada data, apakah terdapat data dengan nilai kosong pada dataset menggunakan fungsi isnull(). Selanjutnya jika terdapat data kosong, data kosong tersebut dibuang menggunakan fungsi drop()
  ## Membagi data menjadi train_data dan test_data
  Sebelum melakukan pembagian data, lakukan pengecekan terhadap persantase nilai selisih harian dari harga penutupan saham, menggunakan visualisasi persentase selisih harga saham penutupan harian. Selanjutnya, membagi dataset menjadi 80% train_data dan 20% test_data beserta visualisasinya.
  Untuk proses training kita menggunakan data pada kolom 'Close'
  ## Normalisasi Menggunakan MinMaxScaler()
  Normalisasi dibutuhkan supaya data berada pada rentang 0 sampai 1, sehingga dapat mempermudah proses kalkulasi pada model.

# Pembuatan Model
  ##Pembuatan model RNN:LSTM
  Penggunaan LSTM karena seiring bertambahnya dataset, LSTM dapat mengingat kembali hasil pelatihan yang lama (memori tidak terkikis), berbeda dengan RNN sederhana yang dapat kehilangan performa seiring bertambahnya data.
  Melakukan summary terhadap model.
  Terdapat 4 layer LSTM pada model.
  ## Pelatihan pada data latih (*Training*)
*   Melakukan deklarasi callbacks EarlyStopping untuk memperkecil resourse yang digunakan
*   Melakukan kompilasi terhadap model menggunakan parameter optimizer = 'adam' dan loss = 'mean_squared_error'
*   Melakukan pelatihan (training) menggunakan model.fit dengan parameter X_train, y_train, nilai epochs=500, step_per_epoch=48, dan batch_size=64
  ## Prediksi
  Prediksi dilakukan dengan menggabungkan train_data dan test_data, setelah itu mengambil test_inputs dan melakukan skalar/normalisasi pada test_inputs.
  Mengambil nilai X_test dari test_data berdasarkan test_inputs dan melakukan prediksi dengan menggunakan fungsi model.predict() dan melakukan skalar terhadap hasil prediksi agar kembali menjadi bentuk asli dari data.

# Evaluasi
  ## Visualisasi harga saham pada pelatihan, harga sebenarnya dan harga prediksi
  Sebelum melakukan perhitungan terhadap error pada hasil prediksi, lakukan visualisasi hasil prediksi terlebih dahulu agar dapat memahami seberapa besar perbandingan hasil prediksi dan harga sebenarnya.
  ### MSE, MAE, dan RMSE
  Perhitungan selisih antara harga saham hasil prediksi dan harga saham sebenarnya menggunakan tiga metriks evaluasi yaitu MSE, MAE, dan RMSE.
  Nilai yang didapat masing-masingnya :
  * mse   : 2160.29893007192
  * mae   : 24.989811694528175
  * rmse  : 46.47901601875754
  Dapat diketahui nilai pada masing-masing variabel evaluasi menunjukkan nilai yang besar, namun ini tidak menjadi masalah karena nilai/harga pada data berupa nilai/harga yang cukup besar berkisar antar Rp.287-Rp.4241. pada **MAE** selisih rata-rata antar harga prediksi dan harga sebenarnya diperoleh sebesar 24.9 yang bisa dikatakan sangat kecil mendekati Rp.25
  Sedangkan pada **RMSE** menunjukkan nilai selisih rata-rata sebesar 46.5 lebih kecil dari Rp.50
  
  ## Prediksi harga saham antm pada hari berikutnya (4 Februari 2021)
  Harga saham prediksi pada hari selanjutnya yaitu pada 4 Februari 2021 adalah Rp.2235
