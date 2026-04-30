# 📊 UTS Machine Learning: Deep Learning vs Classical ML

**Perbandingan Performa Deep Learning dan Metode Konvensional pada 3 Kasus Klasifikasi**

---

## 📋 Tentang Proyek

Proyek ini merupakan pengerjaan Ujian Tengah Semester mata kuliah Machine Learning. Tujuan utama adalah membandingkan performa **Deep Learning** dengan **Machine Learning Konvensional** pada tiga tipe data yang berbeda:

| Kasus | Tipe Data | Dataset | DL Model | Baseline Terbaik |
|-------|-----------|---------|----------|------------------|
| **1** | Tabular | Titanic Survival | MLP | XGBoost |
| **2** | Image | MNIST Digit | CNN | HOG + SVM |
| **3** | Text | Disaster Tweets | Bi-LSTM | TF-IDF + LR |

---

## 🎯 Hasil Utama

### Kasus 1: Titanic (Tabular)
- 🏆 **Winner:** XGBoost (Accuracy: **83.4%**, F1: 0.79)
- ❌ MLP (Accuracy: 82.7%, F1: 0.76)
- **Kesimpulan:** Deep Learning TIDAK mengungguli (dataset terlalu kecil, 891 sampel)

### Kasus 2: MNIST (Image)
- 🏆 **Winner:** CNN (Accuracy: **99.2%**, F1: 0.99)
- ❌ HOG+SVM (Accuracy: 97.5%, F1: 0.975)
- **Kesimpulan:** Deep Learning MENGUNGGULI signifikan

### Kasus 3: Disaster Tweets (NLP)
- 🏆 **Winner:** TF-IDF + Logistic Regression (F1: **0.767**)
- 📊 Bi-LSTM (F1: 0.744)
- **Kesimpulan:** Deep Learning TIDAK mengungguli (dataset kecil, 7.6K tweets)

---

## 📁 Struktur Repository
