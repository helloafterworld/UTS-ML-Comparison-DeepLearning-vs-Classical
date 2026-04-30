# UTS Machine Learning: Deep Learning vs Classical ML

**Author:** Arif Lukman Hakim 
**NIM:** 1202223003
**Kelas:** SI4602
**Semester:** Genap 2025/2026  

---

## 📋 Ringkasan

Proyek ini membandingkan performa **Deep Learning** dengan **Machine Learning Konvensional** pada tiga kasus klasifikasi dari Kaggle:

| Kasus | Tipe Data | Dataset          | DL Model | Baseline Terbaik |
|-------|-----------|------------------|----------|------------------|
| **1** | Tabular   | Titanic Survival | MLP      | XGBoost          |
| **2** | Image     | MNIST Digit      | CNN      | HOG + SVM        |
| **3** | Text      | Disaster Tweets  | Bi-LSTM  | TF-IDF + SVM     |

---

## 🎯 Hasil Utama

### Kasus 1: Titanic
- 🏆 **Winner:** XGBoost (Accuracy: 82.0%, F1: 0.79)
- ❌ MLP (Accuracy: 79.8%, F1: 0.76)
- **Kesimpulan:** Deep Learning TIDAK mengungguli (dataset terlalu kecil)

### Kasus 2: MNIST  
- 🏆 **Winner:** CNN (Accuracy: 99.2%, F1: 0.991)
- ❌ HOG+SVM (Accuracy: 97.5%, F1: 0.975)
- **Kesimpulan:** Deep Learning MENGUNGGULI signifikan

### Kasus 3: Disaster Tweets
- 🏆 **Winner:** Bi-LSTM (F1: 0.78, Accuracy: 80.8%)
- 📊 TF-IDF+SVM (F1: 0.77, Accuracy: 80.0%)
- **Kesimpulan:** Deep Learning sedikit unggul (margin tipis)

---

## 📁 Dataset

Dataset sudah disertakan dalam folder `data/`:
