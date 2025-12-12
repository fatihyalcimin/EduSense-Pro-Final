# 🎓 EduSense Pro | Real-Time Pedagogical AI

**EduSense Pro**, uzaktan eğitimde eğitmenlere sınıfın duygusal durumunu analiz etmeleri için gerçek zamanlı geri bildirim sağlayan, yapay zeka destekli bir **Karar Destek Sistemidir (DSS)**.

Bu proje, yüksek performanslı bir **SaaS (Software as a Service)** mimarisiyle geliştirilmiştir.

---

## 🏗️ Mimari Yapı

Proje iki ana katmandan oluşur:

1.  **Backend (Beyin):** Python & FastAPI
    * YOLOv8-Small yapay zeka modelini barındırır.
    * Gelen görüntüleri işler, duygu analizi yapar ve JSON formatında sonuç döner.
    * Docker ile konteynerize edilmiştir.

2.  **Frontend (Yüz):** HTML5, CSS3 & JavaScript
    * Kullanıcının tarayıcısında çalışır (Client-Side).
    * Web kamerasını donmadan, gecikmesiz (Zero-Latency) görüntüler.
    * Chart.js ile canlı veri görselleştirmesi yapar.

---

## 🛠️ Kullanılan Teknolojiler

![Python](https://img.shields.io/badge/Backend-FastAPI-009688)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-green)
![Docker](https://img.shields.io/badge/Deploy-Docker-blue)
![JavaScript](https://img.shields.io/badge/Frontend-HTML%2FJS-yellow)

* **Yapay Zeka:** `YOLOv8-Small` (AffectNet ile eğitilmiş, ONNX formatına optimize edilmiş).
* **API:** `FastAPI` (Asenkron görüntü işleme).
* **Görüntü İşleme:** `OpenCV` (CLAHE kontrast artırma).
* **Grafikler:** `Chart.js` (İnteraktif zaman çizelgesi ve histogram).

---

## 🚀 Kurulum ve Çalıştırma

Projeyi yerel makinenizde çalıştırmak için:

### 
1. Terminalde şunları yazın:

pip install -r requirements.txt
python main.py

API şu adreste çalışacaktır: http://localhost:8000

2. Arayüzü Başlatın

index.html dosyasını herhangi bir modern tarayıcıda (Chrome, Edge) açın.


Geliştirici: fatihyalcimin




