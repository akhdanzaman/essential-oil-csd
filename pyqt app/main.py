
import sys
import time
from PyQt6.QtWidgets import QApplication, QSplashScreen # QSplashScreen sebagai fallback
from PyQt6.QtGui import QPixmap

# 1. Impor kelas SplashScreen yang ringan
from splash_screen import ModernSplashScreen
from PyQt6.QtWebEngineWidgets import QWebEngineView

try:
    import joblib
    def change_jolib_parallel_prefer(prefer='threads'):
        _original_init = joblib.parallel.Parallel.__init__
        def _monkey_patched_init(self, *args, **kwargs):
            kwargs['prefer'] = prefer
            _original_init(self, *args, **kwargs)
        joblib.parallel.Parallel.__init__ = _monkey_patched_init
    change_jolib_parallel_prefer()
    print("Joblib monkey-patch diterapkan.")
except ImportError:
    print("Joblib tidak ditemukan, patch dilewati.")
# --- Akhir Monkey-Patch ---

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 1. Coba impor ModernSplashScreen, gunakan QSplashScreen standar jika gagal.
    try:
        from splash_screen import ModernSplashScreen
        splash = ModernSplashScreen()
    except ImportError:
        print("PERINGATAN: ModernSplashScreen tidak ditemukan, menggunakan QSplashScreen standar.")
        pixmap = QPixmap("icon.png")
        splash = QSplashScreen(pixmap)
        
    splash.show()
    
    def update_splash(message, progress):
        """Fungsi helper untuk memperbarui splash screen."""
        if hasattr(splash, 'set_message'):
            splash.set_message(message)
            # splash.set_progress(progress)
        else: # Fallback untuk QSplashScreen standar
            splash.showMessage(message, Qt.AlignmentFlag.AlignBottom, Qt.GlobalColor.white)
        QApplication.processEvents()

    # 2. Proses import secara bertahap sambil memperbarui splash screen
    update_splash("Memuat Pustaka Inti (Pandas, Numpy)...", 10)
    import pandas as pd
    import numpy as np
    import requests
    import pickle
    
    update_splash("Memuat Pustaka GUI...", 25)
    from PyQt6.QtWidgets import QMainWindow # Hanya impor yang diperlukan di sini
    from PyQt6.QtGui import QIcon
    
    update_splash("Memuat Komponen Machine Learning...", 40)
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    from hdbscan import HDBSCAN
    
    update_splash("Memuat Pustaka NLP...", 60)
    # Ini adalah bagian yang paling lama dan paling memakan sumber daya
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    
    update_splash("Memuat Pustaka Visualisasi...", 75)
    import matplotlib.pyplot as plt
    import shap
    
    update_splash("Menginisialisasi Jendela Utama...", 90)
    from pyqt import MainWindow # Impor utama dari file aplikasi Anda

    # 3. Inisialisasi jendela utama
    window = MainWindow()

    # 4. Tutup splash screen dan tampilkan jendela utama
    update_splash("Selesai!", 100)
    time.sleep(0.5) # Beri jeda sejenak agar pesan 'Selesai' terlihat
    
    if hasattr(splash, 'close_splash'):
        splash.close_splash(window)
    else:
        splash.finish(window)
        window.show()
    
    # 5. Jalankan aplikasi
    sys.exit(app.exec())
