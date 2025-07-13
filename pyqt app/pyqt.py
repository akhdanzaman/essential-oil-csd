import sys
import requests
import pandas as pd
from PyQt6.QtWidgets import *
import multiprocessing
sys.setrecursionlimit(100000)

print("DONE IMPORTING")
import matplotlib.pyplot as plt
# import nltk
import re
import ast
import joblib
import pickle
from datetime import datetime
import string
import numpy as np
import warnings
print("DONE IMPORTING")

from bertopic import BERTopic
print("DONE BERT")

from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
print("DONE IMPORTING")
# import transformers

print("DONE IMPORTING")
from bertopic.representation import KeyBERTInspired, PartOfSpeech, MaximalMarginalRelevance 

print("DONE IMPORTING")

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDate
from PyQt6.QtGui import QPixmap, QFont, QIcon
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import shap
from PyQt6.QtWebEngineWidgets import QWebEngineView
import tempfile

import os
from io import BytesIO
def change_jolib_parallel_prefer(prefer='threads'):
    """
    Mengubah metode __init__ dari joblib.parallel.Parallel untuk
    selalu memprioritaskan backend 'threads'.
    """
    # Simpan metode __init__ yang asli
    _original_init = joblib.parallel.Parallel.__init__

    # Buat fungsi __init__ baru yang sudah ditambal
    def _monkey_patched_init(self, *args, **kwargs):
        # Paksa nilai 'prefer' menjadi 'threads'
        kwargs['prefer'] = prefer
        # Panggil metode __init__ yang asli dengan argumen yang sudah diubah
        _original_init(self, *args, **kwargs)

    # Ganti metode __init__ yang asli dengan versi yang sudah ditambal
    joblib.parallel.Parallel.__init__ = _monkey_patched_init

# Panggil fungsi patch segera setelah didefinisikan
change_jolib_parallel_prefer()

print("DONE IMPORTING")


print("DONE IMPORTING")

import pandas as pd
import numpy as np
print("DONE IMPORTING")

import pandas as pd
from openai import OpenAI
import time
print("DONE IMPORTING")
import json
import base64 # Untuk konversi ke base64


OPENAI_REPRESENTATION_AVAILABLE = False
try:
    from bertopic.representation import OpenAI as BERTopicOpenAI
    OPENAI_REPRESENTATION_AVAILABLE = True
except ImportError:
    print("BERTopic OpenAI representation model not available. Install with 'pip install bertopic[openai]'")

# Untuk Plotly (perlu diinstal: pip install plotly)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("WARNING: Plotly tidak terinstal. Visualisasi interaktif tidak akan berfungsi.")
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Peringatan: Pustaka OpenAI tidak terinstal. Representasi topik via OpenAI tidak akan berfungsi.")

# Untuk Text Analysis (opsional, perlu sklearn)
try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt # Untuk menyimpan WordCloud ke buffer
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("WARNING: WordCloud atau Matplotlib tidak terinstal. Word Cloud tidak akan berfungsi.")


# --- Helper Function ---
def matplotlib_to_base64_html_img(fig):
    """Konversi figure Matplotlib ke string HTML dengan gambar PNG base64."""
    if not fig.get_axes(): # Jika figure kosong
        plt.close(fig) # Tutup figure kosong untuk membebaskan memori
        return "<p style='text-align:center; color:red;'>Plot tidak dapat dibuat (figure kosong).</p>"
    
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100) # dpi bisa disesuaikan
    plt.close(fig)  # PENTING: Tutup figure untuk membebaskan memori
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    # style ditambahkan untuk memastikan gambar responsif dan tidak terlalu besar
    return f'<div style="text-align:center;"><img src="data:image/png;base64,{img_base64}" alt="Plot" style="max-width:95%; height:auto; border:1px solid #ddd;"></div>'
def placeholder_preprocess_documents(doc_list):
    processed = []
    if not doc_list: return []
    for doc in doc_list:
        if isinstance(doc, str):
            text = doc.lower()
            text = re.sub(r'\W+', ' ', text) # Hapus non-alphanumeric, ganti dengan spasi
            text = ' '.join(text.split()) # Hapus spasi berlebih
            processed.append(text)
        else:
            processed.append("")
    return processed

# QWebEngineView untuk menampilkan HTML
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebEngineCore import QWebEngineSettings # Untuk Javascript
    PYQTWEBENGINE_AVAILABLE = True
except ImportError:
    PYQTWEBENGINE_AVAILABLE = False

class HomePage(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        title = QLabel("Satisfaction Analysis Suite")
        title.setFont(QFont('Arial', 24))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        description = QLabel(
            "This application allows you to scrape product data and reviews from Ecommerce.\n\n"
            "Navigate to the Scraper page to start collecting data."
        )
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setWordWrap(True)
        
        layout.addStretch(1)
        layout.addWidget(title)
        layout.addWidget(description)
        layout.addStretch(1)
        
        self.setLayout(layout)


REVIEW_URL = "https://gql.tokopedia.com/graphql/productReviewList"

# # --- Product Scraper Worker ---
class ProductScraperWorker(QThread):
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int) # items_scraped_so_far, current_page_being_processed

    def __init__(self, keyword, max_pages_to_scrape, search_id, parent=None):
        super().__init__(parent)
        self.keyword = [s.strip() for s in keyword.split(",")]
        # print(self.keyword)
        self.max_pages_to_scrape = max_pages_to_scrape
        self.search_id = search_id
        self.is_running = True

    def _get_params(self,k):
        params_list = []
        # print(j)
        for i in range(1, self.max_pages_to_scrape + 1):
            param = f"device=desktop&l_name=sre&navsource=&ob=5&page={i}&q={k}&related=true&rows=200&safe_search=false&sc={self.search_id}&scheme=https=&shipping=&show_adult=false&source=search&srp_component_id=04.06.00.00&srp_page_id=&srp_page_title=&st=product&start={i*200}&topads_bucket=true&unique_id=170ef45885cd137480a7a34202ecf991&user_id=15548711&variants="
            params_list.append(param)
            print(param)
        return params_list

    def _scrape_single_page_data(self, param,k):
        headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

        payload = [{
            "operationName":"SearchProductV5Query",
            "variables": {
                "params":param

            },
            "query":"query SearchProductV5Query($params: String!) {\n  searchProductV5(params: $params) {\n    header {\n      totalData\n      responseCode\n      keywordProcess\n      keywordIntention\n      componentID\n      isQuerySafe\n      additionalParams\n      backendFilters\n      __typename\n    }\n    data {\n      totalDataText\n      banner {\n        position\n        text\n        applink\n        url\n        imageURL\n        componentID\n        trackingOption\n        __typename\n      }\n      redirection {\n        url\n        __typename\n      }\n      related {\n        relatedKeyword\n        position\n        trackingOption\n        otherRelated {\n          keyword\n          url\n          applink\n          componentID\n          products {\n            oldID: id\n            id: id_str_auto_\n            name\n            url\n            applink\n            mediaURL {\n              image\n              __typename\n            }\n            shop {\n              oldID: id\n              id: id_str_auto_\n              name\n              city\n              tier\n              __typename\n            }\n            badge {\n              oldID: id\n              id: id_str_auto_\n              title\n              url\n              __typename\n            }\n            price {\n              text\n              number\n              __typename\n            }\n            freeShipping {\n              url\n              __typename\n            }\n            labelGroups {\n              position\n              title\n              type\n              url\n              styles {\n                key\n                value\n                __typename\n              }\n              __typename\n            }\n            rating\n            wishlist\n            ads {\n              id\n              productClickURL\n              productViewURL\n              productWishlistURL\n              tag\n              __typename\n            }\n            meta {\n              oldWarehouseID: warehouseID\n              warehouseID: warehouseID_str_auto_\n              componentID\n              __typename\n            }\n            __typename\n          }\n          __typename\n        }\n        __typename\n      }\n      suggestion {\n        currentKeyword\n        suggestion\n        query\n        text\n        componentID\n        trackingOption\n        __typename\n      }\n      ticker {\n        oldID: id\n        id: id_str_auto_\n        text\n        query\n        applink\n        componentID\n        trackingOption\n        __typename\n      }\n      violation {\n        headerText\n        descriptionText\n        imageURL\n        ctaURL\n        ctaApplink\n        buttonText\n        buttonType\n        __typename\n      }\n      products {\n        oldID: id\n        id: id_str_auto_\n        name\n        url\n        applink\n        mediaURL {\n          image\n          image300\n          videoCustom\n          __typename\n        }\n        shop {\n          oldID: id\n          id: id_str_auto_\n          name\n          url\n          city\n          tier\n          __typename\n        }\n        badge {\n          oldID: id\n          id: id_str_auto_\n          title\n          url\n          __typename\n        }\n        price {\n          text\n          number\n          range\n          original\n          discountPercentage\n          __typename\n        }\n        freeShipping {\n          url\n          __typename\n        }\n        labelGroups {\n          position\n          title\n          type\n          url\n          styles {\n            key\n            value\n            __typename\n          }\n          __typename\n        }\n        labelGroupsVariant {\n          title\n          type\n          typeVariant\n          hexColor\n          __typename\n        }\n        category {\n          oldID: id\n          id: id_str_auto_\n          name\n          breadcrumb\n          gaKey\n          __typename\n        }\n        rating\n        wishlist\n        ads {\n          id\n          productClickURL\n          productViewURL\n          productWishlistURL\n          tag\n          __typename\n        }\n        meta {\n          oldParentID: parentID\n          parentID: parentID_str_auto_\n          oldWarehouseID: warehouseID\n          warehouseID: warehouseID_str_auto_\n          isImageBlurred\n          isPortrait\n          __typename\n        }\n        __typename\n      }\n      __typename\n    }\n    __typename\n  }\n}\n"

        }]
        try:
            req = requests.post("https://gql.tokopedia.com/graphql/SearchProductV5Query", json=payload, headers=headers, timeout=20).json()
            rows = req[0]['data']['searchProductV5']['data']['products']
            scraped_page_data = []
            for row in rows:
                scraped_page_data.append({
                    "product_id": row.get("id"), "old_product_id": row.get("meta", {}).get("parentID"),
                    "nama_produk": row.get("name"), "harga": row.get("price", {}).get("number"),
                    "rating": row.get("rating"), "toko": row.get("shop", {}).get("name"),
                    "toko_url": row.get("shop", {}).get("url"), "lokasi": row.get("shop", {}).get("city"),
                    "product_url": row.get("url"), "category": row.get("category", {}).get("name"),
                    "keyword": str(k)
                })
            last_scrap = len(scraped_page_data)

                # print(str(self.keyword))
                # print(str(self.keyword[j]))
            return scraped_page_data, last_scrap
        except requests.exceptions.Timeout:
            # self.error.emit(f"Timeout while scraping page with param: {param[:30]}...") # Terlalu detail untuk worker
            print(f"Timeout for param: {param[:30]}")
            return []
        except Exception as e:
            # self.error.emit(f"Error scraping page data: {str(e)}") # Akan memicu error utama jika diperlukan
            print(f"Error scraping page: {e}") # Log untuk debugging
            return []

    def run(self):
        all_scraped_data = []
        for k in self.keyword:
            params_to_process = self._get_params(k)
            total_pages = len(params_to_process)
            ukuran_per_kategori = total_pages // len(self.keyword)
            for i, param in enumerate(params_to_process):
                if not self.is_running:
                    break
                page_data, last_scrap = self._scrape_single_page_data(param,k)
                all_scraped_data.extend(page_data)
                self.progress.emit(len(all_scraped_data), i + 1) # (total items, current page processed)
                if last_scrap == 0:
                    break
                # print(j)
                # time.sleep(0.5) # Jeda kecil antar request

        if self.is_running and all_scraped_data:
            df = pd.DataFrame(all_scraped_data)
            df["product_id"] = df["product_id"].astype(str)
            
            if "old_product_id" in df.columns:
                df["old_product_id"] = df["old_product_id"].astype(str)
            
            df = df.drop_duplicates(subset="product_id", keep="first")
            df = df.dropna(how='all')
            df = df.reset_index(drop=True) 

            self.finished.emit(df)
        elif self.is_running: # Selesai tanpa data tapi tidak dihentikan
             self.finished.emit(pd.DataFrame()) # Kirim DataFrame kosong
        # Jika dihentikan, tidak emit apa-apa atau emit error khusus jika perlu

    def stop(self):
        self.is_running = False

# # --- Review Scraper Worker ---
class ReviewScraperWorker(QThread):
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int, int) # reviews_scraped_so_far, current_product_index, total_products

    def __init__(self, product_ids_to_scrape, max_pages_per_review, parent=None):
        super().__init__(parent)
        self.product_ids = product_ids_to_scrape
        self.max_review_pages = max_pages_per_review
        self.is_running = True

    def _fetch_single_review_page(self, product_id, page_num):
        headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

        payload = [{
            'operationName':"productReviewList",
            'query': "query productReviewList($productID: String!, $page: Int!, $limit: Int!, $sortBy: String, $filterBy: String) {\n  productrevGetProductReviewList(productID: $productID, page: $page, limit: $limit, sortBy: $sortBy, filterBy: $filterBy) {\n    productID\n    list {\n      id: feedbackID\n      variantName\n      message\n      productRating\n      reviewCreateTime\n      reviewCreateTimestamp\n      isReportable\n      isAnonymous\n      imageAttachments {\n        attachmentID\n        imageThumbnailUrl\n        imageUrl\n        __typename\n      }\n      videoAttachments {\n        attachmentID\n        videoUrl\n        __typename\n      }\n      reviewResponse {\n        message\n        createTime\n        __typename\n      }\n      user {\n        userID\n        fullName\n        image\n        url\n        __typename\n      }\n      likeDislike {\n        totalLike\n        likeStatus\n        __typename\n      }\n      stats {\n        key\n        formatted\n        count\n        __typename\n      }\n      badRatingReasonFmt\n      __typename\n    }\n    shop {\n      shopID\n      name\n      url\n      image\n      __typename\n    }\n    hasNext\n    totalReviews\n    __typename\n  }\n}\n",
            'variables': {
                'productID': product_id, 
                'page': page_num, 
                'limit': 50, 
                'sortBy': "informative_score desc", 
                'filterBy': ""
                }
        }]       
        try:
            r = requests.post(REVIEW_URL, json=payload, headers=headers, timeout=15)
            r.raise_for_status() # Akan raise exception untuk status 4xx/5xx
            data = r.json()[0]['data']['productrevGetProductReviewList']
            return data.get('list', []), data.get('hasNext', False)
        except requests.exceptions.Timeout:
            print(f"Timeout fetching review page {page_num} for product {product_id}")
            return [], False
        except Exception as e:
            print(f"Error fetching review page {page_num} for product {product_id}: {e}")
            return [], False

    def _scrape_reviews_for_product(self, product_id):
        reviews_for_single_product = []
        for p_num in range(1, self.max_review_pages + 1):
            if not self.is_running:
                break
            page_reviews_data, has_next_page = self._fetch_single_review_page(product_id, p_num)
            if not page_reviews_data:
                break 
            for item in page_reviews_data:
                reviews_for_single_product.append({
                    'product_id': product_id, 'ulasan': item.get('message'),
                    'rating': item.get('productRating'), 'time': item.get('reviewCreateTime')
                })
                # print(reviews_for_single_product)
            if not has_next_page:
                break
            # time.sleep(0.3) # Jeda kecil
        return reviews_for_single_product

    def run(self):
        all_reviews_collected = []
        total_products_to_scrape = len(self.product_ids)

        for idx, pid in enumerate(self.product_ids):
            if not self.is_running:
                break
            reviews_from_product = self._scrape_reviews_for_product(pid)
            all_reviews_collected.extend(reviews_from_product)
            self.progress.emit(len(all_reviews_collected), idx + 1, total_products_to_scrape)
            # (total reviews, current product_idx processed, total products)

        if self.is_running and all_reviews_collected:
            df = pd.DataFrame(all_reviews_collected)
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce') # errors='coerce' untuk handle timestamp salah
                # print(df)
            self.finished.emit(df)
        elif self.is_running:
             self.finished.emit(pd.DataFrame()) # Kirim DataFrame kosong jika tidak ada hasil tapi tidak dihentikan

    def stop(self):
        self.is_running = False

def placeholder_preprocess_documents(docs_list):
    # Ganti dengan fungsi pra-pemrosesan teks Anda yang sebenarnya
    # Contoh sederhana:
    processed = []
    if docs_list is None:
        return processed
    for doc in docs_list:
        if isinstance(doc, str):
            processed.append(doc.lower().strip()) # Lowercase dan strip whitespace
        # Anda mungkin ingin menangani tipe data lain atau log jika bukan string
    return processed


class ScraperPage(QWidget):
    def __init__(self):
        super().__init__()
        self.df_prod = None
        self.df_rev = None
        self.merged_df_with_labels = pd.DataFrame()
        self.filtered_subset_for_display = pd.DataFrame()
        self.product_scraper_worker = None
        self.review_scraper_worker = None
        
        # --- Atribut UI yang akan diinisialisasi di initUI ---
        # Panel Kiri
        self.keyword_input = None
        self.max_pages_slider_label = None
        self.max_pages_slider = None
        self.search_id_input = None
        self.scrape_prod_btn = None
        self.stop_prod_scrape_btn = None
        self.download_prod_btn = None
        self.load_prod_btn = None
        self.max_review_pages_slider_label = None
        self.max_review_pages_slider = None
        self.scrape_rev_btn = None
        self.stop_rev_scrape_btn = None
        self.download_rev_btn = None
        self.load_rev_btn = None
        self.review_time_start_edit = None
        self.review_time_end_edit = None
        self.nama_produk_keyword_input = None
        self.merged_location_filter_combo = None
        self.merged_price_min_input = None
        self.merged_price_max_input = None
        self.min_word_input = None
        self.filter_label_input = None
        self.apply_merged_filters_btn = None
        self.apply_and_trim_btn = None
        self.load_merged_data_btn = None
        self.save_merged_data_btn = None
        
        # Panel Kanan
        self.data_tabs = None
        self.product_table = None
        self.review_table = None
        self.merged_filtered_table = None
        self.status_label = None

        # Tab Visualisasi (jika diaktifkan kembali)
        self.visualization_tab_content = None # Pastikan ini didefinisikan
        self.viz_kpi_total_produk_label = None
        self.viz_kpi_total_ulasan_label = None
        self.viz_kpi_avg_rating_label = None
        self.viz_rating_dist_view = None
        self.viz_location_dist_view = None
        self.viz_wordcloud_label = None

        self.initUI()
        self.show_products_table() # Tampilkan tabel produk secara default
        self._update_button_states_after_data_change() # Atur state tombol awal


    def initUI(self):
        main_layout = QHBoxLayout()
        
        # --- Left Panel ---
        left_panel = QFrame(); left_panel.setFrameShape(QFrame.Shape.StyledPanel); left_panel.setFixedWidth(300)
        self.left_layout = QVBoxLayout(); left_layout = self.left_layout # Alias untuk kemudahan
        
        product_label = QLabel("🔍 Product Search"); product_label.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        self.keyword_input = QLineEdit("essential oil")
        self.max_pages_slider_label = QLabel(f"Max pages: 3")
        self.max_pages_slider = QSlider(Qt.Orientation.Horizontal); self.max_pages_slider.setRange(1, 50); self.max_pages_slider.setValue(1)
        self.max_pages_slider_label.setText(f"Max product: {10}")
        self.search_id_input = QLineEdit("2272") # Contoh Search ID (SC)
        self.scrape_prod_btn = QPushButton("Scrape Products 🚀")
        self.stop_prod_scrape_btn = QPushButton("Stop Product Scraping 🛑"); self.stop_prod_scrape_btn.setEnabled(False)
        
        left_layout.addWidget(product_label)
        left_layout.addWidget(QLabel("Keyword:")); left_layout.addWidget(self.keyword_input)
        left_layout.addWidget(self.max_pages_slider_label); left_layout.addWidget(self.max_pages_slider)
        left_layout.addWidget(QLabel("Search Category (SC):")); left_layout.addWidget(self.search_id_input)
        left_layout.addWidget(self.scrape_prod_btn); left_layout.addWidget(self.stop_prod_scrape_btn)

        prod_data_io_layout = QHBoxLayout()
        self.download_prod_btn = QPushButton("Download Products CSV"); self.download_prod_btn.setEnabled(False)
        self.load_prod_btn = QPushButton("Load Products CSV"); self.load_prod_btn.setEnabled(True)
        prod_data_io_layout.addWidget(self.download_prod_btn)
        prod_data_io_layout.addWidget(self.load_prod_btn)
        left_layout.addLayout(prod_data_io_layout)

        separator = QFrame(); separator.setFrameShape(QFrame.Shape.HLine); separator.setFrameShadow(QFrame.Shadow.Sunken)
        left_layout.addWidget(separator)
        
        review_label = QLabel("💬 Reviews"); review_label.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        self.max_review_pages_slider_label = QLabel(f"Max reviews: 50")
        self.max_review_pages_slider = QSlider(Qt.Orientation.Horizontal); self.max_review_pages_slider.setRange(1, 50); self.max_review_pages_slider.setValue(5)
        self.scrape_rev_btn = QPushButton("Scrape Reviews 🚀")
        self.stop_rev_scrape_btn = QPushButton("Stop Review Scraping 🛑"); self.stop_rev_scrape_btn.setEnabled(False)
        
        left_layout.addWidget(review_label)
        left_layout.addWidget(self.max_review_pages_slider_label); left_layout.addWidget(self.max_review_pages_slider)
        left_layout.addWidget(self.scrape_rev_btn); left_layout.addWidget(self.stop_rev_scrape_btn)

        rev_data_io_layout = QHBoxLayout()
        self.download_rev_btn = QPushButton("Download Reviews CSV"); self.download_rev_btn.setEnabled(False)
        self.load_rev_btn = QPushButton("Load Reviews CSV"); self.load_rev_btn.setEnabled(True)
        rev_data_io_layout.addWidget(self.download_rev_btn)
        rev_data_io_layout.addWidget(self.load_rev_btn)
        left_layout.addLayout(rev_data_io_layout)

        separator2 = QFrame(); separator2.setFrameShape(QFrame.Shape.HLine); separator2.setFrameShadow(QFrame.Shadow.Sunken)
        left_layout.addWidget(separator2)

        merged_filter_label = QLabel("📊 Filter & Merge"); merged_filter_label.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        left_layout.addWidget(merged_filter_label)

        filter_form_layout = QFormLayout()
        self.review_time_start_edit = QDateEdit(self); self.review_time_start_edit.setCalendarPopup(True); self.review_time_start_edit.setDate(QDate.currentDate().addYears(-5))
        self.review_time_end_edit = QDateEdit(self); self.review_time_end_edit.setCalendarPopup(True); self.review_time_end_edit.setDate(QDate.currentDate())
        time_filter_layout = QHBoxLayout(); time_filter_layout.addWidget(self.review_time_start_edit); time_filter_layout.addWidget(QLabel("s/d")); time_filter_layout.addWidget(self.review_time_end_edit)
        filter_form_layout.addRow("Rentang Waktu Ulasan:", time_filter_layout)

        self.nama_produk_keyword_input = QLineEdit(); self.nama_produk_keyword_input.setPlaceholderText("Filter nama produk...")
        filter_form_layout.addRow("Keyword Nama Produk:", self.nama_produk_keyword_input)

        self.merged_location_filter_combo = QComboBox()
        self._populate_merged_location_filter_from_merged() # Fungsi ini menambahkan "Semua Lokasi", "Jakarta", "Bandung", dll.
        self.merged_location_filter_combo.setCurrentText("Semua Lokasi")

        filter_form_layout.addRow("Lokasi Toko (Filter):", self.merged_location_filter_combo)

        self.merged_price_min_input = QLineEdit(); self.merged_price_min_input.setPlaceholderText("Harga Min")
        self.merged_price_max_input = QLineEdit(); self.merged_price_max_input.setPlaceholderText("Harga Maks")
        merged_price_layout = QHBoxLayout(); merged_price_layout.addWidget(self.merged_price_min_input); merged_price_layout.addWidget(QLabel("s/d")); merged_price_layout.addWidget(self.merged_price_max_input)
        filter_form_layout.addRow("Rentang Harga (Filter):", merged_price_layout)
        
        self.min_word_input = QLineEdit("2")
        filter_form_layout.addRow("Minimal kata ulasan:", self.min_word_input)
        self.filter_label_input = QLineEdit("Segmen A")
        filter_form_layout.addRow("Nama Label untuk Filter:", self.filter_label_input)
        left_layout.addLayout(filter_form_layout)

        self.apply_merged_filters_btn = QPushButton("Terapkan Filter & Label Gabungan")
        left_layout.addWidget(self.apply_merged_filters_btn)

        self.apply_and_trim_btn = QPushButton(" Terapkan Filter & Hapus Permanen")
        self.apply_and_trim_btn.setToolTip("Terapkan filter dan secara permanen hapus semua baris yang tidak cocok dari data gabungan utama (merged_df_with_labels).")
        left_layout.addWidget(self.apply_and_trim_btn)

        merged_data_io_layout = QHBoxLayout()
        self.save_merged_data_btn = QPushButton("Save Filtered CSV"); self.save_merged_data_btn.setEnabled(False)
        self.load_merged_data_btn = QPushButton("Load Filtered CSV")
        merged_data_io_layout.addWidget(self.save_merged_data_btn)
        merged_data_io_layout.addWidget(self.load_merged_data_btn)
        left_layout.addLayout(merged_data_io_layout)

        left_layout.addStretch(); left_panel.setLayout(left_layout)
        
        # --- Right Panel ---
        right_panel = QFrame(); right_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self.right_layout = QVBoxLayout(); right_layout = self.right_layout # Alias
        self.data_tabs = QTabWidget()

        # Tab 1: Produk
        self.product_table = QTableWidget(); self.product_table.setColumnCount(11)
        self.product_table.setHorizontalHeaderLabels(["Product ID", "Old Product ID", "Name", "Price", "Rating", "Shop", "Shop URL", "Location", "Product URL", "Category", "Keyword"])
        self.product_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.product_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch) # Nama produk stretch
        self.product_table.setAlternatingRowColors(True); self.product_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.data_tabs.addTab(self.product_table, "🛍️ Produk Scraped")

        # Tab 2: Ulasan
        self.review_table = QTableWidget(); self.review_table.setColumnCount(4)
        self.review_table.setHorizontalHeaderLabels(["Product ID", "Review", "Rating", "Time"])
        self.review_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.review_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch) # Kolom ulasan stretch
        self.review_table.setAlternatingRowColors(True); self.review_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.data_tabs.addTab(self.review_table, "💬 Ulasan Scraped")

        # Tab 3: Data Gabungan Terfilter
        self.merged_filtered_table = QTableWidget()
        self.merged_filtered_table.setAlternatingRowColors(True)
        self.merged_filtered_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.data_tabs.addTab(self.merged_filtered_table, "🔗 Data Gabungan Terfilter")
        
        # Inisialisasi atribut visualisasi meskipun tab tidak langsung ditambahkan
        # Ini mencegah error jika _update_visualizations_on_merged_data dipanggil
        self.visualization_tab_content = QWidget() # Konten placeholder
        # from PyQt6.QtWidgets import QGridLayout # Pindahkan impor jika hanya dipakai di sini
        # viz_tab_layout = QGridLayout(self.visualization_tab_content)
        self.viz_kpi_total_produk_label = self._create_viz_label_placeholder("Produk: -")
        self.viz_kpi_total_ulasan_label = self._create_viz_label_placeholder("Ulasan: -")
        self.viz_kpi_avg_rating_label = self._create_viz_label_placeholder("Avg. Rating Ulasan (Difilter): -")
        self.viz_rating_dist_view = self._create_webview_for_plot("Distribusi Rating Ulasan")
        self.viz_location_dist_view = self._create_webview_for_plot("Distribusi Produk per Lokasi")
        self.viz_wordcloud_label = QLabel("Word Cloud akan muncul di sini...")
        # Anda bisa menambahkan tab visualisasi jika dependensi tersedia:
        # if PYQTWEBENGINE_AVAILABLE and PLOTLY_AVAILABLE and WORDCLOUD_AVAILABLE:
        #     # Susun layout untuk viz_tab_layout di sini
        #     # viz_tab_layout.addWidget(self._create_viz_groupbox("Total Produk", self.viz_kpi_total_produk_label), 0, 0)
        #     # ... (widget lainnya)
        #     self.data_tabs.addTab(self.visualization_tab_content, "📈 Visualisasi Gabungan")


        right_layout.addWidget(self.data_tabs)
        
        self.status_label = QLabel("Ready"); self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.status_label)
        right_panel.setLayout(right_layout)
        
        main_layout.addWidget(left_panel); main_layout.addWidget(right_panel, 1) # Beri stretch factor ke panel kanan
        self.setLayout(main_layout)
        
        # Connect signals
        self.max_pages_slider.valueChanged.connect(lambda val: self.max_pages_slider_label.setText(f"Max product: {val*200}"))
        self.max_review_pages_slider.valueChanged.connect(lambda val: self.max_review_pages_slider_label.setText(f"Max reviews per product: {val*50}"))
        self.scrape_prod_btn.clicked.connect(self.start_product_scraping)
        self.stop_prod_scrape_btn.clicked.connect(self.stop_product_scraping)
        self.load_prod_btn.clicked.connect(self.load_products_from_file)
        self.download_prod_btn.clicked.connect(self.download_products_csv)
        self.scrape_rev_btn.clicked.connect(self.start_review_scraping)
        self.stop_rev_scrape_btn.clicked.connect(self.stop_review_scraping)
        self.load_rev_btn.clicked.connect(self.load_reviews_from_file)
        self.download_rev_btn.clicked.connect(self.download_reviews_csv)
        self.apply_merged_filters_btn.clicked.connect(self._apply_and_display_merged_filters)
        self.apply_and_trim_btn.clicked.connect(self._apply_and_trim_data)
        self.save_merged_data_btn.clicked.connect(self.save_filtered_merged_data)
        self.load_merged_data_btn.clicked.connect(self.load_filtered_merged_data)

    def _initial_merge_if_needed(self):
        # This function is called by the filter/trim methods.
        # It checks if a merged DataFrame already exists. If not, it creates one
        # from the available df_prod and df_rev.
        if not self.merged_df_with_labels.empty:
            # A merged frame already exists. We ensure the label column is present.
            if 'filter_segment_label' not in self.merged_df_with_labels.columns:
                self.merged_df_with_labels['filter_segment_label'] = pd.NA
                self.merged_df_with_labels['filter_segment_label'] = self.merged_df_with_labels['filter_segment_label'].astype('object')
            return True

        # A merged frame does not exist, so we try to create one.
        if self.df_prod is None or self.df_rev is None or self.df_prod.empty or self.df_rev.empty:
            if QMessageBox is not None: QMessageBox.warning(self, "Data Tidak Lengkap", "Harap muat atau scrape data produk DAN ulasan terlebih dahulu.")
            return False

        if self.status_label: self.status_label.setText("Menggabungkan data produk dan ulasan awal...");
        if QApplication.instance(): QApplication.processEvents()
        try:
            df_prod_to_merge = self.df_prod.astype({'product_id': str})
            df_rev_to_merge = self.df_rev.astype({'product_id': str})
            # The 'rating' column exists in both, so suffixes will be applied.
            # Other columns like 'time', 'ulasan', 'nama_produk', 'harga' are unique and won't get suffixes.
            merged = pd.merge(df_rev_to_merge, df_prod_to_merge, on="product_id", how="inner", suffixes=('_ulasan', '_produk'))
            if merged.empty:
                if QMessageBox is not None: QMessageBox.information(self, "Merge Kosong", "Tidak ada data yang cocok antara produk dan ulasan.")
                self.merged_df_with_labels = pd.DataFrame()
                return False
            merged['filter_segment_label'] = pd.NA
            merged['filter_segment_label'] = merged['filter_segment_label'].astype('object')
            self.merged_df_with_labels = merged
            if self.status_label: self.status_label.setText(f"Data awal digabungkan: {len(self.merged_df_with_labels)} baris.")
            self._populate_merged_location_filter_from_merged()
            return True
        except Exception as e:
            if QMessageBox is not None: QMessageBox.critical(self, "Error Merge Awal", f"Gagal menggabungkan data: {e}")
            self.merged_df_with_labels = pd.DataFrame()
            return False

    def filter_short_reviews(self, df, column='ulasan', num_word=2):
        if df.empty or column not in df.columns: return df
        if num_word <= 0: return df
        temp_series = df[column].astype(str).fillna("")
        mask = temp_series.apply(lambda x: len(x.split()) >= num_word)
        return df[mask]

    def _apply_and_display_merged_filters(self):
        if self.merged_df_with_labels.empty:
            if not self._initial_merge_if_needed():
                self._clear_merged_filter_outputs()
                return
        
        if self.status_label: self.status_label.setText("Menerapkan filter ke data gabungan...");
        if QApplication.instance(): QApplication.processEvents()
        data_for_selection = self.merged_df_with_labels.copy()
        
        try:
            min_words_str = self.min_word_input.text().strip()
            min_words = int(min_words_str) if min_words_str else 0
            if min_words > 0:
                 data_for_selection = self.filter_short_reviews(data_for_selection, "ulasan", min_words)
        except ValueError:
            if QMessageBox is not None: QMessageBox.warning(self, "Input Jumlah Kata Salah", "Minimal kata harus berupa angka.")

        start_date = self.review_time_start_edit.date().toPyDate()
        end_date = self.review_time_end_edit.date().toPyDate()
        # After merge, review time column is 'time' (it's unique to df_rev). 'waktu_ulasan' is a fallback.
        time_col_to_use = 'waktu_ulasan' if 'waktu_ulasan' in data_for_selection.columns else 'time' if 'time' in data_for_selection.columns else None
        if time_col_to_use and not data_for_selection.empty:
            data_for_selection[time_col_to_use] = pd.to_datetime(data_for_selection[time_col_to_use], errors='coerce')
            valid_time_mask = data_for_selection[time_col_to_use].notna()
            date_filter_mask = pd.Series(False, index=data_for_selection.index)
            if valid_time_mask.any():
                 date_filter_mask.loc[valid_time_mask] = data_for_selection.loc[valid_time_mask, time_col_to_use].dt.date.between(start_date, end_date)
            data_for_selection = data_for_selection[date_filter_mask]

        # Column 'nama_produk' is unique to df_prod, so name is preserved in merge
        nama_keyword = self.nama_produk_keyword_input.text().lower().strip()
        if nama_keyword and 'nama_produk' in data_for_selection.columns and not data_for_selection.empty:
            data_for_selection = data_for_selection[data_for_selection['nama_produk'].astype(str).str.lower().str.contains(nama_keyword, na=False)]

        # Column 'lokasi' is unique to df_prod
        selected_location = self.merged_location_filter_combo.currentText()
        if selected_location != "Semua Lokasi" and 'lokasi' in data_for_selection.columns and not data_for_selection.empty:
            data_for_selection = data_for_selection[data_for_selection['lokasi'] == selected_location]

        if not data_for_selection.empty: # Column 'harga' is unique to df_prod
            try:
                min_price_str = self.merged_price_min_input.text().strip()
                max_price_str = self.merged_price_max_input.text().strip()
                min_price = float(min_price_str) if min_price_str else -np.inf
                max_price = float(max_price_str) if max_price_str else np.inf
                if 'harga' in data_for_selection.columns:
                    data_for_selection['harga'] = pd.to_numeric(data_for_selection['harga'], errors='coerce')
                    data_for_selection = data_for_selection[data_for_selection['harga'].between(min_price, max_price, inclusive='both')]
            except ValueError:
                if QMessageBox is not None: QMessageBox.warning(self, "Input Harga Salah", "Format harga tidak valid.")
        
        selected_indices = data_for_selection.index
        filter_label = self.filter_label_input.text().strip()
        if not filter_label: filter_label = f"Filtered_{datetime.now().strftime('%H%M%S')}"

        if 'filter_segment_label' not in self.merged_df_with_labels.columns:
            self.merged_df_with_labels['filter_segment_label'] = pd.NA 
        self.merged_df_with_labels['filter_segment_label'] = self.merged_df_with_labels['filter_segment_label'].astype('object')
        
        condition_to_clear_label = (self.merged_df_with_labels['filter_segment_label'] == filter_label) & (~self.merged_df_with_labels.index.isin(selected_indices))
        self.merged_df_with_labels.loc[condition_to_clear_label, 'filter_segment_label'] = pd.NA

        if not selected_indices.empty:
            self.merged_df_with_labels.loc[selected_indices, 'filter_segment_label'] = filter_label
        
        if not selected_indices.empty:
            self.filtered_subset_for_display = self.merged_df_with_labels.loc[selected_indices].copy()
        else: 
            self.filtered_subset_for_display = pd.DataFrame(columns=self.merged_df_with_labels.columns)
            if 'filter_segment_label' not in self.filtered_subset_for_display.columns and \
               'filter_segment_label' in self.merged_df_with_labels.columns:
                   self.filtered_subset_for_display['filter_segment_label'] = pd.Series(dtype='object')


        self._display_df_in_table(self.filtered_subset_for_display, self.merged_filtered_table)
        if self.status_label: self.status_label.setText(f"Filter diterapkan. {len(self.filtered_subset_for_display)} baris ditampilkan dengan label '{filter_label}'.")
        if self.save_merged_data_btn: self.save_merged_data_btn.setEnabled(not self.merged_df_with_labels.empty)
        if self.data_tabs and self.merged_filtered_table: self.data_tabs.setCurrentWidget(self.merged_filtered_table)


    def _apply_and_trim_data(self):
        if QMessageBox is None: return 
        reply = QMessageBox.question(self, "Konfirmasi Hapus Data",
                                     "Anda yakin ingin menerapkan filter dan secara permanen "
                                     "menghapus semua data yang tidak cocok dari set data gabungan utama?\n"
                                     "Tindakan ini tidak dapat diurungkan untuk sesi data saat ini.",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No:
            if self.status_label: self.status_label.setText("Operasi hapus data dibatalkan.")
            return

        if self.merged_df_with_labels.empty:
            if not self._initial_merge_if_needed():
                self._clear_merged_filter_outputs()
                if self.status_label: self.status_label.setText("Gagal melakukan merge awal. Operasi dibatalkan.")
                return
        
        if self.status_label: self.status_label.setText("Menerapkan filter untuk memotong data...");
        if QApplication.instance(): QApplication.processEvents()
        data_to_trim_from = self.merged_df_with_labels.copy()
        
        try:
            min_words_str = self.min_word_input.text().strip()
            min_words = int(min_words_str) if min_words_str else 0
            if min_words > 0: data_to_trim_from = self.filter_short_reviews(data_to_trim_from, "ulasan", min_words)
        except ValueError:
            if QMessageBox is not None: QMessageBox.warning(self, "Input Jumlah Kata Salah", "Minimal kata harus berupa angka.")
        
        start_date = self.review_time_start_edit.date().toPyDate()
        end_date = self.review_time_end_edit.date().toPyDate()
        time_col_to_use = 'waktu_ulasan' if 'waktu_ulasan' in data_to_trim_from.columns else 'time' if 'time' in data_to_trim_from.columns else None
        if time_col_to_use and not data_to_trim_from.empty:
            data_to_trim_from[time_col_to_use] = pd.to_datetime(data_to_trim_from[time_col_to_use], errors='coerce')
            valid_time_mask = data_to_trim_from[time_col_to_use].notna()
            date_filter_mask = pd.Series(False, index=data_to_trim_from.index)
            if valid_time_mask.any(): date_filter_mask.loc[valid_time_mask] = data_to_trim_from.loc[valid_time_mask, time_col_to_use].dt.date.between(start_date, end_date)
            data_to_trim_from = data_to_trim_from[date_filter_mask]

        nama_keyword = self.nama_produk_keyword_input.text().lower().strip()
        if nama_keyword and 'nama_produk' in data_to_trim_from.columns and not data_to_trim_from.empty:
            data_to_trim_from = data_to_trim_from[data_to_trim_from['nama_produk'].astype(str).str.lower().str.contains(nama_keyword, na=False)]

        selected_location = self.merged_location_filter_combo.currentText()
        if selected_location != "Semua Lokasi" and 'lokasi' in data_to_trim_from.columns and not data_to_trim_from.empty:
            data_to_trim_from = data_to_trim_from[data_to_trim_from['lokasi'] == selected_location]

        if not data_to_trim_from.empty:
            try:
                min_price_str = self.merged_price_min_input.text().strip()
                max_price_str = self.merged_price_max_input.text().strip()
                min_price = float(min_price_str) if min_price_str else -np.inf
                max_price = float(max_price_str) if max_price_str else np.inf
                if 'harga' in data_to_trim_from.columns:
                    data_to_trim_from['harga'] = pd.to_numeric(data_to_trim_from['harga'], errors='coerce')
                    data_to_trim_from = data_to_trim_from[data_to_trim_from['harga'].between(min_price, max_price, inclusive='both')]
            except ValueError:
                if QMessageBox is not None: QMessageBox.warning(self, "Input Harga Salah", "Format harga tidak valid.")
        
        filtered_df_subset = data_to_trim_from 
        if filtered_df_subset.empty:
            reply_empty_trim = QMessageBox.question(self, "Filter Kosong",
                                                "Filter tidak menghasilkan data apa pun. Melanjutkan akan menghapus SEMUA data gabungan saat ini. Lanjutkan?",
                                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                                QMessageBox.StandardButton.No)
            if reply_empty_trim == QMessageBox.StandardButton.No:
                if self.status_label: self.status_label.setText("Operasi potong dibatalkan karena filter kosong.")
                return

        self.merged_df_with_labels = filtered_df_subset.copy() # Operasi destruktif
        filter_label = self.filter_label_input.text().strip()
        if not filter_label: filter_label = f"Trimmed_Data_{datetime.now().strftime('%H%M%S')}"
        
        if 'filter_segment_label' not in self.merged_df_with_labels.columns:
            self.merged_df_with_labels['filter_segment_label'] = pd.NA
        self.merged_df_with_labels['filter_segment_label'] = self.merged_df_with_labels['filter_segment_label'].astype('object')
        self.merged_df_with_labels['filter_segment_label'] = filter_label

        self.filtered_subset_for_display = self.merged_df_with_labels.copy()
        self._display_df_in_table(self.filtered_subset_for_display, self.merged_filtered_table)
        self._populate_merged_location_filter_from_merged() 
        if self.status_label: self.status_label.setText(f"Data dipotong. Tersisa {len(self.merged_df_with_labels)} baris dengan label '{filter_label}'.")
        if self.save_merged_data_btn: self.save_merged_data_btn.setEnabled(not self.merged_df_with_labels.empty)
        if QMessageBox is not None: QMessageBox.information(self, "Operasi Selesai", f"Data yang tidak cocok telah dihapus.\nTersisa {len(self.merged_df_with_labels)} baris.")


    def _clear_merged_filter_outputs(self):
        cols = self.merged_df_with_labels.columns if not self.merged_df_with_labels.empty else None
        if cols is None and self.merged_filtered_table.columnCount() > 0:
            cols = [self.merged_filtered_table.horizontalHeaderItem(i).text() for i in range(self.merged_filtered_table.columnCount())]

        self.filtered_subset_for_display = pd.DataFrame(columns=cols)
        self._display_df_in_table(self.filtered_subset_for_display, self.merged_filtered_table)
        
        if self.save_merged_data_btn:
            self.save_merged_data_btn.setEnabled(not self.merged_df_with_labels.empty)


    def _populate_merged_location_filter_from_merged(self):
        if not hasattr(self, 'merged_location_filter_combo') or self.merged_location_filter_combo is None: return
        current_selection = self.merged_location_filter_combo.currentText()
        self.merged_location_filter_combo.clear()
        self.merged_location_filter_combo.addItem("Semua Lokasi")
        if not self.merged_df_with_labels.empty and 'lokasi' in self.merged_df_with_labels.columns:
            try:
                unique_locations = sorted(self.merged_df_with_labels['lokasi'].dropna().astype(str).unique().tolist())
                self.merged_location_filter_combo.addItems(unique_locations)
            except Exception as e:
                print(f"Error populating location filter from merged: {e}")

            index = self.merged_location_filter_combo.findText(current_selection)
            if index >= 0: self.merged_location_filter_combo.setCurrentIndex(index)
            else: self.merged_location_filter_combo.setCurrentIndex(0)
        else: self.merged_location_filter_combo.setCurrentIndex(0)

    def _set_scraping_state(self, is_scraping, scrape_type="product"):
        if scrape_type == "product":
            if hasattr(self, 'scrape_prod_btn') and self.scrape_prod_btn: self.scrape_prod_btn.setEnabled(not is_scraping)
            if hasattr(self, 'stop_prod_scrape_btn') and self.stop_prod_scrape_btn: self.stop_prod_scrape_btn.setEnabled(is_scraping)
            if hasattr(self, 'load_prod_btn') and self.load_prod_btn: self.load_prod_btn.setEnabled(not is_scraping)
            if hasattr(self, 'scrape_rev_btn') and self.scrape_rev_btn: self.scrape_rev_btn.setEnabled(not is_scraping) 
        elif scrape_type == "review":
            if hasattr(self, 'scrape_rev_btn') and self.scrape_rev_btn: self.scrape_rev_btn.setEnabled(not is_scraping)
            if hasattr(self, 'stop_rev_scrape_btn') and self.stop_rev_scrape_btn: self.stop_rev_scrape_btn.setEnabled(is_scraping)
            if hasattr(self, 'load_rev_btn') and self.load_rev_btn: self.load_rev_btn.setEnabled(not is_scraping)
            if hasattr(self, 'scrape_prod_btn') and self.scrape_prod_btn: self.scrape_prod_btn.setEnabled(not is_scraping)
        self._update_button_states_after_data_change()


    def save_filtered_merged_data(self):
        data_to_save = self.merged_df_with_labels 

        if data_to_save is None or data_to_save.empty:
            if QMessageBox is not None: QMessageBox.warning(self, "Tidak Ada Data", "Tidak ada data gabungan (berlabel/terpotong) untuk disimpan.")
            return
        if QFileDialog is None: return

        default_filename = f"merged_data_with_labels_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_name, selected_filter = QFileDialog.getSaveFileName(
            self, "Simpan Data Gabungan (Berlabel/Terpotong)", os.path.join(os.path.expanduser("~"), default_filename),
            "CSV Files (*.csv);;Excel Files (*.xlsx)")
        if file_name:
            if self.status_label: self.status_label.setText(f"Menyimpan data ke {file_name}...")
            if QApplication.instance(): QApplication.processEvents()
            try:
                if selected_filter.startswith("CSV"):
                    if not file_name.lower().endswith('.csv'): file_name += '.csv'
                    data_to_save.to_csv(file_name, index=False)
                elif selected_filter.startswith("Excel"):
                    if not file_name.lower().endswith('.xlsx'): file_name += '.xlsx'
                    data_to_save.to_excel(file_name, index=False)
                if self.status_label: self.status_label.setText(f"Data berhasil disimpan ke: {file_name}")
                if QMessageBox is not None: QMessageBox.information(self, "Simpan Berhasil", f"Data disimpan ke:\n{file_name}")
            except Exception as e:
                if QMessageBox is not None: QMessageBox.critical(self, "Error Penyimpanan", f"Gagal menyimpan data:\n{str(e)}")
                if self.status_label: self.status_label.setText(f"Error: Gagal menyimpan data.")

    def load_filtered_merged_data(self):
        if QFileDialog is None: return
        file_path, _ = QFileDialog.getOpenFileName(self, "Muat Data Gabungan dari File", os.path.expanduser("~/"), "Data Files (*.csv *.xlsx *.xls)")
        if file_path:
            if self.status_label: self.status_label.setText(f"Memuat data dari {file_path}...");
            if QApplication.instance(): QApplication.processEvents()
            try:
                if file_path.lower().endswith('.csv'): loaded_df = pd.read_csv(file_path, dtype=str)
                elif file_path.lower().endswith(('.xlsx', '.xls')): loaded_df = pd.read_excel(file_path, dtype=str)
                else:
                    if QMessageBox is not None: QMessageBox.warning(self, "Format Tidak Didukung", "Pilih file CSV atau Excel."); return

                self.merged_df_with_labels = loaded_df
                
                if 'harga' in self.merged_df_with_labels.columns: 
                    self.merged_df_with_labels['harga'] = pd.to_numeric(self.merged_df_with_labels['harga'], errors='coerce')
                if 'rating_ulasan' in self.merged_df_with_labels.columns:
                    self.merged_df_with_labels['rating_ulasan'] = pd.to_numeric(self.merged_df_with_labels['rating_ulasan'], errors='coerce')
                if 'rating_produk' in self.merged_df_with_labels.columns:
                    self.merged_df_with_labels['rating_produk'] = pd.to_numeric(self.merged_df_with_labels['rating_produk'], errors='coerce')
                
                time_col_to_use = None
                if 'waktu_ulasan' in self.merged_df_with_labels.columns: time_col_to_use = 'waktu_ulasan'
                elif 'time' in self.merged_df_with_labels.columns: time_col_to_use = 'time'
                if time_col_to_use:
                    self.merged_df_with_labels[time_col_to_use] = pd.to_datetime(self.merged_df_with_labels[time_col_to_use], errors='coerce')

                if 'filter_segment_label' not in self.merged_df_with_labels.columns:
                    self.merged_df_with_labels['filter_segment_label'] = pd.NA
                self.merged_df_with_labels['filter_segment_label'] = self.merged_df_with_labels['filter_segment_label'].astype('object')
                self.merged_df_with_labels['filter_segment_label'].replace(['nan', 'None', ''], pd.NA, inplace=True)


                self.filtered_subset_for_display = self.merged_df_with_labels.copy()
                self._display_df_in_table(self.filtered_subset_for_display, self.merged_filtered_table)
                self._populate_merged_location_filter_from_merged()
                # self._update_visualizations_on_merged_data()
                
                if self.status_label: self.status_label.setText(f"Berhasil memuat {len(self.merged_df_with_labels)} baris data gabungan.")
                self._update_button_states_after_data_change()
                
                # Invalidate raw data and disable their controls as we are now in "merged mode"
                self.df_prod = None; self.df_rev = None
                self.show_products_table()
                self.show_reviews_table()
                if hasattr(self, 'scrape_prod_btn') and self.scrape_prod_btn: self.scrape_prod_btn.setEnabled(False);
                if hasattr(self, 'load_prod_btn') and self.load_prod_btn: self.load_prod_btn.setEnabled(False)
                if hasattr(self, 'scrape_rev_btn') and self.scrape_rev_btn: self.scrape_rev_btn.setEnabled(False);
                if hasattr(self, 'load_rev_btn') and self.load_rev_btn: self.load_rev_btn.setEnabled(False)
                if hasattr(self, 'download_prod_btn') and self.download_prod_btn: self.download_prod_btn.setEnabled(False);
                if hasattr(self, 'download_rev_btn') and self.download_rev_btn: self.download_rev_btn.setEnabled(False)
                
                if self.data_tabs and self.merged_filtered_table: self.data_tabs.setCurrentWidget(self.merged_filtered_table)

            except Exception as e:
                if QMessageBox is not None: QMessageBox.critical(self, "Error Muat File", f"Gagal memuat data gabungan: {str(e)}")
                if self.status_label: self.status_label.setText(f"Error memuat file.")


    def start_product_scraping(self):
        keyword = self.keyword_input.text()
        max_pages = self.max_pages_slider.value()
        search_id = self.search_id_input.text()
        if not keyword or not search_id:
            if QMessageBox is not None: QMessageBox.warning(self, "Input Missing", "Keyword dan Search ID diperlukan.")
            return

        # --- MODIFICATION ---
        # Reset all downstream data because a new set of products is being scraped.
        # This invalidates old reviews and any previously merged data.
        self.df_prod = pd.DataFrame() 
        self.df_rev = None
        self.merged_df_with_labels = pd.DataFrame()
        self.show_products_table() # Clear product table view
        self.show_reviews_table() # Clear review table view
        self._clear_merged_filter_outputs() # Clear merged table view
        # --- END MODIFICATION ---

        self._set_scraping_state(True, "product")
        if self.status_label: self.status_label.setText(f"Memulai scraping produk untuk '{keyword}'...")

        self.product_scraper_worker = ProductScraperWorker(keyword, max_pages, search_id)
        self.product_scraper_worker.finished.connect(self.handle_product_scrape_finished)
        self.product_scraper_worker.error.connect(self.handle_product_scrape_error)
        self.product_scraper_worker.progress.connect(self.handle_product_scrape_progress)
        self.product_scraper_worker.start()

    def stop_product_scraping(self):
        if self.product_scraper_worker and self.product_scraper_worker.isRunning():
            self.product_scraper_worker.stop()
            if self.status_label: self.status_label.setText("Mengirim sinyal berhenti ke product scraper...")
        else:
            self._set_scraping_state(False, "product")


    def handle_product_scrape_progress(self, items_scraped, current_page):
        if self.status_label: self.status_label.setText(f"Scraped {items_scraped} produk dari {current_page}/{self.max_pages_slider.value()} halaman...")

    def handle_product_scrape_finished(self, df_result):
        self.df_prod = df_result
        if self.df_prod is not None and not self.df_prod.empty:
            if self.status_label: self.status_label.setText(f"Berhasil scrape {len(self.df_prod)} produk.")
            if 'harga' in self.df_prod.columns: self.df_prod['harga'] = pd.to_numeric(self.df_prod['harga'], errors='coerce')
            if 'rating' in self.df_prod.columns: self.df_prod['rating'] = pd.to_numeric(self.df_prod['rating'], errors='coerce')
            self.show_products_table()
            self._populate_merged_location_filter_from_df_prod() 
        elif self.df_prod is not None and self.df_prod.empty:
             if self.status_label: self.status_label.setText("Scraping produk selesai. Tidak ada produk ditemukan.")
             self.show_products_table()
        
        self._set_scraping_state(False, "product")
        self.product_scraper_worker = None
        self._update_button_states_after_data_change()


    def handle_product_scrape_error(self, error_msg):
        if self.status_label: self.status_label.setText(f"Error scraping produk: {error_msg}")
        if QMessageBox is not None: QMessageBox.critical(self, "Scraping Error", f"Error saat scraping produk:\n{error_msg}")
        self.df_prod = pd.DataFrame()
        self.show_products_table()
        self._set_scraping_state(False, "product")
        self.product_scraper_worker = None
        self._update_button_states_after_data_change()


    def start_review_scraping(self):
        if self.df_prod is None or self.df_prod.empty or 'product_id' not in self.df_prod.columns:
            if QMessageBox is not None: QMessageBox.warning(self, "No Products", "Harap muat atau scrape produk (dengan 'product_id') dahulu.")
            return
        product_ids = self.df_prod["product_id"].dropna().unique().tolist()
        if not product_ids:
            if QMessageBox is not None: QMessageBox.warning(self, "No Product IDs", "Tidak ada product ID valid di data produk.")
            return

        # --- MODIFICATION ---
        # Reset data downstream from reviews. The merged data is now invalid.
        self.df_rev = pd.DataFrame()
        self.merged_df_with_labels = pd.DataFrame()
        self.show_reviews_table() # Clear review table view
        self._clear_merged_filter_outputs() # Clear merged table view
        # --- END MODIFICATION ---

        max_rev_pages = self.max_review_pages_slider.value()
        self._set_scraping_state(True, "review")
        if self.status_label: self.status_label.setText(f"Memulai scraping review untuk {len(product_ids)} produk...")

        self.review_scraper_worker = ReviewScraperWorker(product_ids, max_rev_pages)
        self.review_scraper_worker.finished.connect(self.handle_review_scrape_finished)
        self.review_scraper_worker.error.connect(self.handle_review_scrape_error)
        self.review_scraper_worker.progress.connect(self.handle_review_scrape_progress)
        self.review_scraper_worker.start()

    def stop_review_scraping(self):
        if self.review_scraper_worker and self.review_scraper_worker.isRunning():
            self.review_scraper_worker.stop()
            if self.status_label: self.status_label.setText("Mengirim sinyal berhenti ke review scraper...")
        else:
            self._set_scraping_state(False, "review")


    def handle_review_scrape_progress(self, reviews_scraped, product_idx, total_products):
        if self.status_label: self.status_label.setText(f"Scraped {reviews_scraped} review. Produk {product_idx}/{total_products}...")

    def handle_review_scrape_finished(self, df_result):
        self.df_rev = df_result
        if self.df_rev is not None and not self.df_rev.empty:
            if self.status_label: self.status_label.setText(f"Berhasil scrape {len(self.df_rev)} review.")
            # if 'time' in self.df_rev.columns:
            #     try: self.df_rev['time'] = pd.to_datetime(pd.to_numeric(self.df_rev['time'], errors='raise'), unit='s', errors='coerce')
            #     except (ValueError, TypeError): self.df_rev['time'] = pd.to_datetime(self.df_rev['time'], errors='coerce')
            if 'rating' in self.df_rev.columns: self.df_rev['rating'] = pd.to_numeric(self.df_rev['rating'], errors='coerce')
            self.show_reviews_table()
        elif self.df_rev is not None and self.df_rev.empty:
             if self.status_label: self.status_label.setText("Scraping review selesai. Tidak ada review ditemukan.")
             self.show_reviews_table()

        self._set_scraping_state(False, "review")
        self.review_scraper_worker = None
        self._update_button_states_after_data_change()

    def handle_review_scrape_error(self, error_msg):
        if self.status_label: self.status_label.setText(f"Error scraping review: {error_msg}")
        if QMessageBox is not None: QMessageBox.critical(self, "Scraping Error", f"Error saat scraping review:\n{error_msg}")
        self.df_rev = pd.DataFrame()
        self.show_reviews_table()
        self._set_scraping_state(False, "review")
        self.review_scraper_worker = None
        self._update_button_states_after_data_change()
    
    def download_products_csv(self):
        if self.df_prod is not None and not self.df_prod.empty:
            if QFileDialog is None: return
            home_dir = os.path.expanduser("~")
            default_path = os.path.join(home_dir, "products_data.csv")
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Products CSV", default_path, "CSV Files (*.csv)")
            if file_name:
                if not file_name.lower().endswith('.csv'): file_name += '.csv'
                try: 
                    self.df_prod.to_csv(file_name, index=False)
                    if self.status_label: self.status_label.setText(f"Produk disimpan ke {file_name}")
                except Exception as e: 
                    if self.status_label: self.status_label.setText(f"Error saving file: {str(e)}")
                    if QMessageBox is not None: QMessageBox.critical(self, "Save Error", f"Gagal menyimpan file:\n{e}")
        elif self.status_label: self.status_label.setText("Tidak ada data produk untuk disimpan")

    def download_reviews_csv(self):
        if self.df_rev is not None and not self.df_rev.empty:
            if QFileDialog is None: return
            home_dir = os.path.expanduser("~")
            default_path = os.path.join(home_dir, "reviews_data.csv")
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Reviews CSV", default_path, "CSV Files (*.csv)")
            if file_name:
                if not file_name.lower().endswith('.csv'): file_name += '.csv'
                try: 
                    self.df_rev.to_csv(file_name, index=False)
                    if self.status_label: self.status_label.setText(f"Review disimpan ke {file_name}")
                except Exception as e:
                    if self.status_label: self.status_label.setText(f"Error saving file: {str(e)}")
                    if QMessageBox is not None: QMessageBox.critical(self, "Save Error", f"Gagal menyimpan file:\n{e}")
        elif self.status_label: self.status_label.setText("Tidak ada data review untuk disimpan")

    def _update_button_states_after_data_change(self):
        prod_ready = self.df_prod is not None and not self.df_prod.empty
        rev_ready = self.df_rev is not None and not self.df_rev.empty
        merged_ready = not self.merged_df_with_labels.empty
        
        if hasattr(self, 'download_prod_btn') and self.download_prod_btn: self.download_prod_btn.setEnabled(prod_ready)
        if hasattr(self, 'download_rev_btn') and self.download_rev_btn: self.download_rev_btn.setEnabled(rev_ready)
        
        can_filter_or_trim = (prod_ready and rev_ready) or merged_ready
        if hasattr(self, 'apply_merged_filters_btn') and self.apply_merged_filters_btn: self.apply_merged_filters_btn.setEnabled(can_filter_or_trim)
        if hasattr(self, 'apply_and_trim_btn') and self.apply_and_trim_btn: self.apply_and_trim_btn.setEnabled(can_filter_or_trim)
        
        if hasattr(self, 'save_merged_data_btn') and self.save_merged_data_btn: self.save_merged_data_btn.setEnabled(merged_ready)


    def load_products_from_file(self):
        if QFileDialog is None: return
        home_dir = os.path.expanduser("~")
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Products CSV/Excel", home_dir, "Data Files (*.csv *.xlsx *.xls)")
        if file_path:
            try:
                if file_path.lower().endswith('.csv'): loaded_df = pd.read_csv(file_path, dtype=str)
                elif file_path.lower().endswith(('.xlsx', '.xls')): loaded_df = pd.read_excel(file_path, dtype=str)
                else: 
                    if self.status_label: self.status_label.setText("Unsupported file format."); return
                
                # --- MODIFICATION ---
                # Reset all downstream data because a new product file is being loaded.
                self.df_prod = loaded_df
                self.df_rev = None
                self.merged_df_with_labels = pd.DataFrame()
                self.show_reviews_table()
                self._clear_merged_filter_outputs()
                # --- END MODIFICATION ---

                if 'harga' in self.df_prod.columns: 
                    self.df_prod['harga'] = self.df_prod['harga'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                    self.df_prod['harga'] = pd.to_numeric(self.df_prod['harga'], errors='coerce')
                if 'rating' in self.df_prod.columns: 
                    self.df_prod['rating'] = pd.to_numeric(self.df_prod['rating'], errors='coerce')
                
                self.show_products_table()
                if self.status_label: self.status_label.setText(f"Loaded {len(self.df_prod)} products.")
                self._populate_merged_location_filter_from_df_prod() 
                self._update_button_states_after_data_change()
            except Exception as e: 
                if self.status_label: self.status_label.setText(f"Failed to load product file: {e}")
                if QMessageBox is not None: QMessageBox.critical(self, "Load Error", f"Gagal memuat file produk:\n{e}")


    def load_reviews_from_file(self):
        if QFileDialog is None: return
        home_dir = os.path.expanduser("~")
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Reviews CSV/Excel", home_dir, "Data Files (*.csv *.xlsx *.xls)")
        if file_path:
            try:
                if file_path.lower().endswith('.csv'): loaded_df = pd.read_csv(file_path, dtype={'product_id': str})
                elif file_path.lower().endswith(('.xlsx', '.xls')): loaded_df = pd.read_excel(file_path, dtype={'product_id': str})
                else: 
                    if self.status_label: self.status_label.setText("Unsupported file format."); return
                
                # --- MODIFICATION ---
                # Reset merged data because a new review file is being loaded.
                self.df_rev = loaded_df
                self.merged_df_with_labels = pd.DataFrame()
                self._clear_merged_filter_outputs()
                # --- END MODIFICATION ---
                
                if 'time' in self.df_rev.columns:
                    try:
                        self.df_rev['time'] = pd.to_datetime(pd.to_numeric(self.df_rev['time'], errors='raise'), unit='s', errors='coerce')
                    except (ValueError, TypeError):
                        self.df_rev['time'] = pd.to_datetime(self.df_rev['time'], errors='coerce')
                if 'rating' in self.df_rev.columns: 
                    self.df_rev['rating'] = pd.to_numeric(self.df_rev['rating'], errors='coerce')
                
                self.show_reviews_table()
                if self.status_label: self.status_label.setText(f"Loaded {len(self.df_rev)} reviews.")
                self._update_button_states_after_data_change()
            except Exception as e:
                if self.status_label: self.status_label.setText(f"Failed to load review file: {str(e)}")
                if QMessageBox is not None: QMessageBox.critical(self, "Load Error", f"Gagal memuat file review:\n{e}")

    def show_products_table(self):
        if self.data_tabs and self.product_table: self.data_tabs.setCurrentWidget(self.product_table)
        if self.product_table:
            self.product_table.clearContents(); self.product_table.setRowCount(0)
            if self.df_prod is not None and not self.df_prod.empty:
                self.product_table.setRowCount(len(self.df_prod))
                header_labels = ["Product ID", "Old Product ID", "Name", "Price", "Rating", "Shop", "Shop URL", "Location", "Product URL", "Category", "Keyword"]
                col_map = {
                    "Product ID": "product_id", "Old Product ID": "old_product_id", "Name": "nama_produk",
                    "Price": "harga", "Rating": "rating", "Shop": "toko", "Shop URL": "toko_url",
                    "Location": "lokasi", "Product URL": "product_url", "Category": "category", "Keyword": "keyword"
                }

                for i, row_data in self.df_prod.iterrows():
                    for j, header_label in enumerate(header_labels):
                        col_name = col_map.get(header_label)
                        item_value = ""
                        if col_name and col_name in row_data:
                            val = row_data[col_name]
                            if pd.notna(val):
                                if col_name == 'harga' and isinstance(val, (int, float)):
                                    item_value = f"{val:,.0f}".replace(",",".")
                                else:
                                    item_value = str(val)
                            else:
                                item_value = "" 
                        self.product_table.setItem(i, j, QTableWidgetItem(item_value))
                
                self.product_table.resizeColumnsToContents()
                name_col_idx = header_labels.index("Name")
                if name_col_idx != -1:
                       self.product_table.horizontalHeader().setSectionResizeMode(name_col_idx, QHeaderView.ResizeMode.Stretch)


    def show_reviews_table(self):
        if self.data_tabs and self.review_table: self.data_tabs.setCurrentWidget(self.review_table)
        if self.review_table:
            self.review_table.clearContents(); self.review_table.setRowCount(0)
            if self.df_rev is not None and not self.df_rev.empty:
                self.review_table.setRowCount(len(self.df_rev))
                header_labels = ["Product ID", "Review", "Rating", "Time"]
                col_map = {"Product ID": "product_id", "Review": "ulasan", "Rating": "rating", "Time": "time"}

                for i, row_data in self.df_rev.iterrows():
                    for j, header_label in enumerate(header_labels):
                        col_name = col_map.get(header_label)
                        item_value = ""
                        if col_name and col_name in row_data:
                            val = row_data[col_name]
                            if pd.notna(val):
                                if col_name == 'time' and isinstance(val, pd.Timestamp):
                                    item_value = val.strftime('%Y-%m-%d %H:%M:%S')
                                else:
                                    item_value = str(val)
                            else:
                                item_value = ""
                        self.review_table.setItem(i, j, QTableWidgetItem(item_value))

                self.review_table.resizeColumnsToContents()
                review_col_idx = header_labels.index("Review")
                if review_col_idx != -1:
                    self.review_table.horizontalHeader().setSectionResizeMode(review_col_idx, QHeaderView.ResizeMode.Stretch)

    def _display_df_in_table(self, df, table_widget: QTableWidget):
        if table_widget is None: return
        table_widget.clearContents(); table_widget.setRowCount(0)
        if df is None or df.empty: 
            if hasattr(df, 'columns') and not df.columns.empty:
                table_widget.setColumnCount(len(df.columns))
                table_widget.setHorizontalHeaderLabels(df.columns.tolist())
            return

        table_widget.setColumnCount(len(df.columns))
        table_widget.setHorizontalHeaderLabels(df.columns.tolist())
        table_widget.setRowCount(len(df))
        
        for i, row_tuple in enumerate(df.itertuples(index=False)): 
            for j, val in enumerate(row_tuple):
                item_text = str(val) if pd.notna(val) else "" 
                table_widget.setItem(i, j, QTableWidgetItem(item_text))
        
        table_widget.resizeColumnsToContents()
        if 'ulasan' in df.columns:
            try:
                col_idx = df.columns.get_loc('ulasan')
                table_widget.horizontalHeader().setSectionResizeMode(col_idx, QHeaderView.ResizeMode.Stretch)
            except KeyError: pass
        elif 'nama_produk' in df.columns:
            try:
                col_idx = df.columns.get_loc('nama_produk')
                table_widget.horizontalHeader().setSectionResizeMode(col_idx, QHeaderView.ResizeMode.Stretch)
            except KeyError: pass
        elif 'filter_segment_label' in df.columns:
            try:
                col_idx = df.columns.get_loc('filter_segment_label')
                table_widget.horizontalHeader().setSectionResizeMode(col_idx, QHeaderView.ResizeMode.Interactive)
            except KeyError: pass


    def _populate_merged_location_filter_from_df_prod(self):
        if not hasattr(self, 'merged_location_filter_combo') or self.merged_location_filter_combo is None: return
        current_selection = self.merged_location_filter_combo.currentText()
        self.merged_location_filter_combo.clear()
        self.merged_location_filter_combo.addItem("Semua Lokasi")
        if self.df_prod is not None and not self.df_prod.empty and 'lokasi' in self.df_prod.columns:
            try:
                unique_locations = sorted(self.df_prod['lokasi'].dropna().astype(str).unique().tolist())
                self.merged_location_filter_combo.addItems(unique_locations)
            except Exception as e:
                print(f"Error populating location filter from df_prod: {e}")

            index = self.merged_location_filter_combo.findText(current_selection)
            if index >= 0: self.merged_location_filter_combo.setCurrentIndex(index)
            else: self.merged_location_filter_combo.setCurrentIndex(0)
        else: self.merged_location_filter_combo.setCurrentIndex(0)

    # --- Metode Visualisasi (placeholder jika tidak diaktifkan) ---
    def _create_webview_for_plot(self, title="Plot Area"):
        is_real_webview = PYQTWEBENGINE_AVAILABLE and QWebEngineView is not QLabel
        
        if is_real_webview:
            web_view = QWebEngineView()
            if QWebEngineSettings is not None : 
                settings = web_view.settings()
                if hasattr(settings, 'setAttribute'): 
                    settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
            web_view.setHtml(f"<html><body style='display:flex;justify-content:center;align-items:center;height:100%;font-family:Arial,sans-serif;color:#999;'><p>{title} (menunggu data)...</p></body></html>")
            web_view.setMinimumHeight(250)
            return web_view
        else: # Fallback
            label = QLabel(f"{title}\n(PyQtWebEngine tidak tersedia)")
            label.setMinimumHeight(200); label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("border:1px solid #ddd; background:#f0f0f0; color:red;")
            return label 

    def _create_viz_label_placeholder(self, text="Nilai KPI"):
        lbl = QLabel(text); lbl.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); lbl.setMinimumHeight(50)
        return lbl

    def _create_viz_groupbox(self, title, widget_content):
        gb = QGroupBox(title); layout = QVBoxLayout(gb); layout.addWidget(widget_content)
        return gb

    def _update_visualizations_on_merged_data(self, filtered_data_for_viz: pd.DataFrame = None):
        if not all(hasattr(self, attr) for attr in ['viz_kpi_total_produk_label', 'viz_kpi_total_ulasan_label', 
                                                     'viz_kpi_avg_rating_label', 'viz_rating_dist_view', 
                                                     'viz_location_dist_view', 'viz_wordcloud_label']):
            return 

        df_to_plot = filtered_data_for_viz if filtered_data_for_viz is not None and not filtered_data_for_viz.empty else self.merged_df_with_labels
        
        is_rating_view_real_webview = isinstance(self.viz_rating_dist_view, QWebEngineView) and QWebEngineView is not QLabel
        is_location_view_real_webview = isinstance(self.viz_location_dist_view, QWebEngineView) and QWebEngineView is not QLabel

        if not PLOTLY_AVAILABLE:
            self.viz_kpi_total_produk_label.setText("Produk: Plotly N/A")
            self.viz_kpi_total_ulasan_label.setText("Ulasan: Plotly N/A")
            self.viz_kpi_avg_rating_label.setText("Avg. Rating: Plotly N/A")
            if not is_rating_view_real_webview : self.viz_rating_dist_view.setText("Plotly N/A")
            else: self.viz_rating_dist_view.setHtml("<html><body><p>Plotly tidak tersedia.</p></body></html>")
            if not is_location_view_real_webview : self.viz_location_dist_view.setText("Plotly N/A")
            else: self.viz_location_dist_view.setHtml("<html><body><p>Plotly tidak tersedia.</p></body></html>")
            self.viz_wordcloud_label.setText("WordCloud: Plotly N/A"); self.viz_wordcloud_label.setPixmap(QPixmap())
            return

        if df_to_plot is None or df_to_plot.empty:
            self.viz_kpi_total_produk_label.setText("Total Produk: 0")
            self.viz_kpi_total_ulasan_label.setText("Total Ulasan: 0")
            self.viz_kpi_avg_rating_label.setText("Avg. Rating: N/A")
            self._display_plotly_fig(None, self.viz_rating_dist_view) 
            self._display_plotly_fig(None, self.viz_location_dist_view)
            self.viz_wordcloud_label.setText("Tidak ada data untuk Word Cloud."); self.viz_wordcloud_label.setPixmap(QPixmap())
            return

        total_produk_unik = df_to_plot['product_id'].nunique() if 'product_id' in df_to_plot.columns else 0
        self.viz_kpi_total_produk_label.setText(f"Total Produk (Unik): {total_produk_unik:,}")
        
        total_ulasan = len(df_to_plot) 
        self.viz_kpi_total_ulasan_label.setText(f"Total Ulasan (dari Gabungan): {total_ulasan:,}")
        
        # After merge, review rating is 'rating_ulasan'
        rating_col_for_avg = 'rating_ulasan' if 'rating_ulasan' in df_to_plot.columns else 'rating_rev' if 'rating_rev' in df_to_plot.columns else None
        avg_rating_val = np.nan
        if rating_col_for_avg and df_to_plot[rating_col_for_avg].notna().any():
            avg_rating_val = pd.to_numeric(df_to_plot[rating_col_for_avg], errors='coerce').mean()
        self.viz_kpi_avg_rating_label.setText(f"Avg. Rating Ulasan: {avg_rating_val:.2f}" if pd.notna(avg_rating_val) else "Avg. Rating: N/A")

        fig_rating = None
        if rating_col_for_avg and df_to_plot[rating_col_for_avg].notna().any():
            try:
                ratings_numeric = pd.to_numeric(df_to_plot[rating_col_for_avg], errors='coerce').dropna()
                if not ratings_numeric.empty:
                    fig_rating = px.histogram(ratings_numeric, nbins=5, title="Distribusi Rating Ulasan")
                    fig_rating.update_layout(bargap=0.2, height=280, margin=dict(l=10,r=10,t=30,b=10))
            except Exception as e: print(f"Error membuat plot rating: {e}")
        self._display_plotly_fig(fig_rating, self.viz_rating_dist_view)

        fig_loc = None
        if 'lokasi' in df_to_plot.columns and 'product_id' in df_to_plot.columns and df_to_plot['lokasi'].notna().any():
            try:
                loc_counts = df_to_plot.groupby('lokasi')['product_id'].nunique().nlargest(10).reset_index()
                loc_counts.columns = ['Lokasi Toko', 'Jumlah Produk Unik']
                if not loc_counts.empty:
                    fig_loc = px.bar(loc_counts, y='Lokasi Toko', x='Jumlah Produk Unik', orientation='h', text_auto='.2s', title="Top 10 Lokasi Toko")
                    fig_loc.update_layout(height=280, yaxis={'categoryorder':'total ascending'}, margin=dict(l=80,r=10,t=30,b=10))
            except Exception as e: print(f"Error membuat plot lokasi: {e}")
        self._display_plotly_fig(fig_loc, self.viz_location_dist_view)
                
        self._plot_word_cloud_on_merged(self.viz_wordcloud_label, df_to_plot)


    def _display_plotly_fig(self, fig, web_view_widget): 
        is_real_webview = isinstance(web_view_widget, QWebEngineView) and QWebEngineView is not QLabel

        if fig and PLOTLY_AVAILABLE and is_real_webview:
            try:
                plot_div_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True, 'displaylogo': False})
                full_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8" />
                                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                                <style>body{{margin:0;padding:0;overflow:hidden;}} 
                                .plotly-graph-div{{margin:auto;width:100%;height:100%;}} 
                                html, body {{width: 100%; height: 100%;}}</style></head>
                                <body>{plot_div_html}</body></html>"""
                web_view_widget.setHtml(full_html)
            except Exception as e:
                print(f"Error converting plotly fig to HTML: {e}")
                web_view_widget.setHtml(f"<html><body><p>Error menampilkan plot: {e}.</p></body></html>")

        elif is_real_webview:
             web_view_widget.setHtml(f"<html><body style='display:flex;justify-content:center;align-items:center;height:100%;font-family:Arial,sans-serif;color:#aaa;'><p>Tidak ada data untuk visualisasi atau error.</p></body></html>")
        elif isinstance(web_view_widget, QLabel):
            if fig and PLOTLY_AVAILABLE:
                 web_view_widget.setText("Plot dibuat (PyQtWebEngine N/A).")
            elif not PLOTLY_AVAILABLE:
                 web_view_widget.setText("Plotly tidak tersedia.")
            else:
                 web_view_widget.setText("Tidak ada data untuk plot ini.")


    def _plot_word_cloud_on_merged(self, label_widget: QLabel, df_for_wc: pd.DataFrame):
        if not WORDCLOUD_AVAILABLE or not plt:
            if label_widget: label_widget.setText("WordCloud/Matplotlib N/A."); label_widget.setPixmap(QPixmap()); return
        
        # After merge, review column is 'ulasan'
        ulasan_col = 'ulasan_produk' if 'ulasan_produk' in df_for_wc.columns else 'ulasan' if 'ulasan' in df_for_wc.columns else None
        if ulasan_col is None or df_for_wc.empty or df_for_wc[ulasan_col].isna().all():
            if label_widget: label_widget.setText("Data ulasan tidak valid."); label_widget.setPixmap(QPixmap()); return
        
        if self.status_label: self.status_label.setText("Membuat Word Cloud...");
        if QApplication.instance(): QApplication.processEvents()
        try:
            text_list = df_for_wc[ulasan_col].dropna().astype(str).tolist()
            if not text_list: 
                if label_widget: label_widget.setText("Teks ulasan kosong."); label_widget.setPixmap(QPixmap()); return

            processed_text = placeholder_preprocess_documents(text_list)
            all_text = " ".join(processed_text)
            if not all_text.strip():
                if label_widget: label_widget.setText("Tidak ada kata setelah preprocess."); label_widget.setPixmap(QPixmap()); return

            wordcloud = WordCloud(width=400, height=250, background_color='white',
                                    max_words=75, collocations=False, colormap="viridis").generate(all_text)
            
            fig_temp_wc = plt.figure(figsize=(4, 2.5), dpi=100)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.tight_layout(pad=0)

            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig_temp_wc)
            buf.seek(0)
            
            pixmap = QPixmap()
            pixmap.loadFromData(buf.getvalue())
            buf.close()
            
            if label_widget:
                if label_widget.width() > 20 and label_widget.height() > 20:
                    scaled_pixmap = pixmap.scaled(label_widget.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    label_widget.setPixmap(scaled_pixmap)
                else:
                    label_widget.setPixmap(pixmap)
            if self.status_label: self.status_label.setText("Word Cloud dari data gabungan dibuat.")
        except Exception as e:
            print(f"Error saat membuat Word Cloud: {e}")
            if label_widget: label_widget.setText(f"Error WC:\n{str(e)[:60]}..."); label_widget.setPixmap(QPixmap())


    def closeEvent(self, event):
        if self.product_scraper_worker and self.product_scraper_worker.isRunning():
            self.product_scraper_worker.stop()
            self.product_scraper_worker.wait() 
        if self.review_scraper_worker and self.review_scraper_worker.isRunning():
            self.review_scraper_worker.stop()
            self.review_scraper_worker.wait()
        super().closeEvent(event)
class TopicExtractionWorker(QThread):
    started = pyqtSignal()
    finished = pyqtSignal(object, object, object, str) # topics, probs, topic_info, error_message
    progress = pyqtSignal(str)

    def __init__(self, topic_extraction_instance, documents):
        super().__init__()
        self.topic_extractor = topic_extraction_instance
        self.documents = documents

    def run(self):
        self.started.emit()
        try:
            self.progress.emit("Initializing BERTopic model...")
            if not self.topic_extractor.topic_model: # Inisialisasi jika belum
                 self.topic_extractor.initialize_model()
            
            self.progress.emit("Fitting BERTopic model and transforming documents...")
            topics, probs = self.topic_extractor.topic_model.fit_transform(self.documents)
            
            self.progress.emit("Fetching topic information...")
            topic_info = self.topic_extractor.topic_model.get_topic_info()
            
            # Simpan hasil ke instance TopicExtraction utama (opsional, bisa juga hanya diemit)
            self.topic_extractor.topics = topics
            self.topic_extractor.probs = probs
            self.topic_extractor.topic_info = topic_info
            
            self.finished.emit(topics, probs, topic_info, None) # Tidak ada error
        except Exception as e:
            self.finished.emit(None, None, None, str(e))

class TopicExtraction:
    def __init__(self,
                 embedding_model_name="paraphrase-multilingual-MiniLM-L12-v2",
                 umap_n_neighbors=15, umap_n_components=5, umap_min_dist=0.0, umap_metric='cosine',
                 hdbscan_min_cluster_size=15, hdbscan_metric='euclidean', hdbscan_cluster_selection_method='eom',
                 vectorizer_ngram_range=(1, 2),
                 min_topic_size=15, nr_topics="auto", language="multilingual",
                 initial_user_representation_config=None,
                 initial_user_representation_kwargs=None):

        self.embedding_model_name = embedding_model_name
        self.umap_params = {'n_neighbors': umap_n_neighbors, 'n_components': umap_n_components,
                            'min_dist': umap_min_dist, 'metric': umap_metric, 'random_state': 42}
        self.hdbscan_params = {'min_cluster_size': hdbscan_min_cluster_size, 'metric': hdbscan_metric,
                               'cluster_selection_method': hdbscan_cluster_selection_method,
                               'prediction_data': True}
        self.vectorizer_params = {'ngram_range': vectorizer_ngram_range}

        self.embedding_model_instance = embedding_model_name
        self.umap_model_instance = UMAP(**self.umap_params)
        self.hdbscan_model_instance = HDBSCAN(**self.hdbscan_params)
        self.vectorizer_model_instance = CountVectorizer(**self.vectorizer_params)

        self.user_initial_rep_config = initial_user_representation_config
        self.user_initial_rep_kwargs = initial_user_representation_kwargs if initial_user_representation_kwargs else {}
        # from BERTopic import bertopic
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model_instance,
            umap_model=self.umap_model_instance,
            hdbscan_model=self.hdbscan_model_instance,
            vectorizer_model=self.vectorizer_model_instance,
            representation_model=None,
            min_topic_size=min_topic_size,
            nr_topics=nr_topics,
            language=language,
            calculate_probabilities=True,
            verbose=True
        )
        self.documents_processed_for_last_fit = None
        self.topics_ = None
        self.probs_ = None
        self.ctfidf_keywords_per_topic = {}
        self.ctfidf_names_per_topic = {}


    def initialize_model(self):
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
        from hdbscan import HDBSCAN
        from sentence_transformers import SentenceTransformer


        current_nr_topics = self.nr_topics
        if isinstance(current_nr_topics, str) and current_nr_topics.isdigit():
            current_nr_topics = int(current_nr_topics)
        elif current_nr_topics != "auto":
            current_nr_topics = "auto"

        try:
            # Gunakan nama model embedding dari parameter
            sentence_model = SentenceTransformer(self.embedding_model_name)
            
            # Gunakan parameter UMAP dan HDBSCAN
            umap_model = UMAP(**self.umap_params)
            
            # Pastikan min_cluster_size di HDBSCAN sesuai dengan min_topic_size
            hdbscan_model_params = self.hdbscan_params.copy()
            hdbscan_model_params["min_cluster_size"] = self.min_topic_size_bertopic # Gunakan ini untuk HDBSCAN
            hdbscan_model = HDBSCAN(**hdbscan_model_params)
            
            # Gunakan parameter Vectorizer
            # BERTopic akan menangani stop words berdasarkan 'language' jika tidak ada stop_words di sini
            vectorizer_model = CountVectorizer(**self.vectorizer_params) 
            # from BERTopic import bertopic

            self.topic_model = BERTopic(
                embedding_model=sentence_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                min_topic_size=self.min_topic_size_bertopic, # Parameter utama BERTopic
                nr_topics=current_nr_topics,
                language=self.language,
                calculate_probabilities=True,
                verbose=True
            )
        except Exception as e:
            # print(f"Error initializing BERTopic components: {e}")
            raise




    def _create_representation_model(self, model_name_or_config, kwargs): # Diganti dari _from_name
        if not isinstance(model_name_or_config, str):
            return model_name_or_config

        model_name_str = model_name_or_config
        if model_name_str == "MMR":
            diversity = kwargs.get('diversity', 0.7)
            return MaximalMarginalRelevance(diversity=diversity)
        elif model_name_str == "KeyBERTInspired":
            return KeyBERTInspired()
        elif model_name_str == "OpenAI (Prompt Kustom)" and OPENAI_REPRESENTATION_AVAILABLE:
            
            try:
                # Pastikan openai.OpenAI() dipanggil, bukan hanya OpenAI dari bertopic.representation
                import openai as openai_sdk 
                openai_client = openai_sdk.OpenAI()
            except Exception as e:
                raise RuntimeError(f"Gagal membuat OpenAI client: {e}. Pastikan OPENAI_API_KEY terset dan library openai terinstal.") from e
            
            openai_llm_model = kwargs.get('openai_model_name', "gpt-4o-mini")
            openai_prompt_str = kwargs.get('openai_prompt', None)
            
            if not openai_prompt_str:
                print("Warning: Prompt OpenAI kosong, model OpenAI mungkin tidak berfungsi sesuai harapan.")

            return BERTopicOpenAI(
                client=openai_client, # Teruskan client SDK OpenAI
                model=openai_llm_model,
                prompt=openai_prompt_str,
                chat=True,
                exponential_backoff=True
            )
        elif model_name_str == "Default (c-TF-IDF)" or model_name_str is None:
            return None
        else:
            print(f"Warning: Model representasi '{model_name_str}' tidak dikenal. Default (c-TF-IDF) akan digunakan.")
            return None

    def fit_transform(self, documents):
        self.documents_processed_for_last_fit = documents
        try:
            self.topics_, self.probs_ = self.topic_model.fit_transform(documents)

            ctfidf_info_df = self.topic_model.get_topic_info()
            self.ctfidf_keywords_per_topic = {
                row['Topic']: row['Representation']
                for _, row in ctfidf_info_df.iterrows()
            }
            self.ctfidf_names_per_topic = {
                row['Topic']: row['Name']
                for _, row in ctfidf_info_df.iterrows()
            }

            if self.user_initial_rep_config and \
               not (isinstance(self.user_initial_rep_config, str) and "Default (c-TF-IDF)" in self.user_initial_rep_config) :

                actual_initial_rep_model_instance = self._create_representation_model( # Menggunakan nama baru
                    self.user_initial_rep_config,
                    self.user_initial_rep_kwargs
                )
                if actual_initial_rep_model_instance is not None:
                    print(f"Menerapkan representasi kustom awal: {self.user_initial_rep_config}")
                    self.topic_model.update_topics(
                        docs=documents,
                        representation_model=actual_initial_rep_model_instance
                    )
                    self.topic_model.representation_model = actual_initial_rep_model_instance

            final_topic_info_df_for_ui = self.get_topic_info_df()

            return self.topics_, self.probs_, final_topic_info_df_for_ui, None
        except Exception as e:
            self.topics_, self.probs_ = None, None
            self.ctfidf_keywords_per_topic = {}
            self.ctfidf_names_per_topic = {}
            return None, None, None, str(e)

    def update_topic_representation(self, documents, representation_model_config, **kwargs_rep):
        if not self.topic_model or not hasattr(self.topic_model, 'topics_') or self.topic_model.topics_ is None:
            raise ValueError("Model BERTopic belum di-fit. Ekstrak topik terlebih dahulu.")
        if not documents:
             raise ValueError("Dokumen diperlukan untuk update_topics.")

        new_rep_model_instance = self._create_representation_model( # Menggunakan nama baru
            representation_model_config, kwargs_rep
        )

        try:
            self.topic_model.update_topics(docs=documents,
                                           representation_model=new_rep_model_instance)
            self.topic_model.representation_model = new_rep_model_instance
            return self.get_topic_info_df()
        except Exception as e:
            print(f"Error saat update representasi: {e}")
            raise

    def get_topic_info_df(self):
        if not self.topic_model or not hasattr(self.topic_model, 'topics_') or self.topic_model.topics_ is None:
            return pd.DataFrame()

        current_model_topic_info_df = self.topic_model.get_topic_info()

        if not current_model_topic_info_df.empty:
            current_model_topic_info_df['Representasi_Lengkap_CTFIDF'] = \
                current_model_topic_info_df['Topic'].map(self.ctfidf_keywords_per_topic)
            
            current_model_topic_info_df['Representasi_Lengkap_CTFIDF'] = \
                current_model_topic_info_df['Representasi_Lengkap_CTFIDF'].apply(
                    lambda x: x if isinstance(x, list) else []
                )
        return current_model_topic_info_df

    def get_document_info_df(self, documents):
        if not self.topic_model or not hasattr(self.topic_model, 'topics_') or self.topic_model.topics_ is None:
            return pd.DataFrame()
        try:
            doc_info_df = self.topic_model.get_document_info(documents)
            
            topic_info_with_ctfidf = self.get_topic_info_df()
            if not topic_info_with_ctfidf.empty and 'Count' in topic_info_with_ctfidf.columns and 'Topic' in doc_info_df.columns:
                cols_to_drop_from_doc_info = []
                if 'Name' in doc_info_df.columns: cols_to_drop_from_doc_info.append('Name')
                if 'Count' in doc_info_df.columns: cols_to_drop_from_doc_info.append('Count')
                if cols_to_drop_from_doc_info:
                    doc_info_df = doc_info_df.drop(columns=cols_to_drop_from_doc_info, errors='ignore')

                doc_info_df = doc_info_df.merge(
                    topic_info_with_ctfidf[['Topic', 'Name', 'Count']],
                    on="Topic",
                    how="left"
                )
            return doc_info_df
        except Exception as e:
            print(f"Error di get_document_info_df: {e}")
            return pd.DataFrame()

    def visualize_intertopic_map(self, **kwargs):
        if self.topic_model and hasattr(self.topic_model, 'topics_') and self.topic_model.topics_ is not None:
            return self.topic_model.visualize_topics(**kwargs)
        return None

    def visualize_hierarchy(self, **kwargs):
        if self.topic_model and hasattr(self.topic_model, 'topics_') and self.topic_model.topics_ is not None:
            return self.topic_model.visualize_hierarchy(**kwargs)
        return None

    def visualize_barchart(self, **kwargs):
        if self.topic_model and hasattr(self.topic_model, 'topics_') and self.topic_model.topics_ is not None:
            return self.topic_model.visualize_barchart(**kwargs)
        return None

    def visualize_heatmap(self, **kwargs):
        if self.topic_model and hasattr(self.topic_model, 'topics_') and self.topic_model.topics_ is not None:
            return self.topic_model.visualize_heatmap(**kwargs)
        return None

    def save_model(self, path, serialization_method="pickle", include_embedding_model=False): # Ubah default include_embedding_model
        if not self.topic_model:
            raise ValueError("Model BERTopic belum diinisialisasi.")
        if not hasattr(self.topic_model, 'topics_') or self.topic_model.topics_ is None:
            raise ValueError("Model BERTopic belum di-fit. Tidak ada yang bisa disimpan.")

        try:
            self.topic_model.save(
                path,
                serialization=serialization_method,
                save_embedding_model=include_embedding_model # Gunakan argumen ini
            )
            print(f"Model BERTopic disimpan ke: {path} (model embedding {'disertakan' if include_embedding_model else 'tidak disertakan'})")

            if hasattr(self, 'documents_processed_for_last_fit') and self.documents_processed_for_last_fit:
                base, ext = os.path.splitext(path)
                doc_path = f"{base}_docs.pkl"
                try:
                    with open(doc_path, 'wb') as f_docs:
                        pickle.dump(self.documents_processed_for_last_fit, f_docs)
                    print(f"Dokumen yang diproses disimpan ke: {doc_path}")
                except Exception as e:
                    print(f"Peringatan: Gagal menyimpan dokumen yang diproses ke {doc_path}: {e}")
            else:
                print("Tidak ada dokumen yang diproses untuk disimpan bersama model.")

        except Exception as e:
            # Tangkap error pickling di sini jika masih terjadi
            print(f"Error saat menyimpan model BERTopic: {e}")
            raise # Lemparkan kembali error agar UI bisa menangani
    

    @staticmethod
    def load_model(path, **kwargs):
        # from BERTopic import bertopic

        loaded_bertopic_model = BERTopic.load(path, **kwargs)
        
        instance = TopicExtraction() 
        instance.topic_model = loaded_bertopic_model
        instance.topics_ = getattr(loaded_bertopic_model, 'topics_', None)
        instance.probs_ = getattr(loaded_bertopic_model, 'probabilities_', None)

        base, ext = os.path.splitext(path)
        doc_path = f"{base}_docs.pkl"
        try:
            with open(doc_path, 'rb') as f_docs:
                instance.documents_processed_for_last_fit = pickle.load(f_docs)
            print(f"Dokumen yang diproses berhasil dimuat dari: {doc_path}")
        except FileNotFoundError:
            print(f"File dokumen yang diproses tidak ditemukan di: {doc_path}. Fitur yang memerlukan dokumen asli mungkin tidak berfungsi optimal tanpa data baru.")
            instance.documents_processed_for_last_fit = None
        except Exception as e:
            print(f"Peringatan: Gagal memuat dokumen yang diproses dari {doc_path}: {e}")
            instance.documents_processed_for_last_fit = None
            
        if instance.topics_ is not None:
            temp_loaded_info = loaded_bertopic_model.get_topic_info()
            if not temp_loaded_info.empty:
                instance.ctfidf_keywords_per_topic = {
                    row['Topic']: row['Representation']
                    for _, row in temp_loaded_info.iterrows()
                }
                instance.ctfidf_names_per_topic = {
                    row['Topic']: row['Name']
                    for _, row in temp_loaded_info.iterrows()
                }
        else:
            print("Peringatan saat memuat: Model BERTopic tampaknya belum di-fit (tidak ada atribut topics_).")

        try:
            if hasattr(loaded_bertopic_model, 'embedding_model') and loaded_bertopic_model.embedding_model:
                if isinstance(loaded_bertopic_model.embedding_model, str):
                    instance.embedding_model_name = loaded_bertopic_model.embedding_model
                elif hasattr(loaded_bertopic_model.embedding_model, 'model_name_or_path'):
                    instance.embedding_model_name = loaded_bertopic_model.embedding_model.model_name_or_path
                elif hasattr(loaded_bertopic_model.embedding_model, '_model_config') and 'name_or_path' in loaded_bertopic_model.embedding_model._model_config:
                    instance.embedding_model_name = loaded_bertopic_model.embedding_model._model_config['name_or_path']

            if hasattr(loaded_bertopic_model, 'min_topic_size'):
                 instance.min_topic_size = loaded_bertopic_model.min_topic_size
            if hasattr(loaded_bertopic_model, 'language'):
                instance.language = loaded_bertopic_model.language
        except Exception as e:
            print(f"Info: Tidak dapat sepenuhnya menyinkronkan beberapa parameter dari model yang dimuat: {e}")
            
        return instance




class TopicAnalysisPage(QWidget):
    def __init__(self, review_data_df, parent=None): # Ubah nama argumen agar jelas itu DataFrame
        super().__init__(parent)
        # Pastikan review_data adalah DataFrame atau bisa diubah jadi DataFrame
        if isinstance(review_data_df, pd.DataFrame):
            self.review_data_df = review_data_df.copy() # Selalu bekerja dengan salinan
        elif review_data_df is None:
            self.review_data_df = pd.DataFrame() # DataFrame kosong jika tidak ada data awal
        else:
            try: # Coba konversi jika bukan DataFrame (misalnya list of dict)
                self.review_data_df = pd.DataFrame(review_data_df)
            except:
                QMessageBox.critical(self, "Data Error", "Review data format not supported. Expected pandas DataFrame.")
                self.review_data_df = pd.DataFrame()


        self.topic_extraction_logic = None # Ini akan menjadi instance dari TopicExtraction
        self.current_topic_info_df = pd.DataFrame()
        self.current_doc_results_df = pd.DataFrame()
        self.current_viz_html = ""

        self.initUI()
    def set_openai_env_variable(self):
        api_key = self.api_key_input.text().strip()
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
            self.openai_api_key = api_key # Simpan juga di instance untuk penggunaan langsung
            self.progress_label.setText("OPENAI_API_KEY telah diatur untuk sesi ini.")
            # QMessageBox.information(self, "API Key Diatur", 
            #                         "OPENAI_API_KEY telah diatur sebagai variabel lingkungan untuk sesi aplikasi ini.\n"
            #                         "Pustaka OpenAI akan mencoba mengambilnya secara otomatis.")
            # print(f"DEBUG: OPENAI_API_KEY diatur ke: {api_key[:5]}...{api_key[-5:]}") # Jangan print full key
        else:
            QMessageBox.warning(self, "API Key Kosong", "Silakan masukkan API Key OpenAI terlebih dahulu.")
            self.progress_label.setText("Gagal mengatur API Key: Input kosong.")        
    def initUI(self):
        self.setup_controls()
        self.setup_results_display()
        self.setup_visualization()
        
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.control_group)
        
        split_layout = QHBoxLayout()
        split_layout.addWidget(self.results_group, 2) # Proporsi 2 untuk hasil
        split_layout.addWidget(self.viz_group, 1)     # Proporsi 1 untuk visualisasi
        main_layout.addLayout(split_layout)
        
        self.setLayout(main_layout)
        
    def setup_controls(self):
        self.control_group = QGroupBox("Pengaturan Analisis Topik & Komponen Model")
        
        bertopic_params_group = QGroupBox("Parameter BERTopic Utama")
        bertopic_params_layout = QFormLayout()

        self.min_topic_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_topic_slider.setRange(5, 100)
        self.min_topic_slider.setValue(15)
        self.min_topic_slider.setTickInterval(5)
        self.min_topic_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.min_topic_label = QLabel(f"{self.min_topic_slider.value()}")
        self.min_topic_slider.valueChanged.connect(lambda val: self.min_topic_label.setText(str(val)))
        min_topic_layout = QHBoxLayout()
        min_topic_layout.addWidget(self.min_topic_slider)
        min_topic_layout.addWidget(self.min_topic_label)

        self.nr_topics_combo = QComboBox()
        self.nr_topics_combo.addItems(["auto", "5", "10", "15", "20", "25", "30"])
        self.nr_topics_combo.setCurrentText("auto")
        
        self.language_combo = QComboBox()
        self.language_combo.addItems(["multilingual", "english"])
        self.language_combo.setCurrentText("multilingual")
        bertopic_params_layout.addRow("Min. Ukuran Topik (HDBSCAN):", min_topic_layout)
        bertopic_params_layout.addRow("Jumlah Topik (nr_topics):", self.nr_topics_combo)
        bertopic_params_layout.addRow("Bahasa Model:", self.language_combo)
        bertopic_params_group.setLayout(bertopic_params_layout)

        component_params_group = QGroupBox("Parameter Komponen Internal")
        main_component_layout = QHBoxLayout()

        left_layout = QFormLayout()
        self.embedding_model_input = QLineEdit("paraphrase-multilingual-MiniLM-L12-v2")
        left_layout.addRow("Model Embedding:", self.embedding_model_input)

        self.hdbscan_metric_combo = QComboBox()
        self.hdbscan_metric_combo.addItems(['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'hamming', 'canberra'])
        left_layout.addRow("HDBSCAN - metric:", self.hdbscan_metric_combo)
        self.hdbscan_selection_combo = QComboBox()
        self.hdbscan_selection_combo.addItems(['eom', 'leaf'])
        left_layout.addRow("HDBSCAN - method:", self.hdbscan_selection_combo)
        self.vectorizer_ngram_min_spin = QSpinBox()
        self.vectorizer_ngram_min_spin.setRange(1, 5); self.vectorizer_ngram_min_spin.setValue(1)
        self.vectorizer_ngram_max_spin = QSpinBox()
        self.vectorizer_ngram_max_spin.setRange(1, 5); self.vectorizer_ngram_max_spin.setValue(2)
        ngram_layout = QHBoxLayout()
        ngram_layout.addWidget(self.vectorizer_ngram_min_spin)
        ngram_layout.addWidget(QLabel("to"))
        ngram_layout.addWidget(self.vectorizer_ngram_max_spin)
        left_layout.addRow("Vectorizer - N-gram:", ngram_layout)

        right_layout = QFormLayout()
        self.representation_model_combo = QComboBox()
        rep_models_options = ["Default (c-TF-IDF)", "KeyBERTInspired", "MMR"]
        if OPENAI_REPRESENTATION_AVAILABLE:
            rep_models_options.append("OpenAI (Prompt Kustom)")
        self.representation_model_combo.addItems(rep_models_options)
        self.representation_model_combo.setToolTip("Pilih model untuk menghasilkan representasi kata kunci topik.")
        right_layout.addRow("Model Representasi:", self.representation_model_combo)

        self.mmr_diversity_spin = QDoubleSpinBox()
        self.mmr_diversity_spin.setRange(0.0, 1.0); self.mmr_diversity_spin.setValue(0.7); self.mmr_diversity_spin.setSingleStep(0.1)
        self.mmr_diversity_label = QLabel("MMR - Diversity (0.0-1.0):")
        right_layout.addRow(self.mmr_diversity_label, self.mmr_diversity_spin)
        
        if OPENAI_REPRESENTATION_AVAILABLE:
            self.openai_model_label = QLabel("OpenAI - Model:")
            self.api_key_input_label = QLabel("API Key:")
            self.api_key_input = QLineEdit()
            self.api_key_input.setPlaceholderText("Masukkan OpenAI API Key Anda di sini")
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password) # Sembunyikan input

            self.openai_model_input = QLineEdit("gpt-4o-mini")
            self.openai_model_input.setToolTip("Contoh: gpt-4o-mini, gpt-4, gpt-3.5-turbo")
            right_layout.addRow(self.openai_model_label, self.openai_model_input)
            self.openai_model_label.setVisible(False)
            self.openai_model_input.setVisible(False)
            right_layout.addRow(self.api_key_input_label, self.api_key_input)

            self.openai_prompt_label = QLabel("OpenAI - Prompt:")
            self.openai_prompt_text = QTextEdit()
            self.openai_prompt_text.setPlainText(
                "Saya memiliki sebuah topik yang berisi dokumen-dokumen berikut mengenai ulasan produk:\n"
                "[DOCUMENTS]\n\n"
                "Topik ini dideskripsikan oleh kata kunci berikut: [KEYWORDS]\n\n"
                "Berdasarkan informasi di atas, ekstrak sebuah label topik yang pendek (maksimal 3 kata) namun sangat deskriptif. "
                "Label ini harus secara spesifik merepresentasikan dimensi kepuasan pelanggan terhadap produk.\n\n"
                "Contoh dimensi kepuasan yang relevan bisa meliputi (namun tidak terbatas pada):\n"
                "- Kecepatan terbang\n- Ketahanan bulu\n- Kualitas gabus\n- Kestabilan terbang\n"
                "- Kesesuaian harga\n- Konsistensi kualitas slop\n\n"
                "Pastikan output dalam format berikut:: <topic label>"
            )
            self.openai_prompt_text.setFixedHeight(120)
            right_layout.addRow(self.openai_prompt_label, self.openai_prompt_text)
        
        self.update_representation_btn = QPushButton("🔄 Update Representasi Topik")
        self.update_representation_btn.setEnabled(True)
        # self.update_representation_btn.clicked.connect(self.set_openai_env_variable)
        self.update_representation_btn.clicked.connect(self.update_topic_representation_ui)

        right_layout.addWidget(self.update_representation_btn)

        mid_layout = QFormLayout()
        self.umap_neighbors_spin = QSpinBox()
        self.umap_neighbors_spin.setRange(2, 200); self.umap_neighbors_spin.setValue(15)
        mid_layout.addRow("UMAP - n_neighbors:", self.umap_neighbors_spin)
        self.umap_components_spin = QSpinBox()
        self.umap_components_spin.setRange(2, 100); self.umap_components_spin.setValue(5)
        mid_layout.addRow("UMAP - n_components:", self.umap_components_spin)
        self.umap_min_dist_spin = QDoubleSpinBox()
        self.umap_min_dist_spin.setRange(0.0, 0.99); self.umap_min_dist_spin.setValue(0.0); self.umap_min_dist_spin.setSingleStep(0.01)
        mid_layout.addRow("UMAP - min_dist:", self.umap_min_dist_spin)
        self.umap_metric_combo = QComboBox()
        self.umap_metric_combo.addItems(['cosine', 'euclidean', 'manhattan', 'chebyshev', 'minkowski'])
        mid_layout.addRow("UMAP - metric:", self.umap_metric_combo)
        

        main_component_layout.addLayout(left_layout)
        main_component_layout.addLayout(mid_layout)
        main_component_layout.addLayout(right_layout)

        component_params_group.setLayout(main_component_layout)

        cleaning_group = QGroupBox("Opsi Preprocessing Teks (Bahasa Indonesia)")
        cleaning_group_layout = QVBoxLayout()
        main_horizontal_layout = QHBoxLayout()
        column1_layout = QVBoxLayout()
        self.clean_formatting_cb = QCheckBox("Bersihkan Formatting")
        self.clean_formatting_cb.setChecked(True)
        column1_layout.addWidget(self.clean_formatting_cb)
        self.replace_slang_cb = QCheckBox("Kata Gaul")
        self.replace_slang_cb.setChecked(True)
        column1_layout.addWidget(self.replace_slang_cb)
        column2_layout = QVBoxLayout()
        self.stemming_cb = QCheckBox("Terapkan Stemming")
        self.stemming_cb.setChecked(True)
        column2_layout.addWidget(self.stemming_cb)
        self.stopword_cb = QCheckBox("Hapus Stopwords")
        self.stopword_cb.setChecked(True)
        column2_layout.addWidget(self.stopword_cb)
        main_horizontal_layout.addLayout(column1_layout)
        main_horizontal_layout.addLayout(column2_layout)
        cleaning_group_layout.addLayout(main_horizontal_layout)
        custom_stopword_layout = QVBoxLayout()
        self.custom_stopwords_input = QLineEdit()
        self.custom_stopwords_input.setPlaceholderText("Masukkan custom stopwords dipisahkan koma")
        custom_stopword_layout.addWidget(self.custom_stopwords_input)
        cleaning_group_layout.addLayout(custom_stopword_layout)
        cleaning_group.setLayout(cleaning_group_layout)

        button_layout = QVBoxLayout()
        self.extract_btn = QPushButton("🚀 Ekstrak Topik")
        self.extract_btn.setStyleSheet("background-color: #28a745; color: white; font-weight:bold;")
        self.save_model_btn = QPushButton("💾 Simpan Model BERTopic")
        self.save_model_btn.setEnabled(False)
        self.load_model_btn = QPushButton("📂 Muat Model BERTopic")
        button_layout.addWidget(self.extract_btn)
        button_layout.addWidget(self.save_model_btn)
        button_layout.addWidget(self.load_model_btn)
        
        self.status_label = QLabel("Status: Siap untuk analisis topik.")
        self.status_label.setStyleSheet("font-style: italic; color: #555; padding: 5px; border-top: 1px solid #ddd;")
        
        basic_control_hbox = QHBoxLayout()
        basic_control_hbox.addLayout(button_layout)
        basic_control_hbox.addWidget(cleaning_group)
        basic_control_hbox.addWidget(bertopic_params_group)

        main_control_vbox = QVBoxLayout()
        main_control_vbox.addLayout(basic_control_hbox)
        main_control_vbox.addWidget(self.status_label)
        main_control_vbox.addWidget(component_params_group)
        self.control_group.setLayout(main_control_vbox)
        
        self.extract_btn.clicked.connect(self.start_topic_extraction_thread)
        self.save_model_btn.clicked.connect(self.save_bertopic_model)
        self.load_model_btn.clicked.connect(self.load_bertopic_model)
        self.representation_model_combo.currentTextChanged.connect(self._toggle_representation_params_visibility)
        self._toggle_representation_params_visibility(self.representation_model_combo.currentText())


    def _toggle_representation_params_visibility(self, model_name_text):
        is_mmr = model_name_text == "MMR"
        self.mmr_diversity_label.setVisible(is_mmr)
        self.mmr_diversity_spin.setVisible(is_mmr)

        if OPENAI_REPRESENTATION_AVAILABLE: # Pastikan atribut ini ada
            is_openai = model_name_text == "OpenAI (Prompt Kustom)"
            self.openai_model_label.setVisible(False)
            self.api_key_input.setVisible(is_openai)            
            self.api_key_input_label.setVisible(is_openai)            
            self.openai_model_input.setVisible(False)
            self.openai_prompt_label.setVisible(is_openai)
            self.openai_prompt_text.setVisible(is_openai)



    def _toggle_mmr_params_visibility(self, model_name):
        is_mmr = model_name == "MMR"
        self.mmr_diversity_label.setVisible(is_mmr)
        self.mmr_diversity_spin.setVisible(is_mmr)
        # Tambahkan parameter lain untuk model representasi lain di sini jika perlu



    def setup_results_display(self):
        self.results_group = QGroupBox("Hasil Analisis")
        results_layout = QVBoxLayout()
        
        self.topic_table = QTableWidget()
        self.topic_table.setColumnCount(4) # Topic, Name, Count, Representation (Keywords)
        self.topic_table.setHorizontalHeaderLabels(["ID Topik", "Nama Topik (Keywords Utama)", "Jumlah Dokumen", "Representasi Lengkap"])
        self.topic_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.topic_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.topic_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)


        self.doc_table = QTableWidget()
        self.doc_table.setColumnCount(4) # Review, Assigned Topic ID, Topic Name, Probability
        self.doc_table.setHorizontalHeaderLabels(["Teks Review (Potongan)", "ID Topik", "Nama Topik", "Probabilitas"])
        self.doc_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.doc_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.doc_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        
        self.table_tabs = QTabWidget()
        self.table_tabs.addTab(self.topic_table, "Informasi Topik")
        self.table_tabs.addTab(self.doc_table, "Topik per Dokumen")
        
        self.save_results_btn = QPushButton("⬇️ Simpan Hasil Tabel (CSV/Excel)") # Tombol save tabel dipisah
        self.save_results_btn.setEnabled(False)
        self.save_results_btn.clicked.connect(self.save_table_results)

        results_layout.addWidget(self.table_tabs)
        results_layout.addWidget(self.save_results_btn)
        self.results_group.setLayout(results_layout)
        
    def setup_visualization(self):
        self.viz_group = QGroupBox("Visualisasi Topik")
        viz_layout = QVBoxLayout()
        
        self.viz_combo = QComboBox()
        self.viz_combo.addItems([
            "Intertopic Distance Map", # visualize_topics()
            "Topic Hierarchy",         # visualize_hierarchy()
            "Topic Term Importance (Bar Chart)", # visualize_barchart()
            "Topic Similarity Heatmap" # visualize_heatmap()
        ])
        
        if PYQTWEBENGINE_AVAILABLE:
            self.viz_display = QWebEngineView(self)
            # Aktifkan JavaScript jika diperlukan oleh Plotly
            settings = self.viz_display.settings()
            settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.ScrollAnimatorEnabled, True) # smooth scroll
        else:
            self.viz_display = QTextEdit(self)
            self.viz_display.setReadOnly(True)
            self.viz_display.setPlaceholderText("PyQtWebEngine tidak terinstal. Visualisasi interaktif tidak akan tampil.\nSilakan instal: pip install PyQtWebEngine")
            self.viz_display.setStyleSheet("color: red;")

        self.viz_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        viz_control_layout = QHBoxLayout()
        self.viz_btn = QPushButton("📊 Buat Visualisasi")
        self.viz_btn.setEnabled(False)
        self.save_viz_btn = QPushButton("💾 Simpan Visualisasi (HTML)")
        self.save_viz_btn.setEnabled(False)
        
        viz_control_layout.addWidget(self.viz_btn)
        viz_control_layout.addWidget(self.save_viz_btn)
        
        viz_layout.addWidget(QLabel("Tipe Visualisasi:"))
        viz_layout.addWidget(self.viz_combo)
        viz_layout.addLayout(viz_control_layout)
        viz_layout.addWidget(self.viz_display, 1) # Stretch factor for display
        
        self.viz_group.setLayout(viz_layout)
        
        self.viz_btn.clicked.connect(self.generate_visualization)
        self.save_viz_btn.clicked.connect(self.save_visualization_html)

    def start_topic_extraction_thread(self):
        if self.review_data_df is None or self.review_data_df.empty or 'ulasan' not in self.review_data_df.columns:
            QMessageBox.warning(self, "Data Tidak Ada", "Silakan muat data review dengan kolom 'ulasan' terlebih dahulu.")
            return
            
        documents_raw = self.review_data_df['ulasan'].dropna().tolist()
        if not documents_raw:
            QMessageBox.warning(self, "Data Kosong", "Tidak ada teks ulasan yang valid setelah memfilter nilai kosong.")
            return

        self.status_label.setText("Memproses teks...")
        QApplication.processEvents()
        self.documents_processed_for_topic_model = self.preprocess_documents(documents_raw) # <-- DI SINI DIISI

        if not self.documents_processed_for_topic_model or \
           all(not doc.strip() for doc in self.documents_processed_for_topic_model):
            self.status_label.setText("Tidak ada teks valid setelah preprocessing untuk pemodelan topik.")
            QMessageBox.warning(self, "Preprocessing Gagal", "Tidak ada teks yang tersisa setelah tahap preprocessing.")
            return
            
        self.extract_btn.setEnabled(False)
        self.update_representation_btn.setEnabled(True)
        self.status_label.setText("Mengumpulkan parameter untuk ekstraksi topik...")
        
        min_topic_size_val = self.min_topic_slider.value()
        nr_topics_str = self.nr_topics_combo.currentText()
        nr_topics_val = int(nr_topics_str) if nr_topics_str.isdigit() else "auto"
        language_val = self.language_combo.currentText()

        embedding_model_name_val = self.embedding_model_input.text()
        umap_n_neighbors_val = self.umap_neighbors_spin.value()
        umap_n_components_val = self.umap_components_spin.value()
        umap_min_dist_val = self.umap_min_dist_spin.value()
        umap_metric_val = self.umap_metric_combo.currentText()
        
        hdbscan_metric_val = self.hdbscan_metric_combo.currentText()
        hdbscan_selection_val = self.hdbscan_selection_combo.currentText()
        
        vectorizer_ngram_min = self.vectorizer_ngram_min_spin.value()
        vectorizer_ngram_max = self.vectorizer_ngram_max_spin.value()
        vectorizer_ngram_range_val = (vectorizer_ngram_min, vectorizer_ngram_max)

        user_selected_initial_rep_config, user_selected_initial_rep_kwargs = \
            self._get_selected_representation_config()

        self.status_label.setText("Membuat instance TopicExtraction...")
        self.topic_extraction_logic = TopicExtraction(
            embedding_model_name=embedding_model_name_val,
            umap_n_neighbors=umap_n_neighbors_val, umap_n_components=umap_n_components_val,
            umap_min_dist=umap_min_dist_val, umap_metric=umap_metric_val,
            hdbscan_min_cluster_size=min_topic_size_val,
            hdbscan_metric=hdbscan_metric_val, 
            hdbscan_cluster_selection_method=hdbscan_selection_val,
            vectorizer_ngram_range=vectorizer_ngram_range_val,
            min_topic_size=min_topic_size_val,
            nr_topics=nr_topics_val,
            language=language_val,
            initial_user_representation_config=user_selected_initial_rep_config,
            initial_user_representation_kwargs=user_selected_initial_rep_kwargs
        )
        
        self.status_label.setText("Memulai ekstraksi topik (mungkin butuh waktu)...")
        self.worker = TopicExtractionWorker(self.topic_extraction_logic, self.documents_processed_for_topic_model)
        self.worker.started.connect(lambda: self.status_label.setText("Proses BERTopic dimulai..."))
        self.worker.progress.connect(lambda msg: self.status_label.setText(f"Status BERTopic: {msg}"))
        self.worker.finished.connect(self.on_topic_extraction_finished)
        self.worker.start()



    def _get_selected_representation_config(self):
        selected_rep_model_text = self.representation_model_combo.currentText()
        kwargs_rep = {}

        if selected_rep_model_text == "MMR":
            kwargs_rep['diversity'] = self.mmr_diversity_spin.value()
            return "MMR", kwargs_rep
        elif selected_rep_model_text == "KeyBERTInspired":
            return "KeyBERTInspired", kwargs_rep
        elif selected_rep_model_text == "OpenAI (Prompt Kustom)" and OPENAI_REPRESENTATION_AVAILABLE:
            kwargs_rep['openai_model_name'] = self.openai_model_input.text()
            kwargs_rep['openai_prompt'] = self.openai_prompt_text.toPlainText().strip()
            return "OpenAI (Prompt Kustom)", kwargs_rep
        elif selected_rep_model_text == "Default (c-TF-IDF)":
            return "Default (c-TF-IDF)", kwargs_rep
        
        print(f"Peringatan: Teks model representasi tidak dikenal: {selected_rep_model_text}")
        return "Default (c-TF-IDF)", {}


    def on_topic_extraction_finished(self, topics, probs, topic_info_df_augmented, error_message):
        self.extract_btn.setEnabled(True)
        if error_message:
            self.status_label.setText(f"Error ekstraksi topik: {error_message}")
            QMessageBox.critical(self, "Error Ekstraksi Topik", f"Gagal mengekstrak topik:\n{error_message}")
            self.save_model_btn.setEnabled(False)
            self.viz_btn.setEnabled(False)
            self.save_results_btn.setEnabled(False)
            self.update_representation_btn.setEnabled(True)
            return

        if topic_info_df_augmented is None or topic_info_df_augmented.empty:
            self.status_label.setText("Ekstraksi topik selesai, namun tidak ada topik yang ditemukan.")
            QMessageBox.information(self, "Hasil Kosong", "Tidak ada topik yang berhasil diidentifikasi.")
            self.current_topic_info_df = pd.DataFrame()
            self.current_doc_results_df = pd.DataFrame()
            self.save_model_btn.setEnabled(False)
            self.viz_btn.setEnabled(False)
            self.save_results_btn.setEnabled(False)
            self.update_representation_btn.setEnabled(True)
        else:
            self.current_topic_info_df = topic_info_df_augmented
            
            if hasattr(self, 'documents_processed_for_topic_model') and self.documents_processed_for_topic_model:
                 self.current_doc_results_df = self.topic_extraction_logic.get_document_info_df(
                     self.documents_processed_for_topic_model
                 )
            else:
                self.current_doc_results_df = pd.DataFrame()

            self.display_topic_info()
            self.display_doc_info()
            
            non_outlier_topics_count = len(self.current_topic_info_df[self.current_topic_info_df['Topic']!=-1])
            self.status_label.setText(f"Ekstraksi topik selesai! Ditemukan {non_outlier_topics_count} topik (tidak termasuk outlier).")
            self.save_model_btn.setEnabled(True)
            self.viz_btn.setEnabled(True)
            self.save_results_btn.setEnabled(True)
            self.update_representation_btn.setEnabled(True)

    def update_topic_representation_ui(self):
        # if not self.topic_extraction_logic or not self.topic_extraction_logic.topic_model or self.topic_extraction_logic.topics_ is None:
        #     QMessageBox.warning(self, "Model Belum Siap", "Ekstrak topik terlebih dahulu atau muat model yang sudah di-fit.")
        #     return
        api_key = self.api_key_input.text().strip()
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
            self.openai_api_key = api_key # Simpan juga di instance untuk penggunaan langsung
            # self.status_label.setText("OPENAI_API_KEY telah diatur untuk sesi ini.")
            self.status_label.setText( "OPENAI_API_KEY telah diatur sebagai variabel lingkungan untuk sesi aplikasi ini.\n")
            # print(f"DEBUG: OPENAI_API_KEY diatur ke: {api_key[:5]}...{api_key[-5:]}") # Jangan print full key
            if not hasattr(self, 'documents_processed_for_topic_model') or not self.documents_processed_for_topic_model:
                # Cek apakah ada data di self.review_data_df yang bisa diproses
                if self.review_data_df is not None and not self.review_data_df.empty and 'ulasan' in self.review_data_df.columns:
                    msg_box = QMessageBox()
                    msg_box.setIcon(QMessageBox.Icon.Question)
                    msg_box.setText("Data dokumen yang diproses untuk model ini tidak ada.")
                    msg_box.setInformativeText("Apakah Anda ingin menggunakan data ulasan yang saat ini dimuat di aplikasi untuk update representasi?\n"
                                            "Ini akan melalui tahap preprocessing lagi.")
                    msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)
                    ret = msg_box.exec()

                    if ret == QMessageBox.StandardButton.Yes:
                        self.status_label.setText("Memproses ulang dokumen untuk update representasi...")
                        QApplication.processEvents()
                        documents_raw = self.review_data_df['ulasan'].dropna().tolist()
                        if not documents_raw:
                            QMessageBox.warning(self, "Data Kosong", "Data ulasan saat ini kosong.")
                            return
                        self.documents_processed_for_topic_model = self.preprocess_documents(documents_raw)
                        if not self.documents_processed_for_topic_model or \
                        all(not doc.strip() for doc in self.documents_processed_for_topic_model):
                            QMessageBox.warning(self, "Preprocessing Gagal", "Tidak ada teks valid setelah preprocessing.")
                            self.documents_processed_for_topic_model = None # Reset
                            return
                        self.status_label.setText("Dokumen siap untuk update representasi.")
                    else:
                        QMessageBox.information(self, "Update Dibatalkan", "Update representasi memerlukan data dokumen yang telah diproses.")
                        return
                else:
                    QMessageBox.warning(self, "Data Dokumen Hilang",
                                        "Data dokumen yang diproses tidak ditemukan. Silakan muat data ulasan dan ekstrak topik ulang, atau muat data ulasan untuk melanjutkan update.")
                    return

            selected_rep_config, kwargs_rep = self._get_selected_representation_config()

            self.status_label.setText(f"Mengupdate representasi topik dengan {str(selected_rep_config)}...")
            QApplication.processEvents()
            self.update_representation_btn.setEnabled(True)

            try:
                new_augmented_topic_info_df = self.topic_extraction_logic.update_topic_representation(
                    documents=self.documents_processed_for_topic_model,
                    representation_model_config=selected_rep_config,
                    **kwargs_rep
                )

                if new_augmented_topic_info_df is not None:
                    self.current_topic_info_df = new_augmented_topic_info_df
                    self.display_topic_info()
                    
                    if hasattr(self, 'documents_processed_for_topic_model') and self.documents_processed_for_topic_model:
                        self.current_doc_results_df = self.topic_extraction_logic.get_document_info_df(
                            self.documents_processed_for_topic_model
                        )
                    self.display_doc_info()

                    self.status_label.setText(f"Representasi topik berhasil diupdate: {str(selected_rep_config)}.")
                    if self.current_viz_html:
                        self.generate_visualization()
                else:
                    self.status_label.setText(f"Gagal mengupdate representasi: {str(selected_rep_config)}.")
                    QMessageBox.warning(self, "Update Gagal", "Update representasi tidak menghasilkan data topik baru.")
            except Exception as e:
                QMessageBox.critical(self, "Error Update Representasi", f"Gagal mengupdate representasi topik: {str(e)}")
                self.status_label.setText(f"Error update representasi: {str(e)}")
            finally:
                self.update_representation_btn.setEnabled(True)
        else:
            QMessageBox.warning(self, "API Key Kosong", "Silakan masukkan API Key OpenAI terlebih dahulu.")
            # self.status_label.setText("Gagal mengatur API Key: Input kosong.")



    def display_topic_info(self):
        if self.current_topic_info_df.empty:
            self.topic_table.setRowCount(0)
            return

        df_display = self.current_topic_info_df.copy()
        self.topic_table.setRowCount(len(df_display))

        for i, row in df_display.iterrows():
            self.topic_table.setItem(i, 0, QTableWidgetItem(str(row['Topic'])))

            nama_topik_custom = str(row.get('Name', 'N/A'))
            if "_" in nama_topik_custom and nama_topik_custom.split("_", 1)[0] == str(row['Topic']):
                display_name = nama_topik_custom.split("_", 1)[1].replace("_", ", ")
            else:
                display_name = nama_topik_custom
            self.topic_table.setItem(i, 1, QTableWidgetItem(display_name))

            self.topic_table.setItem(i, 2, QTableWidgetItem(str(row.get('Count', 'N/A'))))

            ctfidf_keywords_list = row.get('Representative_Docs', [])
            representation_str = ", ".join(ctfidf_keywords_list) if isinstance(ctfidf_keywords_list, list) else str(ctfidf_keywords_list)
            self.topic_table.setItem(i, 3, QTableWidgetItem(representation_str))

        self.topic_table.resizeColumnsToContents()

    def display_doc_info(self):
        if self.current_doc_results_df.empty or 'Document' not in self.current_doc_results_df.columns:
            self.doc_table.setRowCount(0)
            return
            
        # Gabungkan dengan data review asli untuk mendapatkan teks 'ulasan'
        # Asumsi self.review_data_df memiliki 'ulasan' dan cocok dengan 'Document' di current_doc_results_df setelah preprocessing
        # Ini bisa jadi rumit jika preprocessing mengubah teks secara signifikan.
        # Untuk simple, kita tampilkan 'Document' dari current_doc_results_df
        
        self.doc_table.setRowCount(len(self.current_doc_results_df))
        for i, row in self.current_doc_results_df.iterrows():
            doc_text = str(row.get('Document', 'N/A'))
            self.doc_table.setItem(i, 0, QTableWidgetItem(doc_text[:150] + ("..." if len(doc_text) > 150 else "")))
            self.doc_table.setItem(i, 1, QTableWidgetItem(str(row.get('Topic', 'N/A'))))
            
            topic_name_parts = str(row.get('Name', 'N/A')).split("_", 1)
            display_topic_name = topic_name_parts[1].replace("_", ", ") if len(topic_name_parts) > 1 else str(row.get('Name', 'N/A'))
            self.doc_table.setItem(i, 2, QTableWidgetItem(display_topic_name))
            
            prob_val = row.get('Probability', np.nan)
            self.doc_table.setItem(i, 3, QTableWidgetItem(f"{prob_val:.3f}" if pd.notna(prob_val) else "N/A"))
        self.doc_table.resizeColumnsToContents()

    def generate_visualization(self):
        if not self.topic_extraction_logic or not self.topic_extraction_logic.topic_model:
            QMessageBox.warning(self, "Model Belum Siap", "Silakan ekstrak topik atau muat model terlebih dahulu.")
            return
        if not PYQTWEBENGINE_AVAILABLE:
            QMessageBox.critical(self, "Error Visualisasi", "PyQtWebEngine tidak terinstal. Visualisasi interaktif tidak dapat ditampilkan.")
            return
            
        viz_type = self.viz_combo.currentText()
        self.status_label.setText(f"Membuat visualisasi: {viz_type}...")
        QApplication.processEvents()
        
        fig = None
        try:
            if viz_type == "Intertopic Distance Map":
                fig = self.topic_extraction_logic.visualize_intertopic_map()
            elif viz_type == "Topic Hierarchy":
                fig = self.topic_extraction_logic.visualize_hierarchy()
            elif viz_type == "Topic Term Importance (Bar Chart)":
                fig = self.topic_extraction_logic.visualize_barchart(top_n_topics=10) # Bisa diatur
            elif viz_type == "Topic Similarity Heatmap":
                fig = self.topic_extraction_logic.visualize_heatmap()
            
            if fig:
                if hasattr(fig, 'to_html'):
                    # 1. Dapatkan div HTML untuk plotnya saja, tanpa full HTML dan tanpa skrip Plotly
                    plot_div_html = fig.to_html(full_html=False, include_plotlyjs=False)

                    # 2. Buat string HTML lengkap secara manual, tambahkan tag skrip Plotly.js di <head>
                    self.current_viz_html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta charset="utf-8" />
                        <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script> 
                        <style>
                            body {{ margin: 0; padding: 0; }}
                            /* Pastikan div plot mengambil ruang yang cukup */
                            .plotly-graph-div {{ margin: auto; }} 
                        </style>
                    </head>
                    <body>
                        {plot_div_html}
                    </body>
                    </html>
                    """
                    if PYQTWEBENGINE_AVAILABLE:
                        self.viz_display.setHtml(self.current_viz_html)
                        self.current_viz_html = fig.to_html(full_html=True, include_plotlyjs='cdn') # full_html=True agar berdiri sendiri
                        self.viz_display.setHtml(self.current_viz_html)
                        self.save_viz_btn.setEnabled(True)
                        self.status_label.setText(f"Visualisasi '{viz_type}' berhasil dibuat.")
                    else:
                        raise TypeError("Objek visualisasi yang dikembalikan bukan figure Plotly yang valid.")
            else:
                self.status_label.setText(f"Visualisasi '{viz_type}' tidak menghasilkan output.")
                self.viz_display.setHtml("<p style='text-align:center;color:orange;'>Tidak ada visualisasi untuk ditampilkan.</p>")
                self.save_viz_btn.setEnabled(False)
                
        except Exception as e:
            error_msg = f"Gagal membuat visualisasi '{viz_type}': {str(e)}"
            self.status_label.setText(error_msg)
            QMessageBox.critical(self, "Error Visualisasi", error_msg)
            self.viz_display.setHtml(f"<p style='text-align:center;color:red;'>Error: {e}</p>")
            self.save_viz_btn.setEnabled(False)
    
    def save_table_results(self): # Mengganti nama dari save_results
        if self.current_doc_results_df.empty and self.current_topic_info_df.empty:
            QMessageBox.warning(self, "Tidak Ada Data", "Tidak ada hasil tabel untuk disimpan.")
            return
            
        options = QFileDialog.Option(0) # Default options
        file_name, selected_filter = QFileDialog.getSaveFileName(
            self, "Simpan Hasil Tabel", "",
            "Excel Files (*.xlsx);;CSV Files - Info Topik (*_topic_info.csv);;CSV Files - Topik Dokumen (*_doc_topics.csv)",
            options=options
        )
        
        if file_name:
            try:
                if selected_filter.startswith("Excel"):
                    if not file_name.endswith('.xlsx'): file_name += '.xlsx'
                    with pd.ExcelWriter(file_name) as writer:
                        if not self.current_topic_info_df.empty:
                            self.current_topic_info_df.to_excel(writer, sheet_name='Topic Info', index=False)
                        if not self.current_doc_results_df.empty:
                            # Pilih kolom yang relevan untuk disimpan dari current_doc_results_df
                            cols_to_save = ['Document', 'Topic', 'Name', 'Probability']
                            save_df = self.current_doc_results_df[[col for col in cols_to_save if col in self.current_doc_results_df.columns]]
                            save_df.to_excel(writer, sheet_name='Document Topics', index=False)
                    self.status_label.setText(f"Hasil tabel disimpan ke {file_name}")
                elif "Info Topik" in selected_filter:
                    if not file_name.endswith('_topic_info.csv'): file_name = file_name.replace(".csv","") + '_topic_info.csv'
                    if not self.current_topic_info_df.empty:
                         self.current_topic_info_df.to_csv(file_name, index=False)
                         self.status_label.setText(f"Info topik disimpan ke {file_name}")
                    else: QMessageBox.information(self,"Info", "Tidak ada data info topik untuk disimpan sebagai CSV.")
                elif "Topik Dokumen" in selected_filter:
                    if not file_name.endswith('_doc_topics.csv'): file_name = file_name.replace(".csv","") + '_doc_topics.csv'
                    if not self.current_doc_results_df.empty:
                        cols_to_save = ['Document', 'Topic', 'Name', 'Probability']
                        save_df = self.current_doc_results_df[[col for col in cols_to_save if col in self.current_doc_results_df.columns]]
                        save_df.to_csv(file_name, index=False)
                        self.status_label.setText(f"Topik dokumen disimpan ke {file_name}")
                    else: QMessageBox.information(self,"Info", "Tidak ada data topik dokumen untuk disimpan sebagai CSV.")
                
            except Exception as e:
                QMessageBox.critical(self, "Error Menyimpan", f"Gagal menyimpan: {str(e)}")

    def save_visualization_html(self): # Mengganti nama dari save_visualization
        if not self.current_viz_html:
            QMessageBox.warning(self, "Tidak Ada Visualisasi", "Tidak ada visualisasi untuk disimpan.")
            return
            
        options = QFileDialog.Option(0)
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Simpan Visualisasi HTML", "", "HTML Files (*.html)", options=options
        )
        
        if file_name:
            try:
                if not file_name.lower().endswith('.html'): file_name += '.html'
                with open(file_name, 'w', encoding='utf-8') as f:
                    f.write(self.current_viz_html) # current_viz_html sudah full_html=True
                self.status_label.setText(f"Visualisasi disimpan ke {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error Menyimpan", f"Gagal menyimpan: {str(e)}")

    # Di dalam kelas TopicAnalysisPage:

    def save_bertopic_model(self):
        if not self.topic_extraction_logic or not self.topic_extraction_logic.topic_model:
            QMessageBox.warning(self, "Model Tidak Tersedia", 
                                "Tidak ada model BERTopic yang aktif untuk disimpan. "
                                "Silakan ekstrak topik atau muat model terlebih dahulu.")
            return

        # Dapatkan path file dari pengguna
        options = QFileDialog.Option(0) # Opsi default
        # Sarankan ekstensi .btm (umum untuk model BERTopic) atau .pkl
        # BERTopic sendiri biasanya tidak menambahkan ekstensi secara otomatis saat save,
        # jadi lebih baik pengguna yang menentukan atau kita tambahkan jika tidak ada.
        file_name, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Simpan Model BERTopic",
            f"bertopic_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}", # Nama file default dengan timestamp
            "BERTopic Model Files (*.btm *.pkl);;All Files (*)",
            options=options
        )

        if file_name:
            # Pastikan ekstensi file ada jika pengguna tidak mengetikkannya
            # BERTopic.save() tidak secara otomatis menambahkan ekstensi.
            # Ekstensi .pkl juga umum jika menggunakan pickle_model=True (default)
            # atau .safetensors jika menggunakan serialization='safetensors'.
            # Untuk umum, .btm bisa jadi konvensi atau .pkl jika metode default digunakan.
            # Jika selected_filter adalah "BERTopic Model Files (*.btm *.pkl)"
            # dan file_name tidak memiliki ekstensi, kita bisa tambahkan .pkl sebagai default.
            if not any(file_name.lower().endswith(ext) for ext in ['.btm', '.pkl']):
                if "." not in file_name: # Tidak ada ekstensi sama sekali
                     file_name += ".pkl" # Default ke .pkl jika menggunakan metode simpan standar
                # Jika ada ekstensi lain, biarkan saja atau ganti. Untuk sekarang, biarkan.
            
            self.status_label.setText(f"Menyimpan model BERTopic ke {file_name}...")
            QApplication.processEvents()

            try:
                # Panggil dengan include_embedding_model=False
                self.topic_extraction_logic.save_model(
                    file_name,
                    include_embedding_model=False # <--- PENTING
                ) 
                
                self.status_label.setText(f"Model BERTopic berhasil disimpan ke: {file_name}")
                QMessageBox.information(self, "Simpan Berhasil", 
                                        f"Model BERTopic telah berhasil disimpan ke:\n{file_name}")

            except AttributeError:
                 # Ini terjadi jika self.topic_extraction_logic adalah None
                 QMessageBox.critical(self, "Error", "Objek TopicExtraction tidak diinisialisasi.")
                 self.status_label.setText("Error: Gagal menyimpan model (objek tidak ada).")
            except ValueError as ve:
                 # Ini terjadi jika model di TopicExtraction belum di-fit (dari pengecekan di TopicExtraction.save_model)
                 QMessageBox.warning(self, "Model Belum Siap", str(ve))
                 self.status_label.setText(f"Gagal menyimpan: {str(ve)}")
            except Exception as e:
                QMessageBox.critical(self, "Error Penyimpanan", 
                                     f"Gagal menyimpan model BERTopic:\n{str(e)}")
                self.status_label.setText(f"Error: Gagal menyimpan model ke {file_name}.")

    def load_bertopic_model(self):
        options = QFileDialog.Option(0)
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Muat Model BERTopic", "", "BERTopic Model Files (*.btm *.pkl);;All Files (*)", options=options
        )
        
        if file_name:
            self.status_label.setText(f"Memuat model dari {file_name}...")
            QApplication.processEvents()
            try:
                self.topic_extraction_logic = TopicExtraction.load_model(file_name)
                self.current_topic_info_df = self.topic_extraction_logic.get_topic_info_df()

                # Setelah model dimuat, coba tampilkan info topik dan aktifkan tombol
                if not self.current_topic_info_df.empty:
                    self.display_topic_info()
                    # Untuk doc_info, kita perlu dokumen. Pengguna mungkin perlu menjalankan ulang pada data baru
                    # atau kita bisa mencoba memuat dokumen yang disimpan bersama model (jika ada).
                    # Untuk saat ini, kita kosongkan doc_table.
                    self.doc_table.setRowCount(0)
                    self.current_doc_results_df = pd.DataFrame()

                    self.status_label.setText(f"Model BERTopic berhasil dimuat dari {file_name}. {len(self.current_topic_info_df)-1} topik ditemukan.")
                    self.save_model_btn.setEnabled(True) # Model sudah ada, bisa disimpan ulang
                    self.viz_btn.setEnabled(True)
                    self.save_results_btn.setEnabled(False) # Hasil tabel dokumen perlu di-generate ulang
                else:
                    self.status_label.setText(f"Model BERTopic dimuat, tapi tidak ada info topik. Model mungkin perlu di-fit ulang.")
                    self.save_model_btn.setEnabled(False)
                    self.viz_btn.setEnabled(False)
                    self.save_results_btn.setEnabled(False)
                
                self.display_topic_info()

            except Exception as e:
                QMessageBox.critical(self, "Error Memuat Model", f"Gagal memuat model: {str(e)}")
                self.status_label.setText(f"Gagal memuat model.")
                self.topic_extraction_logic = None
                self.current_topic_info_df = pd.DataFrame()
                self.current_doc_results_df = pd.DataFrame()
                self.display_topic_info()
                self.display_doc_info()
                self.save_model_btn.setEnabled(False)
                self.viz_btn.setEnabled(False)
                self.update_representation_btn.setEnabled(True)
                self.save_results_btn.setEnabled(False)
            finally: # Pastikan tombol update dinonaktifkan jika tidak ada dokumen yang diproses
                if not hasattr(self, 'documents_processed_for_topic_model') or not self.documents_processed_for_topic_model:
                    self.update_representation_btn.setEnabled(True)

    def preprocess_documents(self, documents_list): # Terima list, kembalikan list
        # Fungsi preprocessing ini spesifik untuk Bahasa Indonesia
        # Pastikan library yang dibutuhkan sudah diimpor di awal file atau di sini
        # (re, string, pandas, Sastrawi, nltk)
        processed_docs = []
        
        try:
            if self.replace_slang_cb.isChecked():
                # Load kamus alay hanya jika diperlukan dan belum ada
                if not hasattr(self, 'kamus_alay_df'):
                    try:
                        self.kamus_alay_df = pd.read_csv('https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv')
                        self.kamus_alay_df = self.kamus_alay_df.filter(['slang', 'formal'], axis=1).drop_duplicates(subset=['slang'], keep='first').set_index('slang')
                    except Exception as e_kamus:
                        # print(f"Warning: Gagal memuat kamus alay online: {e_kamus}. Penggantian slang dilewati.")
                        self.replace_slang_cb.setChecked(False) # Matikan jika gagal load

            if self.stemming_cb.isChecked():
                if not hasattr(self, 'stemmer_sastrawi'):
                    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
                    factory = StemmerFactory()
                    self.stemmer_sastrawi = factory.create_stemmer()
            
            if self.stopword_cb.isChecked():
                if not hasattr(self, 'stopwords_indonesian'):
                    
                    import nltk
                    nltk.download("stopwords")
                    from nltk.corpus import stopwords
                    self.stopwords_indonesian = list(stopwords.words('indonesian'))
                    self.stopwords_indonesian.extend(["sih", "lho", "nya", "iya", "tdk", "ga", "gak", "gk", "yg", "jg", "aja", "dll", "dsb", "dst"]) # Tambah custom
                custom_input = self.custom_stopwords_input.text()
                if custom_input:
                    custom_words = [word.strip() for word in custom_input.split(",") if word.strip()]
                    self.stopwords_indonesian.extend(custom_words)
        except ImportError as e_dep:
            QMessageBox.warning(self, "Dependency Error", f"Preprocessing library missing: {e_dep}. Beberapa langkah preprocessing mungkin tidak berjalan.")
            # Matikan checkbox yang relevan jika library-nya tidak ada
            if 'Sastrawi' in str(e_dep): self.stemming_cb.setChecked(False)
            if 'nltk' in str(e_dep): self.stopword_cb.setChecked(False)


        for text_idx, text in enumerate(documents_list):
            QApplication.processEvents() # Agar UI tidak freeze saat loop panjang
            if (text_idx + 1) % 100 == 0: # Update status setiap 100 dokumen
                self.status_label.setText(f"Memproses teks: Dokumen {text_idx + 1}/{len(documents_list)}...")

            if not isinstance(text, str): # Skip jika bukan string
                processed_docs.append("") 
                continue

            original_text = text # Simpan teks asli untuk debug jika perlu

            if self.clean_formatting_cb.isChecked():
                # 1. Hapus emoji
                emoji_pattern = re.compile(
                    "[\U0001F600-\U0001F64F]|"  # emoticons
                    "[\U0001F300-\U0001F5FF]|"  # symbols & pictographs
                    "[\U0001F680-\U0001F6FF]|"  # transport & map symbols
                    "[\U0001F1E0-\U0001F1FF]+", # flags (iOS)
                    flags=re.UNICODE)
                text = emoji_pattern.sub(r"", text)
                # 2. Hapus URL
                text = re.sub(r'http\S+', '', text)
                # 3. Hapus mention dan hashtag
                text = re.sub(r'(@\w+|#\w+)', '', text)
                # 4. Hapus tag HTML
                text = re.sub(r'<.*?>', '', text)
                # 5. Hapus angka (opsional, tergantung kebutuhan analisis topik)
                # text = re.sub(r'\d+', '', text) 
                # 6. Hapus tanda baca dan ganti dengan spasi, lalu bersihkan karakter non-alfabet
                text = ''.join(' ' if c in string.punctuation else c for c in text)
                text = re.sub(r'[^a-zA-Z\s]', '', text) # Hanya pertahankan huruf dan spasi
                # 7. Hapus baris baru
                text = text.replace("\n", " ")
                # 8. Case folding
                text = text.lower()
                # 9. Hapus kata-kata umum yang tidak bermakna (jika ada, ini contoh)
                text = re.sub(r"\b(username|user|url|rt|xf|fx|xe|xa|nya)\b", "", text, flags=re.IGNORECASE)
                # 10. Hapus karakter berulang (misalnya, "baguuusss" menjadi "bagus")
                text = re.sub(r'(\w)\1{2,}', r'\1', text)
                # 11. Hapus huruf tunggal yang berdiri sendiri (setelah tokenisasi implisit oleh split)
                text = re.sub(r"\b[a-zA-Z]\b", "", text)
                # 12. Normalisasi spasi berlebih
                text = ' '.join(text.split())

            if self.replace_slang_cb.isChecked() and hasattr(self, 'kamus_alay_df'):
                words = text.split()
                # Performa lebih baik jika kamus alay adalah dict
                if not hasattr(self, 'kamus_alay_dict'):
                    self.kamus_alay_dict = self.kamus_alay_df['formal'].to_dict()

                replaced_words = [self.kamus_alay_dict.get(word, word) for word in words]
                text = ' '.join(replaced_words)

            if self.stemming_cb.isChecked() and hasattr(self, 'stemmer_sastrawi'):
                text = self.stemmer_sastrawi.stem(text)
            
            if self.stopword_cb.isChecked() and hasattr(self, 'stopwords_indonesian'):
                words = text.split()
                text = ' '.join([word for word in words if word not in self.stopwords_indonesian and len(word) > 1]) # Hapus juga kata pendek

            if not text.strip() and original_text.strip(): # Jika teks menjadi kosong setelah proses
                # print(f"Warning: Teks '{original_text[:50]}...' menjadi kosong setelah preprocessing.")
                pass # Biarkan kosong jika memang hasilnya kosong, BERTopic akan handle

            processed_docs.append(text)
        
        self.status_label.setText("Preprocessing teks selesai.")
        return processed_docs

class LabelingPage(QWidget):
    def __init__(self):
        super().__init__()
        self.df = None
        self.current_index = 0
        self.running = False
        self.initUI()
        
    def initUI(self):
        main_layout = QHBoxLayout()
        
        # Left panel (controls)
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        left_panel.setFixedWidth(300)
        self.left_layout = QVBoxLayout()
        
        # CSV Section
        csv_label = QLabel("📁 CSV Processing")
        csv_label.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        
        self.load_csv_btn = QPushButton("Load CSV")
        self.start_btn = QPushButton("Start Labeling")
        self.pause_btn = QPushButton("Pause")
        self.download_btn = QPushButton("Download Labeled CSV")
        
        # Progress display
        self.progress_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        
        self.left_layout.addWidget(csv_label)
        self.left_layout.addWidget(self.load_csv_btn)
        self.left_layout.addWidget(self.start_btn)
        self.left_layout.addWidget(self.pause_btn)
        self.left_layout.addWidget(self.download_btn)
        self.left_layout.addWidget(self.progress_bar)
        self.left_layout.addWidget(self.progress_label)
        self.left_layout.addStretch()

        # --- OpenAI API Key Section (BARU) ---
        openai_group = QGroupBox("🔑 OpenAI API Key")
        openai_layout = QFormLayout(openai_group) # Gunakan QFormLayout di dalam group

        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Masukkan OpenAI API Key Anda di sini")
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password) # Sembunyikan input
        openai_layout.addRow("API Key:", self.api_key_input)

        self.set_env_var_btn = QPushButton("Set API Key sebagai ENV Variable")
        self.set_env_var_btn.setToolTip(
            "Mengatur OPENAI_API_KEY untuk sesi ini. "
            "Anda mungkin perlu mengatur ini secara permanen di sistem Anda."
        )
        self.set_env_var_btn.clicked.connect(self.set_openai_env_variable)
        openai_layout.addRow(self.set_env_var_btn)
        
        self.left_layout.addWidget(openai_group)

        initial_csd = {
            "Aroma": ["harum", "wangi", "bau menyengat", "aroma segar", "tidak wangi"],
            "Bonus": ["bonus", "hadiah", "free gift", "tidak dapat bonus"],
            "Harga": ["murah", "mahal", "worth it", "tidak sesuai harga"],
            "Kemasan": ["botol", "dus", "plastik", "bocor", "kemasan rapi"],
            "Kesesuaian Deskripsi": ["sesuai deskripsi", "beda dengan foto", "tidak sesuai info"],
            "Kondisi Paket": ["penyok", "hancur", "aman", "utuh", "terbuka"],
            "Kualitas": ["bagus", "berkualitas", "murahan", "cacat", "tidak tahan lama"],
            "Manfaat": ["efektif", "menyegarkan", "relaksasi", "tidak terasa efeknya"],
            "Pelayanan": ["ramah", "fast response", "slow response", "tidak ditanggapi"],
            "Pengiriman": ["cepat", "lama", "delay", "ekspedisi"],
            "Aplikasi Humidifier/Diffuser": ["cocok di diffuser", "menyumbat", "larut", "bisa dicampur"]
        }
        

        # === CSD List Section ===
        csd_label = QLabel("Customer Satisfaction Dimensions (CSD):")
        self.csd_table = QTableWidget()
        self.csd_table.setColumnCount(2)
        self.csd_table.setHorizontalHeaderLabels(["CSD", "Kata Kunci (dipisah koma)"])
        self.csd_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.csd_table.setMinimumHeight(300)
        self.set_csd_table_from_dict(initial_csd)

        # Tombol tambah/hapus baris
        csd_button_layout = QHBoxLayout()
        self.add_row_btn = QPushButton("➕ Tambah Baris")
        self.remove_row_btn = QPushButton("➖ Hapus Baris")
        csd_button_layout.addWidget(self.add_row_btn)
        csd_button_layout.addWidget(self.remove_row_btn)

        # Tambahkan ke layout
        self.left_layout.addWidget(csd_label)
        self.left_layout.addWidget(self.csd_table)
        self.left_layout.addLayout(csd_button_layout)

        self.add_row_btn.clicked.connect(lambda: self.csd_table.insertRow(self.csd_table.rowCount()))
        self.remove_row_btn.clicked.connect(self.remove_csd_row)

        
        left_panel.setLayout(self.left_layout)
        
        # Right panel (results)
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self.right_layout = QVBoxLayout()
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5) # Sesuai dengan update_table Anda
        self.results_table.setHorizontalHeaderLabels(["Review", "Rating", "Dimensi (JSON)", "One Hot Encoded", "Status"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive) # Kolom Review
        self.results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive) # Kolom Dimensi
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers) # Pastikan read-only
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)


        self.right_layout.addWidget(self.results_table)
        right_panel.setLayout(self.right_layout)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1) # Beri right_panel stretch factor
        self.setLayout(main_layout)
        
        # Connect signals (tetap sama)
        self.load_csv_btn.clicked.connect(self.load_csv)
        self.start_btn.clicked.connect(self.start_labeling)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.download_btn.clicked.connect(self.download_csv)
        self.add_row_btn.clicked.connect(self.add_csd_row) # Pastikan ini lambda atau metode
        self.remove_row_btn.clicked.connect(self.remove_csd_row)
        
        self.pause_btn.setEnabled(False)
        self.download_btn.setEnabled(False)



    def set_openai_env_variable(self):
        api_key = self.api_key_input.text().strip()
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
            self.openai_api_key = api_key # Simpan juga di instance untuk penggunaan langsung
            self.progress_label.setText("OPENAI_API_KEY telah diatur untuk sesi ini.")
            QMessageBox.information(self, "API Key Diatur", 
                                    "OPENAI_API_KEY telah diatur sebagai variabel lingkungan untuk sesi aplikasi ini.\n"
                                    "Pustaka OpenAI akan mencoba mengambilnya secara otomatis.")
            # print(f"DEBUG: OPENAI_API_KEY diatur ke: {api_key[:5]}...{api_key[-5:]}") # Jangan print full key
        else:
            QMessageBox.warning(self, "API Key Kosong", "Silakan masukkan API Key OpenAI terlebih dahulu.")
            self.progress_label.setText("Gagal mengatur API Key: Input kosong.")



    def add_csd_row(self):
        row = self.csd_table.rowCount()
        self.csd_table.insertRow(row)
        self.csd_table.setItem(row, 0, QTableWidgetItem(""))
        self.csd_table.setItem(row, 1, QTableWidgetItem(""))

    def remove_csd_row(self):
        selected = self.csd_table.currentRow()
        if selected >= 0:
            self.csd_table.removeRow(selected)

    def set_csd_table_from_dict(self, csd_dict):
        self.csd_table.setRowCount(0)
        for key, keywords in csd_dict.items():
            row = self.csd_table.rowCount()
            self.csd_table.insertRow(row)
            self.csd_table.setItem(row, 0, QTableWidgetItem(key))
            self.csd_table.setItem(row, 1, QTableWidgetItem(", ".join(keywords)))


    def get_csd_dict(self):
        csd_dict = {}
        for row in range(self.csd_table.rowCount()):
            key_item = self.csd_table.item(row, 0)
            val_item = self.csd_table.item(row, 1)
            if key_item and val_item:
                key = key_item.text().strip()
                val = [v.strip() for v in val_item.text().split(",") if v.strip()]
                if key:
                    csd_dict[key] = val
        return csd_dict

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv)")
        
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.current_index = 0
                self.update_table()
                self.progress_label.setText(f"Loaded {len(self.df)} rows")
                self.progress_bar.setMaximum(len(self.df))
                self.download_btn.setEnabled(True)
            except Exception as e:
                self.progress_label.setText(f"Error loading CSV: {str(e)}")
                
    def update_table(self):
        if self.df is not None:
            self.results_table.setRowCount(len(self.df))
            for i, row in self.df.iterrows():
                self.results_table.setItem(i, 0, QTableWidgetItem(str(row.get('ulasan', ''))))
                self.results_table.setItem(i, 1, QTableWidgetItem(str(row.get('rating_ulasan') or row.get('rating') or '')))
                self.results_table.setItem(i, 2, QTableWidgetItem(str(row.get('dimensi', ''))))
                status = "Processed" if pd.notna(row.get('dimensi')) else "Pending"
                self.results_table.setItem(i, 3, QTableWidgetItem(str(row.get('dimensi_expanded', ''))))

                self.results_table.setItem(i, 4, QTableWidgetItem(status))
                
    def start_labeling(self):
        try:
            self.running = True
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)

            system_prompt = f"""ANALISIS ULASAN PRODUK 

                            Tugas:
                            Lakukan analisis terhadap ulasan produk ini dengan langkah-langkah berikut:
                            1. Deteksi CSD (Customer Satisfaction Dimension) jika disebutkan secara eksplisit terkait pembelian saat ini (bukan pembelian sebelumnya atau harapan).

                            2. Untuk setiap CSD:
                            - Tentukan sentimen: positif, netral, atau negatif.

                            3. Jika tidak ada CSD:
                            - Isi keterangan dengan salah satu dari:
                            a. harapan: hanya berisi harapan/ekspektasi
                            b. kepuasan_umum: puas tanpa menyebut alasan
                            c. emosi_umum: ekspresi perasaan umum tanpa konteks
                            d. sarkasme: gaya bahasa ambigu/sindirian
                            e. lainnya: jika tidak sesuai kategori di atas

                            4. Jika sentimen mengandung sindiran, sarkasme, atau ambiguitas, beri label keterangan: "sarkasme" walaupun CSD terdeteksi.

                            5. Output wajib dalam format JSON dengan struktur:
                            - csd: list objek {{name, sentiment}}
                            - keterangan: null atau string kategori (jika tidak ada CSD)
                            - jangan tulis sama sekali ```json... ``` didepan outputnya

                            CSD yang tersedia:
                            CSD: Kata Kunci Contoh
                            {self.get_csd_dict()}


                            Format output dalam JSON:
                                Contoh Input 1:
                                "Barang datang cepat, kemasan rapi. Aroma lavender sangat autentik."

                                Contoh Output 1:
                                {{
                                "csd": [
                                    {{"name": "Pengiriman", "sentiment": "positif"}},
                                    {{"name": "Kemasan", "sentiment": "positif"}},
                                    {{"name": "Aroma", "sentiment": "positif"}}
                                ],
                                "keterangan": null
                                }}

                                Contoh Input 2:
                                "Semoga next order dapat bonus."

                                Contoh Output 2:
                                {{
                                "csd": [],
                                "keterangan": "harapan"
                                }}

                                """  # Your system prompt here
            


            dimensi_urut = list(self.get_csd_dict().keys())
            for idx in range(self.current_index, len(self.df)):
                if not self.running:
                    print("BREAK")
                    break
                # print("MASUK FOR")    
                row = self.df.iloc[idx]
                if pd.notna(row.get('ulasan')):
                    try:
                        text = row['ulasan']
                        bintang = row.get('rating_ulasan') or row.get('rating') or ''
                        client = OpenAI()
                        # OpenAI API call
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": f"Ulasan yang akan diproses: {bintang}, Ulasan yang akan diproses: {text}"},
                            ],
                            max_tokens=1000,
                            temperature=0.4
                        )
                        sentiment_map = {"positif": 1, "negatif": -1, "netral": 0}
 
                        result = (completion.choices[0].message.content.strip())
                        result = json.loads(result)

                        dimensi_expanded = [0] * len(dimensi_urut)

                        for item in result.get("csd", []):
                            name = item.get("name")
                            sentiment = item.get("sentiment", "netral")
                            if name in dimensi_urut:
                                idx_csd = dimensi_urut.index(name)
                                dimensi_expanded[idx_csd] = sentiment_map.get(sentiment, 0)
                        
                        expanded = []
                        for val in dimensi_expanded:
                            if val == 1:
                                expanded.extend([1, 0])
                            elif val == -1:
                                expanded.extend([0, 1])
                            else:
                                expanded.extend([0, 0])




                        self.df.at[idx, 'dimensi'] = str(result)
                        self.df.at[idx, 'one_hot'] = str(expanded)

                        
                        # Update UI
                        self.progress_bar.setValue(idx+1)
                        self.progress_label.setText(f"Processing row {idx+1}/{len(self.df)}")
                        self.update_table()
                        QApplication.processEvents()
                        
                        # Save every 50 rows
                        if (idx+1) % 50 == 0:
                            self.save_progress()
                        # print(system_prompt)
                        # print(text)
                        self.current_index = self.current_index+ 1
                        
                    except Exception as e:
                        self.df.at[idx, 'dimensi'] = f"Error: {str(e)}"
                        print(e)
                        continue
                else:
                    print("KOSONG")
                        
            self.running = False
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.save_progress()
        except Exception as e:
            print(e)
        
    def toggle_pause(self):
        self.running = False
        self.pause_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        
    def save_progress(self):
        try:
            self.df.to_csv("auto_save_labeled.csv", index=False)
        except Exception as e:
            self.progress_label.setText(f"Error saving progress: {str(e)}")
            
    def download_csv(self):
        if self.df is not None:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Save Labeled CSV", "", "CSV Files (*.csv)")
            
            if file_name:
                try:
                    self.df.to_csv(file_name, index=False)
                    self.progress_label.setText(f"Saved to {file_name}")
                except Exception as e:
                    self.progress_label.setText(f"Error saving: {str(e)}")

# --- Training Worker Thread ---
class TrainingWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    # Modified results signal to include all data snapshots
    results = pyqtSignal(str, object, dict, 
                         pd.DataFrame, pd.Series, # X_test, y_test
                         pd.DataFrame, pd.Series, # X_train, y_train
                         pd.DataFrame, pd.Series) # X_val, y_val
    error = pyqtSignal(str)

    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, model_name, param_grid_str, cv_folds):
        super().__init__()
        self.X_train_orig = X_train.copy() if X_train is not None else pd.DataFrame()
        self.y_train_orig = y_train.copy() if y_train is not None else pd.Series(dtype='float64')
        self.X_val_orig = X_val.copy() if X_val is not None else pd.DataFrame()
        self.y_val_orig = y_val.copy() if y_val is not None else pd.Series(dtype='float64')
        self.X_test_orig = X_test.copy() if X_test is not None else pd.DataFrame()
        self.y_test_orig = y_test.copy() if y_test is not None else pd.Series(dtype='float64')
        
        self.model_name = model_name
        self.param_grid_str = param_grid_str
        self.cv_folds = cv_folds

    def _parse_param_grid(self):
        parsed_grid = {}
        try:
            for param, value_str in self.param_grid_str.items():
                if not value_str.strip():
                    self.status.emit(f"Skipping empty hyperparameter: {param}")
                    continue
                stripped_value_str = value_str.strip()
                if stripped_value_str.startswith('[') and stripped_value_str.endswith(']'):
                    try:
                        parsed_list = ast.literal_eval(stripped_value_str)
                        if isinstance(parsed_list, list):
                            parsed_grid[param] = parsed_list
                            continue
                    except Exception:
                        pass 
                values = [v.strip() for v in stripped_value_str.split(',')]
                casted_values = []
                for v_str in values:
                    if not v_str: continue
                    if v_str.lower() == 'none': casted_values.append(None)
                    elif v_str.lower() == 'true': casted_values.append(True)
                    elif v_str.lower() == 'false': casted_values.append(False)
                    else:
                        try:
                            if '.' in v_str or 'e' in v_str.lower(): casted_values.append(float(v_str))
                            else: casted_values.append(int(v_str))
                        except ValueError: casted_values.append(v_str)
                if not casted_values: continue
                parsed_grid[param] = casted_values
            if not parsed_grid:
                self.error.emit("Error: Hyperparameter grid is empty after parsing.")
                return None
            return parsed_grid
        except Exception as e:
            self.error.emit(f"Critical error parsing hyperparameter grid: {str(e)}")
            return None
        
    def run(self):
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        try:
            import xgboost as xgb
            XGB_AVAILABLE = True
        except ImportError:
            XGB_AVAILABLE = False
        try:
            import lightgbm as lgb
            LGBM_AVAILABLE = True
        except ImportError:
            LGBM_AVAILABLE = False

        if self.model_name in ["XGBoost", "LightGBM"] and not (XGB_AVAILABLE if self.model_name == "XGBoost" else LGBM_AVAILABLE):
            self.error.emit(f"{self.model_name} library not installed.")
            return

        self.status.emit("🚀 Starting training process...")
        self.progress.emit(5)
        param_grid = self._parse_param_grid()
        if param_grid is None:
            self.progress.emit(0)
            return
        self.status.emit(f"Parsed Grid: {param_grid}")
        self.progress.emit(10)

        model = None
        if self.model_name == "Random Forest": model = RandomForestRegressor(random_state=42)
        elif self.model_name == "Gradient Boosting": model = GradientBoostingRegressor(random_state=42)
        elif self.model_name == "XGBoost" and XGB_AVAILABLE: model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
        elif self.model_name == "LightGBM" and LGBM_AVAILABLE: model = lgb.LGBMRegressor(random_state=42, verbosity=-1)
        else:
            self.error.emit(f"Model {self.model_name} error or library missing.")
            self.progress.emit(0)
            return

        self.status.emit(f"GridSearchCV for {self.model_name} (CV: {self.cv_folds})...")
        self.progress.emit(20)
        
        if self.X_train_orig.empty or self.y_train_orig.empty:
            self.error.emit("Training data (X_train_orig or y_train_orig) is missing or empty.")
            self.progress.emit(0)
            return

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                                   scoring='neg_mean_absolute_error', cv=self.cv_folds, verbose=0, n_jobs=-1)
        
        try:
            grid_search.fit(self.X_train_orig, self.y_train_orig)
        except Exception as e_fit:
            self.error.emit(f"Error during GridSearchCV.fit: {str(e_fit)}")
            self.progress.emit(0)
            return
            
        self.progress.emit(70)
        best_model = grid_search.best_estimator_
        
        actual_best_params = grid_search.best_params_
        best_params_str = f"Best Parameters: {actual_best_params}"

        self.status.emit(best_params_str)
        self.progress.emit(80)

        metrics_data = {"params": actual_best_params, "sets":{}}
        output_text_summary = f"\n✅ Training {self.model_name} DONE.\n{best_params_str}\n"

        eval_sets = {"Train": (self.X_train_orig, self.y_train_orig)}
        if not self.X_val_orig.empty and not self.y_val_orig.empty:
            eval_sets["Validation"] = (self.X_val_orig, self.y_val_orig)
        if not self.X_test_orig.empty and not self.y_test_orig.empty:
            eval_sets["Test"] = (self.X_test_orig, self.y_test_orig)
        else:
            self.status.emit("Warning: Test data snapshots are missing for final evaluation in worker.")

        for name, (X_set, y_set) in eval_sets.items():
            if not X_set.empty and not y_set.empty:
                preds = best_model.predict(X_set)
                mae = mean_absolute_error(y_set, preds)
                rmse = np.sqrt(mean_squared_error(y_set, preds))
                r2 = r2_score(y_set, preds)
                metrics_data["sets"][name] = {"MAE": mae, "RMSE": rmse, "R²": r2}
                output_text_summary += f"\n--- {name} Set ---\nMAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}\n"
        
        self.progress.emit(100)
        self.status.emit("🏁 Training finished!")
        self.results.emit(output_text_summary, best_model, metrics_data, 
                          self.X_test_orig, self.y_test_orig,
                          self.X_train_orig, self.y_train_orig,
                          self.X_val_orig, self.y_val_orig)

# --- Training Page ---
class TrainingPage(QWidget):
    def __init__(self):
        super().__init__()
        self.df_raw = None 
        self.df_for_processing = None 
        self.df_processed = None 
        self.trained_model = None 
        self.X_test_final_active = None 
        self.y_test_final_active = None
        self.X_train_active_recalled = None
        self.y_train_active_recalled = None
        self.X_val_active_recalled = None
        self.y_val_active_recalled = None
        self.training_runs_history = []

        self.initUI()
        self._update_button_states()
        self._setup_evaluation_table()

    def initUI(self):
        main_layout = QVBoxLayout()
        top_group_layout = QHBoxLayout()

        data_group = QGroupBox("1. Data Input & Seleksi")
        data_layout_form = QFormLayout() 

        self.load_btn = QPushButton("📁 Load Labeled CSV")
        self.load_btn.clicked.connect(self.load_csv)
        data_layout_form.addRow(self.load_btn)

        self.target_column_combo = QComboBox()
        data_layout_form.addRow("🎯 Target Column (untuk y):", self.target_column_combo)

        self.use_segment_filter_checkbox = QCheckBox("Gunakan Filter Segmen Data?")
        self.use_segment_filter_checkbox.setChecked(False)
        self.use_segment_filter_checkbox.toggled.connect(self._on_segment_filter_toggled)

        segment_layout = QHBoxLayout()
        self.segment_label_value_combo = QComboBox()
        self.segment_label_value_combo.setEnabled(False)

        segment_layout.addWidget(self.use_segment_filter_checkbox)
        segment_layout.addWidget(self.segment_label_value_combo)
        data_layout_form.addRow(segment_layout)
        
        self.y_transform_checkbox = QCheckBox("Kurangi 1 dari Target (misal, rating 1-5 jadi 0-4)")
        self.y_transform_checkbox.setChecked(True)
        data_layout_form.addRow(self.y_transform_checkbox)
        data_group.setLayout(data_layout_form)
        
        config_group = QGroupBox("2. Model & Training Setup")
        config_form_layout = QFormLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"])
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        config_form_layout.addRow("🤖 Choose Model:", self.model_combo)
        self.cv_spinbox = QSpinBox()
        self.cv_spinbox.setRange(2, 20); self.cv_spinbox.setValue(3)
        config_form_layout.addRow("🔄 CV Folds:", self.cv_spinbox)
        split_layout = QHBoxLayout()
        self.test_size_spinbox = QDoubleSpinBox(); self.test_size_spinbox.setRange(0.1, 0.5); self.test_size_spinbox.setValue(0.2); self.test_size_spinbox.setSingleStep(0.05)
        split_layout.addWidget(QLabel("Test Size:")); split_layout.addWidget(self.test_size_spinbox)
        self.val_size_spinbox = QDoubleSpinBox(); self.val_size_spinbox.setRange(0.0, 0.9); self.val_size_spinbox.setValue(0.25); self.val_size_spinbox.setSingleStep(0.05)
        self.val_size_spinbox.setToolTip("Proporsi dari (Train+Val) untuk validasi. Set 0 jika tidak ada validasi (hanya CV).")
        split_layout.addWidget(QLabel("Val Size (dari Train+Val):")); split_layout.addWidget(self.val_size_spinbox)
        config_form_layout.addRow("📊 Data Split:", split_layout)
        config_group.setLayout(config_form_layout)
        top_group_layout.addWidget(data_group,1) 
        top_group_layout.addWidget(config_group,1)
        main_layout.addLayout(top_group_layout)

        combined_layout = QHBoxLayout()
        self.hyperparam_group = QGroupBox("3. Hyperparameters")
        hyperparam_layout = QVBoxLayout(); self.hyperparam_stack = QStackedWidget()
        self._init_hyperparameter_widgets(); hyperparam_layout.addWidget(self.hyperparam_stack)
        self.hyperparam_group.setLayout(hyperparam_layout)
        train_output_group = QGroupBox("4. Training Process")
        train_output_layout = QVBoxLayout()
        control_layout = QHBoxLayout()
        self.train_btn = QPushButton("🚂 Train Model"); self.train_btn.setFont(QFont('Arial', 12, QFont.Weight.Bold)); self.train_btn.clicked.connect(self.start_training_thread)
        control_layout.addWidget(self.train_btn)
        self.save_model_btn = QPushButton("💾 Save Active Model"); self.save_model_btn.clicked.connect(self.save_model)
        control_layout.addWidget(self.save_model_btn)
        train_output_layout.addLayout(control_layout)
        self.progress_bar = QProgressBar(); self.progress_bar.setTextVisible(True); self.progress_bar.setFormat("%p% - %v/%m")
        train_output_layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Status: Idle"); self.status_label.setStyleSheet("font-style: italic; color: gray;")
        train_output_layout.addWidget(self.status_label)
        self.output_box = QTextEdit(); self.output_box.setReadOnly(True); self.output_box.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        train_output_layout.addWidget(self.output_box)
        self.clear_output_btn = QPushButton("🧹 Clear Log"); self.clear_output_btn.clicked.connect(lambda: self.output_box.clear())
        train_output_layout.addWidget(self.clear_output_btn)
        train_output_group.setLayout(train_output_layout)
        combined_layout.addWidget(self.hyperparam_group, 1); combined_layout.addWidget(train_output_group, 2)
        main_layout.addLayout(combined_layout)

        history_group = QGroupBox("5. Model Training History & Recall")
        history_layout = QVBoxLayout()
        self.evaluation_table = QTableWidget(); self.evaluation_table.setAlternatingRowColors(True); self.evaluation_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers); self.evaluation_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows); self.evaluation_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        history_layout.addWidget(QLabel("Evaluation Summary:")); history_layout.addWidget(self.evaluation_table)
        recall_layout = QHBoxLayout(); self.recall_model_combo = QComboBox(); recall_layout.addWidget(QLabel("Recall Model:")); recall_layout.addWidget(self.recall_model_combo, 1)
        self.recall_load_btn = QPushButton("🔄 Recall Selected"); self.recall_load_btn.clicked.connect(self._recall_selected_model)
        recall_layout.addWidget(self.recall_load_btn); history_layout.addLayout(recall_layout)
        history_group.setLayout(history_layout); main_layout.addWidget(history_group)
        
        self.on_model_changed(self.model_combo.currentText())
        self.setLayout(main_layout)

    def _setup_evaluation_table(self):
        headers = ["Run ID", "Model", "Timestamp", "Params", 
                   "Train MAE", "Train RMSE", "Train R²",
                   "Val MAE", "Val RMSE", "Val R²",
                   "Test MAE", "Test RMSE", "Test R²"]
        self.evaluation_table.setColumnCount(len(headers))
        self.evaluation_table.setHorizontalHeaderLabels(headers)
        self.evaluation_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.evaluation_table.horizontalHeader().setStretchLastSection(True)

    def _refresh_training_history_ui(self):
        self.evaluation_table.setRowCount(0) 
        for i, run_data in enumerate(self.training_runs_history):
            self.evaluation_table.insertRow(i)
            self.evaluation_table.setItem(i, 0, QTableWidgetItem(run_data["run_id"]))
            self.evaluation_table.setItem(i, 1, QTableWidgetItem(run_data["model_name_display"]))
            self.evaluation_table.setItem(i, 2, QTableWidgetItem(run_data["timestamp"].strftime("%Y-%m-%d %H:%M:%S")))
            self.evaluation_table.setItem(i, 3, QTableWidgetItem(str(run_data["best_params"])))
            
            metrics_map = {
                "Train": (4, 5, 6), "Validation": (7, 8, 9), "Test": (10, 11, 12)
            }
            for setName, cols in metrics_map.items():
                if setName in run_data["metrics"]["sets"]:
                    m = run_data["metrics"]["sets"][setName]
                    self.evaluation_table.setItem(i, cols[0], QTableWidgetItem(f"{m.get('MAE', float('nan')):.4f}"))
                    self.evaluation_table.setItem(i, cols[1], QTableWidgetItem(f"{m.get('RMSE', float('nan')):.4f}"))
                    self.evaluation_table.setItem(i, cols[2], QTableWidgetItem(f"{m.get('R²', float('nan')):.4f}"))
                else: 
                    for col_idx in cols:
                         self.evaluation_table.setItem(i, col_idx, QTableWidgetItem("N/A"))
        self.evaluation_table.resizeColumnsToContents()

        current_recall_id = self.recall_model_combo.currentData()
        self.recall_model_combo.clear()
        for run_data in self.training_runs_history:
            self.recall_model_combo.addItem(f"{run_data['run_id']} ({run_data['model_name_display']})", userData=run_data["run_id"])
        
        idx = self.recall_model_combo.findData(current_recall_id)
        if idx != -1:
            self.recall_model_combo.setCurrentIndex(idx)
        
        self.recall_load_btn.setEnabled(len(self.training_runs_history) > 0)

    def on_training_complete(self, output_text_summary, best_model_obj, metrics_data_dict, 
                             X_test_run, y_test_run, 
                             X_train_run, y_train_run, 
                             X_val_run, y_val_run):
        self._log(output_text_summary)
        
        self.trained_model = best_model_obj
        self.X_test_final_active = X_test_run.copy() if X_test_run is not None else None
        self.y_test_final_active = y_test_run.copy() if y_test_run is not None else None

        run_timestamp = datetime.now()
        model_display_name = self.model_combo.currentText()
        run_id = f"{model_display_name.replace(' ','')}_{run_timestamp.strftime('%H%M%S')}"

        current_run_info = {
            "run_id": run_id,
            "model_name_display": model_display_name,
            "timestamp": run_timestamp,
            "best_params": metrics_data_dict.get("params", {}),
            "metrics": metrics_data_dict,
            "model_object": best_model_obj,
            "X_test_snapshot": X_test_run.copy() if X_test_run is not None else None,
            "y_test_snapshot": y_test_run.copy() if y_test_run is not None else None,
            "X_train_snapshot": X_train_run.copy() if X_train_run is not None else None, 
            "y_train_snapshot": y_train_run.copy() if y_train_run is not None else None, 
            "X_val_snapshot": X_val_run.copy() if X_val_run is not None else None,       
            "y_val_snapshot": y_val_run.copy() if y_val_run is not None else None        
        }
        self.training_runs_history.append(current_run_info)
        self._refresh_training_history_ui()

        self.status_label.setText("Status: Training successful!")
        self.progress_bar.setFormat("Completed - %p%")
        QMessageBox.information(self, "Training Complete", "Model training finished successfully!")
        self._update_button_states()

    def _recall_selected_model(self):
        selected_run_id = self.recall_model_combo.currentData()
        if not selected_run_id:
            QMessageBox.information(self, "No Model Selected", "Please select a model from the recall dropdown.")
            return

        recalled_run = next((run for run in self.training_runs_history if run["run_id"] == selected_run_id), None)

        if recalled_run:
            self.trained_model = recalled_run["model_object"]
            self.X_test_final_active = recalled_run.get("X_test_snapshot")
            self.y_test_final_active = recalled_run.get("y_test_snapshot")
            
            self.X_train_active_recalled = recalled_run.get("X_train_snapshot")
            self.y_train_active_recalled = recalled_run.get("y_train_snapshot") 
            self.X_val_active_recalled = recalled_run.get("X_val_snapshot")
            self.y_val_active_recalled = recalled_run.get("y_val_snapshot") 
            
            self._log(f"🔄 Model '{recalled_run['run_id']}' recalled and set as active.")
            self._log(f"   Type: {recalled_run['model_name_display']}")
            self._log(f"   Best Params: {recalled_run['best_params']}")
            self._log(f"   Associated X_train shape: {self.X_train_active_recalled.shape if self.X_train_active_recalled is not None else 'N/A'}")
            self._log(f"   Associated X_val shape: {self.X_val_active_recalled.shape if self.X_val_active_recalled is not None else 'N/A'}")
            self._log(f"   Associated X_test shape: {self.X_test_final_active.shape if self.X_test_final_active is not None else 'N/A'}")
            
            self.model_combo.setCurrentText(recalled_run["model_name_display"])
            
            QMessageBox.information(self, "Model Recalled", 
                                    f"Model '{recalled_run['run_id']}' is now the active trained model.\n"
                                    f"Associated X_train, X_val, and X_test snapshots are also loaded from this run.")
            self._update_button_states()
        else:
            QMessageBox.warning(self, "Recall Error", "Could not find the selected model in history. Try refreshing.")

    def _init_hyperparameter_widgets(self): 
        self.hyperparam_widgets = {} 
        rf_widget = QWidget(); rf_layout = QFormLayout(rf_widget)
        self.hyperparam_widgets["Random Forest"] = { "n_estimators": QLineEdit("50,100"), "max_depth": QLineEdit("None,10"), "min_samples_split": QLineEdit("2,5"), "min_samples_leaf": QLineEdit("1,2"), "bootstrap": QComboBox()}
        self.hyperparam_widgets["Random Forest"]["bootstrap"].addItems(["True", "False"])
        for name, widget in self.hyperparam_widgets["Random Forest"].items(): rf_layout.addRow(f"{name}:", widget)
        self.hyperparam_stack.addWidget(rf_widget)

        gb_widget = QWidget(); gb_layout = QFormLayout(gb_widget)
        self.hyperparam_widgets["Gradient Boosting"] = { "n_estimators": QLineEdit("50,100"), "learning_rate": QLineEdit("0.05,0.1"), "max_depth": QLineEdit("3,5"), "subsample": QLineEdit("0.8,1.0")}
        for name, widget in self.hyperparam_widgets["Gradient Boosting"].items(): gb_layout.addRow(f"{name}:", widget)
        self.hyperparam_stack.addWidget(gb_widget)

        xgb_widget = QWidget(); xgb_layout = QFormLayout(xgb_widget)
        self.hyperparam_widgets["XGBoost"] = { "n_estimators": QLineEdit("50,100"), "learning_rate": QLineEdit("0.05,0.1"), "max_depth": QLineEdit("3,5"), "subsample": QLineEdit("0.8"), "colsample_bytree": QLineEdit("0.8"), "gamma": QLineEdit("0,0.1")}
        for name, widget in self.hyperparam_widgets["XGBoost"].items(): xgb_layout.addRow(f"{name}:", widget)
        self.hyperparam_stack.addWidget(xgb_widget)

        lgbm_widget = QWidget(); lgbm_layout = QFormLayout(lgbm_widget)
        self.hyperparam_widgets["LightGBM"] = { "n_estimators": QLineEdit("50,100"), "learning_rate": QLineEdit("0.05,0.1"), "max_depth": QLineEdit("-1,5"), "num_leaves": QLineEdit("31,40"), "subsample": QLineEdit("0.8"), "colsample_bytree": QLineEdit("0.8")}
        for name, widget in self.hyperparam_widgets["LightGBM"].items(): lgbm_layout.addRow(f"{name}:", widget)
        self.hyperparam_stack.addWidget(lgbm_widget)

    def on_model_changed(self, model_name):
        if model_name == "Random Forest": self.hyperparam_stack.setCurrentIndex(0)
        elif model_name == "Gradient Boosting": self.hyperparam_stack.setCurrentIndex(1)
        elif model_name == "XGBoost": self.hyperparam_stack.setCurrentIndex(2)
        elif model_name == "LightGBM": self.hyperparam_stack.setCurrentIndex(3)
        if hasattr(self, 'output_box') and self.output_box: # Check if output_box exists
             if self.output_box.document().toPlainText().count('\n') > 500: # Simple line count limit
                self.output_box.clear()
                self._log("Log cleared due to length.")
             self._log(f"Selected model: {model_name}. Configure its hyperparameters above.")
    
    def _update_button_states(self):
        can_train = self.df_raw is not None and not self.df_raw.empty
        self.train_btn.setEnabled(can_train)
        self.save_model_btn.setEnabled(self.trained_model is not None)
        self.recall_load_btn.setEnabled(len(self.training_runs_history) > 0)

    def _log(self, message):
        if not hasattr(self, 'output_box') or not self.output_box: return 
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.output_box.append(f"[{timestamp}] {message}")
        QApplication.processEvents() 

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Labeled CSV File", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.df_raw = pd.read_csv(file_path)
                if 'filter_segment_label' in self.df_raw.columns:
                    self.df_raw['filter_segment_label'] = self.df_raw['filter_segment_label'].astype(str)
                self._log(f"📄 Raw data loaded: {self.df_raw.shape[0]} rows, {self.df_raw.shape[1]} columns.")
                
                self.target_column_combo.clear()
                if self.df_raw is not None:
                    self.target_column_combo.addItems(self.df_raw.columns)
                    if "bintang" in self.df_raw.columns:
                        self.target_column_combo.setCurrentText("bintang")
                
                self._populate_segment_value_combo()
                
                self.df_processed = None 
                self.df_for_processing = None
                self.trained_model = None 
                self.X_test_final_active = None
                self.y_test_final_active = None
                self.X_train_active_recalled = None 
                self.y_train_active_recalled = None
                self.X_val_active_recalled = None
                self.y_val_active_recalled = None
                # self.training_runs_history.clear() # Decide if history should clear on new CSV
                # self._refresh_training_history_ui()

            except Exception as e:
                QMessageBox.critical(self, "Error Loading CSV", str(e))
                self.df_raw = None
            self._update_button_states()

    def _on_segment_filter_toggled(self, checked):
        self.segment_label_value_combo.setEnabled(checked)
        if checked:
            self._populate_segment_value_combo()
        else:
            self.segment_label_value_combo.clear()
            self.segment_label_value_combo.addItem("Filter Segmen Tidak Aktif")

    def _populate_segment_value_combo(self):
        self.segment_label_value_combo.clear()
        if not self.use_segment_filter_checkbox.isChecked() or \
           self.df_raw is None or self.df_raw.empty:
            self.segment_label_value_combo.addItem("N/A (Filter Tidak Aktif)")
            self.segment_label_value_combo.setEnabled(False)
            return

        segment_column_name = 'filter_segment_label'
        
        if segment_column_name not in self.df_raw.columns:
            self.segment_label_value_combo.addItem(f"Kolom '{segment_column_name}' Tidak Ada")
            self.segment_label_value_combo.setEnabled(False)
            self.use_segment_filter_checkbox.setChecked(False) 
            QMessageBox.warning(self, "Kolom Tidak Ditemukan", 
                                f"Kolom '{segment_column_name}' tidak ditemukan. Filter segmen dinonaktifkan.")
            return

        try:
            unique_values = self.df_raw[segment_column_name].dropna().unique().tolist()
            self.segment_label_value_combo.addItem("Semua Nilai dari Segmen") 
            self.segment_label_value_combo.addItems([str(val) for val in unique_values])
            self.segment_label_value_combo.setEnabled(True)
        except Exception as e:
            self._log(f"Error mendapatkan nilai unik untuk '{segment_column_name}': {e}")
            self.segment_label_value_combo.addItem("Error Memuat Nilai")
            self.segment_label_value_combo.setEnabled(False)

    def _on_segment_value_changed(self, selected_value_text): 
        if self.df_raw is not None and not self.df_raw.empty:
            self._log(f"Pilihan nilai segmen diubah ke '{selected_value_text}'.")
        # else: # This log might be too noisy if df_raw is not yet loaded
            # self._log(f"Pilihan nilai segmen diubah ke '{selected_value_text}', tapi df_raw kosong.")
    
    def _on_segment_column_changed(self, selected_column_name): 
        pass 

    def preprocess_data_for_training(self):
        if self.df_for_processing is None or self.df_for_processing.empty:
            self._log("❌ Tidak ada data untuk dipreprocessing (df_for_processing kosong).")
            self.df_processed = pd.DataFrame()
            return

        self._log("⚙️ Memulai preprocessing data untuk training...")
        df_to_process = self.df_for_processing.copy()

        if "dimensi_expanded" in df_to_process.columns:
            try:
                # Ensure 'dimensi_expanded' is string before ast.literal_eval, handle potential errors
                def safe_literal_eval(val):
                    try:
                        return ast.literal_eval(str(val))
                    except (ValueError, SyntaxError):
                        return None # Or some other placeholder like []
                df_to_process["dimensi_expanded"] = df_to_process["dimensi_expanded"].apply(safe_literal_eval)
                df_to_process.dropna(subset=["dimensi_expanded"], inplace=True) # Remove rows where eval failed

                original_len = len(df_to_process)
                
                if df_to_process['dimensi_expanded'].empty:
                    self._log("⚠️ Kolom 'dimensi_expanded' tidak memiliki nilai list valid setelah evaluasi.")
                    self.df_processed = pd.DataFrame() 
                    return

                # Determine expected_dim_len from the first valid list
                first_valid_list = next((item for item in df_to_process['dimensi_expanded'] if isinstance(item, list)), None)
                if first_valid_list is None:
                    self._log("⚠️ Tidak ada list valid di 'dimensi_expanded' untuk menentukan panjang dimensi.")
                    self.df_processed = pd.DataFrame()
                    return
                expected_dim_len = len(first_valid_list)

                # Filter rows where 'dimensi_expanded' is a list of the expected length
                df_to_process = df_to_process[df_to_process["dimensi_expanded"].apply(lambda x: isinstance(x, list) and len(x) == expected_dim_len)].reset_index(drop=True)
                self._log(f"Memfilter 'dimensi_expanded' (panjang {expected_dim_len}): Menyimpan {len(df_to_process)} dari {original_len} baris.")

                if not df_to_process.empty:
                    dimensi_df_cols = [f"csd{i//2+1}_{i%2}" for i in range(expected_dim_len)]
                    dimensi_df = pd.DataFrame(df_to_process["dimensi_expanded"].tolist(), columns=dimensi_df_cols, index=df_to_process.index) 
                
                    target_col_name = self.target_column_combo.currentText()
                    
                    if target_col_name in df_to_process:
                        self.df_processed = df_to_process[[target_col_name]].copy()
                    else: 
                        self._log(f"❌ Target column '{target_col_name}' not in df_to_process during concat.")
                        self.df_processed = pd.DataFrame()
                        return

                    self.df_processed = pd.concat([self.df_processed, dimensi_df], axis=1)
                    self.df_processed = self.df_processed.loc[:,~self.df_processed.columns.duplicated()]
                    self._log(f"Data selesai dipreprocessing (dengan dimensi): {self.df_processed.shape[0]} baris, {self.df_processed.shape[1]} kolom.")
                else:
                    self.df_processed = pd.DataFrame()
                    self._log("⚠️ 'dimensi_expanded' menghasilkan DataFrame kosong setelah filter panjang.")
            except Exception as e:
                self._log(f"❌ Error pada preprocessing 'dimensi_expanded': {str(e)}")
                self.df_processed = df_to_process.drop(columns=['dimensi_expanded'], errors='ignore') 
        else:
            self.df_processed = df_to_process.copy() 
            self._log("Kolom 'dimensi_expanded' tidak ditemukan. Menggunakan data apa adanya (setelah filter segmen).")
        
        if self.df_processed.empty:
            self._log("❌ Preprocessing menghasilkan DataFrame kosong.")
        else:
            target_col = self.target_column_combo.currentText()
            if target_col not in self.df_processed.columns:
                if target_col in self.df_for_processing.columns: 
                    self.df_processed[target_col] = self.df_for_processing.loc[self.df_processed.index, target_col] 
                    self._log(f"Menambahkan kembali kolom target '{target_col}' ke df_processed.")
                else:
                    self._log(f"❌ PERINGATAN: Kolom target '{target_col}' tidak ditemukan di df_processed.")
                    QMessageBox.critical(self, "Error Preprocessing", f"Kolom target '{target_col}' hilang.")
                    self.df_processed = pd.DataFrame()
                    return
            self._log(f"✅ Preprocessing data untuk training selesai. Ukuran df_processed: {self.df_processed.shape}")
        self._update_button_states()

    def start_training_thread(self):
        if self.df_raw is None or self.df_raw.empty:
            QMessageBox.warning(self, "Data Belum Dimuat", "Silakan muat file CSV terlebih dahulu."); return
        
        self.df_for_processing = self.df_raw.copy()

        if self.use_segment_filter_checkbox.isChecked():
            selected_segment_value = self.segment_label_value_combo.currentText()
            segment_column_name = 'filter_segment_label'

            if segment_column_name in self.df_for_processing.columns and \
               selected_segment_value not in ["N/A (Filter Tidak Aktif)", f"Kolom '{segment_column_name}' Tidak Ada", "Error Memuat Nilai"]:
                if selected_segment_value != "Semua Nilai dari Segmen":
                    try:
                        self.df_for_processing = self.df_for_processing[
                            self.df_for_processing[segment_column_name].astype(str) == str(selected_segment_value)
                        ]
                        self._log(f"Filter segmen: '{segment_column_name}' == '{selected_segment_value}'. Baris: {len(self.df_for_processing)}")
                    except Exception as e_filter:
                        self._log(f"Error filter segmen: {e_filter}. Menggunakan semua data mentah.")
                        self.df_for_processing = self.df_raw.copy() 
                else: 
                    self.df_for_processing = self.df_for_processing[self.df_for_processing[segment_column_name].notna()]
                    self._log(f"Filter segmen: semua nilai non-null di '{segment_column_name}'. Baris: {len(self.df_for_processing)}")
            else:
                self._log("Filter segmen aktif tapi tidak valid. Menggunakan semua data mentah.")
        else:
            self._log("Filter segmen tidak aktif. Menggunakan semua data mentah.")
        
        if self.df_for_processing.empty:
            QMessageBox.warning(self, "Data Kosong", "Tidak ada data setelah filter segmen. Training dibatalkan.")
            return
        
        self.preprocess_data_for_training()

        if self.df_processed is None or self.df_processed.empty:
            QMessageBox.warning(self, "Preprocessing Gagal", "Data kosong setelah preprocessing. Training dibatalkan.")
            return

        target_col = self.target_column_combo.currentText()
        if not target_col or target_col not in self.df_processed.columns:
            QMessageBox.warning(self, "Target Error", f"Kolom target '{target_col}' tidak ditemukan di data yang diproses."); return
        
        try:
            X = self.df_processed.drop(columns=[target_col])
            y = self.df_processed[target_col].copy()
        except KeyError: 
            QMessageBox.critical(self, "Error", f"Gagal memisahkan X/y dari data yang diproses dengan target '{target_col}'."); return

        if self.y_transform_checkbox.isChecked(): y = y - 1
            
        from sklearn.model_selection import train_test_split 
        test_prop = self.test_size_spinbox.value()
        val_prop_relative_to_train_val = self.val_size_spinbox.value() 

        X_train_val, X_test_snap, y_train_val, y_test_snap = train_test_split(X, y, test_size=test_prop, random_state=42, stratify=y if y.nunique() > 1 else None)
        
        X_train_snap, X_val_snap, y_train_snap, y_val_snap = pd.DataFrame(), pd.Series(dtype='float64'), pd.DataFrame(), pd.Series(dtype='float64')
        if val_prop_relative_to_train_val > 0.001 and len(X_train_val) > 1 : 
            try:
                X_train_snap, X_val_snap, y_train_snap, y_val_snap = train_test_split(
                    X_train_val, y_train_val, test_size=val_prop_relative_to_train_val, random_state=42, stratify=y_train_val if y_train_val.nunique() > 1 else None
                )
                self._log(f"Data split: Train ({X_train_snap.shape[0]}), Val ({X_val_snap.shape[0] if X_val_snap is not None and not X_val_snap.empty else 0}), Test ({X_test_snap.shape[0]})")
            except ValueError as e_split: 
                self._log(f"Warning: Gagal membuat validation split ({e_split}). Menggunakan Train/Test saja.")
                X_train_snap, y_train_snap = X_train_val.copy(), y_train_val.copy() 
                X_val_snap, y_val_snap = pd.DataFrame(), pd.Series(dtype='float64') 
        else:
            X_train_snap, y_train_snap = X_train_val.copy(), y_train_val.copy()
            X_val_snap, y_val_snap = pd.DataFrame(), pd.Series(dtype='float64') 
            self._log(f"Data split: Train ({X_train_snap.shape[0]}), Test ({X_test_snap.shape[0]}). Tidak ada validation set.")
        
        if X_train_snap is None or X_train_snap.empty:
            QMessageBox.critical(self, "Data Split Error", "Data training kosong setelah split."); return

        model_name_selected = self.model_combo.currentText()
        cv_folds_selected = self.cv_spinbox.value()
        current_hyperparams_str_map = {
            name: widget.text() if isinstance(widget, QLineEdit) else widget.currentText()
            for name, widget in self.hyperparam_widgets.get(model_name_selected, {}).items()
        }
        
        self.train_btn.setEnabled(False); self.save_model_btn.setEnabled(False)
        self.progress_bar.setValue(0); self.status_label.setText("Status: Menyiapkan training...")

        self.worker = TrainingWorker(X_train_snap, y_train_snap, X_val_snap, y_val_snap, X_test_snap, y_test_snap, 
                                     model_name_selected, current_hyperparams_str_map, cv_folds_selected)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(lambda msg: self.status_label.setText(f"Status: {msg}"))
        self.worker.results.connect(self.on_training_complete)
        self.worker.error.connect(self.on_training_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_training_error(self, error_msg):
        self._log(f"❌ Training Error: {error_msg}")
        QMessageBox.critical(self, "Training Error", error_msg)
        self.progress_bar.setValue(0); self.progress_bar.setFormat("Error")
        self.status_label.setText("Status: Error.")
        self.train_btn.setEnabled(True) 

    def on_worker_finished(self):
        self._log("Worker thread finished.")
        self.train_btn.setEnabled(self.df_processed is not None and not self.df_processed.empty) 
        self._update_button_states()

    def save_model(self):
        if self.trained_model is None:
            QMessageBox.warning(self, "No Active Model", "No model is active (train or recall one first)."); return
        
        active_model_name_suggestion = "trained_model"
        current_recalled_id = self.recall_model_combo.currentData()
        active_run = None
        if current_recalled_id:
            active_run = next((run for run in self.training_runs_history if run["run_id"] == current_recalled_id and run["model_object"] is self.trained_model), None)
        
        if active_run: 
            active_model_name_suggestion = active_run["run_id"]
        else: 
            active_model_name_suggestion = self.model_combo.currentText().replace(' ', '_')

        default_filename = f"{active_model_name_suggestion}_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Active Model", default_filename, "Pickle Files (*.pkl);;Joblib Files (*.sav)")
        
        if file_path:
            try:
                joblib.dump(self.trained_model, file_path)
                self._log(f"✅ Active model saved to: {file_path}")
                QMessageBox.information(self, "Model Saved", f"Model saved to:\n{file_path}")
            except Exception as e:
                self._log(f"❌ Error saving model: {str(e)}"); QMessageBox.critical(self, "Save Error", str(e))

# --- Rename Features Dialog ---
class RenameFeaturesDialog(QDialog):
    def __init__(self, feature_names, current_renames, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rename Feature Dimensions")
        self.feature_names = feature_names
        self.current_renames = current_renames
        self.renamed_features = current_renames.copy() # Work on a copy
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.entries = {}
        form_layout = QFormLayout()
        for name in self.feature_names:
            edit = QLineEdit(self.current_renames.get(name, '')) # Use current or empty
            form_layout.addRow(f"'{name}':", edit)
            self.entries[name] = edit
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content_widget = QWidget()
        content_widget.setLayout(form_layout)
        scroll_area.setWidget(content_widget)
        scroll_area.setMinimumHeight(300)
        scroll_area.setMinimumWidth(400)

        layout.addWidget(scroll_area)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept_changes)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        self.setLayout(layout)

    def accept_changes(self):
        for original_name, edit_widget in self.entries.items():
            new_name = edit_widget.text().strip()
            if new_name and new_name != original_name: # If new name is provided and different
                self.renamed_features[original_name] = new_name
            elif not new_name and original_name in self.renamed_features: # If new name is empty, remove existing rename
                del self.renamed_features[original_name]
            # If new_name is same as original or same as current_rename, no change needed to self.renamed_features
        self.accept()

    def get_renames(self):
        return self.renamed_features

class ShapWorker(QThread): # Tetap sama
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str, int)
    def __init__(self, model, X_input, explainer_type, background_size=50):
        super().__init__()
        self.model = model
        self.X_input = X_input.copy()
        self.explainer_type = explainer_type
        self.background_size = background_size
    def run(self):
        try:
            self.progress.emit("Initializing explainer...", 10)
            QThread.msleep(100)
            if self.explainer_type == "TreeExplainer":
                explainer = shap.TreeExplainer(self.model)
            elif self.explainer_type == "KernelExplainer":
                num_samples = min(self.background_size, len(self.X_input))
                if num_samples < 1:
                    self.error.emit("Input data for background too small")
                    return
                background = shap.sample(self.X_input, num_samples, random_state=42)
                if hasattr(self.model, 'predict_proba'):
                    predict_fn = self.model.predict_proba
                elif hasattr(self.model, 'predict'):
                    predict_fn = self.model.predict
                else:
                    self.error.emit("Model needs 'predict' or 'predict_proba'")
                    return
                explainer = shap.KernelExplainer(predict_fn, background)
            elif self.explainer_type == "LinearExplainer":
                explainer = shap.LinearExplainer(self.model, self.X_input)
            else:
                self.error.emit(f"Unknown explainer: {self.explainer_type}")
                return
            
            self.progress.emit("Calculating SHAP values for combined data...", 50)
            QThread.msleep(100)
            explanation_object = explainer(self.X_input)
            self.progress.emit("Finalizing...", 90)
            QThread.msleep(100)
            self.finished.emit(explanation_object)
        except Exception as e:
            self.error.emit(f"SHAP worker error: {e}")

class ExplainabilityPage(QWidget):
    def __init__(self, training_page_ref=None):
        super().__init__()
        self.model = None
        self.X_train = None # Will hold recalled X_train from training session
        self.X_test = None  # Will hold recalled X_test from training session
        self.X_val = None   # Will hold recalled X_val from training session
        self.data_for_shap = None # This will be the combined dataset for SHAP
        self.explanation_obj = None
        self.training_page = training_page_ref # Reference to TrainingPage instance
        self.current_html_content = ""
        self.feature_renames = {} 

        self.summary_table_widget = None
        self.tab_widget = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("SHAP Explanation Studio")
        main_layout = QVBoxLayout(self)

        top_row_outer_layout = QHBoxLayout()
        
        load_file_group = QGroupBox("Load from File")
        load_file_layout = QVBoxLayout()
        self.load_model_btn = QPushButton("📁 Load Model File (.pkl/.sav)")
        self.load_model_btn.clicked.connect(self.load_model_from_file)
        
        self.load_train_data_btn = QPushButton("📄 Load Training Data (X_train)") # For manual override
        self.load_train_data_btn.clicked.connect(self.load_train_data_from_file)
        
        self.load_test_data_btn = QPushButton("📄 Load Test Data (X_test)") # For manual override
        self.load_test_data_btn.clicked.connect(self.load_test_data_from_file)
        
        self.load_val_data_btn = QPushButton("📄 Load Validation Data (X_val)") # For manual override
        self.load_val_data_btn.clicked.connect(self.load_val_data_from_file)
        
        self.load_shap_btn = QPushButton("🧮 Load SHAP File (.shap)")
        self.load_shap_btn.clicked.connect(self.load_shap_from_file)
        
        load_file_layout.addWidget(self.load_model_btn)
        load_file_layout.addWidget(self.load_train_data_btn)
        load_file_layout.addWidget(self.load_test_data_btn)
        load_file_layout.addWidget(self.load_val_data_btn)
        load_file_layout.addWidget(self.load_shap_btn)
        load_file_group.setLayout(load_file_layout)
        top_row_outer_layout.addWidget(load_file_group, 1)

        trained_model_group = QGroupBox("Use Model from Training Session")
        trained_model_layout = QVBoxLayout()
        self.refresh_trained_models_btn = QPushButton("🔄 Refresh Trained Models List")
        self.refresh_trained_models_btn.clicked.connect(self.refresh_trained_models_list)
        trained_model_layout.addWidget(self.refresh_trained_models_btn)
        recall_combo_layout = QHBoxLayout()
        recall_combo_layout.addWidget(QLabel("Select Trained Model:"))
        self.trained_model_combo = QComboBox()
        recall_combo_layout.addWidget(self.trained_model_combo, 1)
        trained_model_layout.addLayout(recall_combo_layout)
        self.use_trained_model_btn = QPushButton("LOAD Selected Trained Model & Data")
        self.use_trained_model_btn.clicked.connect(self.use_selected_trained_model)
        trained_model_layout.addWidget(self.use_trained_model_btn)
        trained_model_group.setLayout(trained_model_layout)
        top_row_outer_layout.addWidget(trained_model_group, 1)

        config_group = QGroupBox("Explanation Settings")
        config_grid_layout = QFormLayout()
        self.explainer_combo = QComboBox()
        self.explainer_combo.addItems(["TreeExplainer", "KernelExplainer", "LinearExplainer"])
        self.explainer_combo.currentTextChanged.connect(self._update_ui_for_explainer)
        config_grid_layout.addRow("Explainer Type:", self.explainer_combo)
        self.kernel_bg_size_spinbox = QSpinBox()
        self.kernel_bg_size_spinbox.setRange(10, 1000); self.kernel_bg_size_spinbox.setValue(50)
        self.kernel_bg_size_label = QLabel("Kernel BG Samples:")
        config_grid_layout.addRow(self.kernel_bg_size_label, self.kernel_bg_size_spinbox)
        self.visual_combo = QComboBox()
        self.visual_combo.addItems([
            "Force Plot (Interactive HTML)", "Summary Plot (as Image)",
            "Bar Plot (Summary, as Image)", "Waterfall Plot (Single Instance, as Image)",
            "Dependence Plot (as Image)", "Decision Plot (All Instances, as Image)"
        ])
        self.visual_combo.currentTextChanged.connect(self._update_ui_for_visualization)
        config_grid_layout.addRow("Visualization Type:", self.visual_combo)
        self.feature_combo = QComboBox()
        self.feature_combo_label = QLabel("Feature (for Dependence):")
        config_grid_layout.addRow(self.feature_combo_label, self.feature_combo)
        self.instance_spin = QSpinBox(); self.instance_spin.setMinimum(0)
        self.instance_spin.valueChanged.connect(self.regenerate_plot_if_needed)
        self.instance_spin_label = QLabel("Instance Index:")
        config_grid_layout.addRow(self.instance_spin_label, self.instance_spin)
        self.max_display_spinbox = QSpinBox()
        self.max_display_spinbox.setRange(1,100); self.max_display_spinbox.setValue(15)
        self.max_display_label = QLabel("Max Display Features:")
        config_grid_layout.addRow(self.max_display_label, self.max_display_spinbox)
        config_group.setLayout(config_grid_layout)
        top_row_outer_layout.addWidget(config_group, 1)
        main_layout.addLayout(top_row_outer_layout)

        rename_layout = QHBoxLayout()
        self.rename_features_btn = QPushButton("✏️ Edit Feature Dimension Names")
        self.rename_features_btn.clicked.connect(self.open_rename_features_dialog)
        rename_layout.addStretch()
        rename_layout.addWidget(self.rename_features_btn)
        rename_layout.addStretch()
        main_layout.addLayout(rename_layout)

        self.tab_widget = QTabWidget()
        viz_tab_widget = QWidget()
        viz_layout = QVBoxLayout(viz_tab_widget); viz_layout.setContentsMargins(0,0,0,0)
        if PYQTWEBENGINE_AVAILABLE:
            self.html_viewer = QWebEngineView()
            settings = self.html_viewer.settings()
            if settings: # Check if settings is not None
                settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
                settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)
        else:
            self.html_viewer = QLabel("PyQtWebEngine not installed. HTML plots unavailable.")
            self.html_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.html_viewer.setStyleSheet("color: red; font-weight: bold; border: 1px dashed red; padding: 20px;")
        self.html_viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll_viz = QScrollArea(); scroll_viz.setWidgetResizable(True); scroll_viz.setWidget(self.html_viewer)
        viz_layout.addWidget(scroll_viz)
        self.tab_widget.addTab(viz_tab_widget, "📊 Visualization")

        summary_tab_widget = QWidget()
        summary_layout = QVBoxLayout(summary_tab_widget)
        self.summary_table_widget = QTableWidget()
        self.summary_table_widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.summary_table_widget.setAlternatingRowColors(True)
        summary_layout.addWidget(self.summary_table_widget)
        self.tab_widget.addTab(summary_tab_widget, "📈 SHAP Summary (Pos/Neg)")

        kano_tab_widget = QWidget()
        kano_layout = QVBoxLayout(kano_tab_widget); kano_layout.setContentsMargins(0,0,0,0)
        if PYQTWEBENGINE_AVAILABLE:
            self.kano_plot_viewer = QWebEngineView()
            settings_kano = self.kano_plot_viewer.settings()
            if settings_kano:
                settings_kano.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        else:
            self.kano_plot_viewer = QLabel("PyQtWebEngine not installed. KANO plot cannot be shown here (fallback).")
            self.kano_plot_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll_kano = QScrollArea(); scroll_kano.setWidgetResizable(True); scroll_kano.setWidget(self.kano_plot_viewer)
        kano_layout.addWidget(scroll_kano)
        self.tab_widget.addTab(kano_tab_widget, "🗺️ KANO Classification")
        
        main_layout.addWidget(self.tab_widget, 1)

        control_btn_group = QGroupBox("Actions")
        btn_layout = QHBoxLayout()
        self.explain_btn = QPushButton("⚡ Generate Explanation")
        self.explain_btn.clicked.connect(self.start_explanation_thread)
        self.save_shap_btn = QPushButton("💾 Save SHAP Explanation")
        self.save_shap_btn.clicked.connect(self.save_shap_explanation)
        self.export_btn = QPushButton("🖼️ Export Current View")
        self.export_btn.clicked.connect(self.export_content)
        btn_layout.addStretch()
        btn_layout.addWidget(self.explain_btn); btn_layout.addWidget(self.save_shap_btn); btn_layout.addWidget(self.export_btn)
        btn_layout.addStretch()
        control_btn_group.setLayout(btn_layout)
        main_layout.addWidget(control_btn_group)

        self.setLayout(main_layout)
        self.resize(1200, 850)
        self._update_ui_for_explainer(self.explainer_combo.currentText())
        self._update_ui_for_visualization(self.visual_combo.currentText())
        self.refresh_trained_models_list()
        self.update_shap_summary_table()
        self.generate_kano_plot() 

    def load_train_data_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Training Data (X_train)", "", "CSV (*.csv);;Excel (*.xlsx *.xls)")
        if file_path:
            try:
                self.X_train = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
                QMessageBox.information(self, "Success", "X_train data loaded manually!")
                self.explanation_obj = None 
                self.data_for_shap = None 
                self.update_shap_summary_table()
                self.generate_kano_plot()
                self._reset_visualization_and_controls()
                if PYQTWEBENGINE_AVAILABLE and isinstance(self.html_viewer, QWebEngineView): 
                    self.html_viewer.setHtml("<p>Manual X_train loaded. Combine with other data if needed and generate explanation.</p>")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load X_train data: {e}")
                self.X_train = None

    def load_test_data_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Test Data (X_test)", "", "CSV (*.csv);;Excel (*.xlsx *.xls)")
        if file_path:
            try:
                self.X_test = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
                QMessageBox.information(self, "Success", "X_test data loaded manually!")
                self.explanation_obj = None
                self.data_for_shap = None
                self.update_shap_summary_table()
                self.generate_kano_plot()
                self._reset_visualization_and_controls()
                if PYQTWEBENGINE_AVAILABLE and isinstance(self.html_viewer, QWebEngineView):
                    self.html_viewer.setHtml("<p>Manual X_test loaded. Combine with other data if needed and generate explanation.</p>")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load X_test data: {e}")
                self.X_test = None
    
    def load_val_data_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Validation Data (X_val)", "", "CSV (*.csv);;Excel (*.xlsx *.xls)")
        if file_path:
            try:
                self.X_val = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
                QMessageBox.information(self, "Success", "X_val data loaded manually!")
                self.explanation_obj = None
                self.data_for_shap = None
                self.update_shap_summary_table()
                self.generate_kano_plot()
                self._reset_visualization_and_controls()
                if PYQTWEBENGINE_AVAILABLE and isinstance(self.html_viewer, QWebEngineView):
                    self.html_viewer.setHtml("<p>Manual X_val loaded. Combine with other data if needed and generate explanation.</p>")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load X_val data: {e}")
                self.X_val = None
    
    def _reset_visualization_and_controls(self):
        """ Helper to reset UI elements when data changes significantly. """
        self.explanation_obj = None
        self.data_for_shap = None
        self.update_shap_summary_table()
        self.generate_kano_plot() # Will show placeholder
        if PYQTWEBENGINE_AVAILABLE and isinstance(self.html_viewer, QWebEngineView):
            self.html_viewer.setHtml("<p>Data loaded/changed. Please generate explanation.</p>")
        else:
            self.html_viewer.setText("Data loaded/changed. Please generate explanation.")
        
        self.instance_spin.setMaximum(0) # Reset instance spinner
        self.feature_combo.clear()       # Clear feature combo
        self.feature_combo.setEnabled(False)
        self._update_ui_for_visualization(self.visual_combo.currentText()) # Refresh UI based on current viz type


    def use_selected_trained_model(self):
        if not self.training_page or not hasattr(self.training_page, 'training_runs_history'):
            QMessageBox.warning(self, "Error", "Training page reference not available.")
            return

        selected_run_id = self.trained_model_combo.currentData()
        if not selected_run_id:
            QMessageBox.information(self, "No Model Selected", "Please select a model from the list.")
            return

        recalled_run = next((r for r in self.training_page.training_runs_history if r["run_id"] == selected_run_id), None)
        
        if recalled_run:
            self.model = recalled_run.get("model_object")
            
            # Load all data snapshots from the recalled run
            self.X_train = recalled_run.get("X_train_snapshot")
            self.X_val = recalled_run.get("X_val_snapshot")
            self.X_test = recalled_run.get("X_test_snapshot")
            
            # Ensure they are DataFrames, even if None was stored (convert None to empty DataFrame)
            self.X_train = self.X_train.copy() if isinstance(self.X_train, pd.DataFrame) else pd.DataFrame()
            self.X_val = self.X_val.copy() if isinstance(self.X_val, pd.DataFrame) else pd.DataFrame()
            self.X_test = self.X_test.copy() if isinstance(self.X_test, pd.DataFrame) else pd.DataFrame()

            log_message = f"Model '{selected_run_id}' loaded.\n"
            log_message += f"  X_train snapshot loaded (Shape: {self.X_train.shape if not self.X_train.empty else 'N/A'}).\n"
            log_message += f"  X_val snapshot loaded (Shape: {self.X_val.shape if not self.X_val.empty else 'N/A'}).\n"
            log_message += f"  X_test snapshot loaded (Shape: {self.X_test.shape if not self.X_test.empty else 'N/A'})."
            
            QMessageBox.information(self, "Model and Data Loaded", log_message)
            
            self._reset_visualization_and_controls()
            if PYQTWEBENGINE_AVAILABLE and isinstance(self.html_viewer, QWebEngineView):
                self.html_viewer.setHtml("<p>Model and associated data from training session loaded. You can now generate an explanation.</p>")
            else:
                self.html_viewer.setText("Model and data loaded. Generate explanation.")

        else:
            QMessageBox.warning(self, "Recall Error", f"Could not find training run with ID '{selected_run_id}'.")
            self.model = None
            self.X_train = None
            self.X_val = None
            self.X_test = None
            self._reset_visualization_and_controls()

    def _csd_sort_key(self, col_name): 
        match = re.match(r"csd(\d+)_(\d+)", col_name)
        if match:
            csd_num = int(match.group(1))
            sub_num = int(match.group(2))
            return (csd_num, sub_num)
        return (9999, 9999) 

    def _classify_kano(self, pos_val, neg_val, threshold=0.0002): 
        pos_val = float(pos_val) if pd.notna(pos_val) else 0.0
        neg_val = float(neg_val) if pd.notna(neg_val) else 0.0

        if pd.isna(pos_val) or pd.isna(neg_val):
            return 'Indifferent' 

        if abs(pos_val) < threshold and abs(neg_val) < threshold: 
            return 'Indifferent'
        elif pos_val <= 0 and neg_val <= 0: 
            return 'Must-be' 
        elif pos_val <= 0 and neg_val >= 0:
            return 'Reverse'  
        elif pos_val > 0 and neg_val < 0:
            return 'Performance' 
        elif pos_val > 0 and neg_val >= 0:
            return 'Excitement' 
        return 'Unclassified'

    def generate_kano_plot(self):
        if not PYQTWEBENGINE_AVAILABLE or self.kano_plot_viewer is None: return

        if self.explanation_obj is None or self.data_for_shap is None or self.data_for_shap.empty:
            placeholder_html = "<p style='text-align:center; padding:50px;'>SHAP explanation or its data (data_for_shap) not available for KANO Plot.</p>"
            if isinstance(self.kano_plot_viewer, QWebEngineView): self.kano_plot_viewer.setHtml(placeholder_html)
            else: self.kano_plot_viewer.setText("SHAP/data_for_shap not available.")
            return

        try:
            all_feature_names_from_shap = getattr(self.explanation_obj, 'feature_names', None)
            if all_feature_names_from_shap and len(all_feature_names_from_shap) == self.explanation_obj.values.shape[1]:
                all_feature_names = all_feature_names_from_shap
            elif not self.data_for_shap.empty and len(self.data_for_shap.columns) == self.explanation_obj.values.shape[1]:
                 all_feature_names = self.data_for_shap.columns.tolist()
            else:
                msg = "Cannot determine consistent feature names for KANO plot."
                if isinstance(self.kano_plot_viewer, QWebEngineView): self.kano_plot_viewer.setHtml(f"<p style='color:red;'>{msg}</p>")
                else: self.kano_plot_viewer.setText(msg)
                return

            all_shap_values = self.explanation_obj.values
            
            if len(all_shap_values.shape) == 3: # Handle multi-output SHAP values
                all_shap_values = np.mean(all_shap_values, axis=2) # Example: take mean over outputs

            if len(all_feature_names) != all_shap_values.shape[1]:
                 msg = f"Feature name count ({len(all_feature_names)}) differs from SHAP values dimension ({all_shap_values.shape[1]}) for KANO plot."
                 if isinstance(self.kano_plot_viewer, QWebEngineView): self.kano_plot_viewer.setHtml(f"<p style='color:red;'>{msg}</p>")
                 else: self.kano_plot_viewer.setText(msg)
                 return

            try:
                # Ensure data_for_shap has the same columns as all_feature_names for masking
                X_values_for_mask_df = self.data_for_shap[all_feature_names]
                X_values_for_mask = X_values_for_mask_df.values
            except KeyError as e:
                missing_cols = set(all_feature_names) - set(self.data_for_shap.columns)
                error_msg = (f"Columns for KANO mask not in data_for_shap: {missing_cols}. Error: {e}")
                if isinstance(self.kano_plot_viewer, QWebEngineView): self.kano_plot_viewer.setHtml(f"<p style='color:red;'>{error_msg}</p>")
                else: self.kano_plot_viewer.setText(error_msg)
                return

            mask = (X_values_for_mask == 1) # Assuming binary features for active/inactive
            active_shap_values = np.where(mask, all_shap_values, np.nan)
            
            median_shap_active = {}
            for i, feature_name in enumerate(all_feature_names):
                col_active_shap = active_shap_values[:, i]
                if np.all(np.isnan(col_active_shap)):
                    median_shap_active[feature_name] = 0.0 
                else:
                    with warnings.catch_warnings(): # Suppress warning for all-NaN slice if it still occurs
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        median_shap_active[feature_name] = np.nanmean(col_active_shap) 

            kano_results_dict = {}
            plot_data_points = []
            csd_base_names_found = set()
            for fname in all_feature_names:
                match = re.match(r"csd(\d+)_([01])", fname)
                if match:
                    csd_base_names_found.add(f"csd{match.group(1)}")
            
            if not csd_base_names_found:
                msg = "No csdX_0 or csdX_1 features found for KANO plot."
                if isinstance(self.kano_plot_viewer, QWebEngineView): self.kano_plot_viewer.setHtml(f"<p>{msg}</p>")
                else: self.kano_plot_viewer.setText(msg)
                return

            sorted_csd_base_names = sorted(list(csd_base_names_found), key=lambda x: int(re.match(r"csd(\d+)", x).group(1)))

            for base_csd_name in sorted_csd_base_names:
                pos_val = median_shap_active.get(f"{base_csd_name}_0", 0.0)
                neg_val = median_shap_active.get(f"{base_csd_name}_1", 0.0)
                renamed_label = self.feature_renames.get(base_csd_name, base_csd_name)
                kano_class = self._classify_kano(pos_val, neg_val)
                kano_results_dict[renamed_label] = kano_class
                plot_data_points.append({
                    'label': renamed_label, 'x': float(pos_val),
                    'y': float(neg_val), 'class': kano_class
                })
            
            if not plot_data_points:
                msg = "No data points for KANO plot after processing."
                if isinstance(self.kano_plot_viewer, QWebEngineView): self.kano_plot_viewer.setHtml(f"<p>{msg}</p>")
                else: self.kano_plot_viewer.setText(msg)
                return

            x_coords = [p['x'] for p in plot_data_points]
            y_coords = [p['y'] for p in plot_data_points]
            kano_classes = [p['class'] for p in plot_data_points]

            fig, ax = plt.subplots(figsize=(12, 9.6))
            colors = {'Indifferent': 'gray', 'Must-be': 'red', 'Reverse': 'orange',
                      'Performance': 'blue', 'Excitement': 'green', 'Unclassified': 'black'}
            
            ax.scatter(x_coords, y_coords, c=[colors.get(cls, 'black') for cls in kano_classes],
                       s=180, edgecolor='black', alpha=0.75, linewidth=0.5)

            for i, data_point in enumerate(plot_data_points):
                ax.annotate(f"{data_point['label']}\n({data_point['class']})", (data_point['x'], data_point['y']),
                            textcoords="offset points", xytext=(0,10), ha='center', fontsize=8.5, alpha=0.85)

            max_abs_x = max(abs(val) for val in x_coords) if x_coords else 0.05
            max_abs_y = max(abs(val) for val in y_coords) if y_coords else 0.1
            
            x_limit = max(max_abs_x, 0.01) * 1.2 
            y_limit = max(max_abs_y, 0.01) * 1.2

            ax.set_xlim(-x_limit, x_limit)
            ax.set_ylim(-y_limit, y_limit)
            
            ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
            ax.axvline(0, color="black", linestyle="--", linewidth=0.7)
            ax.grid(True, linestyle=':', alpha=0.4)

            quad_props = {'ha':'center', 'va':'center', 'fontsize':11, 'color':'darkgray', 'alpha':0.6, 'fontweight':'bold'}
            ax.text(x_limit*0.5, y_limit*0.5, 'Excitement', **quad_props)
            ax.text(-x_limit*0.5, y_limit*0.5, 'Reverse', **quad_props) 
            ax.text(-x_limit*0.5, -y_limit*0.5, 'Must-be', **quad_props)
            ax.text(x_limit*0.5, -y_limit*0.5, 'Performance', **quad_props)
            
            ax.set_xlabel("Dampak Positif (Functional / Feature Present)", fontsize=11)
            ax.set_ylabel("Dampak Negatif (Dysfunctional / Feature Absent)", fontsize=11)
            ax.set_title("Klasifikasi KANO Berdasarkan Dampak Fitur (Median SHAP Aktif)\n", fontsize=15, fontweight='bold')
            
            html_output = matplotlib_to_base64_html_img(fig)
            if isinstance(self.kano_plot_viewer, QWebEngineView): self.kano_plot_viewer.setHtml(html_output)
            else: self.kano_plot_viewer.setText("KANO Plot generated (cannot display as image here).")

        except Exception as e:
            error_html = f"<p style='color:red; text-align:center; padding:20px;'><b>Error generating KANO plot:</b><br>{str(e)}</p>"
            if isinstance(self.kano_plot_viewer, QWebEngineView): self.kano_plot_viewer.setHtml(error_html)
            else: self.kano_plot_viewer.setText(f"Error KANO plot: {e}")
            import traceback
            print(f"KANO Plot Error: {e}\n{traceback.format_exc()}")

    def open_rename_features_dialog(self):
        # Use self.data_for_shap if available, otherwise try self.X_test (from manual load)
        data_source_for_columns = None
        if self.data_for_shap is not None and not self.data_for_shap.empty:
            data_source_for_columns = self.data_for_shap
        elif self.X_test is not None and not self.X_test.empty: # Fallback to manually loaded X_test
            data_source_for_columns = self.X_test
        elif self.X_train is not None and not self.X_train.empty: # Fallback to manually loaded X_train
             data_source_for_columns = self.X_train


        if data_source_for_columns is None or data_source_for_columns.empty:
            QMessageBox.information(self, "No Data", "Load data (via training session recall, SHAP file, or manual load) first to identify base features.")
            return

        base_feature_names_set = set()
        for col_name in data_source_for_columns.columns:
            if col_name.endswith("_0") or col_name.endswith("_1"):
                base_feature_names_set.add(col_name[:-2])
            else:
                base_feature_names_set.add(col_name)
        
        unique_base_names = sorted(list(base_feature_names_set), key=self._csd_sort_key_base)

        if not unique_base_names:
            QMessageBox.information(self, "No Features", "No base features could be identified from the current data columns.")
            return

        dialog = RenameFeaturesDialog(unique_base_names, self.feature_renames, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.feature_renames = dialog.get_renames()
            self.update_shap_summary_table() 
            self.generate_kano_plot() 
            QMessageBox.information(self, "Success", "Feature dimension names updated.")
            
    def _csd_sort_key_base(self, base_name):
        match = re.match(r"csd(\d+)", base_name)
        if match:
            return (int(match.group(1)), base_name) 
        return (9999, base_name) # Place non-csd items at the end, sorted by name

    def _update_ui_for_explainer(self, explainer_type): 
        is_kernel = explainer_type == "KernelExplainer"
        self.kernel_bg_size_label.setVisible(is_kernel)
        self.kernel_bg_size_spinbox.setVisible(is_kernel)

    def _update_ui_for_visualization(self, visual_type):
        is_instance_specific = visual_type in ["Force Plot (Interactive HTML)", "Waterfall Plot (Single Instance, as Image)"]
        self.instance_spin_label.setVisible(is_instance_specific)
        self.instance_spin.setVisible(is_instance_specific)

        is_dependence = visual_type == "Dependence Plot (as Image)"
        self.feature_combo_label.setVisible(is_dependence)
        
        # Enable feature combo if dependence plot and data_for_shap is available
        current_data_source = self.data_for_shap
        if current_data_source is None or current_data_source.empty: # Fallback if data_for_shap not set
            if self.X_test is not None and not self.X_test.empty:
                current_data_source = self.X_test
            elif self.X_train is not None and not self.X_train.empty:
                current_data_source = self.X_train
        
        self.feature_combo.setEnabled(is_dependence and current_data_source is not None and not current_data_source.empty)
        
        shows_max_display = visual_type in [
            "Summary Plot (as Image)", "Bar Plot (Summary, as Image)",
            "Waterfall Plot (Single Instance, as Image)", "Decision Plot (All Instances, as Image)"
        ]
        self.max_display_label.setVisible(shows_max_display)
        self.max_display_spinbox.setVisible(shows_max_display)
        
        if self.explanation_obj is not None: # Regenerate if an explanation already exists
             self.generate_visualization()

    def regenerate_plot_if_needed(self): 
        if self.explanation_obj is not None and self.tab_widget.currentIndex() == 0: # Only if viz tab is active
            self.generate_visualization()

    def load_model_from_file(self): 
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "Model Files (*.pkl *.sav *.joblib)")
        if file_path:
            try:
                self.model = joblib.load(file_path)
                QMessageBox.information(self, "Success", "Model loaded from file!")
                self._reset_visualization_and_controls() # Reset SHAP related things
                if PYQTWEBENGINE_AVAILABLE and isinstance(self.html_viewer, QWebEngineView): 
                    self.html_viewer.setHtml("<p>Model loaded. Load data and generate explanation.</p>")
            except Exception as e: 
                QMessageBox.critical(self, "Error", f"Failed load model: {e}")
                self.model = None
    
    # load_data_from_file (X_test) is now load_test_data_from_file

    def load_shap_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load SHAP Explanation", "", "SHAP Files (*.shap *.pkl)")
        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    saved_data = pickle.load(f)
                if isinstance(saved_data, dict) and 'explanation_obj' in saved_data and 'data_for_shap' in saved_data:
                    self.explanation_obj = saved_data['explanation_obj']
                    self.data_for_shap = saved_data['data_for_shap'] # This is the key data
                    self.feature_renames = saved_data.get('feature_renames', {})
                    QMessageBox.information(self, "Success", "SHAP explanation and corresponding data_for_shap loaded!")
                    
                    # Clear manually loaded X_train, X_test, X_val as data_for_shap takes precedence
                    self.X_train = None 
                    self.X_test = None
                    self.X_val = None

                    if self.data_for_shap is not None and not self.data_for_shap.empty:
                        self.instance_spin.setMaximum(max(0, len(self.data_for_shap) - 1))
                        self.feature_combo.clear()
                        self.feature_combo.addItems(self.data_for_shap.columns.tolist())
                        self.feature_combo.setEnabled(True)
                    else:
                        self.instance_spin.setMaximum(0)
                        self.feature_combo.clear()
                        self.feature_combo.setEnabled(False)
                        
                    self.model = None # SHAP file doesn't store the model, user needs to load it if re-explaining
                    self.generate_visualization()
                    self.update_shap_summary_table()
                    self.generate_kano_plot()
                else:
                    raise ValueError("Invalid SHAP file format. Expected 'explanation_obj' and 'data_for_shap'.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load SHAP explanation: {e}")
                self.explanation_obj = None
                self.data_for_shap = None
                self._reset_visualization_and_controls()


    def save_shap_explanation(self):
        if self.explanation_obj is None or self.data_for_shap is None or self.data_for_shap.empty:
            QMessageBox.warning(self, "No Data", "No SHAP explanation or its corresponding data (data_for_shap) available to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Save SHAP Explanation", "", "SHAP Files (*.shap *.pkl)")
        if file_path:
            if not file_path.endswith((".shap", ".pkl")):
                file_path += ".shap"
            
            data_to_save = {
                'explanation_obj': self.explanation_obj,
                'data_for_shap': self.data_for_shap, # Save the combined data
                'feature_renames': self.feature_renames
            }
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(data_to_save, f)
                QMessageBox.information(self, "Success", f"SHAP explanation, data_for_shap, and renames saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save: {e}")
    
    def refresh_trained_models_list(self): 
        self.trained_model_combo.clear()
        if self.training_page and hasattr(self.training_page, 'training_runs_history') and self.training_page.training_runs_history:
            for run_data in self.training_page.training_runs_history: 
                self.trained_model_combo.addItem(f"{run_data['run_id']} ({run_data['model_name_display']})", userData=run_data["run_id"])
            self.trained_model_combo.setEnabled(True); self.use_trained_model_btn.setEnabled(True)
        else:
            self.trained_model_combo.addItem("No models trained or N/A."); self.trained_model_combo.setEnabled(False); self.use_trained_model_btn.setEnabled(False)


    def start_explanation_thread(self):
        if self.model is None:
            QMessageBox.warning(self, "Input Missing", "Please load or recall a model first.")
            return

        # Consolidate data for SHAP
        # Priority: Manually loaded data > Recalled data from training session
        # If X_train, X_test, or X_val are manually loaded, they might override what was recalled.
        # For simplicity, let's assume if any manual data is loaded, it's the primary source.
        # Otherwise, use the recalled data. If neither, then error.

        datasets_to_combine = []
        # Check for manually loaded data first
        manual_data_loaded = False
        if self.X_train is not None and not self.X_train.empty:
            datasets_to_combine.append(self.X_train)
            manual_data_loaded = True
        if self.X_test is not None and not self.X_test.empty:
            datasets_to_combine.append(self.X_test)
            manual_data_loaded = True
        if self.X_val is not None and not self.X_val.empty:
            datasets_to_combine.append(self.X_val)
            manual_data_loaded = True
        
        # If no manual data, and a model was recalled, use its associated data
        # This part is implicitly handled by how use_selected_trained_model sets self.X_train etc.
        # So, the datasets_to_combine list will be populated correctly based on current state of self.X_train etc.

        if not datasets_to_combine:
            QMessageBox.warning(self, "Input Missing", "No data (X_train, X_test, or X_val) available for explanation. Load data manually or recall a training session with data.")
            return
        
        try:
            # Check for column consistency before concat
            if len(datasets_to_combine) > 1:
                first_cols = datasets_to_combine[0].columns
                for i, df in enumerate(datasets_to_combine[1:], 1):
                    if not first_cols.equals(df.columns):
                        QMessageBox.critical(self, "Data Column Mismatch", 
                                             f"Columns in dataset {i+1} do not match the first dataset. Cannot combine for SHAP.")
                        return
            
            self.data_for_shap = pd.concat(datasets_to_combine).reset_index(drop=True)
        except Exception as e:
            QMessageBox.critical(self, "Data Concatenation Error", f"Failed to combine datasets: {e}")
            self.data_for_shap = None
            return
            
        if self.data_for_shap.empty:
            QMessageBox.warning(self, "Input Missing", "Combined dataset (data_for_shap) is empty.")
            self.data_for_shap = None
            return

        self.explain_btn.setEnabled(False)
        self.progress_dialog = QProgressDialog("Calculating SHAP values...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(False) # Keep it open until explicitly closed
        self.progress_dialog.setValue(0)
        
        self.progress_dialog.canceled.connect(self.cancel_shap_worker) # Connect cancel signal

        self.shap_worker = ShapWorker(
            self.model,
            self.data_for_shap, # Use the combined data
            self.explainer_combo.currentText(),
            background_size=self.kernel_bg_size_spinbox.value()
        )
        self.shap_worker.finished.connect(self.handle_shap_results)
        self.shap_worker.error.connect(self.handle_shap_error)
        self.shap_worker.progress.connect(self.update_progress_dialog)
        self.shap_worker.start()
        self.progress_dialog.exec() # Show and run the dialog
        # explain_btn re-enabled in handle_shap_results or handle_shap_error

    def update_progress_dialog(self, message, value):
        if hasattr(self, 'progress_dialog') and self.progress_dialog is not None:
            if self.progress_dialog.wasCanceled(): # Check if user pressed Cancel
                if hasattr(self, 'shap_worker') and self.shap_worker and self.shap_worker.isRunning():
                    self.shap_worker.terminate() # Attempt to stop the thread
                    # self.shap_worker.wait() # Optionally wait
                self.explain_btn.setEnabled(True)
                self.progress_dialog.close() # Close the dialog
                self.progress_dialog = None
                QMessageBox.information(self, "Cancelled", "SHAP calculation was cancelled by the user.")
                return
            self.progress_dialog.setLabelText(message)
            self.progress_dialog.setValue(value)

    def cancel_shap_worker(self): 
        if hasattr(self, 'shap_worker') and self.shap_worker and self.shap_worker.isRunning():
            self.shap_worker.terminate()
            # self.shap_worker.wait() # Consider if waiting is necessary or could hang UI
        self.explain_btn.setEnabled(True)
        if hasattr(self, 'progress_dialog') and self.progress_dialog is not None:
            self.progress_dialog.close()
            self.progress_dialog = None
        # QMessageBox.information(self, "Cancelled", "SHAP calculation cancelled.") # Message now in update_progress_dialog

    def handle_shap_results(self, explanation_obj_from_worker):
        self.explanation_obj = explanation_obj_from_worker
        if hasattr(self, 'progress_dialog') and self.progress_dialog is not None : 
            self.progress_dialog.setValue(100)
            self.progress_dialog.close() 
            self.progress_dialog = None 

        self.explain_btn.setEnabled(True)
        QMessageBox.information(self, "SHAP Calculation Complete", "SHAP values have been generated for the combined dataset.")

        if self.data_for_shap is not None and not self.data_for_shap.empty:
            self.instance_spin.setMaximum(max(0, len(self.data_for_shap) - 1))
            self.feature_combo.clear()
            self.feature_combo.addItems(self.data_for_shap.columns.tolist())
            self.feature_combo.setEnabled(True)
        else: # Should not happen if start_explanation_thread checks correctly
            self.instance_spin.setMaximum(0)
            self.feature_combo.clear()
            self.feature_combo.setEnabled(False)
            
        self.generate_visualization()
        self.update_shap_summary_table()
        self.generate_kano_plot()

    def handle_shap_error(self, error_msg): 
        if hasattr(self, 'progress_dialog') and self.progress_dialog is not None: 
            self.progress_dialog.close()
            self.progress_dialog = None
        self.explain_btn.setEnabled(True)
        QMessageBox.critical(self, "SHAP Error", f"Failed: {error_msg}")
        self.explanation_obj = None # Clear previous explanation on error
        self.update_shap_summary_table() # Update table to show no data
        if PYQTWEBENGINE_AVAILABLE and isinstance(self.html_viewer, QWebEngineView):
            self.html_viewer.setHtml(f"<p style='color:red;'>SHAP Error: {error_msg}</p>")
        else:
            self.html_viewer.setText(f"SHAP Error: {error_msg}")


    def update_shap_summary_table(self):
        if self.summary_table_widget is None: return
        self.summary_table_widget.setRowCount(0)
        self.summary_table_widget.setColumnCount(10) 
        self.summary_table_widget.setHorizontalHeaderLabels([
            "No", "Customer Satisfaction Dimension",
            "Pos Max", "Pos Min", "Pos Mean", "Pos Median",
            "Neg Max", "Neg Min", "Neg Mean", "Neg Median"
        ])

        if self.explanation_obj is None or self.data_for_shap is None or self.data_for_shap.empty:
            self.summary_table_widget.setRowCount(1)
            item = QTableWidgetItem("No SHAP data available. Generate or load an explanation with its data.")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.summary_table_widget.setItem(0, 0, item)
            self.summary_table_widget.setSpan(0, 0, 1, 10)
            return

        try:
            shap_values_raw = self.explanation_obj.values
            
            feature_names_from_shap = getattr(self.explanation_obj, 'feature_names', None)
            if feature_names_from_shap and len(feature_names_from_shap) == shap_values_raw.shape[1]:
                 feature_names_for_table = feature_names_from_shap
            elif not self.data_for_shap.empty and len(self.data_for_shap.columns) == shap_values_raw.shape[1]:
                 feature_names_for_table = self.data_for_shap.columns.tolist()
            else:
                QMessageBox.critical(self, "Feature Name Error", "Cannot reconcile feature names between SHAP object and data_for_shap for summary table.")
                return

            current_shap_values_2d = shap_values_raw
            if len(shap_values_raw.shape) == 3:
                current_shap_values_2d = shap_values_raw[:, :, 0] 
            
            if len(feature_names_for_table) != current_shap_values_2d.shape[1]:
                QMessageBox.warning(self, "Data Mismatch",
                                    f"Feature name count ({len(feature_names_for_table)}) "
                                    f"differs from processed SHAP values dimension ({current_shap_values_2d.shape[1]}). Summary table might be incorrect.")
                # Attempt to reconcile if possible, or return
                if len(feature_names_for_table) > current_shap_values_2d.shape[1]:
                    feature_names_for_table = feature_names_for_table[:current_shap_values_2d.shape[1]]
                # else: cannot extend if shap values are shorter

            try:
                data_for_shap_subset = self.data_for_shap[feature_names_for_table]
            except KeyError as e:
                missing_cols = set(feature_names_for_table) - set(self.data_for_shap.columns)
                QMessageBox.critical(self, "Data Column Error",
                                     f"data_for_shap is missing columns for summary: {missing_cols}. Error: {e}")
                return
            
            data_values_for_mask = data_for_shap_subset.values
            mask = (data_values_for_mask == 1)
            active_shap_values = np.where(mask, current_shap_values_2d, np.nan)
            
            processed_data = {} 
            for i, original_col_name in enumerate(feature_names_for_table):
                if i >= active_shap_values.shape[1]: continue 
                col_active_shap_values = active_shap_values[:, i]
                
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', r'All-NaN slice encountered')
                    warnings.filterwarnings('ignore', r'Mean of empty slice')
                    if np.all(np.isnan(col_active_shap_values)):
                        stats = (np.nan, np.nan, np.nan)
                    else:
                        stats = (np.nanmax(col_active_shap_values), 
                                 np.nanmin(col_active_shap_values),
                                 np.nanmean(col_active_shap_values), 
                                 np.nanmedian(col_active_shap_values))

                base_name = original_col_name
                entry_type = None
                if original_col_name.endswith("_0"):
                    base_name = original_col_name[:-2]
                    entry_type = "pos_stats"
                elif original_col_name.endswith("_1"):
                    base_name = original_col_name[:-2]
                    entry_type = "neg_stats"

                if entry_type:
                    if base_name not in processed_data:
                        processed_data[base_name] = {"pos_stats": None, "neg_stats": None}
                    processed_data[base_name][entry_type] = stats
            
            def csd_numerical_sort_key(csd_base_name):
                match = re.match(r"csd(\d+)", csd_base_name)
                if match: return int(match.group(1))
                return 9999 

            sorted_base_names = sorted(processed_data.keys(), key=csd_numerical_sort_key)
            self.summary_table_widget.setRowCount(len(sorted_base_names))

            for row_idx, base_name in enumerate(sorted_base_names):
                display_name = self.feature_renames.get(base_name, base_name)
                data_entry = processed_data[base_name]

                self.summary_table_widget.setItem(row_idx, 0, QTableWidgetItem(str(row_idx + 1)))
                self.summary_table_widget.setItem(row_idx, 1, QTableWidgetItem(display_name))

                pos_stats = data_entry.get("pos_stats")
                if pos_stats and not all(np.isnan(s) for s in pos_stats):
                    self.summary_table_widget.setItem(row_idx, 2, QTableWidgetItem(f"{pos_stats[0]:.4f}" if pd.notna(pos_stats[0]) else "N/A"))
                    self.summary_table_widget.setItem(row_idx, 3, QTableWidgetItem(f"{pos_stats[1]:.4f}" if pd.notna(pos_stats[1]) else "N/A"))
                    self.summary_table_widget.setItem(row_idx, 4, QTableWidgetItem(f"{pos_stats[2]:.4f}" if pd.notna(pos_stats[2]) else "N/A"))
                    self.summary_table_widget.setItem(row_idx, 5, QTableWidgetItem(f"{pos_stats[3]:.4f}" if pd.notna(pos_stats[2]) else "N/A"))

                else:
                    for c_idx in range(2,6): self.summary_table_widget.setItem(row_idx, c_idx, QTableWidgetItem("N/A"))

                neg_stats = data_entry.get("neg_stats")
                if neg_stats and not all(np.isnan(s) for s in neg_stats):
                    self.summary_table_widget.setItem(row_idx, 6, QTableWidgetItem(f"{neg_stats[0]:.4f}" if pd.notna(neg_stats[0]) else "N/A"))
                    self.summary_table_widget.setItem(row_idx, 7, QTableWidgetItem(f"{neg_stats[1]:.4f}" if pd.notna(neg_stats[1]) else "N/A"))
                    self.summary_table_widget.setItem(row_idx, 8, QTableWidgetItem(f"{neg_stats[2]:.4f}" if pd.notna(neg_stats[2]) else "N/A"))
                    self.summary_table_widget.setItem(row_idx, 9, QTableWidgetItem(f"{neg_stats[3]:.4f}" if pd.notna(neg_stats[2]) else "N/A"))

                else:
                    for c_idx in range(6,10): self.summary_table_widget.setItem(row_idx, c_idx, QTableWidgetItem("N/A"))
            
            self.summary_table_widget.resizeColumnsToContents()
            if self.summary_table_widget.columnCount() > 1 :
                self.summary_table_widget.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
                self.summary_table_widget.setColumnWidth(1, 250) # Give more space for dimension name

        except Exception as e:
            self.summary_table_widget.setRowCount(1)
            error_item = QTableWidgetItem(f"Error populating summary table: {str(e)}")
            self.summary_table_widget.setItem(0,0, error_item)
            self.summary_table_widget.setSpan(0,0,1,8) 
            import traceback
            print(f"Error in update_shap_summary_table: {e}\n{traceback.format_exc()}")
    
    def generate_visualization(self):
        if not PYQTWEBENGINE_AVAILABLE and self.visual_combo.currentText().endswith("(Interactive HTML)"):
            if isinstance(self.html_viewer, QLabel): self.html_viewer.setText("PyQtWebEngine required for HTML plots.")
            self.current_html_content = self.html_viewer.text() if isinstance(self.html_viewer, QLabel) else ""
            return

        if self.explanation_obj is None or self.data_for_shap is None or self.data_for_shap.empty:
            msg = "<p>Generate or load SHAP explanation and its corresponding data (data_for_shap) first.</p>"
            self.current_html_content = msg
            if isinstance(self.html_viewer, QWebEngineView): self.html_viewer.setHtml(msg)
            elif isinstance(self.html_viewer, QLabel): self.html_viewer.setText("Generate/load SHAP explanation and data_for_shap.")
            return

        visual_type = self.visual_combo.currentText()
        instance_idx = self.instance_spin.value()
        selected_feature = self.feature_combo.currentText()
        max_disp = self.max_display_spinbox.value()
        html_output = ""
        self.current_html_content = ""

        num_shap_instances = len(self.explanation_obj) if hasattr(self.explanation_obj, '__len__') else 0
        
        if instance_idx >= len(self.data_for_shap): 
            instance_idx = 0
            self.instance_spin.setValue(0)
        
        if num_shap_instances > 0 and instance_idx >= num_shap_instances : 
            instance_idx = 0
            self.instance_spin.setValue(0)
            
        try:
            if num_shap_instances == 0 and visual_type not in ["Summary Plot (as Image)", "Bar Plot (Summary, as Image)", "Decision Plot (All Instances, as Image)"]: 
                 raise ValueError("SHAP Explanation object appears to have no instances or is not indexable for this plot type.")
            
            current_explanation_instance = self.explanation_obj[instance_idx] if num_shap_instances > 0 else self.explanation_obj

            # Determine feature names for plotting, prioritizing SHAP object's names if consistent
            feature_names_for_plot = self.data_for_shap.columns.tolist()
            if hasattr(self.explanation_obj, 'feature_names') and self.explanation_obj.feature_names is not None:
                if len(self.explanation_obj.feature_names) == self.data_for_shap.shape[1]: 
                     feature_names_for_plot = self.explanation_obj.feature_names
                elif len(self.explanation_obj.feature_names) == self.explanation_obj.values.shape[1]: # If values shape matches feature_names
                     feature_names_for_plot = self.explanation_obj.feature_names


            if visual_type == "Force Plot (Interactive HTML)":
                if not PYQTWEBENGINE_AVAILABLE: raise ImportError("PyQtWebEngine required.")
                
                plot_base_values = self.explanation_obj.base_values
                # Handle base_values that might be a single value or an array
                if isinstance(plot_base_values, np.ndarray) and plot_base_values.ndim > 0:
                    if plot_base_values.shape[0] == num_shap_instances: # Array per instance
                        plot_base_values = plot_base_values[instance_idx]
                    elif plot_base_values.shape[0] == 1: # Single base value for all
                        plot_base_values = plot_base_values[0]
                    # Add more conditions if base_values can have other shapes (e.g., per class)
                
                # Ensure current_explanation_instance.values is 1D for single instance force plot
                shap_values_for_force = current_explanation_instance.values
                if shap_values_for_force.ndim > 1 and shap_values_for_force.shape[0] == 1: # If it's (1, num_features)
                    shap_values_for_force = shap_values_for_force.ravel()


                force_plot_obj = shap.force_plot(plot_base_values, 
                                                 shap_values_for_force, 
                                                 features=self.data_for_shap.iloc[instance_idx],
                                                 feature_names=feature_names_for_plot,
                                                 show=False, matplotlib=False) # Ensure matplotlib=False for HTML
                if force_plot_obj is None: raise ValueError("Could not generate force plot.")
                
                fd, tmp_path = tempfile.mkstemp(suffix=".html")
                try:
                    shap.save_html(tmp_path, force_plot_obj)
                    with open(tmp_path, 'r', encoding='utf-8') as f: html_output = f.read()
                finally:
                    os.close(fd); os.remove(tmp_path)
            else: # Matplotlib based plots
                plt.close('all') # Close any existing figures
                # temp_fig = plt.figure(figsize=(9, 6.5), dpi=100) # Not needed, SHAP creates its own
                
                if visual_type == "Summary Plot (as Image)":
                    shap.summary_plot(self.explanation_obj, self.data_for_shap, feature_names=feature_names_for_plot, max_display=max_disp, show=False)
                elif visual_type == "Bar Plot (Summary, as Image)":
                    shap.summary_plot(self.explanation_obj, self.data_for_shap, feature_names=feature_names_for_plot, plot_type="bar", max_display=max_disp, show=False)
                elif visual_type == "Waterfall Plot (Single Instance, as Image)":
                    shap.waterfall_plot(current_explanation_instance, max_display=max_disp, show=False)
                elif visual_type == "Dependence Plot (as Image)":
                    if not selected_feature: 
                        fig_err, ax_err = plt.subplots(); ax_err.text(0.5,0.5, "Select a feature for dependence plot.", ha='center'); plt.show(block=False)
                    elif selected_feature not in self.data_for_shap.columns: 
                        fig_err, ax_err = plt.subplots(); ax_err.text(0.5,0.5, f"Feature '{selected_feature}' not in data.", ha='center'); plt.show(block=False)
                    else:
                        shap.dependence_plot(selected_feature, self.explanation_obj.values, self.data_for_shap, 
                                             feature_names=feature_names_for_plot,
                                             interaction_index="auto", show=False)
                elif visual_type == "Decision Plot (All Instances, as Image)":
                    # Decision plot needs base_values that matches the number of outputs if multi-output
                    base_values_for_decision = self.explanation_obj.base_values
                    if isinstance(base_values_for_decision, np.ndarray) and base_values_for_decision.ndim > 0 and \
                       len(self.explanation_obj.values.shape) == 3 and base_values_for_decision.shape[0] != self.explanation_obj.values.shape[2]:
                        # If base_values is a single array but SHAP values are multi-output, repeat base_value for each output
                        if base_values_for_decision.shape[0] == 1:
                             base_values_for_decision = np.repeat(base_values_for_decision, self.explanation_obj.values.shape[2])
                        # Add other logic if base_values structure is different for multi-output

                    shap.decision_plot(base_values_for_decision, self.explanation_obj.values,
                                       features=self.data_for_shap, 
                                       feature_names=feature_names_for_plot,
                                       show=False, auto_size_plot=True, 
                                       highlight=self.data_for_shap.iloc[[instance_idx]] if num_shap_instances >0 and instance_idx < len(self.data_for_shap) else None)
                
                html_output = matplotlib_to_base64_html_img(plt.gcf()) 
            
            self.current_html_content = html_output
            if isinstance(self.html_viewer, QWebEngineView): self.html_viewer.setHtml(self.current_html_content)
            elif isinstance(self.html_viewer, QLabel): self.html_viewer.setText("Matplotlib plot generated (cannot display here). Export to view.")

        except ImportError as e_imp: # Should not happen if PYQTWEBENGINE_AVAILABLE is handled
            msg = f"<p style='color:red;'><b>Missing Dependency:</b><br>{e_imp}</p>"
            self.current_html_content = msg
            if isinstance(self.html_viewer, QWebEngineView): self.html_viewer.setHtml(msg)
            elif isinstance(self.html_viewer, QLabel): self.html_viewer.setText(f"Missing Dependency: {e_imp}")
        except Exception as e_viz:
            msg = f"<p style='color:red;'><b>Visualization Error:</b><br>{e_viz}</p>"
            self.current_html_content = msg
            if isinstance(self.html_viewer, QWebEngineView): self.html_viewer.setHtml(msg)
            elif isinstance(self.html_viewer, QLabel): self.html_viewer.setText(f"Visualization Error: {e_viz}")
            import traceback
            print(f"Visualization error: {e_viz}\n{traceback.format_exc()}")


    def export_content(self): 
        current_tab_index = self.tab_widget.currentIndex()
        if current_tab_index == 0: # Visualisasi
            if not self.current_html_content or self.current_html_content.startswith("<p"): 
                QMessageBox.warning(self, "No Content", "No HTML visualization to export."); return
            default_filename = f"shap_viz_{self.visual_combo.currentText().split('(')[0].strip().replace(' ','_')}.html"
            file_path, _ = QFileDialog.getSaveFileName(self, "Save HTML Visualization", default_filename, "HTML (*.html)")
            if file_path:
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        html_to_save = self.current_html_content
                        if not (html_to_save.lower().strip().startswith("<!doctype html") or html_to_save.lower().strip().startswith("<html")):
                            html_to_save = f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>SHAP Export</title></head><body>{html_to_save}</body></html>"
                        f.write(html_to_save)
                    QMessageBox.information(self, "Success", "HTML visualization saved!")
                except Exception as e: QMessageBox.critical(self, "Error", f"Failed to save HTML: {e}")
        elif current_tab_index == 1: # Tabel Ringkasan
            if self.summary_table_widget.rowCount() == 0 or \
               (self.summary_table_widget.rowCount() == 1 and "No SHAP data" in self.summary_table_widget.item(0,0).text()):
                QMessageBox.warning(self, "No Content", "No summary data to export."); return
            default_filename = "shap_summary_pos_neg.csv"
            file_path, _ = QFileDialog.getSaveFileName(self, "Save SHAP Summary Table", default_filename, "CSV (*.csv)")
            if file_path:
                try:
                    data = []; headers = [self.summary_table_widget.horizontalHeaderItem(i).text() for i in range(self.summary_table_widget.columnCount())]
                    for row in range(self.summary_table_widget.rowCount()):
                        row_data = [self.summary_table_widget.item(row, col).text() for col in range(self.summary_table_widget.columnCount())]
                        data.append(row_data)
                    df_summary = pd.DataFrame(data, columns=headers)
                    df_summary.to_csv(file_path, index=False)
                    QMessageBox.information(self, "Success", "SHAP summary table saved as CSV!")
                except Exception as e: QMessageBox.critical(self, "Error", f"Failed to save summary CSV: {e}")
        elif current_tab_index == 2: # KANO Plot
            if isinstance(self.kano_plot_viewer, QWebEngineView):
                # For QWebEngineView, we need to get its current HTML content
                self.kano_plot_viewer.page().toHtml(self._save_kano_plot_html_callback) # Async, use callback
            elif isinstance(self.kano_plot_viewer, QLabel) and self.kano_plot_viewer.text().startswith("<img src="): # If it's an image in a QLabel
                 self._save_kano_plot_html(self.kano_plot_viewer.text()) # Pass the HTML directly
            else:
                 QMessageBox.warning(self, "No KANO Content", "No KANO plot (as HTML image) to export from this tab.")


    def _save_kano_plot_html_callback(self, html_content):
        """Callback for QWebEngineView.toHtml to save KANO plot."""
        self._save_kano_plot_html(html_content)


    def _save_kano_plot_html(self, html_content):
        if not html_content or html_content.startswith("<p style='text-align:center; padding:50px;'>Data SHAP"):
            QMessageBox.warning(self, "No KANO Content", "No KANO plot generated to export.")
            return
        default_filename = "kano_classification_plot.html"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save KANO Plot", default_filename, "HTML Files (*.html)")
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    if not (html_content.lower().strip().startswith("<!doctype html") or html_content.lower().strip().startswith("<html")):
                        html_to_save = f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>KANO Classification Plot</title></head><body>{html_content}</body></html>"
                    else:
                        html_to_save = html_content
                    f.write(html_to_save)
                QMessageBox.information(self, "Success", "KANO plot HTML saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save KANO plot HTML: {str(e)}")

import traceback # Untuk mencetak traceback error

class DebugConsoleWindow(QWidget):
    """
    Jendela terpisah untuk menjalankan perintah Python secara manual
    dalam konteks MainWindow. Output akan muncul di CLI.
    """
    def __init__(self, main_window_ref, parent=None):
        super().__init__(parent)
        self.main_window_ref = main_window_ref  # Referensi ke instance MainWindow
        self.setWindowTitle("Konsole Debug Aplikasi")
        self.setGeometry(200, 200, 600, 350) # Sedikit lebih besar

        layout = QVBoxLayout(self)

        info_label = QLabel(
            "Ketik perintah Python di bawah ini. Output dan error akan muncul di terminal/CLI utama.\n"
            "Gunakan variabel yang tersedia:\n"
            "  - `mw`: Instance MainWindow aplikasi utama.\n"
            "  - `currentPage`: Instance dari halaman yang sedang aktif/terbuka di MainWindow.\n"
            "  - `pd`: Modul pandas.\n"
            "  - `np`: Modul numpy.\n"
            "Contoh:\n"
            "  `print(currentPage.df_prod.head())` (jika halaman aktif punya df_prod)\n"
            "  `currentPage.some_method_on_active_page()`\n"
            "  `print(mw.scraper_page.df_rev.columns)` (mengakses page spesifik via mw)"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; padding: 8px; border-radius: 4px;")
        layout.addWidget(info_label)

        self.command_input = QTextEdit(self)
        self.command_input.setPlaceholderText("Contoh: print(currentPage.df_prod.shape)")
        self.command_input.setFont(QFont("Consolas", 10)) # Font monospace
        self.command_input.setMinimumHeight(100)
        layout.addWidget(self.command_input)

        button_layout = QHBoxLayout()
        self.clear_button = QPushButton("🧹 Bersihkan Input")
        self.clear_button.clicked.connect(self.command_input.clear)
        
        self.send_button = QPushButton("🚀 Kirim Perintah ke CLI")
        self.send_button.setStyleSheet("background-color: #007bff; color: white; font-weight: bold; padding: 8px;")
        self.send_button.clicked.connect(self._execute_command)

        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()
        button_layout.addWidget(self.send_button)
        layout.addLayout(button_layout)

        self.setMinimumSize(500, 300)

    def _execute_command(self):
        command_text = self.command_input.toPlainText().strip()
        if not command_text:
            print("\nDEBUG CONSOLE: Tidak ada perintah untuk dijalankan.")
            return

        print(f"\nDEBUG CONSOLE: Menjalankan perintah >>\n{command_text}\n{'-'*50}")
        
        mw_instance = self.main_window_ref
        current_page_widget = None
        if hasattr(mw_instance, 'stacked_widget'): # Pastikan stacked_widget ada
            current_page_widget = mw_instance.stacked_widget.currentWidget()

        # Siapkan environment untuk exec()
        exec_globals = {
            '__builtins__': __builtins__,
            'mw': mw_instance,                 # Instance MainWindow
            'currentPage': current_page_widget, # Instance halaman yang sedang aktif
            'pd': pd,                         # Modul pandas
            'np': np,                          # Modul numpy
            # Anda bisa menambahkan referensi ke semua halaman jika diinginkan,
            # tapi ini bisa membuat namespace global menjadi ramai.
            # 'home': mw_instance.home_page if hasattr(mw_instance, 'home_page') else None,
            # 'scraper': mw_instance.scraper_page if hasattr(mw_instance, 'scraper_page') else None,
            # 'labeling': mw_instance.labeling_page if hasattr(mw_instance, 'labeling_page') else None,
            # 'topic_analyzer': mw_instance.topic_analysis_page if hasattr(mw_instance, 'topic_analysis_page') else None,
            # 'trainer': mw_instance.training_page if hasattr(mw_instance, 'training_page') else None,
            # 'explainer': mw_instance.explainable_page if hasattr(mw_instance, 'explainable_page') else None,
            # 'dataviz': mw_instance.dataviz_page if hasattr(mw_instance, 'dataviz_page') else None,
        }
        
        # Filter None values jika Anda menambahkan referensi opsional di atas
        # exec_globals = {k: v for k, v in exec_globals.items() if v is not None}

        try:
            # Jalankan perintah. Output dari print() dan error akan ke CLI.
            exec(command_text, exec_globals)
            print(f"{'-'*50}\nDEBUG CONSOLE: Perintah selesai dijalankan.")
        except Exception as e:
            print(f"{'-'*50}\nDEBUG CONSOLE: Error saat menjalankan perintah!")
            print(f"  Tipe Error  : {type(e).__name__}")
            print(f"  Pesan Error : {e}")
            print(f"  Traceback   :\n{traceback.format_exc()}")
            print(f"{'-'*50}")
            QMessageBox.critical(self, "Execution Error", 
                                 f"Error: {type(e).__name__}: {e}\n\nLihat CLI untuk traceback lengkap.")

    def showEvent(self, event):
        """Dipanggil saat jendela ditampilkan, bisa digunakan untuk refresh info."""
        super().showEvent(event)
        # Anda bisa menambahkan logika di sini jika perlu memperbarui sesuatu saat jendela muncul
        print("DEBUG CONSOLE: Jendela debug dibuka.")
        if hasattr(self.main_window_ref, 'stacked_widget'):
            current_page = self.main_window_ref.stacked_widget.currentWidget()
            print(f"DEBUG CONSOLE: Halaman aktif saat ini di MainWindow adalah: {type(current_page).__name__}")


    def closeEvent(self, event):
        # Sembunyikan jendela, jangan hancurkan, agar bisa dibuka lagi
        self.hide()
        event.ignore() 
        print("DEBUG CONSOLE: Jendela debug ditutup (disembunyikan).")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Satisfaction Analysis Suite")
        self.setWindowIcon(QIcon("icon.png"))

        self.setGeometry(100, 100, 1200, 800)
        
        # Create stacked widget for pages
        self.stacked_widget = QStackedWidget()
        
        # Create pages
        self.home_page = HomePage()
        self.scraper_page = ScraperPage()
        self.labeling_page = LabelingPage()
        self.training_page = TrainingPage()
        self.explainable_page = ExplainabilityPage(training_page_ref=self.training_page)
        # self.dataviz_page = DataVisualizationPage(None, None) 
        
        # Di __init__ MainWindow:
        self.stacked_widget.addWidget(self.home_page)
        self.stacked_widget.addWidget(self.scraper_page)
        self.stacked_widget.addWidget(self.labeling_page)  # Index 2
        self.stacked_widget.addWidget(self.training_page)  # Index 3 (jika ada)
        self.stacked_widget.addWidget(self.explainable_page)  # Index 3 (jika ada)
        # # self.stacked_widget.addWidget(self.dataviz_page)  # Index 4 (jika ada)
        # self.dataviz_page = None        # Placeholder, akan dibuat on-demand

        

        
        
        # Create sidebar
        self.sidebar = QFrame()
        self.sidebar.setFrameShape(QFrame.Shape.StyledPanel)
        self.sidebar.setFixedWidth(150)
        
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setContentsMargins(10, 20, 10, 20)
        
        # Home button
        self.home_btn = QPushButton("🏠 Home")
        self.home_btn.setCheckable(True)
        self.home_btn.setChecked(True)
        self.home_btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton:checked {
                background-color: #e0e0e0;
                font-weight: bold;
            }
        """)
        
        # Scraper button
        self.scraper_btn = QPushButton("🛒 Scraper")
        self.scraper_btn.setCheckable(True)
        self.scraper_btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton:checked {
                background-color: #e0e0e0;
                font-weight: bold;
            }
        """)
        
        # Topic Analysis button (new)
        self.topic_btn = QPushButton("📊 Topic Analysis")
        self.topic_btn.setCheckable(True)
        self.topic_btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton:checked {
                background-color: #e0e0e0;
                font-weight: bold;
            }
        """)

        # Data Labeling button (new)
        self.labeling_btn = QPushButton("🏷️ Data Labeling")
        self.labeling_btn.setCheckable(True)
        self.labeling_btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton:checked {
                background-color: #e0e0e0;
                font-weight: bold;
            }
        """)

        # Training Model button (new)
        self.training_btn = QPushButton("⚙️ Model Training")
        self.training_btn.setCheckable(True)
        self.training_btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton:checked {
                background-color: #e0e0e0;
                font-weight: bold;
            }
        """)
            
        # Explain button (new)
        self.explainable_button = QPushButton("🔎 Explain Model")
        self.explainable_button.setCheckable(True)
        self.explainable_button.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton:checked {
                background-color: #e0e0e0;
                font-weight: bold;
            }
        """)

        # # dataviz button (new)
        # self.dataviz_button = QPushButton("🖼️ Datavis Model")
        # self.dataviz_button.setCheckable(True)
        # self.dataviz_button.setStyleSheet("""
        #     QPushButton {
        #         text-align: left;
        #         padding: 8px;
        #         font-size: 14px;
        #     }
        #     QPushButton:checked {
        #         background-color: #e0e0e0;
        #         font-weight: bold;
        #     }
        # """)

        # Add buttons to sidebar
        sidebar_layout.addWidget(self.home_btn)
        sidebar_layout.addWidget(self.scraper_btn)
        sidebar_layout.addWidget(self.topic_btn)
        sidebar_layout.addWidget(self.labeling_btn)
        sidebar_layout.addWidget(self.training_btn)
        sidebar_layout.addWidget(self.explainable_button)
        # sidebar_layout.addWidget(self.dataviz_button)

        sidebar_layout.addStretch()
        
        self.sidebar.setLayout(sidebar_layout)
        
        # Main layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.stacked_widget)
        
        # Central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.debug_console_window = None # Placeholder untuk jendela debug
        
        # Connect signals
        self.home_btn.clicked.connect(self.show_home)
        self.scraper_btn.clicked.connect(self.show_scraper)
        self.topic_btn.clicked.connect(self.show_topic_analysis)
        self.labeling_btn.clicked.connect(self.show_labeling)  # Pindahkan koneksi ke button yang benar
        self.training_btn.clicked.connect(self.show_training)  # Pindahkan koneksi ke button yang benar
        self.explainable_button.clicked.connect(self.show_explainable)  # Pindahkan koneksi ke button yang benar
        # self.dataviz_button.clicked.connect(self.show_dataviz)  # Pindahkan koneksi ke button yang benar
        self.show_debug_console

    def show_debug_console(self):
        # Tombol debug tidak mengubah halaman utama di stacked_widget,
        # jadi kita tidak memanggil _update_button_states dengan self.debug_btn
        # Biarkan tombol halaman yang aktif sebelumnya tetap aktif.
        # self.debug_btn.setChecked(True) # Ini akan membatalkan check tombol halaman lain jika di QButtonGroup
        
        if self.debug_console_window is None:
            self.debug_console_window = DebugConsoleWindow(main_window_ref=self)
        
        if self.debug_console_window.isVisible():
            self.debug_console_window.raise_() # Bawa ke depan jika sudah terlihat
            self.debug_console_window.activateWindow()
        else:
            self.debug_console_window.show()
    
    def show_home(self):
        self.stacked_widget.setCurrentIndex(0)
        self.home_btn.setChecked(True)
        self.scraper_btn.setChecked(False)
        self.topic_btn.setChecked(False)
        self.labeling_btn.setChecked(False)
        self.training_btn.setChecked(False)
        self.explainable_button.setChecked(False)
        # self.dataviz_button.setChecked(False)

    
    def show_scraper(self):
        self.stacked_widget.setCurrentIndex(1)
        self.home_btn.setChecked(False)
        self.scraper_btn.setChecked(True)
        self.topic_btn.setChecked(False)
        self.labeling_btn.setChecked(False)
        self.training_btn.setChecked(False)
        self.explainable_button.setChecked(False)
        # self.dataviz_button.setChecked(False)

    
    def show_topic_analysis(self):
        # First check if we have review data
        if hasattr(self.scraper_page, 'df_rev') and self.scraper_page.df_rev is not None:
            # Create topic analysis page if it doesn't exist
            if not hasattr(self, 'topic_analysis_page'):
                self.topic_analysis_page = TopicAnalysisPage(self.scraper_page.df_rev)
                self.stacked_widget.addWidget(self.topic_analysis_page)
            
            self.stacked_widget.setCurrentWidget(self.topic_analysis_page)
            self.home_btn.setChecked(False)
            self.scraper_btn.setChecked(False)
            self.topic_btn.setChecked(True)
            self.labeling_btn.setChecked(False)
            self.training_btn.setChecked(False)
            self.explainable_button.setChecked(False)
            # self.dataviz_button.setChecked(False)

        else:
            # Show warning if no review data exists
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "No Data Available",
                "Please scrape reviews first before analyzing topics.",
                QMessageBox.StandardButton.Ok
            )
            self.topic_btn.setChecked(False)
            self.scraper_btn.setChecked(True)
            self.labeling_btn.setChecked(False)
            self.training_btn.setChecked(False)
            self.explainable_button.setChecked(False)
            # self.dataviz_button.setChecked(False)

            self.stacked_widget.setCurrentIndex(1)

    def show_labeling(self):
        self.stacked_widget.setCurrentWidget(self.labeling_page)
        self.home_btn.setChecked(False)
        self.scraper_btn.setChecked(False)
        self.topic_btn.setChecked(False)
        self.labeling_btn.setChecked(True)
        self.training_btn.setChecked(False)
        self.explainable_button.setChecked(False)
        # self.dataviz_button.setChecked(False)

    def show_training(self):
        self.stacked_widget.setCurrentWidget(self.training_page)
        self.home_btn.setChecked(False)
        self.scraper_btn.setChecked(False)
        self.topic_btn.setChecked(False)
        self.labeling_btn.setChecked(False)
        self.training_btn.setChecked(True)
        self.explainable_button.setChecked(False)
        # self.dataviz_button.setChecked(False)

    def show_explainable(self):
        self.stacked_widget.setCurrentWidget(self.explainable_page)
        self.home_btn.setChecked(False)
        self.scraper_btn.setChecked(False)
        self.topic_btn.setChecked(False)
        self.labeling_btn.setChecked(False)
        self.training_btn.setChecked(False)
        self.explainable_button.setChecked(True)
        # self.dataviz_button.setChecked(False)

    def show_dataviz(self):
        # Cek apakah data produk atau ulasan sudah ada di ScraperPage
        prod_data_exists = hasattr(self.scraper_page, 'df_prod') and \
                           self.scraper_page.df_prod is not None and \
                           not self.scraper_page.df_prod.empty
        rev_data_exists = hasattr(self.scraper_page, 'df_rev') and \
                          self.scraper_page.df_rev is not None and \
                          not self.scraper_page.df_rev.empty

        if not (prod_data_exists or rev_data_exists):
            QMessageBox.warning(
                self,
                "Data Tidak Tersedia",
                "Silakan lakukan scraping atau muat data produk dan/atau ulasan pada halaman 'Scraper Data' terlebih dahulu.",
                QMessageBox.StandardButton.Ok
            )
            # self._update_button_states(self.scraper_btn) # Kembalikan fokus ke scraper atau halaman sebelumnya
            self.stacked_widget.setCurrentWidget(self.scraper_page)
            self.home_btn.setChecked(False)
            self.scraper_btn.setChecked(False)
            self.topic_btn.setChecked(True)
            self.labeling_btn.setChecked(False)
            self.training_btn.setChecked(False)
            self.explainable_button.setChecked(False)
            # self.dataviz_button.setChecked(False)
            return

        # if self.dataviz_page is None: # Buat instance jika belum ada (Lazy Initialization)
        #     self.dataviz_page = DataVisualizationPage(
        #         df_prod=self.scraper_page.df_prod if prod_data_exists else None,
        #         df_rev=self.scraper_page.df_rev if rev_data_exists else None
        #     )
        #     self.stacked_widget.addWidget(self.dataviz_page)
        # else:
        #     # Jika instance sudah ada, panggil metode update_data
        #     self.dataviz_page.update_data(
        #         df_prod=self.scraper_page.df_prod if prod_data_exists else None,
        #         df_rev=self.scraper_page.df_rev if rev_data_exists else None
        #     )

        self.stacked_widget.setCurrentWidget(self.dataviz_page)
        self.home_btn.setChecked(False)
        self.scraper_btn.setChecked(False)
        self.topic_btn.setChecked(False)
        self.labeling_btn.setChecked(False)
        self.training_btn.setChecked(False)
        self.explainable_button.setChecked(False)
        # self.dataviz_button.setChecked(True)
        # self._update_button_states(self.dataviz_button)

# if __name__ == "__main__":
#     REVIEW_URL = "https://gql.tokopedia.com/graphql/productReviewList"
#     multiprocessing.freeze_support()

#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec())