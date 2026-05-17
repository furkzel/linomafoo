"""Construct main_presentation_linoalgo.ipynb from three source notebooks."""
import json
from pathlib import Path

ROOT = Path('.')

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text}

def load_source_cells(path):
    with open(path, 'r', encoding='utf-8') as fp:
        nb = json.load(fp)
    out = []
    for c in nb['cells']:
        src = c.get('source', '')
        if isinstance(src, list):
            src = ''.join(src)
        out.append((c['cell_type'], src))
    return out

src0 = load_source_cells('ozi_the_pca_master.ipynb')
src1 = load_source_cells('ozi_the_pca_master_1.ipynb')
src2 = load_source_cells('ozi_the_pca_master_2.ipynb')

cells = []

# =================================================================
# COVER CELL
# =================================================================
cover = """# Cancer Genetics için Principal Component Analysis

**Ders:** Lineer Cebir ve Algoritmaları
**Konu:** Principal Component Analysis (PCA) — teori ve uygulama
**Veri Setleri:** TCGA-PANCAN-HiSeq (gene expression, $n{=}801$, $p{=}20\\,531$) • Breast Cancer Wisconsin Diagnostic (WDBC, $n{=}569$, $p{=}30$)
**Sunum Tarihi:** 17 Mayıs 2026

## Özet

Bu çalışma, Principal Component Analysis tekniğini hem matematiksel temelleriyle hem de iki yapısal olarak zıt kanser veri seti üzerindeki uygulamasıyla bütünleşik biçimde sunmaktadır. PCA, önce Eigendecomposition ve Singular Value Decomposition formülasyonları üzerinden inşa edilmekte, ardından NumPy ile sıfırdan uygulanıp `scikit-learn` referansıyla doğrulanmaktadır. Yüksek-boyutlu TCGA-PANCAN RNA-seq verisi ve düşük-boyutlu WDBC klinik morfoloji verisi üzerinde tam EDA → PCA → Loading analizi → KNN değerlendirmesi döngüsü çalıştırılmakta; Marchenko-Pastur yasası ve BBP faz geçişi gibi rastgele matris teorisi araçlarıyla iki rejim arasındaki davranış farkları açıklanmaktadır. Sonuçlar, PCA'nın salt görselleştirme aracı olmadığını, gerçek bir Feature Extraction tekniği olduğunu sayısal kanıtlarla ortaya koymaktadır.

## İçindekiler

1. [Bölüm 0 — Kütüphaneler ve Sabitler](#Bölüm-0-—-Kütüphaneler-ve-Sabitler)
2. [Bölüm 1 — PCA'nın Matematiksel Temelleri](#Bölüm-1-—-PCA'nın-Matematiksel-Temelleri)
3. [Bölüm 2 — NumPy PCA: Sıfırdan İmplementasyon](#Bölüm-2-—-NumPy-PCA:-Sıfırdan-İmplementasyon)
4. [Bölüm 3 — TCGA-PANCAN Gene Expression](#Bölüm-3-—-TCGA-PANCAN-Gene-Expression)
5. [Bölüm 4 — Breast Cancer Wisconsin Diagnostic](#Bölüm-4-—-Breast-Cancer-Wisconsin-Diagnostic)
6. [Bölüm 5 — Karşılaştırmalı Analiz](#Bölüm-5-—-Karşılaştırmalı-Analiz)
7. [Bölüm 6 — Sonuçlar ve Tartışma](#Bölüm-6-—-Sonuçlar-ve-Tartışma)
"""
cells.append(md(cover))

# =================================================================
# BÖLÜM 0 — Unified imports + constants
# =================================================================
bolum0_intro = """# Bölüm 0 — Kütüphaneler ve Sabitler

Bu bölümde notebook boyunca kullanılacak tüm bağımlılıklar **tek bir blokta** içe aktarılmaktadır. Sabitler de aynı şekilde tek bir yerde tanımlanır: `RANDOM_STATE`, tüm stokastik adımlarda (örneğin `KNeighborsClassifier` öncesi `train_test_split`, `cross_val_score` katlamaları) deterministik sonuçlar elde etmek için sabitlenmiştir. `N_COMPONENTS = 10`, PCA için varsayılan üst sınırdır; ihtiyaca göre yerel olarak değiştirilebilir. Renk paletleri sınıflar arası görsel tutarlılık için her plotta aynı sınıfa aynı rengi atamaktadır.
"""
cells.append(md(bolum0_intro))

imports_code = """import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_COMPONENTS = 10

CANCER_COLORS = {
    'BRCA': '#E74C3C',
    'COAD': '#3498DB',
    'KIRC': '#2ECC71',
    'LUAD': '#F39C12',
    'PRAD': '#9B59B6',
}
WDBC_COLORS = {'M': '#E74C3C', 'B': '#2ECC71'}

TCGA_DATA_PATH = './TCGA-PANCAN-HiSeq-801x20531/data.csv'
TCGA_LABELS_PATH = './TCGA-PANCAN-HiSeq-801x20531/labels.csv'
WDBC_DATA_PATH = './breast+cancer+wisconsin+diagnostic/wdbc.data'

np.random.seed(RANDOM_STATE)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
"""
cells.append(code(imports_code))

# =================================================================
# BÖLÜM 1 — From notebook 0, cells 3..13
# =================================================================
for i in range(3, 14):
    ctype, csrc = src0[i]
    cells.append(md(csrc) if ctype == 'markdown' else code(csrc))

# =================================================================
# BÖLÜM 2 — From notebook 1, with renumbered subsection headers
# =================================================================
# Cell 2 (intro)
cells.append(md(src1[2][1]))
# 2.1 header + class
cells.append(md(src1[3][1]))   # ## 2.1 — PCA From Scratch
cells.append(code(src1[4][1]))  # PCAFromScratch class

# 2.2 — Sklearn comparison + sign ambiguity + speed (merge sub-headers under 2.2)
bolum_2_2_header = """## 2.2 — Sklearn ile Karşılaştırma, İşaret Belirsizliği ve Hız

İmplementasyonun doğruluğu üç düzeyde test edilmektedir: (i) `explained_variance_ratio_` değerleri makine epsilon mertebesinde aynı olmalıdır, (ii) skorlar **işarete kadar** eşleşmelidir (Eigenvector belirsizliği), (iii) altuzaylar aynı olduğu için reconstruction birebir aynı olmalıdır. Ayrıca farklı $n, p$ rejimlerinde hız karşılaştırması yapılmaktadır.
"""
cells.append(md(bolum_2_2_header))

# Sub-sections: keep cells 5,6,7,8,9,10,11
for i in [5, 6, 7, 8, 9, 10, 11]:
    ctype, csrc = src1[i]
    cells.append(md(csrc) if ctype == 'markdown' else code(csrc))

# 2.3 — Reconstruction + Eckart-Young proof
bolum_2_3_header = src1[12][1].replace('## 2.2 — Reconstruction ve Hata Analizi',
                                        '## 2.3 — Reconstruction ve Hata Analizi')
cells.append(md(bolum_2_3_header))
cells.append(code(src1[13][1]))  # reconstruction error curve

# NEW Eckart-Young SVD derivation cell
eckart_young = """### SVD ile Reconstruction Error'ın Türetilmesi (Eckart–Young–Mirsky)

Yukarıdaki sayısal eşitliğin matematiksel kanıtı doğrudan SVD'den izlemektedir. $\\tilde{X} \\in \\mathbb{R}^{n \\times p}$'in thin SVD'si $\\tilde{X} = U \\Sigma V^\\top$ olsun ve rank-$k$ Approximation $\\hat{X}^{(k)} = U_k \\Sigma_k V_k^\\top$ ile tanımlansın (burada $U_k \\in \\mathbb{R}^{n \\times k}$, $\\Sigma_k \\in \\mathbb{R}^{k \\times k}$, $V_k \\in \\mathbb{R}^{p \\times k}$ ilk $k$ singular bileşeni temsil eder). O zaman:

$$\\lVert \\tilde{X} - \\hat{X}^{(k)} \\rVert_F^2 \\;=\\; \\lVert U \\Sigma V^\\top - U_k \\Sigma_k V_k^\\top \\rVert_F^2 \\;=\\; \\lVert U_{>k}\\, \\Sigma_{>k}\\, V_{>k}^\\top \\rVert_F^2.$$

$U$ ve $V$ ortonormal sütunlara sahip olduğu için Frobenius normu unitary dönüşümler altında değişmez ve sadece $\\Sigma$ üzerinde toplanır:

$$\\lVert U_{>k}\\, \\Sigma_{>k}\\, V_{>k}^\\top \\rVert_F^2 \\;=\\; \\sum_{i > k} \\sigma_i^2 \\;=\\; (n-1) \\sum_{i > k} \\lambda_i,$$

burada Bölüm 1.3'te türetilen $\\lambda_i = \\sigma_i^2 / (n-1)$ özdeşliği kullanılmaktadır. Paydadaki normalleştirme $\\lVert \\tilde{X} \\rVert_F^2 = \\sum_i \\sigma_i^2 = (n-1) \\sum_i \\lambda_i$ ile bölündüğünde:

$$\\boxed{\\; \\mathrm{ReconErr}(k) \\;=\\; \\frac{\\sum_{i > k} \\lambda_i}{\\sum_i \\lambda_i} \\;=\\; 1 - \\mathrm{CumPVE}(k). \\;}$$

Bu özdeşlik, bir önceki kod hücresindeki sayısal kontrolün ($\\mathrm{ReconErr}(k) - (1 - \\mathrm{CumPVE}(k))$ farkının makine epsilonu mertebesinde sıfır çıkması) **kapalı-form ispatıdır**: PCA'nın "varyansı maksimize etme" formülasyonu, Frobenius norm anlamında "Reconstruction Error'ı minimize etme" formülasyonuna birebir eşdeğerdir.
"""
cells.append(md(eckart_young))

# 2.4 — Helper functions (FIXED: plot_scree with legends)
bolum_2_4_header = """## 2.4 — Yardımcı Fonksiyonlar

Aşağıdaki dört fonksiyon, bu notebook'un geri kalanında — Bölüm 3 (TCGA), Bölüm 4 (WDBC) ve Bölüm 5 (karşılaştırmalı analiz) — kullanılacaktır. Tek bir yerde tanımlı olmaları, tüm downstream görselleştirmelerin tutarlı renk, eksen ve format kullanmasını garanti eder. `plot_scree` fonksiyonu hem bireysel hem de Cumulative PVE eğrilerini iki ayrı `y` ekseninde gösterir ve her iki eksen için legend etiketleri sağlar.
"""
cells.append(md(bolum_2_4_header))

helpers_code = """def plot_pca_2d(scores, labels, color_dict, title, pc_x=1, pc_y=2, pve=None):
    \"\"\"Plot 2D PCA projection with labeled groups.\"\"\"
    fig, ax = plt.subplots(figsize=(10, 8))
    labels = np.asarray(labels)
    for label, color in color_dict.items():
        mask = labels == label
        if not mask.any():
            continue
        ax.scatter(scores[mask, pc_x - 1], scores[mask, pc_y - 1],
                   c=color, label=label, alpha=0.75, s=32,
                   edgecolor='white', linewidth=0.4)
    xl = f'PC{pc_x}' + (f' ({pve[pc_x - 1] * 100:.1f}% var)' if pve is not None else '')
    yl = f'PC{pc_y}' + (f' ({pve[pc_y - 1] * 100:.1f}% var)' if pve is not None else '')
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title)
    ax.legend(title='Class', loc='best', framealpha=0.9)
    fig.tight_layout()
    return fig, ax


def plot_scree(pca_model, n_components=20, title='Scree Plot'):
    \"\"\"Plot eigenvalue bar chart with cumulative variance line and dual legends.\"\"\"
    pve = np.asarray(pca_model.explained_variance_ratio_)[:n_components]
    cum = np.cumsum(pve)
    x = np.arange(1, len(pve) + 1)
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(x, pve * 100, color='#3498DB', alpha=0.75, label='Individual PVE')
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Individual PVE (%)', color='#3498DB')
    ax1.tick_params(axis='y', labelcolor='#3498DB')
    ax1.set_xticks(x)
    ax2 = ax1.twinx()
    ax2.plot(x, cum * 100, color='#E74C3C', marker='o', label='Cumulative PVE')
    ax2.set_ylabel('Cumulative PVE (%)', color='#E74C3C')
    ax2.tick_params(axis='y', labelcolor='#E74C3C')
    ax2.set_ylim(0, 105)
    ax2.axhline(80, ls='--', color='gray', alpha=0.5)
    ax2.axhline(90, ls='--', color='gray', alpha=0.5)
    ax1.set_title(title)
    ax1.legend(loc='upper left', framealpha=0.9)
    ax2.legend(loc='center right', framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_loadings(components, feature_names, pc_index=1, top_n=20):
    \"\"\"Plot top_n absolute loadings for a given principal component.\"\"\"
    pc = np.asarray(components[pc_index - 1])
    feature_names = np.asarray(feature_names)
    top_idx = np.argsort(np.abs(pc))[::-1][:top_n]
    values = pc[top_idx]
    names = feature_names[top_idx]
    colors = ['#E74C3C' if v < 0 else '#2ECC71' for v in values]
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(values))
    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(0, color='black', lw=0.6)
    ax.set_xlabel('Loading value')
    ax.set_title(f'Top-{top_n} loadings for PC{pc_index} (green: positive, red: negative)')
    fig.tight_layout()
    return fig


def knn_accuracy_vs_components(X, y, ks, cv=5, n_neighbors=5, scale=False):
    \"\"\"Return DataFrame of k -> cross-validated KNN accuracy on PCA-reduced X.\"\"\"
    rows = []
    X_in = StandardScaler().fit_transform(X) if scale else np.asarray(X, dtype=np.float64)
    for k in ks:
        pca_k = PCA(n_components=k, random_state=RANDOM_STATE)
        X_k = pca_k.fit_transform(X_in)
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        scores_cv = cross_val_score(clf, X_k, y, cv=cv, n_jobs=-1)
        rows.append({
            'n_components': k,
            'accuracy_mean': scores_cv.mean(),
            'accuracy_std': scores_cv.std(),
        })
    return pd.DataFrame(rows)
"""
cells.append(code(helpers_code))

# =================================================================
# BÖLÜM 3 — From notebook 1, cells 16..47
# =================================================================
for i in range(16, 48):
    ctype, csrc = src1[i]
    cells.append(md(csrc) if ctype == 'markdown' else code(csrc))

# =================================================================
# BÖLÜM 4 — From notebook 2, cells 3..21, with biplot inserted after cell 16
# =================================================================
biplot_code = """arrow_scale = 3.0
score_pc1 = scores_wdbc_cor[:, 0]
score_pc2 = scores_wdbc_cor[:, 1]
max_score_range = max(score_pc1.max() - score_pc1.min(),
                      score_pc2.max() - score_pc2.min())
loadings_pc1 = pca_wdbc_cor.components_[0]
loadings_pc2 = pca_wdbc_cor.components_[1]
scale_x = arrow_scale * np.sqrt(pve_wdbc_cor[0]) * max_score_range
scale_y = arrow_scale * np.sqrt(pve_wdbc_cor[1]) * max_score_range

fig, ax_scores = plt.subplots(figsize=(12, 10))
for cls, color in WDBC_COLORS.items():
    mask = y_wdbc == cls
    ax_scores.scatter(score_pc1[mask], score_pc2[mask],
                      c=color, label=cls, alpha=0.4, s=20,
                      edgecolor='white', linewidth=0.3)

for j, name in enumerate(feature_names_wdbc):
    dx = loadings_pc1[j] * scale_x
    dy = loadings_pc2[j] * scale_y
    ax_scores.annotate('', xy=(dx, dy), xytext=(0, 0),
                       arrowprops=dict(arrowstyle='->', color='#34495E',
                                       lw=1.2, alpha=0.75))
    ax_scores.text(dx * 1.10, dy * 1.10, name, fontsize=8,
                   color='#2C3E50', ha='center', va='center')

ax_scores.axhline(0, color='gray', lw=0.4)
ax_scores.axvline(0, color='gray', lw=0.4)
ax_scores.set_xlabel(f'PC1 score ({pve_wdbc_cor[0] * 100:.1f}% var)')
ax_scores.set_ylabel(f'PC2 score ({pve_wdbc_cor[1] * 100:.1f}% var)')
ax_scores.set_title('WDBC — Correlation PCA Biplot (PC1 × PC2)')
ax_scores.legend(title='Diagnosis', loc='upper right', framealpha=0.9)

xlim = ax_scores.get_xlim()
ylim = ax_scores.get_ylim()
ax_loadings_x = ax_scores.twiny()
ax_loadings_x.set_xlim(xlim[0] / scale_x, xlim[1] / scale_x)
ax_loadings_x.set_xlabel('PC1 loading')
ax_loadings_y = ax_scores.twinx()
ax_loadings_y.set_ylim(ylim[0] / scale_y, ylim[1] / scale_y)
ax_loadings_y.set_ylabel('PC2 loading')

fig.tight_layout()
plt.show()
"""

biplot_explanation = """**Biplot Yorumu.** Biplot, gözlemleri (Score'lar — noktalar) ve özellikleri (Loading'ler — oklar) **aynı PC1 × PC2 düzleminde** birleştiren standart bir PCA görselleştirmesidir. Alt/sol ekseni Score'ları orijinal birimlerde, üst/sağ ekseni ise Loading'leri $[-1, 1]$ aralığında okutmaktadır. Yorumlama anahtarları:

- **Ok uzunluğu.** Bir özelliğin PC1 × PC2 düzleminde ne kadar varyans katkısı yaptığını gösterir; daha uzun oklar (`*_worst` ile `radius/perimeter/area_*` aileleri) PC eksenlerine daha güçlü katkı verir, kısa oklar bu iki bileşene marjinal katkıda bulunur.
- **Ok yönü.** Özelliğin yüksek değerlerinin PC düzleminde hangi yönde uzandığını söyler. **Aynı yöne** işaret eden oklar pozitif korele özelliklerdir (örneğin `radius_mean`, `perimeter_mean`, `area_mean` neredeyse paralel uzanır — beklenildiği gibi, çünkü matematiksel olarak da bağlıdırlar); **zıt yönde** oklar negatif korele, **dik açıdaki** oklar ise yaklaşık ilişkisizdir.
- **Nokta–ok hizalanması.** Bir gözlem bulutu hangi okların yönüne yatkın yerleşmişse, o özellikler bakımından yüksek değer taşır. Malign (kırmızı) örnekler `*_worst` ve büyüklük oklarının yönünde toplanmakta; bu, bir önceki adımdaki Cohen's $d$ analizi ile tutarlıdır — sınıf ayrımının taşıyıcısı bu özellik aileleridir.

Biplot bu nedenle "PC eksenleri biyolojik olarak ne anlama geliyor?" sorusunu **görsel olarak** yanıtlayan en doğrudan araçtır; Loading bar chart'ları kadar sayısaldır ancak özellikler arası mekânsal ilişkiyi de korur.
"""

for i in range(3, 22):
    ctype, csrc = src2[i]
    cells.append(md(csrc) if ctype == 'markdown' else code(csrc))
    if i == 16:
        # After the loading bar charts cell, insert biplot + explanation
        cells.append(code(biplot_code))
        cells.append(md(biplot_explanation))

# =================================================================
# BÖLÜM 5 — From notebook 2, cells 22..33, with MP spike interpretation
# =================================================================
mp_spike_interp = """### Spike Sayısının Biyolojik Yorumu

Marchenko-Pastur bulk'unun üst sınırı $\\lambda_+$'ı aşan eigenvalue sayısı — yani **Spike sayısı** — verinin taşıdığı bağımsız gerçek sinyal eksenlerinin sayısına karşılık gelir. TCGA-PANCAN için bu sayının biyolojik olarak makul aralığını şu şekilde tahmin edebiliriz:

- Veri setinde **5 farklı kanser tipi** (BRCA, COAD, KIRC, LUAD, PRAD) bulunmaktadır. 5 sınıf merkezi $\\mathbb{R}^p$'de en fazla 4 boyutlu bir afin alt uzayda yer alır; dolayısıyla yalnızca tipler-arası ortalama farkı bile **en az 4 ek Spike** üretmesi beklenir.
- Buna ek olarak her kanser tipi içinde bilinen biyolojik alt yapı vardır: örneğin **BRCA** için literatürde tanımlı **luminal-A, luminal-B, HER2-enriched ve basal-like** moleküler alt sınıfları her biri ayrı bir Spike olarak görünebilir.
- Bu iki katmanı birleştirince TCGA için beklenen aralık yaklaşık **$5 \\leq n_{\\text{spikes}} \\leq 10\\text{–}15$**'tir; gözlemlenen Spike sayısı bu aralıkla **tutarlıdır**, yani PCA'nın yakaladığı düşük-boyutlu yapı biyolojik beklentiyi doğrulamaktadır.

Eğer $n_{\\text{spikes}} \\gg 5$ olsaydı bu, batch effect, RNA kalite farklılıkları veya hasta-spesifik konfounder gibi **biyolojik olmayan kovaryasyon kaynaklarının** da spektruma sızdığına işaret ederdi ve önişleme adımlarının (örneğin ComBat ile batch correction veya surrogate variable analysis) yeniden gözden geçirilmesi gerekirdi.
"""

for i in range(22, 35):
    ctype, csrc = src2[i]
    cells.append(md(csrc) if ctype == 'markdown' else code(csrc))
    if i == 28:
        # After the n_spikes print code cell, insert the interpretation markdown
        cells.append(md(mp_spike_interp))

# =================================================================
# Assemble notebook JSON
# =================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# Add cell ids per nbformat 4.5
import uuid
for c in nb['cells']:
    c['id'] = uuid.uuid4().hex[:12]

out_path = ROOT / 'main_presentation_linoalgo.ipynb'
with open(out_path, 'w', encoding='utf-8') as fp:
    json.dump(nb, fp, ensure_ascii=False, indent=1)

# Quality gates
import io, sys
sys.stdout.reconfigure(encoding='utf-8')

all_sources = [
    (''.join(c['source']) if isinstance(c['source'], list) else c['source'])
    for c in nb['cells']
]
all_code = [s for c, s in zip(nb['cells'], all_sources) if c['cell_type'] == 'code']
all_md = [s for c, s in zip(nb['cells'], all_sources) if c['cell_type'] == 'markdown']

# Count import cells (containing 'import matplotlib' as fingerprint)
n_imports = sum(1 for s in all_code if 'import matplotlib.pyplot as plt' in s)
assert n_imports == 1, f'Expected 1 import cell, got {n_imports}'

# Count plot_pca_2d definitions
n_plot_pca = sum(s.count('def plot_pca_2d(') for s in all_code)
assert n_plot_pca == 1, f'plot_pca_2d defined {n_plot_pca} times'
n_plot_scree = sum(s.count('def plot_scree(') for s in all_code)
assert n_plot_scree == 1, f'plot_scree defined {n_plot_scree} times'
n_plot_loadings = sum(s.count('def plot_loadings(') for s in all_code)
assert n_plot_loadings == 1, f'plot_loadings defined {n_plot_loadings} times'
n_knn_helper = sum(s.count('def knn_accuracy_vs_components(') for s in all_code)
assert n_knn_helper == 1, f'knn_accuracy_vs_components defined {n_knn_helper} times'

# Eckart-Young cell present
assert any('Eckart' in s for s in all_md), 'Eckart-Young markdown missing'

# Biplot present
assert any('Biplot' in s for s in all_md), 'Biplot markdown missing'
assert any('biplot' in s.lower() and 'twiny' in s for s in all_code), 'Biplot code missing'

# MP spike interpretation present
assert any('Spike Sayısının Biyolojik Yorumu' in s for s in all_md), 'MP spike interpretation missing'

# No references to original notebook names
for s in all_sources:
    assert 'ozi_the_pca_master' not in s, f'Found leftover reference to original notebook'

# plot_scree legend check
scree_src = [s for s in all_code if 'def plot_scree(' in s][0]
assert 'ax1.legend(' in scree_src and 'ax2.legend(' in scree_src, 'plot_scree legends missing'

print(f'main_presentation_linoalgo.ipynb yazıldı. Toplam hücre sayısı: {len(nb["cells"])}')
