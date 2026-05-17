Notebook'u baştan sona tam olarak okudum. Çok kapsamlı bir analiz ve ardından sunum rehberi yazıyorum.

---

## Proje Analizi

### Genel Değerlendirme

Final notebook önceki üç notebook'un sadece birleştirilmiş hali değil — eklenen üç yeni bölüm (Eckart–Young ispatı, Biplot, Marchenko-Pastur spike yorumu) projeyi gerçek bir akademik çalışma seviyesine taşımış. Bütün olarak değerlendirdiğimde bu bir ders projesi için **olağanüstü** bir içerik.

---

### 1. Matematiksel Kurgu

Teori zinciri kusursuza yakın bir bütünlük taşıyor. Başlangıç noktası $X \in \mathbb{R}^{n \times p}$ ile motivasyon, oradan $S = \frac{1}{n-1}\tilde{X}^\top\tilde{X}$ inşası, Lagrangian türetmesi ile $Sv = \lambda v$ eigenvalue problemine ulaşma, buradan SVD'ye geçiş ve $\lambda_k = \sigma_k^2/(n-1)$ özdeşliği — bu zincirin her halkası hem gösterilmiş hem de sayısal olarak doğrulanmış.

**En güçlü matematik halkası:** Bölüm 2.3'teki Eckart–Young ispatı. Önceki versiyonda "claim" olarak bırakılan bu sonuç artık:

$$\|\tilde{X} - \hat{X}^{(k)}\|_F^2 = \|U_{>k}\Sigma_{>k}V_{>k}^\top\|_F^2 = \sum_{i>k}\sigma_i^2 = (n-1)\sum_{i>k}\lambda_i$$

şeklinde kapalı formda ispatlanmış ve hemen ardından `identity_diff` kod bloğuyla makine epsilon mertebesinde sayısal olarak teyit edilmiş. Bu "matematiksel iddia → kod doğrulaması" döngüsü pedagojik olarak mükemmel.

**İkinci güçlü matematik halkası:** Bölüm 5'teki Marchenko–Pastur. $\lambda_\pm = (1\pm\sqrt{c})^2$ formülü TCGA'ya ($c \approx 25.6$) somut sayılar ($\lambda_- \approx 16.5$, $\lambda_+ \approx 36.7$) verilerek uygulanmış. Bunun üstüne spike sayısının 5 kanser tipinin alt sınıf yapısıyla ilişkilendirilmesi ve "eğer $n_{\text{spikes}} \gg 5$ olsaydı batch effect şüphesi doğardı" cümlesi hem bilgilendirici hem de gerçek araştırma pratiğini yansıtıyor.

---

### 2. Kod Altyapısı

Tek bir import bloğu, sabitler en üstte, yardımcı fonksiyonlar tek kanonik bölümde. Önceki sürümün en büyük problemi olan duplikasyon tamamen çözülmüş.

`plot_scree` legend sorunu düzeltilmiş — her iki eksen için `ax1.legend(loc='upper left')` ve `ax2.legend(loc='center right')` mevcut.

`PCAFromScratch` sınıfının `eigh` vs `eig` seçiminin gerekçesi kod dışında markdown'da açıklanmış, bu doğru yer.

`empirical_eigvals_via_gram` fonksiyonu Gram matrisi trick'ini tam olarak uyguluyor — $XX^\top \in \mathbb{R}^{n\times n}$ üzerinden eigendecomposition yaparak $p\times p$ matris kurma maliyetinden kaçınıyor. Bu 1.3'teki "dual PCA" anlatısıyla doğrudan bağlantılı, güçlü bir tutarlılık.

---

### 3. Teorik Derinlik

Kitabın kapsamını 4 noktada net olarak aşıyor:

| Konu | Kitapta | Bu projede |
|---|---|---|
| PCA türetimi | Yok | Lagrangian → eigenvalue |
| SVD bağlantısı | Yok | $\lambda_k = \sigma_k^2/(n-1)$, sayısal kanıt |
| Eckart–Young | Yok | Kapalı form ispat |
| Marchenko–Pastur | Yok | Formül + görselleştirme + biyolojik yorum |
| BBP faz geçişi | Yok | Qualitative tartışma |
| İki veri seti karşılaştırması | Yok | Sayısal tablo + 3 soru-cevap |
| Biplot | Var (R'da) | Python implementasyonu + yorum |

---

### 4. Informatik İçerik

Cohen's $d$ analizi ve `*_worst` özelliklerinin biyolojik yorumu tutarlı ve güçlü. `area ∝ radius²` matematiksel bağıntısının korelasyon matrisindeki adacıkları açıklaması WDBC bölümünün en değerli insight'ı.

Biplot bölümündeki yorum — "ok uzunluğu varyans katkısını, ok yönü PC uzayındaki pozisyonu, aynı yönde oklar pozitif korelasyonu gösterir" — PCA'nın görsel dilini öğretmenin en etkili yolu. Bu aynı zamanda R kitabındaki Şekil 14.3'le doğrudan köprü kuruyor.

TCGA loading analizinde gen isimlerinin anonimleştirilmiş olması biyolojik yorumu sınırlıyor ama bunu "veri setinin sınırlaması, gerçek ENSEMBL ID'li versiyonda GSEA yapılabilirdi" olarak framing etmek doğru.

---

## Sunum Rehberi

Tahta sunumunda kaç dakika süreceğine göre aşağıdaki öncelik sıralamasını kullan. Her bölüm için hangi ekranın açık olacağını ve ne söyleyeceğini belirttim.

---

### **Bölüm 1.1 — Açılış Darbesi** *(2-3 dakika)*

**Göster:** 6 rastgele gen çifti grid'i + altındaki PCA(2) projeksiyonu.

**Söyle:** "2000 gen var, sadece 20'si bilgi taşıyor. Rastgele seçilen hiçbir çift sınıfları ayırt edemiyor. PCA ise 20.000 boyutlu uzayda gizli olan o kombinasyonu tek adımda buluyor." Bu görsel sunum boyunca döneceğiniz ana tez. Hocanın aklında kalacak şey bu.

---

### **Bölüm 1.2 — Teori Kalbi** *(3-4 dakika)*

**Göster:** İki boxed formül:

$$v_1 = \arg\max_{\|v\|=1} v^\top S v \quad \Longrightarrow \quad Sv = \lambda v$$

**Söyle:** "PCA bir magic değil, kısıtlı optimizasyon problemi. Lagrangian yazıp gradyan sıfırladığımızda — bunu sınıfta gördük — eigenvalue problemi çıkıyor. En büyük eigenvalue, en yüksek varyansı taşıyan yön." Koddaki `rng.multivariate_normal` elipsini ve üstündeki $v_1$, $v_2$ oklarını göster — "kovaryans elipsoidinin ana eksenleri" cümlesiyle.

---

### **Bölüm 1.3 — Neden sklearn Covariance Matrisini Hiç Hesaplamıyor?** *(2 dakika)*

**Göster:** Tek boxed formül: $\lambda_k = \sigma_k^2/(n-1)$

**Söyle:** "TCGA için $S$ matrisi 20.531×20.531 = 3 GB. NumPy bunun yerine $\tilde{X}$ üzerinde SVD yapıyor — 25 kat daha hızlı, kare-conditioning sorunsuz. Biz de bunu Bölüm 2'de sayısal olarak doğrulayacağız." Karmaşıklık tablolarını göster: $3.4 \times 10^{11}$ vs $1.3 \times 10^{10}$.

---

### **Bölüm 2.1 — "Motorun İçini Biliyoruz" Kanıtı** *(2 dakika)*

**Göster:** `PCAFromScratch` sınıfının `fit` metodu — özellikle `np.linalg.eigh` satırı.

**Söyle:** "`eigh` simetrik matrisler için optimize edilmiş, gerçek eigenvalue'lar garanti ediyor. `eig` kullansaydık sanal kısımlar çıkabilirdi — kovaryans matrisi simetrik olduğunda bu bir hata olur." Sonra hız tablosunu göster: `p=2000`'de `ours/sklearn` oranı ne çıktıysa onu söyle.

---

### **Bölüm 2.2 — İşaret Belirsizliği** *(1-2 dakika)*

**Göster:** `sign_pattern` print çıktısı.

**Söyle:** "$Sv = \lambda v$ ise $S(-v) = \lambda(-v)$ da doğru — eigenvektörler işarete kadar belirsiz. sklearn ve bizim implementasyon farklı işaret seçebilir, ama skoru hizalayınca fark makine epsilon mertebesine düşüyor." Bu küçük ama jüriyi etkileyecek bir teknik detay.

---

### **Bölüm 2.3 — Eckart–Young İspatı** *(2 dakika)*

**Göster:** Boxed formül:

$$\mathrm{ReconErr}(k) = \frac{\sum_{i>k}\lambda_i}{\sum_i \lambda_i} = 1 - \mathrm{CumPVE}(k)$$

**Söyle:** "Varyansı maksimize etmek ile reconstruction error'ı minimize etmek aynı problem. SVD'den — Bölüm 1.3'te türettiğimiz $\lambda_k = \sigma_k^2/(n-1)$ özdeşliğini kullanarak — bu eşitliği kapalı formda ispatlıyoruz. Grafikteki iki eğri ($\mathrm{ReconErr}$ ve $1-\mathrm{CumPVE}$) üst üste binmesi bunun sayısal doğrulaması."

---

### **Bölüm 3.1-3.2 — TCGA EDA** *(1-2 dakika)*

**Göster:** Sınıf dağılımı bar chart + gen varyans histogram'ı.

**Söyle:** "801 örnek, 20.531 gen. 267 sıfır-varyanslı gen filtrelendi — bunlar correlation PCA'da sıfıra bölme hatası üretirdi. Geri kalanlar için log10 ölçekte dağılıma bakınca medyan varyans ~0.016: çoğu gen düşük bilgi taşıyor, az sayıda gen yüksek varyanslı."

---

### **Bölüm 3.3 — TCGA Covariance PCA** *(3 dakika)*

**Göster:** Scree plot → 2D projeksiyon → **3D Plotly** (bu en etkileyici görsel).

**Söyle:** "PC1+PC2 toplam varyansın sadece ~%20-25'ini açıklıyor. Ama 2B projeksiyona bakın — 5 kanser tipi net kümeleniyor. 3D'de döndürdüğümüzde PC3'ün BRCA ile LUAD'ı ayırdığını görüyoruz." Sonra yüzde eşik tablosunu göster: "%90 PVE için kaç component gerekiyor?"

---

### **Bölüm 3.4 — Covariance vs Correlation Yan Yana** *(1 dakika)*

**Göster:** İki scatter yan yana.

**Söyle:** "Correlation PCA'da tüm genler eşitlenince varyans düzleşiyor — aynı %80 PVE için çok daha fazla bileşen gerekiyor. Bu biyolojik bilginin silinmesinin bedeli."

---

### **Bölüm 3.5 — Loading Analizi** *(2 dakika)*

**Göster:** PC1 ve PC2 loading bar chart'ları.

**Söyle:** "PC1 yüklemeleri dengelidir — her biri ~0.01 mertebesinde, binlerce genin küçük katkısı. Bu bir 'genel transkripsiyonel imza' eksenidir. PC2 daha konsantre — daha spesifik bir biyolojik eksen. Top-20 overlabı az, çünkü ortogonalite zorunluluğu farklı genlerin baskın olmasını sağlıyor. Gerçek bir çalışmada bu genler KEGG/Reactome'a götürülürdü."

---

### **Bölüm 3.6 — PCA + KNN Eğrisi** *(2 dakika)*

**Göster:** Error bar'lı KNN accuracy eğrisi, ham veri baseline'ıyla.

**Söyle:** "Burası projenin pratik cevabı. 20.000 genle başladık, sadece k=X component'le aynı ya da daha iyi accuracy elde ettik — %99.X boyut azaltma. PCA görselleştirme oyuncağı değil, gerçek bir feature extraction tekniği."

---

### **Bölüm 4.2 — WDBC Cohen's d** *(1-2 dakika)*

**Göster:** Cohen's d bar chart.

**Söyle:** "`*_worst` suffix'li özellikler d>1.2 ile en güçlü ayırt ediciler. Biyolojik yorum: malign tümörler en kötü hücrelerinde karakteristik — ortalama değil, uç istatistikler ayırt edici. Bu aynı zamanda neden PCA'nın PC1'inin `*_worst` özelliklerine yüksek loading verdiğini açıklıyor."

---

### **Bölüm 4.3 — Biplot** *(3 dakika — sunumun görsel klimaksı)*

**Göster:** Biplot.

**Söyle:** "Biplot gözlemleri ve özellikleri aynı PC uzayında birleştiriyor — R kitabındaki Şekil 14.3'ün Python versiyonu. Üç okuma kuralı: ok uzunluğu PC katkısını, yön korelasyon yapısını, nokta-ok hizalaması özellik değerini söylüyor. `radius/perimeter/area` okları neredeyse paralel — `area ∝ radius²` matematiksel bağıntısının PCA'daki yansıması. Kırmızı noktalar `*_worst` oklarının yönüne toplanıyor — Cohen's d ile tutarlı."

---

### **Bölüm 5 — Marchenko–Pastur** *(3-4 dakika — projenin teorik klimaksı)*

**Göster:** Eigenvalue histogram (mavi gürültü + kırmızı TCGA) + iki dikey kesik çizgi.

**Söyle:** "Soru şu: TCGA'da PC1+PC2 neden sadece %20 varyans açıklıyor? Cevap rastgele matris teorisinde. $c = p/n \approx 25.6$ için saf gürültü bile $[16.5, 36.7]$ aralığında eigenvalue üretiyor — bu Marchenko-Pastur bulk. 20.000 genin çoğu gürültüden ibaret. Gerçek sinyal sadece bulk'un üstüne çıkan spike'larda. TCGA'da beklenen spike sayısı 5 kanser tipi + alt sınıflar nedeniyle 5-15 arası — gözlemlediğimiz sayı bununla tutarlı. Eğer çok fazla spike görseydk batch effect şüphemiz olurdu."

Sonra Soru 2'yi anlat: $\mathrm{Var}_\text{total} = \mathrm{Var}_\text{within} + \mathrm{Var}_\text{between}$. "Toplam varyansın %80'i gürültüde, ama sınıf ayrımı between-class varyansında — ve PCA bu varyansın peşine gidiyor."

---

### **Bölüm 5 — Soru 3: Covariance vs Correlation Tablosu** *(1 dakika)*

**Göster:** `cov_vs_cor_df` tablosu.

**Söyle:** "Evrensel kural yok. TCGA'da covariance kazanıyor çünkü varyans biyolojik bilgi. WDBC'de correlation kazanıyor çünkü ölçek farklılıkları var. Karar semantiğe bağlı."

---

### **Bölüm 6.3 — Kapanış** *(1 dakika)*

**Göster:** 8 maddelik lineer cebir kavramları listesi.

**Söyle:** "Eigendecomposition, SVD, ortogonal projeksiyon, basis change, rank-k approximation, quadratic form maximization, Lagrange multipliers, trace özdeşlikleri — bunlar bu projede soyut kavramlar değil, gerçek kanser veri seti üzerinde çalışan araçlar oldu."

---

### Hazırlıklı Olunması Gereken Sorular

Jüri büyük ihtimalle şu sorulardan birini soracak:

**"PCA neden eigendecomposition yerine SVD kullanıyor?"** → Bölüm 1.3. $\kappa(\tilde{X})^2$ vs $\kappa(\tilde{X})$, 25 kat hız farkı.

**"İşaret belirsizliğini nasıl çözdünüz?"** → Bölüm 2.2. `sign_pattern = np.sign(skl/ours)` ve aligned_diff.

**"TCGA'da %20 PVE neden yeterli?"** → Bölüm 5 Soru 2. $\mathrm{Var} = \mathrm{Var}_{W} + \mathrm{Var}_{B}$ ayrışması.

**"Biplot'ta okların ölçeği nasıl seçildi?"** → `scale_x = 3.0 × np.sqrt(pve[0]) × max_score_range`. PVE ile ağırlıklı ölçekleme, büyük PVE'li PC'nin okları daha uzun.

**"Covariance vs correlation PCA karar kuralı nedir?"** → Bölüm 1.5 tablosu + Bölüm 5 Soru 3 özet kutusu.