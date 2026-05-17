# Sıkça Sorulabilecek Sorular (FAQ) — `main_presentation_linoalgo.ipynb`

> Bu doküman, **Cancer Genetics için Principal Component Analysis** sunumu sırasında hem teknik (jüri, hocalar) hem de teknik olmayan dinleyicilerden gelebilecek olası soruları ve bu sorulara verilebilecek kapsamlı ama özlü cevapları toplar. Her cevap üç katmanlıdır: *Teknik Açıklama*, *Matematiksel Arka Plan* ve *Gerçek Hayattan Örnek*.

---

### Soru 1: PCA tam olarak hangi problemi çözüyor? Bir cümle ile özetleyebilir misiniz?

- **Teknik Açıklama:** PCA, $p$ boyutlu verinin **kovaryans matrisinin eigendecomposition**'ını (veya verinin SVD'sini) hesaplayarak, verinin **varyansını maksimize eden ortogonal yönler** bulur. İlk birkaç yöne projeksiyon, orijinal bilginin büyük kısmını çok daha az boyutta tutar.
- **Matematiksel Arka Plan:** Problem $v_1 = \arg\max_{\|v\|=1} v^\top S v$ kısıtlı maksimizasyonudur; Lagrangian türetildiğinde $Sv = \lambda v$ eigenvalue problemine eşdeğerdir. En büyük $\lambda_1$, en yüksek varyanslı PC'nin varyansıdır.
- **Gerçek Hayattan Örnek:** Bir fotoğraf stüdyosu, çekim açısını seçerken modelin yüzünün **en geniş profilini gösteren** açıyı seçer. PCA tam olarak bunu yapar: verinin "en geniş profilini" gösteren açıyı bulur, sonra ona dik en geniş ikinci açıyı seçer, böyle devam eder.

---

### Soru 2: Neden sadece varyansı yüksek olan birkaç özelliği seçmek yerine PCA gibi karmaşık bir yöntem kullanıyoruz?

- **Teknik Açıklama:** Sadece tek tek özelliklere bakmak (**feature selection**) var olan eksenleri filtreler; bilgi tek bir gende değil, **binlerce genin lineer kombinasyonunda** saklıysa kaybolur. PCA ise **feature extraction** yapar: tamamen yeni eksenler üretir.
- **Matematiksel Arka Plan:** PCA, $v \in \mathbb{R}^p$ uzayındaki **tüm yönler** arasında varyansı maksimize eden lineer kombinasyonu bulur — yani $\mathrm{PC}_1 = \sum_{j=1}^{p} v_{1,j} \cdot \text{gene}_j$. Notebook'taki 6 rastgele gen-çifti grid'i, bilgi taşıyan 20 geni rastgele yakalama olasılığının pratikte sıfır olduğunu gösterir.
- **Gerçek Hayattan Örnek:** Bir müzik karması (mix) hayal edin: hiçbir enstrüman tek başına şarkıyı anlatmaz, ama doğru oranlarda birleştirildiğinde melodi ortaya çıkar. PCA, "hangi enstrümanların hangi oranlarla karıştığında en zengin sesin çıktığını" otomatik bulan bir prodüktör gibidir.

---

### Soru 3: Neden scikit-learn covariance matrisini hiç hesaplamıyor da SVD kullanıyor? `eig` yerine `eigh` seçiminizin önemi nedir?

- **Teknik Açıklama:** $p = 20.531$ için $S = \tilde{X}^\top\tilde{X}/(n-1)$ matrisi $p \times p \approx 3$ GB belleğe ihtiyaç duyar. SVD ise doğrudan $\tilde{X}$ üzerinde çalışır, ne $S$'i kurma ne de $X^\top X$ çarpımının numerik instabilitesini yaşar. `eigh`, simetrik matrisler için optimize edilmiştir ve **gerçek eigenvalue garantisi** verir; `eig` genel matrisler için tasarlandığından küçük sanal kısımlar dönebilir.
- **Matematiksel Arka Plan:** Kritik özdeşlik: $\lambda_k = \sigma_k^2 / (n-1)$. SVD'nin sağ singular vektörleri $V$, PCA'nın eigenvektörleridir. Conditioning sayıları açısından eigendecomp $\kappa(\tilde{X})^2$ ile, SVD ise $\kappa(\tilde{X})$ ile sınırlıdır — yani $X^\top X$ kurmak hatayı **karesel olarak şişirir**. Karmaşıklık karşılaştırması: TCGA için $\mathcal{O}(np^2) \approx 3.4 \times 10^{11}$ vs $\mathcal{O}(n^2p) \approx 1.3 \times 10^{10}$ ≈ **~25 kat hız farkı**.
- **Gerçek Hayattan Örnek:** Bir kütüphanedeki tüm kitapların hangi rafta olduğunu bulmak için iki yöntem var: (a) bütün kitapları yere döküp tek tek bakmak (eigendecomp), (b) önceden hazırlanmış bir indeks fişine bakmak (SVD). İkincisi hem hızlı hem de hata yapma olasılığı çok daha düşük.

---

### Soru 4: PCA'da işaret belirsizliği (sign ambiguity) ne demek? Kendi implementasyonunuz ile sklearn farklı sonuçlar üretirken nasıl "doğrulama" yapıyorsunuz?

- **Teknik Açıklama:** Bir PC vektörü $v$ ise, $-v$ de aynı eigenvalue'ya ait geçerli bir PC'dir. LAPACK rutinleri (NumPy ve sklearn'ün ikisi de bunu kullanır) sayısal kararlılığa göre işareti belirler; iki bağımsız çağrı farklı işaret seçebilir. Doğrulama için `sign_pattern = np.sign(scores_skl[0] / scores_ours[0])` ile sütun bazında işareti hizalıyor ve sonra **makine epsilon mertebesinde** ($\sim 10^{-13}$) eşitlik kontrol ediyoruz.
- **Matematiksel Arka Plan:** $Sv = \lambda v$ doğru ise $S(-v) = -Sv = -\lambda v = \lambda(-v)$ da doğrudur — eigenvalue problemi işarete göre **homojen**dir. Tek istisna katlı (degenerate) eigenvalue'lar — o zaman tüm eigenspace serbestçe döndürülebilir, ama TCGA gibi farklı eigenvalue'lara sahip veride bu durum oluşmaz.
- **Gerçek Hayattan Örnek:** Bir GPS, "kuzey-güney aksı boyunca git" diyebilir; ama haritayı baş aşağı çevirdiğinizde aynı aks "güney-kuzey" olur. Yön belirsizdir, ama aks aynıdır. PCA'nın PC1'i de aynı şeyi söyler: "bu eksen önemlidir", + veya − işareti sadece konvansiyondur.

---

### Soru 5: Açıklanan Varyans Oranı (PVE) ile Reconstruction Error neden tam olarak birbirini tamamlıyor? Bu Eckart–Young teoremi neyi söylüyor?

- **Teknik Açıklama:** İlk $k$ PC ile rekonstrüksiyon: $\hat{X}^{(k)} = \bar{x} + T_{:,:k} V_{:,:k}^\top$. Normalleştirilmiş Frobenius reconstruction error tam olarak $1 - \mathrm{CumPVE}(k)$'ya eşittir. Notebook'taki `identity_diff` kod bloğu bu eşitliği makine epsilon mertebesinde sayısal olarak doğrulayarak gösterir.
- **Matematiksel Arka Plan:** Eckart–Young teoremi, **Frobenius norma altında en iyi rank-$k$ yaklaşımının ilk $k$ singular value/vector ile verilen yaklaşım olduğunu** garanti eder:
  $$\|\tilde{X} - \hat{X}^{(k)}\|_F^2 = \sum_{i>k}\sigma_i^2 = (n-1)\sum_{i>k}\lambda_i.$$
  Bu iki yorumun (varyansı maksimize etme ↔ hatayı minimize etme) **matematiksel olarak eşdeğer** olduğunu söyler.
- **Gerçek Hayattan Örnek:** Bir JPEG sıkıştırması düşünün: dosya boyutunu azaltmak ile görüntü kalitesini korumak aynı paranın iki yüzüdür. Sıkıştırma oranını arttırırsanız (daha az "PC" tutarsanız), dosya küçülür ama kayıp artar — aralarındaki ilişki matematiksel olarak tam ve tahmin edilebilir.

---

### Soru 6: TCGA verisinde PC1+PC2 toplam varyansın sadece ~%20'sini açıklıyor, buna rağmen 5 kanser tipi 2B'de net ayrışıyor. Bu çelişki değil mi?

- **Teknik Açıklama:** Çelişki değil — **toplam varyans ile sınıf ayırt etme gücü farklı şeylerdir**. PCA, sınıf etiketinden habersiz olarak sadece toplam varyansı maksimize eder; ancak TCGA'da farklı kanser tipleri kökten farklı transkripsiyonel programlar çalıştırdığı için **sınıflar-arası varyans, sınıf-içi varyandan çok daha büyüktür** ve ilk birkaç PC istemeden de olsa sınıfsal eksenleri yakalar.
- **Matematiksel Arka Plan:** Varyans ayrışması: $\mathrm{Var}_\text{toplam} = \mathrm{Var}_\text{within} + \mathrm{Var}_\text{between}$. TCGA'da paydadaki gen-içi gürültü (20.000+ genin küçük varyansları) toplam varyansı şişirir, bu yüzden PVE düşük görünür; ama spike eigenvalue'lar — Marchenko-Pastur bulk'un üstüne çıkan — **between-class varyansını** taşır. BBP (Baik-Ben Arous-Péché) faz geçişi: spike $> (1+\sqrt{c})^2$ ise PC sinyalin yönünü tutarlı şekilde estime eder.
- **Gerçek Hayattan Örnek:** Bir kalabalık konser salonunda iki grup insan farklı renk tişört giyiyor olsun (kırmızı vs mavi). Salonun "toplam görsel varyansının" %80'i kıyafet detaylarındaki ufak tefek farklardır (gürültü), %20'si ise tişört renkleri (sinyal). Ama sadece o %20'ye odaklandığınızda iki grubu net görürsünüz — çünkü o %20, **bilgi taşıyan** kısımdır.

---

### Soru 7: Marchenko–Pastur yasası nedir? Neden hemen TCGA verisi ile konuşurken bahsediyorsunuz?

- **Teknik Açıklama:** Marchenko-Pastur (MP), saf gürültü matrisinin ($n \times p$, IID girişler) eigenvalue dağılımının **deterministik bir desteğe** ($[\lambda_-, \lambda_+]$) yakınsadığını söyleyen rastgele matris teorisinden bir sonuçtur. TCGA'da $c = p/n \approx 25.6$ için bu sınırlar $[16.5, 36.7]$ olur. Notebook'taki histogramda mavi (saf gürültü) bu aralığa düşer; kırmızı (gerçek TCGA) aynı bulk'a sahiptir ama **bulk'un üstüne çıkan spike'lar** içerir — bunlar gerçek biyolojik sinyaldir.
- **Matematiksel Arka Plan:**
  $$\rho_{\mathrm{MP}}(\lambda) = \frac{1}{2\pi c \lambda}\sqrt{(\lambda_+-\lambda)(\lambda-\lambda_-)}, \quad \lambda_\pm = (1\pm\sqrt{c})^2.$$
  TCGA'da spike sayısı ~5–15 arasında, 5 kanser tipi + alt sınıflarla tutarlı. Eğer çok daha fazla spike görseydik **batch effect** veya teknik artefakt şüphesi doğardı — bu, gerçek genomik araştırma pratiğinde kullanılan bir kalite kontrolü adımıdır.
- **Gerçek Hayattan Örnek:** Bir radyoda "shhhh" tarzı parazit dinlersiniz — bu, **rastgele frekanslarda dağılmış gürültüdür** (MP bulk). Bir müzik istasyonuna geçtiğinizde, parazitin üstünde net olarak çıkan birkaç frekans (vokal, gitar, bas) duyarsınız — bunlar **sinyal spike'larıdır**. Sinyali gürültüden ayırt etmenin matematiği MP yasasıdır.

---

### Soru 8: Covariance PCA mı yoksa Correlation PCA mı kullanmalıyım? Kararı neye göre veriyorsunuz?

- **Teknik Açıklama:** Karar veri setinin **semantik yapısına** bağlıdır. Notebook'ta iki örnek var: TCGA'da gen değerleri aynı birimde (log-RNA-seq) ölçülmüş ve varyans biyolojik bilgi taşıyor → **covariance PCA** kazanıyor (PC1+PC2 ~%20 PVE). WDBC'de ise `area_mean` (mm²) ile `smoothness_mean` (oransız) tamamen farklı ölçeklerde → **correlation PCA** kazanıyor; covariance versiyonunda PC1 tek başına `area` ailesi tarafından domine edilip varyansın ~%98'ini "yutmuş" gibi gösteriyor ki bu yanıltıcı.
- **Matematiksel Arka Plan:** Correlation PCA, $Z = \tilde{X} D^{-1}$ ($D = \mathrm{diag}(\hat\sigma_j)$) standardizasyonu sonrası uygulanır. Bu lineer bir dönüşümdür ama rotation değildir — sonuçtaki eigenvektörler/eigenvalue'lar **birbirinden basit bir formülle elde edilemez**. Karar kuralı: özellikler **homojen+anlamlı varyans** → covariance; **heterojen birim/ölçek** → correlation.
- **Gerçek Hayattan Örnek:** Bir araç değerlendirme listesi düşünün: motor gücü (HP), yakıt tüketimi (l/100km), bagaj hacmi (litre), fiyat (₺). Eğer ham veriyle PCA yaparsanız, fiyat (milyonlar) tüm analizi domine eder — diğer özellikleri sanki yokmuş gibi gösterir. Önce her özelliği standardize ederseniz (correlation PCA), her özellik "kendi ölçeğinde adil" karşılaştırılır. Ama eğer tüm özellikler **fiyatlar** olsaydı (örn. 10 farklı ayın fiyatları), o zaman varyansların kendisi anlamlı bilgi olurdu — standardize etmek bu bilgiyi siler.

---

### Soru 9: Loadings (yüklemeler) nedir, scores'tan farkı nedir, biyolojik olarak nasıl yorumlanır?

- **Teknik Açıklama:** **Loadings** ($v_{k,j}$), her PC'nin orijinal özelliklerin lineer kombinasyonu olarak ifadesindeki katsayılardır: $\mathrm{PC}_k = \sum_j v_{k,j} \cdot \mathrm{feature}_j$. **Scores** ise her bir örneğin PC eksenlerine olan izdüşüm değerleridir. Notebook'ta TCGA PC1 yüklemeleri **dengeli** (binlerce genin küçük katkısı, ~0.01 mertebesinde — "genel transkripsiyonel imza"), PC2 ise **daha konsantredir** (daha spesifik bir eksen). Top-20 overlap'i azdır çünkü ortogonalite zorunluluğu farklı genlerin baskın olmasını sağlar.
- **Matematiksel Arka Plan:** $V \in \mathbb{R}^{p \times k}$ matrisinin sütunları loadings'i, $T = \tilde{X}V \in \mathbb{R}^{n \times k}$ matrisi scores'u verir. Ortogonalite: $V^\top V = I$, bu yüzden farklı PC'ler bağımsız bilgi taşır. WDBC'de PC1 yüklemeleri `radius/perimeter/area/concave_points` ailesine yüksek pozitif katsayı verir → "tümör büyüklüğü/şiddeti" ekseni.
- **Gerçek Hayattan Örnek:** Bir yemek tarifi düşünün. **Scores**: "bu pasta tarifinde %30 lezzet, %25 sunum, %20 doku, %25 koku puanı aldı" (bir örneğin PC eksenlerindeki konumu). **Loadings**: "lezzet skoru = 0.4×şeker + 0.3×tereyağı + 0.2×vanilya + 0.1×tuz" (bu ekseni hangi malzemelerin oluşturduğu). Pastanın puanını bilmek bir şey, pastayı oluşturan tarifi bilmek başka şey — ikisi farklı sorular.

---

### Soru 10: Biplot nedir? Notebook'taki biplot grafiğinde neye bakmalıyım?

- **Teknik Açıklama:** Biplot, **gözlemleri (scatter)** ve **özellik vektörlerini (oklar)** aynı PC uzayında birleştiren bir görselleştirmedir — R kitabındaki Şekil 14.3'ün Python karşılığı. WDBC biplot'unda `radius/perimeter/area` okları neredeyse paralel uzanır (yüksek korelasyon), `*_worst` ok yönüne kırmızı (M) noktalar toplanır (Cohen's d analiziyle tutarlı). Okların ölçeklemesi `scale_x = 3.0 × np.sqrt(pve[0]) × max_score_range` formülüyle PVE-ağırlıklı yapılır, böylece yüksek PVE'li PC'nin okları doğal olarak daha uzun çıkar.
- **Matematiksel Arka Plan:** Üç okuma kuralı: (i) **ok uzunluğu** → özelliğin PC'ye katkısı (loading büyüklüğü), (ii) **ok yönü** → özelliğin PC uzayındaki pozisyonu (aynı yöndeki oklar pozitif korelasyon, zıt yön negatif korelasyon, dik yön korelasyonsuzluk gösterir), (iii) **nokta-ok hizalaması** → o örnekte o özelliğin değerinin yüksek/düşük olduğunu söyler.
- **Gerçek Hayattan Örnek:** Bir şehir haritası düşünün: hem turistler (gözlemler/noktalar) hem de yön tabelaları (özellik okları) aynı haritada görünür. "Müze oku batıya bakıyor, bizim oradaki turistler de batıdaysa, demek ki müzeye gidiyorlar." Biplot da benzer biçimde, hangi örneğin hangi özellik yönünde "daha güçlü" olduğunu tek bakışta gösterir.

---

### Soru 11: TCGA'da 20.000 genle başladınız, en sonunda PCA+KNN ile çok daha az bileşenle benzer accuracy aldınız. Bu pratik olarak ne anlama geliyor?

- **Teknik Açıklama:** Notebook'taki PCA+KNN deneyi, $k \in \{2, 5, 10, 25, 50, 100\}$ component sayıları için 5-fold cross-validation accuracy ölçer. En iyi $k$'da ham veri baseline'ı (20.000 genle çalışan KNN) ile **karşılaştırılabilir veya daha iyi** sonuç alınır; üstelik **%99.X boyut azaltma** sağlanır. Yüksek boyutta KNN, **concentration of distances** etkisinden zarar görür — tüm noktaların birbirine "eşit uzaklıkta" görünmesi mesafe metriğini etkisizleştirir.
- **Matematiksel Arka Plan:** Yüksek boyutta $\frac{\max_j \|x_i - x_j\| - \min_j \|x_i - x_j\|}{\min_j \|x_i - x_j\|} \to 0$. PCA, **gürültü boyutlarını eleyerek** sinyal-gürültü oranını arttırır — bu yüzden boyut azaltma sonrası KNN'in mesafe metriği yeniden anlamlı hale gelir. Pratik olarak: 20.531 boyutlu uzayda K-en yakın komşu hesabı hem yavaş hem de bilgi-azdır; PC uzayında ise hem hızlı hem de doğrudur.
- **Gerçek Hayattan Örnek:** 100 maddelik bir kişilik testi düşünün. Tüm 100 maddeye birden bakarak iki kişinin "ne kadar benzediğini" söylemek zor; ama testin altında yatan 5 ana boyutu (Big Five: extraversion, openness, vs) çıkarırsanız, sadece o 5 boyutta karşılaştırmak hem daha hızlı hem de **gürültüden arınmış** bir benzerlik ölçer. PCA, "Big Five"ı veriden otomatik öğrenen yöntemdir.

---

### Soru 12: PCA'nın sınırlamaları nelerdir? Bu projede karşılaştığınız bir zayıf yönü var mı?

- **Teknik Açıklama:** PCA dört temel sınırlama taşır: (i) **lineerdir** — eğri manifoldları yakalayamaz; (ii) **outlier-duyarlıdır** çünkü varyans karesel olarak girer; (iii) **unsupervised'tır** — sınıf etiketinden habersizdir, bu yüzden ille de en iyi sınıf-ayırıcı yön en yüksek varyanslı yön olmaz; (iv) **önişleme kararına duyarlıdır** (covariance vs correlation). Notebook'ta TCGA'da spike eigenvalue'ların sınıf yapısıyla "şanslı hizalandığı" tartışılmıştır — bu hizalanma her zaman garantili değildir.
- **Matematiksel Arka Plan:** Bu sınırlamalar için alternatifler vardır: **t-SNE / UMAP** (non-lineer manifold öğrenme), **Robust PCA** ($\tilde{X} = L + S$ düşük-rank + seyrek ayrışımı, outlier-dayanıklı), **Linear Discriminant Analysis (LDA)** ($\mathrm{tr}(S_B)/\mathrm{tr}(S_W)$ oranını maksimize eder, supervised), **Sparse PCA** (loadings'i L1 düzenlileştirme ile seyrekleştirir, biyolojik yorumlanabilirliği arttırır).
- **Gerçek Hayattan Örnek:** PCA, çizgisel bir cetvel gibidir — düz mesafe ölçmekte mükemmeldir ama eğri bir yolu (örn. bir sarmal merdiveni) doğru ölçemez. Eğri yolları ölçmek için "esnek metre" (t-SNE/UMAP) gerekir. Aynı şekilde, eğer veride bir-iki çok aykırı örnek varsa, klasik cetvel onlara "yerçekimi gibi çekilir"; bunun için "darbe sönümleyici cetvel" (Robust PCA) kullanılır.

---

### Soru 13: Projeyi tek cümle ile özetlerseniz, lineer cebir dersine katkısı nedir?

- **Teknik Açıklama:** Proje, **eigendecomposition, SVD, ortogonal projeksiyon, basis change, rank-$k$ approximation, quadratic form maximization, Lagrange multipliers, trace özdeşlikleri** gibi dersin sekiz temel kavramının gerçek bir kanser veri seti üzerinde nasıl somut araçlara dönüştüğünü gösterir. Aynı algoritmanın $p = 30$ (WDBC) ve $p = 20.531$ (TCGA) gibi iki uçta nasıl farklı davrandığı, **boyutun matematiğinin uygulamayla iç içe geçmesini** ortaya koyar.
- **Matematiksel Arka Plan:** Tek bir denklem zincirinde özetlersek: $v_1 = \arg\max v^\top Sv \Rightarrow Sv = \lambda v \Leftrightarrow \tilde{X} = U\Sigma V^\top \Rightarrow \lambda_k = \sigma_k^2/(n-1) \Rightarrow \mathrm{ReconErr}(k) = 1 - \mathrm{CumPVE}(k)$. Bu zincirin her halkası hem teorik olarak türetilmiş hem de notebook'ta sayısal olarak doğrulanmıştır.
- **Gerçek Hayattan Örnek:** Bir orkestra şefinin işine benzer: tek tek müzisyenler (lineer cebir kavramları) bireysel olarak güçlüdür, ama gerçek değer **onları tek bir senfonide (gerçek veri problemi) bir araya getirmektir**. Proje, "PCA senfonisi" boyunca dersin tüm kavramlarının nasıl uyumlu çaldığını dinletir.
