import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ================================
# CONFIG & THEME
# ================================
st.set_page_config(
    page_title="Sistem Rekomendasi Wisata Yogyakarta",
    page_icon="🏯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif;
}

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border-left: 6px solid #e94560;
}

.main-header h1 {
    color: #ffffff;
    font-size: 2rem;
    margin: 0;
}

.main-header p {
    color: #a8b2d8;
    margin: 0.5rem 0 0 0;
    font-size: 0.95rem;
}

.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    border-top: 3px solid #e94560;
}

.metric-card h3 {
    color: #e94560;
    font-size: 2rem;
    margin: 0;
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
}

.metric-card p {
    color: #a8b2d8;
    margin: 0.3rem 0 0 0;
    font-size: 0.85rem;
}

.cluster-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    border-left: 5px solid #e94560;
}

.cluster-card h3 {
    color: #e2e8f0;
    margin: 0 0 0.5rem 0;
    font-size: 1.1rem;
}

.cluster-card p {
    color: #a8b2d8;
    margin: 0.2rem 0;
    font-size: 0.88rem;
}

.rec-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 1.3rem;
    margin-bottom: 0.8rem;
    border-left: 5px solid #e94560;
}

.rec-card h3 {
    color: #e2e8f0 !important;
    margin: 0 0 0.4rem 0;
    font-size: 1rem;
}

.rec-card p {
    color: #a8b2d8;
    margin: 0.15rem 0;
    font-size: 0.85rem;
}

.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.3rem;
}

.badge-budaya { background: #2d1b69; color: #c4b5fd; }
.badge-alam { background: #14532d; color: #86efac; }
.badge-bahari { background: #0c4a6e; color: #7dd3fc; }
.badge-hiburan { background: #7c2d12; color: #fdba74; }
.badge-belanja { background: #4a1942; color: #f0abfc; }

.stMetric {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 1rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}

section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

.sidebar-info {
    background: rgba(233, 69, 96, 0.1);
    border: 1px solid #e94560;
    border-radius: 8px;
    padding: 0.8rem;
    margin-top: 1rem;
    font-size: 0.82rem;
    color: #a8b2d8;
}
</style>
""", unsafe_allow_html=True)

# ================================
# LOAD DATA
# ================================
@st.cache_data
def load_data():
    tour_url = "https://raw.githubusercontent.com/AwalDinz/rekomendasi-wisata-ml/main/dataset/tour.csv"
    rating_url = "https://raw.githubusercontent.com/AwalDinz/rekomendasi-wisata-ml/main/dataset/tour_rating.csv"
    user_url = "https://raw.githubusercontent.com/AwalDinz/rekomendasi-wisata-ml/main/dataset/user.csv"

    tour = pd.read_csv(tour_url)
    rating = pd.read_csv(rating_url)
    user = pd.read_csv(user_url)
    return tour, rating, user

tour, rating, user = load_data()

# ================================
# FEATURE ENGINEERING UNTUK CLUSTERING
# ================================
@st.cache_data
def build_user_features(tour, rating, user):
    # Gabungkan rating dengan info tempat wisata
    merged = rating.merge(tour[['Place_Id', 'Category', 'Price']], on='Place_Id', how='left')

    # Fitur 1: Jumlah tempat yang dirating per user
    jumlah_rated = rating.groupby('User_Id')['Place_Id'].count().reset_index()
    jumlah_rated.columns = ['User_Id', 'Jumlah_Tempat_Dirating']

    # Fitur 2: Rata-rata rating yang diberikan per user
    avg_rating = rating.groupby('User_Id')['Place_Ratings'].mean().reset_index()
    avg_rating.columns = ['User_Id', 'Avg_Rating']

    # Fitur 3: Kategori favorit per user (kategori dengan rating terbanyak)
    cat_rating = merged.groupby(['User_Id', 'Category'])['Place_Ratings'].count().reset_index()
    fav_cat = cat_rating.loc[cat_rating.groupby('User_Id')['Place_Ratings'].idxmax()]
    fav_cat = fav_cat[['User_Id', 'Category']].rename(columns={'Category': 'Kategori_Favorit'})

    # Fitur 4: Rata-rata harga tiket yang dikunjungi per user
    avg_price = merged.groupby('User_Id')['Price'].mean().reset_index()
    avg_price.columns = ['User_Id', 'Avg_Harga']

    # Gabungkan semua fitur dengan data user (Age)
    user_features = user[['User_Id', 'Age']].copy()
    user_features = user_features.merge(jumlah_rated, on='User_Id', how='left')
    user_features = user_features.merge(avg_rating, on='User_Id', how='left')
    user_features = user_features.merge(fav_cat, on='User_Id', how='left')
    user_features = user_features.merge(avg_price, on='User_Id', how='left')
    user_features = user_features.fillna(0)

    return user_features

user_features = build_user_features(tour, rating, user)

# ================================
# K-MEANS CLUSTERING
# ================================
@st.cache_data
def run_kmeans(user_features, n_clusters=3):
    # Encode kategori favorit
    le = LabelEncoder()
    user_features_encoded = user_features.copy()
    user_features_encoded['Kategori_Encoded'] = le.fit_transform(
        user_features_encoded['Kategori_Favorit'].astype(str)
    )

    # Pilih fitur numerik untuk clustering
    fitur_clustering = ['Age', 'Jumlah_Tempat_Dirating', 'Avg_Rating', 'Avg_Harga', 'Kategori_Encoded']
    X = user_features_encoded[fitur_clustering].values

    # Normalisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Silhouette Score
    sil_score = silhouette_score(X_scaled, labels)

    # PCA untuk visualisasi 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    return labels, sil_score, X_pca, X_scaled, scaler, le, fitur_clustering

# Elbow Method
@st.cache_data
def elbow_method(user_features):
    le = LabelEncoder()
    user_features_encoded = user_features.copy()
    user_features_encoded['Kategori_Encoded'] = le.fit_transform(
        user_features_encoded['Kategori_Favorit'].astype(str)
    )
    fitur_clustering = ['Age', 'Jumlah_Tempat_Dirating', 'Avg_Rating', 'Avg_Harga', 'Kategori_Encoded']
    X = user_features_encoded[fitur_clustering].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = []
    sil_scores = []
    K_range = range(2, 9)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, km.labels_))

    return list(K_range), inertias, sil_scores

# ================================
# REKOMENDASI BERBASIS CLUSTER
# ================================
@st.cache_data
def get_cluster_recommendations(tour, rating, cluster_labels, user_features, n_rec=5):
    user_features_copy = user_features.copy()
    user_features_copy['Cluster'] = cluster_labels

    recs = {}
    for cluster_id in sorted(user_features_copy['Cluster'].unique()):
        cluster_users = user_features_copy[user_features_copy['Cluster'] == cluster_id]['User_Id'].tolist()
        cluster_ratings = rating[rating['User_Id'].isin(cluster_users)]
        top_places = (
            cluster_ratings.groupby('Place_Id')['Place_Ratings']
            .agg(['mean', 'count'])
            .reset_index()
        )
        top_places.columns = ['Place_Id', 'Avg_Rating', 'Jumlah_Rating']
        top_places = top_places[top_places['Jumlah_Rating'] >= 2]
        top_places = top_places.sort_values('Avg_Rating', ascending=False).head(n_rec)
        top_places = top_places.merge(tour, on='Place_Id', how='left')
        recs[cluster_id] = top_places

    return recs

# ================================
# LABEL CLUSTER OTOMATIS
# ================================
def get_cluster_label(cluster_id, user_features_clustered):
    cluster_data = user_features_clustered[user_features_clustered['Cluster'] == cluster_id]
    fav = cluster_data['Kategori_Favorit'].mode()[0] if len(cluster_data) > 0 else "Umum"
    avg_age = cluster_data['Age'].mean()

    label_map = {
        'Budaya': '🏛️ Pecinta Wisata Budaya',
        'Taman Hiburan': '🎡 Wisatawan Hiburan',
        'Cagar Alam': '🌿 Pecinta Alam',
        'Bahari': '🌊 Wisatawan Bahari',
        'Pusat Perbelanjaan': '🛍️ Wisatawan Belanja',
    }
    return label_map.get(fav, f'🗺️ Cluster {cluster_id}')

# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.markdown("### 🏯 Panel Kontrol")
    st.divider()
    menu = st.radio("Pilih Menu:", [
        "📊 Dashboard Analisis",
        "🔍 Clustering Pengguna",
        "🎯 Cari Rekomendasi"
    ])
    st.divider()

    n_clusters = st.slider("Jumlah Cluster (K)", min_value=2, max_value=6, value=3)

    st.markdown("""
    <div class='sidebar-info'>
    <b>📌 Metode:</b><br>
    K-Means Clustering + Cosine Similarity<br><br>
    <b>📁 Dataset:</b><br>
    126 Destinasi Wisata<br>
    300 Pengguna<br>
    2870 Data Rating
    </div>
    """, unsafe_allow_html=True)

# ================================
# HEADER
# ================================
st.markdown("""
<div class='main-header'>
    <h1>🏯 Sistem Rekomendasi Tempat Wisata Yogyakarta</h1>
    <p>Penerapan K-Means Clustering untuk Segmentasi Pengguna — Mata Kuliah Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ================================
# TAB 1: DASHBOARD ANALISIS
# ================================
if menu == "📊 Dashboard Analisis":

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-card'><h3>{len(tour)}</h3><p>Total Destinasi Wisata</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'><h3>{len(user)}</h3><p>Total Pengguna</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'><h3>{len(rating)}</h3><p>Total Rating</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='metric-card'><h3>{tour['Category'].nunique()}</h3><p>Kategori Wisata</p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏆 Top 10 Wisata Terpopuler")
        top10 = rating.merge(tour, on='Place_Id')[['Place_Name']].value_counts().head(10).reset_index()
        top10.columns = ['Place_Name', 'Jumlah_Rating']
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        bars = ax.barh(top10['Place_Name'], top10['Jumlah_Rating'],
                       color=['#e94560' if i == 0 else '#0f3460' for i in range(len(top10))])
        ax.set_xlabel('Jumlah Rating', color='#a8b2d8')
        ax.tick_params(colors='#a8b2d8', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#0f3460')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("🗂️ Distribusi Kategori Wisata")
        cat_data = tour['Category'].value_counts()
        colors = ['#e94560', '#0f3460', '#533483', '#2ec4b6', '#ff9f1c', '#e71d36']
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        wedges, texts, autotexts = ax.pie(
            cat_data,
            labels=cat_data.index,
            autopct='%1.1f%%',
            colors=colors[:len(cat_data)],
            startangle=140,
            textprops={'color': '#e2e8f0', 'fontsize': 9}
        )
        for at in autotexts:
            at.set_color('#ffffff')
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("⭐ Distribusi Rating Pengguna")
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    rating_counts = rating['Place_Ratings'].value_counts().sort_index()
    ax.bar(rating_counts.index, rating_counts.values, color='#e94560', width=0.6, alpha=0.9)
    ax.set_xlabel('Nilai Rating', color='#a8b2d8')
    ax.set_ylabel('Jumlah', color='#a8b2d8')
    ax.tick_params(colors='#a8b2d8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#0f3460')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("👥 Distribusi Usia Pengguna")
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    ax.hist(user['Age'], bins=15, color='#0f3460', edgecolor='#e94560', alpha=0.9)
    ax.set_xlabel('Usia', color='#a8b2d8')
    ax.set_ylabel('Jumlah Pengguna', color='#a8b2d8')
    ax.tick_params(colors='#a8b2d8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#0f3460')
    plt.tight_layout()
    st.pyplot(fig)

# ================================
# TAB 2: CLUSTERING PENGGUNA
# ================================
elif menu == "🔍 Clustering Pengguna":

    st.subheader("🔍 K-Means Clustering — Segmentasi Pengguna")

    # Elbow Method
    st.markdown("#### 📈 Elbow Method & Silhouette Score")
    K_range, inertias, sil_scores = elbow_method(user_features)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        ax.plot(K_range, inertias, marker='o', color='#e94560', linewidth=2, markersize=8)
        ax.set_xlabel('Jumlah Cluster (K)', color='#a8b2d8')
        ax.set_ylabel('Inertia', color='#a8b2d8')
        ax.set_title('Elbow Method', color='#e2e8f0')
        ax.tick_params(colors='#a8b2d8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#0f3460')
        ax.axvline(x=n_clusters, color='#533483', linestyle='--', alpha=0.7, label=f'K={n_clusters}')
        ax.legend(labelcolor='#a8b2d8', facecolor='#1a1a2e')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        ax.plot(K_range, sil_scores, marker='s', color='#2ec4b6', linewidth=2, markersize=8)
        ax.set_xlabel('Jumlah Cluster (K)', color='#a8b2d8')
        ax.set_ylabel('Silhouette Score', color='#a8b2d8')
        ax.set_title('Silhouette Score per K', color='#e2e8f0')
        ax.tick_params(colors='#a8b2d8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#0f3460')
        ax.axvline(x=n_clusters, color='#533483', linestyle='--', alpha=0.7, label=f'K={n_clusters}')
        ax.legend(labelcolor='#a8b2d8', facecolor='#1a1a2e')
        plt.tight_layout()
        st.pyplot(fig)

    # Jalankan Clustering
    labels, sil_score, X_pca, X_scaled, scaler, le, fitur = run_kmeans(user_features, n_clusters)
    user_features_clustered = user_features.copy()
    user_features_clustered['Cluster'] = labels

    st.markdown(f"#### ✅ Hasil Clustering dengan K = {n_clusters}")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>{sil_score:.3f}</h3>
            <p>Silhouette Score<br><small>(Semakin mendekati 1, semakin baik)</small></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Jumlah Anggota per Cluster:**")
        cluster_counts = user_features_clustered['Cluster'].value_counts().sort_index()
        for cid, count in cluster_counts.items():
            label = get_cluster_label(cid, user_features_clustered)
            st.markdown(f"""
            <div class='cluster-card'>
                <h3>{label}</h3>
                <p>👥 {count} pengguna</p>
                <p>📍 Cluster {cid}</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("**Visualisasi Cluster (PCA 2D):**")
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        cluster_colors = ['#e94560', '#2ec4b6', '#ff9f1c', '#533483', '#86efac', '#f0abfc']
        for cid in range(n_clusters):
            mask = labels == cid
            label_name = get_cluster_label(cid, user_features_clustered)
            ax.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                c=cluster_colors[cid],
                label=label_name,
                alpha=0.7,
                s=60,
                edgecolors='white',
                linewidths=0.3
            )
        ax.set_xlabel('PCA Komponen 1', color='#a8b2d8')
        ax.set_ylabel('PCA Komponen 2', color='#a8b2d8')
        ax.set_title('Visualisasi Cluster Pengguna', color='#e2e8f0')
        ax.tick_params(colors='#a8b2d8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#0f3460')
        legend = ax.legend(facecolor='#1a1a2e', labelcolor='#e2e8f0', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

    # Karakteristik Cluster
    st.markdown("#### 📊 Karakteristik Tiap Cluster")
    cluster_summary = user_features_clustered.groupby('Cluster').agg(
        Jumlah_User=('User_Id', 'count'),
        Rata_Usia=('Age', 'mean'),
        Rata_Tempat_Dirating=('Jumlah_Tempat_Dirating', 'mean'),
        Rata_Rating=('Avg_Rating', 'mean'),
        Rata_Harga=('Avg_Harga', 'mean'),
        Kategori_Favorit=('Kategori_Favorit', lambda x: x.mode()[0])
    ).reset_index()

    cluster_summary['Rata_Usia'] = cluster_summary['Rata_Usia'].round(1)
    cluster_summary['Rata_Tempat_Dirating'] = cluster_summary['Rata_Tempat_Dirating'].round(1)
    cluster_summary['Rata_Rating'] = cluster_summary['Rata_Rating'].round(2)
    cluster_summary['Rata_Harga'] = cluster_summary['Rata_Harga'].round(0).astype(int)
    cluster_summary['Label'] = cluster_summary['Cluster'].apply(
        lambda x: get_cluster_label(x, user_features_clustered)
    )

    st.dataframe(
        cluster_summary[['Label', 'Jumlah_User', 'Rata_Usia', 'Rata_Tempat_Dirating',
                          'Rata_Rating', 'Rata_Harga', 'Kategori_Favorit']],
        use_container_width=True,
        hide_index=True
    )

# ================================
# TAB 3: CARI REKOMENDASI
# ================================
elif menu == "🎯 Cari Rekomendasi":

    st.subheader("🎯 Rekomendasi Wisata Berdasarkan Cluster")

    # Jalankan clustering
    labels, sil_score, X_pca, X_scaled, scaler, le, fitur = run_kmeans(user_features, n_clusters)
    user_features_clustered = user_features.copy()
    user_features_clustered['Cluster'] = labels

    # Rekomendasi per cluster
    cluster_recs = get_cluster_recommendations(tour, rating, labels, user_features, n_rec=5)

    col1, col2 = st.columns(2)
    with col1:
        target_user = st.selectbox("👤 Pilih User ID", sorted(user['User_Id'].tolist()))
    with col2:
        n_rec = st.slider("📋 Jumlah Rekomendasi", 3, 10, 5)

    if st.button("✨ Tampilkan Rekomendasi", use_container_width=True):
        user_cluster = user_features_clustered[user_features_clustered['User_Id'] == target_user]['Cluster'].values

        if len(user_cluster) == 0:
            st.error("User tidak ditemukan dalam data.")
        else:
            cid = user_cluster[0]
            cluster_label = get_cluster_label(cid, user_features_clustered)
            user_info = user_features_clustered[user_features_clustered['User_Id'] == target_user].iloc[0]

            st.success(f"User {target_user} masuk ke **{cluster_label}** (Cluster {cid})")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""<div class='metric-card'><h3>{int(user_info['Age'])}</h3><p>Usia</p></div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class='metric-card'><h3>{int(user_info['Jumlah_Tempat_Dirating'])}</h3><p>Tempat Dirating</p></div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""<div class='metric-card'><h3>{user_info['Avg_Rating']:.1f}</h3><p>Rata-rata Rating</p></div>""", unsafe_allow_html=True)

            st.markdown(f"#### 🏆 Rekomendasi Wisata untuk {cluster_label}")

            recs = cluster_recs.get(cid, pd.DataFrame())

            # Exclude places already rated by user
            user_rated = rating[rating['User_Id'] == target_user]['Place_Id'].tolist()
            recs = recs[~recs['Place_Id'].isin(user_rated)].head(n_rec)

            if recs.empty:
                # Fallback: tampilkan top rated di cluster tanpa filter
                recs = cluster_recs.get(cid, pd.DataFrame()).head(n_rec)

            if not recs.empty:
                for _, row in recs.iterrows():
                    cat = str(row.get('Category', ''))
                    badge_class = {
                        'Budaya': 'badge-budaya',
                        'Cagar Alam': 'badge-alam',
                        'Bahari': 'badge-bahari',
                        'Taman Hiburan': 'badge-hiburan',
                        'Pusat Perbelanjaan': 'badge-belanja'
                    }.get(cat, 'badge-budaya')

                    price = int(row.get('Price', 0)) if pd.notna(row.get('Price')) else 0
                    avg_r = round(row.get('Avg_Rating', 0), 2)

                    st.markdown(f"""
                    <div class='rec-card'>
                        <h3>🏛️ {row.get('Place_Name', '-')}</h3>
                        <p>📍 {row.get('City', '-')} &nbsp;|&nbsp; <span class='badge {badge_class}'>{cat}</span></p>
                        <p>🎟️ Tiket: Rp {price:,} &nbsp;|&nbsp; ⭐ Rating Cluster: {avg_r}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Rekomendasi tidak tersedia untuk cluster ini.")

    st.divider()
    st.markdown("#### 📋 Semua Rekomendasi per Cluster")
    for cid, recs in cluster_recs.items():
        label = get_cluster_label(cid, user_features_clustered)
        with st.expander(f"{label} — Top 5 Rekomendasi"):
            if not recs.empty:
                display_cols = ['Place_Name', 'Category', 'City', 'Price', 'Avg_Rating', 'Jumlah_Rating']
                available = [c for c in display_cols if c in recs.columns]
                st.dataframe(recs[available], use_container_width=True, hide_index=True)
            else:
                st.info("Tidak ada rekomendasi untuk cluster ini.")

st.markdown("---")
st.caption("Final Project — Machine Learning | Universitas Amikom Yogyakarta 2026")
