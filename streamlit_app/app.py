"""
=============================================================================
  DiabetesPredict — Interface Streamlit de Prédiction
  Design : Tesla-Inspired Dark UI
=============================================================================
"""

import json
import sys

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

sys.path.insert(0, "/app")

from pyspark.sql import SparkSession
from spark_ml.predict import load_model, predict

# =============================================================================
# CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="DiabetesPredict — Prédiction du Risque de Diabète",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS — TESLA DARK DESIGN SYSTEM
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ==============================
       BASE
    ============================== */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #171a20 !important;
        color: #ffffff !important;
        -webkit-font-smoothing: antialiased;
    }

    .main .block-container {
        padding: 2rem 2.5rem 4rem 2.5rem !important;
        max-width: 1400px !important;
        background-color: #171a20 !important;
    }

    /* ==============================
       SIDEBAR
    ============================== */
    [data-testid="stSidebar"] {
        background-color: #1e2229 !important;
        border-right: 1px solid #2e3038 !important;
    }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    [data-testid="stSidebar"] .stMetricValue { color: #ffffff !important; font-size: 1.4rem !important; font-weight: 700 !important; }
    [data-testid="stSidebar"] .stMetricLabel { color: #8a8d95 !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.08em; }

    .sb-logo {
        padding: 1rem 0 1.5rem;
        border-bottom: 1px solid #2e3038;
        margin-bottom: 1.5rem;
    }
    .sb-logo-name {
        font-size: 1.1rem; font-weight: 800;
        color: #ffffff !important; letter-spacing: -0.02em;
    }
    .sb-logo-sub {
        font-size: 0.72rem; color: #e82127 !important;
        font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em;
        margin-top: 3px;
    }
    .sb-sep { font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: #3e4148 !important; margin: 1.5rem 0 0.75rem; }
    .sb-kpi-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 0.75rem; }
    .sb-kpi {
        background: #23262d; border: 1px solid #2e3038; border-radius: 10px;
        padding: 0.85rem 0.8rem;
    }
    .sb-kpi-val { font-size: 1.4rem; font-weight: 800; color: #ffffff !important; letter-spacing: -0.03em; line-height: 1; }
    .sb-kpi-lbl { font-size: 0.62rem; color: #8a8d95 !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 3px; }
    .model-tag {
        background: linear-gradient(90deg, rgba(232,33,39,0.12), rgba(232,33,39,0.04));
        border: 1px solid rgba(232,33,39,0.35); border-radius: 10px;
        padding: 0.8rem 1rem; margin-bottom: 1rem;
    }
    .model-tag-lbl { font-size: 0.6rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: #e82127 !important; }
    .model-tag-val { font-size: 0.95rem; font-weight: 700; color: #ffffff !important; margin-top: 2px; }

    /* ==============================
       HERO BANNER
    ============================== */
    .tesla-hero {
        background: linear-gradient(135deg, #1e2229 0%, #171a20 60%, #1a1215 100%);
        border: 1px solid #2e3038; border-radius: 20px;
        padding: 3rem 3.5rem; margin-bottom: 2rem;
        position: relative; overflow: hidden;
    }
    .tesla-hero::before {
        content: '';
        position: absolute; top: 0; right: 0;
        width: 400px; height: 100%;
        background: radial-gradient(ellipse at 80% 50%, rgba(232,33,39,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-tag {
        display: inline-block;
        background: #e82127; color: #ffffff;
        font-size: 0.65rem; font-weight: 800;
        text-transform: uppercase; letter-spacing: 0.15em;
        padding: 4px 12px; border-radius: 4px; margin-bottom: 1.25rem;
    }
    .hero-title {
        font-size: 3rem; font-weight: 900;
        line-height: 1.05; letter-spacing: -0.04em;
        color: #ffffff; margin-bottom: 0.75rem;
    }
    .hero-title .red { color: #e82127; }
    .hero-sub {
        font-size: 1rem; color: #8a8d95; font-weight: 400;
        line-height: 1.6; max-width: 600px; margin-bottom: 2rem;
    }
    .hero-stats { display: flex; gap: 2.5rem; }
    .h-stat-val { font-size: 2rem; font-weight: 800; color: #ffffff; letter-spacing: -0.04em; line-height: 1; }
    .h-stat-lbl { font-size: 0.73rem; color: #8a8d95; font-weight: 500; margin-top: 4px; }
    .h-stat-div { width: 1px; background: #2e3038; height: 40px; align-self: center; }

    /* ==============================
       SECTION HEADERS
    ============================== */
    .ts-section {
        display: flex; align-items: center; gap: 12px;
        margin: 2rem 0 1.25rem;
    }
    .ts-section-bar { width: 4px; height: 1.4rem; background: #e82127; border-radius: 2px; }
    .ts-section-title { font-size: 1.05rem; font-weight: 700; color: #ffffff; letter-spacing: -0.02em; }

    /* ==============================
       FORM CARDS
    ============================== */
    .ts-card {
        background: #1e2229; border: 1px solid #2e3038; border-radius: 14px;
        padding: 1.5rem; margin-bottom: 0;
        transition: border-color 0.2s;
    }
    .ts-card:hover { border-color: #3e4148; }
    .ts-card-head {
        font-size: 0.75rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.1em;
        color: #8a8d95; border-bottom: 1px solid #2e3038;
        padding-bottom: 0.75rem; margin-bottom: 1.25rem;
        display: flex; align-items: center; gap: 8px;
    }
    .ts-card-head .dot { width: 6px; height: 6px; border-radius: 50%; background: #e82127; }

    /* ==============================
       WIDGETS OVERRIDES  
    ============================== */
    label, .stSlider > label {
        color: #d0d2d6 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }
    .stSelectbox > div > div {
        background: #23262d !important;
        border: 1px solid #3e4148 !important;
        border-radius: 8px !important;
        color: #ffffff !important;
    }
    .stSlider > div > div > div > div { background-color: #e82127 !important; }
    p { color: #d0d2d6 !important; }

    /* ==============================
       BOUTON D'ANALYSE
    ============================== */
    .stButton > button {
        background: #e82127 !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        padding: 0.85rem 3rem !important;
        border-radius: 8px !important;
        border: none !important;
        letter-spacing: 0.02em !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 20px rgba(232,33,39,0.3) !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background: #c41b21 !important;
        box-shadow: 0 6px 30px rgba(232,33,39,0.5) !important;
        transform: translateY(-2px) !important;
    }

    /* ==============================
       RÉSULTATS
    ============================== */
    .r-card {
        border-radius: 14px; padding: 1.75rem 2rem;
        border: 1px solid; margin: 1rem 0;
    }
    .r-danger { background: rgba(232,33,39,0.06); border-color: rgba(232,33,39,0.3); }
    .r-success { background: rgba(52,199,89,0.06); border-color: rgba(52,199,89,0.3); }
    .r-badge {
        display: inline-block; font-size: 0.65rem; font-weight: 800;
        text-transform: uppercase; letter-spacing: 0.12em;
        padding: 3px 10px; border-radius: 4px; margin-bottom: 0.75rem;
    }
    .r-badge-danger { background: #e82127; color: #fff; }
    .r-badge-success { background: #34c759; color: #fff; }
    .r-title { font-size: 1.6rem; font-weight: 800; letter-spacing: -0.03em; margin-bottom: 0.5rem; }
    .r-danger .r-title { color: #ff8080 !important; }
    .r-success .r-title { color: #5edd7e !important; }
    .r-body { font-size: 0.9rem; color: #8a8d95 !important; line-height: 1.65; }

    /* ==============================
       PROBABILITÉ DISPLAY
    ============================== */
    .prob-display {
        background: #1e2229; border: 1px solid #2e3038;
        border-radius: 14px; padding: 2rem; text-align: center;
    }
    .prob-num { font-size: 4.5rem; font-weight: 900; letter-spacing: -0.06em; line-height: 1; }
    .prob-lbl { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #8a8d95; margin-top: 0.5rem; }
    .prob-danger .prob-num { color: #e82127; }
    .prob-success .prob-num { color: #34c759; }

    /* ==============================
       INSIGHTS CARD
    ============================== */
    .insight-card {
        background: #1e2229; border: 1px solid #2e3038;
        border-radius: 12px; padding: 1.5rem;
    }
    .insight-title { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #8a8d95; margin-bottom: 1rem; }
    .insight-item { display: flex; justify-content: space-between; align-items: center; padding: 0.6rem 0; border-bottom: 1px solid #2e3038; }
    .insight-item:last-child { border-bottom: none; }
    .i-feat { font-size: 0.85rem; font-weight: 500; color: #d0d2d6; }
    .i-val { font-size: 0.85rem; font-weight: 700; color: #e82127; }

    /* ==============================
       HR & MISC
    ============================== */
    hr { border-color: #2e3038 !important; }
    .stAlert { background: #1e2229 !important; border-radius: 10px !important; }

    /* ==============================
       FOOTER
    ============================== */
    .ts-footer {
        border-top: 1px solid #2e3038; margin-top: 4rem;
        padding: 2rem 0 1rem; text-align: center;
        color: #3e4148; font-size: 0.78rem; line-height: 1.9;
    }
    .ts-footer a { color: #8a8d95; text-decoration: none; }
    .ts-footer a:hover { color: #ffffff; }

</style>
""", unsafe_allow_html=True)

# =============================================================================
# SPARK & MODÈLE — CHARGEMENT UNIQUE EN CACHE
# =============================================================================
@st.cache_resource
def init_spark():
    spark = SparkSession.builder \
        .appName("DiabetesPredict-Inference") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

@st.cache_resource
def get_model(_spark):
    try: return load_model(_spark, "/app/model/best_model")
    except: return None

@st.cache_data
def load_metrics():
    try:
        with open("/app/model/metrics.json", "r") as f: return json.load(f)
    except: return None

spark   = init_spark()
model   = get_model(spark)
metrics = load_metrics()

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-name">🩺 DiabetesPredict</div>
        <div class="sb-logo-sub">Moteur IA de risque médical</div>
    </div>
    """, unsafe_allow_html=True)

    if metrics:
        best_name = metrics.get("best_model", "Inconnu")
        best      = metrics.get("best_metrics", {})

        st.markdown('<div class="sb-sep">🏆 Modèle sélectionné</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="model-tag">
            <div class="model-tag-lbl">Meilleur par AUC-ROC</div>
            <div class="model-tag-val">{best_name}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-sep">📊 Métriques de performance</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sb-kpi-row">
            <div class="sb-kpi"><div class="sb-kpi-val">{best.get('accuracy',0)*100:.1f}%</div><div class="sb-kpi-lbl">Précision</div></div>
            <div class="sb-kpi"><div class="sb-kpi-val">{best.get('auc_roc',0)*100:.1f}%</div><div class="sb-kpi-lbl">AUC-ROC</div></div>
            <div class="sb-kpi"><div class="sb-kpi-val">{best.get('f1_score',0)*100:.1f}%</div><div class="sb-kpi-lbl">F1-Score</div></div>
            <div class="sb-kpi"><div class="sb-kpi-val">{best.get('training_time_sec',0):.0f}s</div><div class="sb-kpi-lbl">Durée entraîn.</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Comparaison modèles
        all_models = metrics.get("all_models", {})
        if all_models:
            st.markdown('<div class="sb-sep">📈 Comparaison des modèles</div>', unsafe_allow_html=True)
            noms = list(all_models.keys())
            noms_courts = ["Rég. Log.", "Forêt Al.", "GBT"]
            aucs = [all_models[m]["auc_roc"] * 100 for m in noms]
            cols_bar = ["#3e4148", "#3e4148", "#e82127"]

            fig_bar = go.Figure()
            for n, a, c in zip(noms_courts, aucs, cols_bar):
                fig_bar.add_trace(go.Bar(x=[n], y=[a], marker_color=c, marker_line_width=0,
                                         text=[f"{a:.1f}%"], textposition="outside",
                                         textfont=dict(color="#8a8d95", size=9)))
            fig_bar.update_layout(
                height=165, showlegend=False, barmode="group",
                margin=dict(l=0,r=0,t=20,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, tickfont=dict(color="#8a8d95", size=10)),
                yaxis=dict(range=[78,88], gridcolor="#2e3038", tickfont=dict(size=8, color="#3e4148")),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        ds = metrics.get("dataset_info", {})
        if ds:
            st.markdown('<div class="sb-sep">📋 Données CDC</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size:0.8rem; line-height:2.2; color:#8a8d95;">
                Patients &ensp;<b style="color:#d0d2d6;">{ds.get('total_rows',0):,}</b><br>
                Variables &ensp;<b style="color:#d0d2d6;">{ds.get('total_features',0)}</b><br>
                Entraînement &ensp;<b style="color:#d0d2d6;">{ds.get('train_rows',0):,}</b><br>
                Test &ensp;<b style="color:#d0d2d6;">{ds.get('test_rows',0):,}</b>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("En attente des données d'entraînement...")

# =============================================================================
# HERO BANNER
# =============================================================================
st.markdown("""
<div class="tesla-hero">
    <div class="hero-tag">Moteur de Prédiction IA · Apache Spark ML</div>
    <div class="hero-title">Évaluez votre risque de<br><span class="red">diabète en quelques secondes.</span></div>
    <div class="hero-sub">
        Un modèle de Machine Learning entraîné sur <b style="color:#d0d2d6;">253 680 patients</b> de l'étude CDC analysera
        vos indicateurs de santé pour estimer votre risque de développer un diabète de type 2.
    </div>
    <div class="hero-stats">
        <div>
            <div class="h-stat-val">253 680</div>
            <div class="h-stat-lbl">Patients analysés</div>
        </div>
        <div class="h-stat-div"></div>
        <div>
            <div class="h-stat-val">86.4%</div>
            <div class="h-stat-lbl">Précision du modèle</div>
        </div>
        <div class="h-stat-div"></div>
        <div>
            <div class="h-stat-val">82.8%</div>
            <div class="h-stat-lbl">Score AUC-ROC</div>
        </div>
        <div class="h-stat-div"></div>
        <div>
            <div class="h-stat-val">21</div>
            <div class="h-stat-lbl">Indicateurs cliniques</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Modèle non disponible. Le service d'entraînement est peut-être encore en cours d'initialisation.")
    st.stop()

# =============================================================================
# FORMULAIRE PATIENT
# =============================================================================
st.markdown("""
<div class="ts-section">
    <div class="ts-section-bar"></div>
    <div class="ts-section-title">Profil du Patient — Saisie des données cliniques</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown('<div class="ts-card"><div class="ts-card-head"><span class="dot"></span>Indicateurs Cliniques</div>', unsafe_allow_html=True)
    bmi           = st.slider("Indice de Masse Corporelle (IMC)", 10, 100, 25, help="Normale : 18.5 – 24.9")
    high_bp       = st.selectbox("Hypertension artérielle",        [0,1], format_func=lambda x: "✔ Oui" if x else "✖ Non")
    high_chol     = st.selectbox("Cholestérol élevé",              [0,1], format_func=lambda x: "✔ Oui" if x else "✖ Non")
    chol_check    = st.selectbox("Contrôle cholestérol (5 ans)",   [0,1], format_func=lambda x: "✔ Oui" if x else "✖ Non", index=1)
    smoker        = st.selectbox("Tabagisme",                      [0,1], format_func=lambda x: "✔ Oui" if x else "✖ Non")
    stroke        = st.selectbox("Antécédent d'AVC",               [0,1], format_func=lambda x: "✔ Oui" if x else "✖ Non")
    heart_disease = st.selectbox("Maladie cardiovasculaire",       [0,1], format_func=lambda x: "✔ Oui" if x else "✖ Non")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="ts-card"><div class="ts-card-head"><span class="dot"></span>Habitudes & Mode de Vie</div>', unsafe_allow_html=True)
    phys_activity = st.selectbox("Activité physique régulière",    [0,1], format_func=lambda x: "✔ Actif(ve)" if x else "✖ Sédentaire", index=1)
    fruits        = st.selectbox("Consommation de fruits / jour",  [0,1], format_func=lambda x: "✔ Oui" if x else "✖ Non", index=1)
    veggies       = st.selectbox("Consommation de légumes / jour", [0,1], format_func=lambda x: "✔ Oui" if x else "✖ Non", index=1)
    heavy_alcohol = st.selectbox("Alcool en excès",                [0,1], format_func=lambda x: "✔ Oui" if x else "✖ Non")
    any_healthcare= st.selectbox("Couverture médicale",            [0,1], format_func=lambda x: "✔ Oui" if x else "✖ Non", index=1)
    no_doc_cost   = st.selectbox("Renoncé au médecin (coût)",      [0,1], format_func=lambda x: "✔ Oui" if x else "✖ Non")
    diff_walk     = st.selectbox("Difficulté à marcher / monter",  [0,1], format_func=lambda x: "✔ Oui" if x else "✖ Non")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="ts-card"><div class="ts-card-head"><span class="dot"></span>Profil Socio-Démographique</div>', unsafe_allow_html=True)
    gen_hlth  = st.slider("Santé générale (1 = Excellent · 5 = Mauvaise)", 1, 5, 3)
    ment_hlth = st.slider("Jours de mal-être mental (/ 30 jours)",         0, 30, 0)
    phys_hlth = st.slider("Jours de mal-être physique (/ 30 jours)",       0, 30, 0)
    sex       = st.selectbox("Sexe biologique", [0,1], format_func=lambda x: "👨 Homme" if x else "👩 Femme")
    age       = st.slider("Tranche d'âge  (1 = 18-24 ans · 13 = 80+)",    1, 13, 5, help="1: 18-24, 2: 25-29, ..., 13: 80+")
    education = st.slider("Niveau d'éducation  (1 = Aucune · 6 = Bac+5)", 1, 6, 4)
    income    = st.slider("Revenu annuel  (1 = < 10k$ · 8 = ≥ 75k$)",     1, 8, 5)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# BOUTON D'ANALYSE
# =============================================================================
st.markdown("<br>", unsafe_allow_html=True)
col_btn = st.columns([1, 2, 1])
with col_btn[1]:
    analyse = st.button("🔬 Lancer l'analyse du risque de diabète", use_container_width=True)

# =============================================================================
# RÉSULTATS
# =============================================================================
if analyse:
    input_dict = {
        "HighBP": float(high_bp), "HighChol": float(high_chol), "CholCheck": float(chol_check),
        "BMI": float(bmi), "Smoker": float(smoker), "Stroke": float(stroke),
        "HeartDiseaseorAttack": float(heart_disease), "PhysActivity": float(phys_activity),
        "Fruits": float(fruits), "Veggies": float(veggies), "HvyAlcoholConsump": float(heavy_alcohol),
        "AnyHealthcare": float(any_healthcare), "NoDocbcCost": float(no_doc_cost),
        "GenHlth": float(gen_hlth), "MentHlth": float(ment_hlth), "PhysHlth": float(phys_hlth),
        "DiffWalk": float(diff_walk), "Sex": float(sex), "Age": float(age),
        "Education": float(education), "Income": float(income),
    }

    with st.spinner("Analyse en cours via le pipeline Spark ML..."):
        try:
            prediction, probability = predict(spark, model, input_dict)

            st.markdown("""
            <div class="ts-section" style="margin-top:2.5rem;">
                <div class="ts-section-bar"></div>
                <div class="ts-section-title">Résultats de l'Analyse — Rapport de Risque</div>
            </div>
            """, unsafe_allow_html=True)

            r1, r2 = st.columns([3, 1], gap="medium")
            with r1:
                # Récupération du nom du meilleur modèle pour l'affichage dynamique
                best_model_name = metrics.get("best_model", "Modèle IA") if metrics else "Modèle IA"
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class="r-card r-danger">
                        <div class="r-badge r-badge-danger">⚠ Risque élevé détecté</div>
                        <div class="r-title">Marqueurs de diabète<br>identifiés dans ce profil.</div>
                        <div class="r-body">
                            Le modèle <b>{best_model_name}</b> a détecté des patterns statistiques associés à un risque
                            de pré-diabète ou de diabète de type 2. Ce résultat ne constitue pas un diagnostic médical —
                            une consultation auprès d'un professionnel de santé est <b style="color:#ff6b6b;">fortement recommandée.</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="r-card r-success">
                        <div class="r-badge r-badge-success">✔ Aucun risque significatif</div>
                        <div class="r-title">Profil clinique favorable<br>selon le modèle IA.</div>
                        <div class="r-body">
                            Aucun indicateur clinique significatif de diabète n'a été identifié dans ce profil.
                            Il est tout de même recommandé de maintenir un <b style="color:#5edd7e;">suivi médical régulier</b>
                            et un mode de vie actif pour préserver ce statut.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with r2:
                pc = "prob-danger" if prediction == 1 else "prob-success"
                cl = "#e82127" if prediction == 1 else "#34c759"
                st.markdown(f"""
                <div class="prob-display {pc}">
                    <div class="prob-num" style="color:{cl};">{probability * 100:.1f}%</div>
                    <div class="prob-lbl">Probabilité de diabète</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Graphiques Analytiques ─────────────────────────────────────
            st.markdown("""
            <div class="ts-section" style="margin-top:2rem;">
                <div class="ts-section-bar"></div>
                <div class="ts-section-title">Analyse Graphique du Profil</div>
            </div>
            """, unsafe_allow_html=True)

            gc1, gc2 = st.columns(2, gap="medium")

            bar_color = "#e82127" if prediction == 1 else "#34c759"

            with gc1:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": "%", "font": {"size": 36, "color": "#ffffff", "family": "Inter"}},
                    title={"text": "Score de Risque Global", "font": {"size": 13, "color": "#8a8d95"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#2e3038", "tickfont": {"size": 9, "color": "#3e4148"}},
                        "bar": {"color": bar_color, "thickness": 0.22},
                        "bgcolor": "#23262d",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0,  33], "color": "rgba(52, 199, 89, 0.07)"},
                            {"range": [33, 66], "color": "rgba(255, 150, 0, 0.07)"},
                            {"range": [66,100], "color": "rgba(232, 33, 39, 0.07)"},
                        ],
                        "threshold": {"line": {"color": bar_color, "width": 3}, "thickness": 0.85, "value": probability * 100},
                    }
                ))
                fig_g.update_layout(
                    height=290, margin=dict(l=20,r=20,t=50,b=20),
                    paper_bgcolor="#1e2229", font=dict(family="Inter"),
                )
                st.plotly_chart(fig_g, use_container_width=True)

            with gc2:
                risk_factors = {
                    "IMC": min(bmi/50, 1.0), "Hypertension": float(high_bp),
                    "Cholestérol": float(high_chol), "Tabac": float(smoker),
                    "Santé Gén.": (gen_hlth-1)/4, "Âge": (age-1)/12,
                    "Sédentarité": 1-float(phys_activity), "Cardio": float(heart_disease)
                }
                cats = list(risk_factors.keys())
                vals = list(risk_factors.values()) + [list(risk_factors.values())[0]]
                fill = f"rgba({'232,33,39' if prediction==1 else '52,199,89'}, 0.12)"

                fig_r = go.Figure(go.Scatterpolar(
                    r=vals, theta=cats+[cats[0]], fill="toself", fillcolor=fill,
                    line=dict(color=bar_color, width=2.5),
                ))
                fig_r.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2e3038", linecolor="#2e3038", tickfont=dict(size=8, color="#3e4148")),
                        angularaxis=dict(gridcolor="#2e3038", linecolor="#2e3038", tickfont=dict(color="#8a8d95", size=11)),
                        bgcolor="#1e2229"
                    ),
                    showlegend=False,
                    height=290, margin=dict(l=55,r=55,t=20,b=20),
                    paper_bgcolor="#1e2229", font=dict(family="Inter"),
                )
                st.plotly_chart(fig_r, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur lors du calcul : {e}")

# =============================================================================
# FEATURE IMPORTANCES
# =============================================================================
if metrics and metrics.get("feature_importances"):
    fi = metrics["feature_importances"]
    mn = metrics.get("best_model", "Inconnu")

    st.markdown("""
    <div class="ts-section" style="margin-top:2rem;">
        <div class="ts-section-bar"></div>
        <div class="ts-section-title">Transparence du Modèle — Variables les plus influentes</div>
    </div>
    """, unsafe_allow_html=True)

    fi_col1, fi_col2 = st.columns([3, 2], gap="medium")

    with fi_col1:
        fi_df = pd.DataFrame(list(fi.items()), columns=["Variable", "Importance"]) \
                  .sort_values("Importance", ascending=True).tail(12)
        n = len(fi_df)
        bar_colors = [f"rgba(232, 33, 39, {0.25 + 0.75*(i/max(n-1,1))})" for i in range(n)]

        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Variable"], orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{v*100:.1f}%" for v in fi_df["Importance"]],
            textposition="outside", textfont=dict(color="#3e4148", size=10),
        ))
        fig_fi.update_layout(
            height=400, margin=dict(l=0, r=60, t=10, b=0),
            paper_bgcolor="#1e2229", plot_bgcolor="#1e2229",
            font=dict(family="Inter", color="#8a8d95"),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(title="", tickfont=dict(size=11, color="#d0d2d6"), gridcolor="#23262d"),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with fi_col2:
        # Top 5 summary card
        top5 = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:5]
        items_html = "".join([
            f'<div class="insight-item"><span class="i-feat">#{i+1} — {feat}</span><span class="i-val">{score*100:.1f}%</span></div>'
            for i, (feat, score) in enumerate(top5)
        ])
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">Top 5 des facteurs déterminants ({mn})</div>
            {items_html}
            <p style="font-size:0.78rem; color:#3e4148; margin-top:1rem; line-height:1.6;">
                Ces variables ont le plus grand poids statistique dans les décisions du modèle
                Gradient Boosted Trees, conformément aux données épidémiologiques du CDC américain.
            </p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div class="ts-footer">
    <strong style="color:#8a8d95;">DiabetesPredict</strong> &mdash; Système d'aide à la décision médicale basé sur l'IA<br>
    Données : <a href="https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators" target="_blank">CDC Diabetes Health Indicators</a>
    · 253 680 patients · 21 variables cliniques · Sources : CDC / UCI ML Repository<br>
    <span style="font-size:0.7rem; color:#2e3038;">⚠️ Usage strictement académique. Ne se substitue en aucun cas à un diagnostic médical professionnel.</span>
</div>
""", unsafe_allow_html=True)
