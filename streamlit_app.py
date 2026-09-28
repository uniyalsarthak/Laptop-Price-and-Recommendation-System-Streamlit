import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="Laptop Price Prediction & Recommendation",
    page_icon="💻",
    layout="wide",
)

# ----------------------------------------------------------------------
# Custom CSS (clean light theme — green accent, matches original site look)
# ----------------------------------------------------------------------

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: Arial, Helvetica, sans-serif;
    }

    /* App background */
    .stApp {
        background: #ffffff;
    }

    .block-container {
        max-width: 1000px;
        padding-top: 2rem !important;
    }

    /* Title */
    .app-title {
        text-align: center;
        color: #087d3b;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .app-subtitle {
        text-align: center;
        color: #666666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    hr, [data-testid="stDivider"] {
        border-color: #dddddd !important;
    }

    /* Field labels */
    label, .stSelectbox label p {
        color: #087d3b !important;
        font-weight: bold !important;
        font-size: 0.9rem !important;
    }

    /* Selectbox widget */
    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1px solid #cccccc !important;
        border-radius: 6px !important;
        color: #000000 !important;
    }
    div[data-baseweb="select"] span {
        color: #000000 !important;
    }
    div[data-baseweb="select"]:hover {
        border-color: #2563eb !important;
    }

    /* Predict button */
    .stButton > button {
        display: block;
        width: 100%;
        background: #087d3b;
        color: #ffffff;
        font-size: 1rem;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        padding: 0.7rem 0;
        margin-top: 1rem;
        transition: 0.2s;
    }
    .stButton > button:hover {
        background: #045527;
        color: #ffffff;
    }
    .stButton > button:active {
        transform: scale(0.98);
    }

    /* Result block */
    .result-box {
        margin-top: 1.5rem;
        text-align: center;
        border-top: 1px solid #dddddd;
        padding-top: 1.6rem;
    }
    .result-box .label {
        color: #666666;
        font-size: 1.1rem;
        font-weight: normal;
        margin-bottom: 0.4rem;
    }
    .result-box .price {
        color: #087d3b;
        font-size: 2.6rem;
        font-weight: 700;
    }

    /* Recommendation section heading */
    .rec-heading {
        color: #087d3b;
        font-size: 1.15rem;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
    }

    /* Recommendation cards */
    .rec-card {
        background: #ffffff;
        border: 1px solid #dddddd;
        border-radius: 8px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
        transition: 0.2s ease;
    }
    .rec-card:hover {
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.12);
        transform: translateY(-2px);
    }
    .rec-card h4 {
        color: #096631;
        font-size: 1.05rem;
        margin-bottom: 0.5rem;
    }
    .rec-card p {
        color: #444444;
        font-size: 0.9rem;
        margin: 0.3rem 0;
        line-height: 1.4;
    }
    .rec-card .rec-price {
        color: #087d3b;
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    .rec-card .rec-similarity {
        color: #22c55e;
        font-weight: 600;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Load models (cached so they only load once per session)
# ----------------------------------------------------------------------


@st.cache_resource
def load_models():
    xgb_model = pickle.load(open("models/laptop_price_predictor.pkl", "rb"))
    recommend_preprocessor = pickle.load(
        open("models/recommend_preprocessor.pkl", "rb")
    )
    knn = pickle.load(open("models/knn_model.pkl", "rb"))
    recommend_df = pickle.load(open("models/recommend_dataset.pkl", "rb"))
    return xgb_model, recommend_preprocessor, knn, recommend_df


xgb_model, recommend_preprocessor, knn, recommend_df = load_models()

# ----------------------------------------------------------------------
# Static option data (mirrors script.js / index.html)
# ----------------------------------------------------------------------

TYPE_DEFAULT = {
    "Apple": "Ultrabook",
    "Dell": "Notebook",
    "HP": "Notebook",
    "Lenovo": "Notebook",
    "Asus": "Notebook",
    "Acer": "Notebook",
    "MSI": "Gaming",
    "Toshiba": "Notebook",
    "Samsung": "Ultrabook",
    "Huawei": "Ultrabook",
    "LG": "Ultrabook",
}

COMPANY_OPTIONS = {
    "Apple": {"cpu": ["Intel Core i5", "Intel Core i7"], "os": ["Mac"]},
    "Dell": {
        "cpu": [
            "Intel Core i3",
            "Intel Core i5",
            "Intel Core i7",
            "AMD",
            "Intel Celeron",
            "Intel Pentium",
            "Other Intel",
        ],
        "os": ["Windows", "Linux", "Other"],
    },
    "HP": {
        "cpu": ["Intel Core i3", "Intel Core i5", "Intel Core i7", "AMD"],
        "os": ["Windows", "Linux", "Other"],
    },
    "Lenovo": {
        "cpu": ["Intel Core i3", "Intel Core i5", "Intel Core i7", "AMD"],
        "os": ["Windows", "Linux", "Other"],
    },
    "Asus": {
        "cpu": ["Intel Core i3", "Intel Core i5", "Intel Core i7", "AMD"],
        "os": ["Windows", "Linux", "Other"],
    },
    "Acer": {
        "cpu": ["Intel Core i3", "Intel Core i5", "Intel Core i7", "AMD", "Intel Celeron"],
        "os": ["Windows", "Linux", "Other"],
    },
    "MSI": {"cpu": ["Intel Core i5", "Intel Core i7"], "os": ["Windows"]},
    "Toshiba": {"cpu": ["Intel Core i3", "Intel Core i5", "Intel Core i7"], "os": ["Windows"]},
    "Samsung": {"cpu": ["Intel Core i3", "Intel Core i5", "Intel Core i7"], "os": ["Windows"]},
    "Huawei": {"cpu": ["Intel Core i5", "Intel Core i7"], "os": ["Windows"]},
    "LG": {"cpu": ["Intel Core i5", "Intel Core i7"], "os": ["Windows"]},
}

INCHES_OPTIONS = [
    10.1, 11.6, 12.0, 12.3, 12.5, 13.0, 13.3, 13.5, 14.0, 15.0, 15.6, 17.0, 17.3, 18.4,
]
RESOLUTION_OPTIONS = [
    "1366x768", "1600x900", "1920x1080", "2560x1440",
    "2560x1600", "2880x1800", "3200x1800", "3840x2160",
]
RAM_OPTIONS = [2, 4, 8, 12, 16, 24, 32, 64]
SSD_OPTIONS = [0, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
HDD_OPTIONS = [0, 128, 256, 500, 512, 1000, 2000]
WEIGHT_OPTIONS = [
    0.69, 0.92, 1.00, 1.20, 1.30, 1.37, 1.50, 1.80,
    2.00, 2.20, 2.50, 2.80, 3.00, 3.50, 4.00, 4.42,
]
GPU_OPTIONS = ["Intel", "Nvidia", "AMD"]

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------

st.markdown("<div class='app-title'>Laptop Price Prediction</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='app-subtitle'>Predict laptop prices and get similar laptop recommendations.</div>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------

row1 = st.columns(4)
with row1[0]:
    company = st.selectbox("Company", list(COMPANY_OPTIONS.keys()))
with row1[1]:
    inches = st.selectbox("Screen Size (Inches)", INCHES_OPTIONS, index=INCHES_OPTIONS.index(15.6))
with row1[2]:
    resolution = st.selectbox("Resolution", RESOLUTION_OPTIONS)
with row1[3]:
    ram = st.selectbox("RAM (GB)", RAM_OPTIONS, index=RAM_OPTIONS.index(8))

row2 = st.columns(4)
with row2[0]:
    ssd = st.selectbox("SSD (GB)", SSD_OPTIONS, index=SSD_OPTIONS.index(256))
with row2[1]:
    hdd = st.selectbox("HDD (GB)", HDD_OPTIONS, index=HDD_OPTIONS.index(0))
with row2[2]:
    weight = st.selectbox("Weight (kg)", WEIGHT_OPTIONS, index=WEIGHT_OPTIONS.index(1.00))
with row2[3]:
    touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])

row3 = st.columns(4)
with row3[0]:
    ips = st.selectbox("IPS Display", ["No", "Yes"])
with row3[1]:
    cpu = st.selectbox("CPU", COMPANY_OPTIONS[company]["cpu"])
with row3[2]:
    gpu = st.selectbox("GPU", GPU_OPTIONS)
with row3[3]:
    os_choice = st.selectbox("Operating System", COMPANY_OPTIONS[company]["os"])

_, btn_col, _ = st.columns([1, 1, 1])
with btn_col:
    predict_clicked = st.button("Predict Price", use_container_width=True)

# ----------------------------------------------------------------------
# Prediction + recommendation logic (mirrors app.py's /predict route)
# ----------------------------------------------------------------------

if predict_clicked:
    typename = TYPE_DEFAULT[company]

    x_res, y_res = map(int, resolution.split("x"))
    ppi = np.sqrt(x_res**2 + y_res**2) / float(inches)

    user_df = pd.DataFrame(
        [
            {
                "Company": company,
                "TypeName": typename,
                "Ram": int(ram),
                "Weight": float(weight),
                "Touchscreen": 1 if touchscreen == "Yes" else 0,
                "Ips": 1 if ips == "Yes" else 0,
                "Cpu Brand": cpu,
                "PPI": ppi,
                "SSD": int(ssd),
                "HDD": int(hdd),
                "Flash": 0,
                "Hybrid": 0,
                "Gpu Brand": gpu,
                "OS": os_choice,
            }
        ]
    )

    # --- Price prediction ---
    log_price = float(xgb_model.predict(user_df)[0])
    predicted_price = int(round(np.exp(log_price)))

    # --- Recommendations ---
    recommend_df["Actual_Price"] = np.exp(recommend_df["Price"])

    filtered_df = recommend_df[
        (recommend_df["Ram"] == int(ram))
        & (recommend_df["SSD"] == int(ssd))
        & (recommend_df["Actual_Price"] >= predicted_price - 5000)
        & (recommend_df["Actual_Price"] <= predicted_price + 5000)
    ].copy()

    filtered_df = filtered_df.drop_duplicates(
        subset=["Company", "TypeName", "Cpu Brand", "Ram", "SSD", "Gpu Brand", "OS"]
    )

    if len(filtered_df) < 5:
        filtered_df = recommend_df.copy()

    filtered_features = recommend_preprocessor.transform(
        filtered_df[
            [
                "Company", "TypeName", "Ram", "Weight", "Touchscreen", "Ips",
                "Cpu Brand", "PPI", "SSD", "HDD", "Flash", "Hybrid",
                "Gpu Brand", "OS",
            ]
        ]
    )

    user_vector = recommend_preprocessor.transform(user_df)

    knn_temp = NearestNeighbors(n_neighbors=min(4, len(filtered_df)), metric="cosine")
    knn_temp.fit(filtered_features)
    distances, indices = knn_temp.kneighbors(user_vector)

    recommendations = filtered_df.iloc[indices[0]].copy()
    recommendations["Price"] = np.exp(recommendations["Price"]).round().astype(int)
    recommendations["Similarity"] = ((1 - distances[0]) * 100).round(2)

    # --- Display result ---
    st.markdown(
        f"""
        <div class="result-box">
            <div class="label">Predicted Price</div>
            <div class="price">₹ {predicted_price:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='rec-heading'>Similar Laptops</div>", unsafe_allow_html=True)

    rec_cols = st.columns(4)
    for i, (_, row) in enumerate(recommendations.iterrows()):
        with rec_cols[i % 4]:
            st.markdown(
                f"""
                <div class="rec-card">
                    <h4>{row['Company']} — {row['TypeName']}</h4>
                    <p>{row['Cpu Brand']}</p>
                    <p>{int(row['Ram'])} GB RAM · {int(row['SSD'])} GB SSD</p>
                    <p>{row['Gpu Brand']} · {row['OS']}</p>
                    <div class="rec-price">₹ {int(row['Price']):,}</div>
                    <div class="rec-similarity">Similarity: {row['Similarity']}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )