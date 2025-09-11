import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import branca.colormap as cm
import altair as alt   # untuk chart interaktif

# --------------------------
# Konfigurasi Halaman
# --------------------------
st.set_page_config(
    page_title="SIPANGAN Dashboard Monitoring",
    layout="wide"
)

st.title("📊 SIPANGAN Dashboard Monitoring")
st.caption("Monitoring Indeks Ketahanan Pangan (IKP) dari data historis 2019–2024")

# --------------------------
# Load Dataset Lokal
# --------------------------
DATA_PATH = "datasec.xlsx"

try:
    if DATA_PATH.endswith(".csv"):
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_excel(DATA_PATH, engine="openpyxl")

    df = df.fillna(0)
    df = df.rename(columns=lambda x: x.strip().replace(" ", "_"))
    df["id"] = range(len(df))
except Exception as e:
    st.error(f"❌ Gagal membaca dataset SEC 2025: {e}")
    st.stop()

# --------------------------
# Tentukan kolom Provinsi
# --------------------------
if "Provinsi" in df.columns:
    prov_col = "Provinsi"
elif "Nama_Provinsi" in df.columns:
    prov_col = "Nama_Provinsi"
else:
    st.error("❌ Dataset harus punya kolom 'Provinsi' atau 'Nama_Provinsi'")
    st.stop()

# --------------------------
# Preview Data
# --------------------------
st.write("---")
st.subheader("🔍 Data Preview")
st.dataframe(df.head())

# --------------------------
# Peta IKP per Provinsi
# --------------------------
st.write("---")
st.subheader("🗺️ Peta Indonesia — IKP per Provinsi")

try:
    tahun_list = sorted(df["Tahun"].unique())
    tahun_peta = st.selectbox("Pilih Tahun untuk Peta", tahun_list)

    df_filtered = df[df["Tahun"] == tahun_peta].copy()

    url = "https://raw.githubusercontent.com/ans-4175/peta-indonesia-geojson/master/indonesia-prov.geojson"
    gdf = gpd.read_file(url)

    gdf[prov_col] = gdf["Propinsi"].str.title()
    df_filtered[prov_col] = df_filtered[prov_col].str.title()

    ikp_min, ikp_max = df_filtered["IKP"].min(), df_filtered["IKP"].max()
    st.write(f"Range IKP ({tahun_peta}): {ikp_min:.2f} – {ikp_max:.2f}")

    bins = [0, 37.61, 48.27, 57.11, 65.96, 74.40, max(100, ikp_max + 1)]

    colormap = cm.StepColormap(
        colors=['#4B0000', '#FF3333', '#FF9999',
                '#CCFF99', '#66CC66', '#006600'],
        vmin=bins[0], vmax=bins[-1], index=bins,
        caption="Indeks Ketahanan Pangan (IKP)"
    )

    m = folium.Map(location=[-2.5, 118], zoom_start=5)

    gdf = gdf.merge(df_filtered[[prov_col, "IKP"]], on=prov_col, how="left")

    def style_function(feature):
        value = feature["properties"]["IKP"]
        if value is None:
            return {"fillOpacity": 0.3, "weight": 0.5, "color": "gray"}
        return {
            "fillColor": colormap(value),
            "fillOpacity": 0.7,
            "weight": 0.5,
            "color": "black"
        }

    folium.GeoJson(
        gdf,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=[prov_col, "IKP"])
    ).add_to(m)

    colormap.add_to(m)
    st_folium(m, width=1000, height=600)

except Exception as e:
    st.error(f"❌ Gagal memuat peta: {e}")

# --------------------------
# Detail Provinsi
# --------------------------
st.write("---")
st.subheader("📍 Detail Provinsi")

prov = st.selectbox("Pilih Provinsi", df[prov_col].unique())
prov_data = df[df[prov_col] == prov].copy()

st.write(f"### {prov} — IKP 2019–2024")

# Filter tahun 2019–2024
prov_data_filtered = (
    prov_data[prov_data["Tahun"].between(2019, 2024)][["Tahun", "IKP"]]
    .copy()
    .reset_index(drop=True)
)
prov_data_filtered["Tahun"] = prov_data_filtered["Tahun"].astype(int)

# Inject CSS untuk tabel lebih rapih & font lebih besar
st.markdown(
    """
    <style>
    table {
        font-size: 18px !important;
        text-align: center !important;
        width: 100% !important;
        border-collapse: collapse !important;
    }
    th, td {
        padding: 8px 16px !important;
        text-align: center !important;
    }
    thead th {
        font-weight: bold !important;
        border-bottom: 2px solid #999 !important;
    }
    tbody tr:nth-child(even) {
        background-color: #f9f9f9 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Layout tabel + chart
col1, col2 = st.columns([1, 1.2])

with col1:
    st.table(prov_data_filtered)

with col2:
    base = alt.Chart(prov_data_filtered).encode(
        x=alt.X("Tahun:O", title="Tahun", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("IKP:Q", title="IKP")
    )

    line = base.mark_line(point=True).encode(
        tooltip=["Tahun", "IKP"]
    )

    text = base.mark_text(
        align="left", dx=5, dy=-8, fontSize=12, color="black"
    ).encode(
        text=alt.Text("IKP:Q", format=".2f")
    )

    chart = (line + text).properties(
        width=700,   # 🔥 dilebarkan (dari 500 → 700)
        height=420,
        title="IKP 5 Tahun"
    ).configure_axis(
        labelFontSize=14,
        titleFontSize=16
    ).configure_title(
        fontSize=20
    ).configure_point(
        size=70
    )

    st.altair_chart(chart, use_container_width=False)

# --------------------------
# Box indikator variabel X
# --------------------------
st.write(f"### 📊 Indikator {prov} — Tahun {tahun_pilih}")

# Pilih data sesuai tahun yang dipilih
data_tahun = prov_data[prov_data["Tahun"] == tahun_pilih].copy()

# Ambil semua kolom kecuali yang umum
exclude_cols = ["Tahun", "IKP", prov_col, "Latitude", "Longitude", "id"]
indikator_cols = [c for c in data_tahun.columns if c not in exclude_cols]

if not data_tahun.empty:
    indikator_data = data_tahun[indikator_cols].iloc[0]

    # Buat grid 3 kolom biar rapih
    cols = st.columns(3)
    for i, (label, value) in enumerate(indikator_data.items()):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div style="border:1px solid #ddd; border-radius:8px; padding:12px; margin-bottom:10px;">
                    <div style="font-size:14px; color:#555;">{label.replace('_',' ')}</div>
                    <div style="font-size:20px; font-weight:bold;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    st.warning("⚠️ Data tidak tersedia untuk tahun ini.")
