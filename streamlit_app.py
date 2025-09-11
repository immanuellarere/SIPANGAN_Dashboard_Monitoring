import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import branca.colormap as cm

from gtnnwr_wrapper import GTNNWRWrapper   # wrapper GTNNWR buatanmu


# --------------------------
# Konfigurasi Halaman
# --------------------------
st.set_page_config(
    page_title="SIPANGAN Dashboard Monitoring",
    layout="wide"
)

st.title("📊 SIPANGAN Dashboard Monitoring")
st.caption("Monitoring Indeks Ketahanan Pangan (IKP) berbasis GTNNWR Pretrained")


# --------------------------
# Load Dataset
# --------------------------
DATA_PATH = "datasec.xlsx"
MODEL_PATH = "./GTNNWR_DSi_model.pth"

try:
    if DATA_PATH.endswith(".csv"):
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_excel(DATA_PATH, engine="openpyxl")

    df = df.fillna(0)
    df = df.rename(columns=lambda x: x.strip().replace(" ", "_"))
    df["id"] = range(len(df))

except Exception as e:
    st.error(f"❌ Gagal membaca dataset: {e}")
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
# Analisis GTNNWR (Load dari .pth)
# --------------------------
st.write("---")
st.subheader("🤖 Analisis GTNNWR (Load Pretrained)")

coef_df = None
try:
    x_columns = [c for c in df.columns if c not in
                 [prov_col, "Tahun", "IKP", "Longitude", "Latitude", "id"]]

    model = GTNNWRWrapper(x_columns, y_column="IKP")
    model.load_pretrained(df, MODEL_PATH)

    coef_df = model.get_coefs()

    if coef_df is not None and not coef_df.empty:
        coef_filtered = coef_df[(coef_df["Tahun"] >= 2019) & (coef_df["Tahun"] <= 2024)]
        st.subheader("📑 Koefisien GTNNWR per Provinsi (2019–2024)")
        st.dataframe(coef_filtered)

        st.download_button(
            label="📥 Download Koefisien 2019–2024 (CSV)",
            data=coef_filtered.to_csv(index=False).encode("utf-8"),
            file_name="koefisien_2019_2024.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ Tidak ada koefisien yang bisa ditampilkan.")

except Exception as e:
    st.error(f"❌ Gagal menjalankan GTNNWR: {e}")


# --------------------------
# Detail Provinsi
# --------------------------
st.write("---")
st.subheader("📍 Detail Provinsi")

prov = st.selectbox("Pilih Provinsi", df[prov_col].unique())
prov_data = df[df[prov_col] == prov].copy()

prov_data["Tahun"] = prov_data["Tahun"].astype(int)
prov_data = prov_data.sort_values("Tahun")

st.write(f"### {prov} — Tren IKP 2019–2024")

# Layout tabel + grafik
col1, col2 = st.columns([1, 2])

with col1:
    st.dataframe(prov_data[["Tahun", "IKP"]].reset_index(drop=True))

with col2:
    st.line_chart(
        prov_data.set_index("Tahun")[["IKP"]],
        height=300,
        use_container_width=True
    )

# --------------------------
# Kartu indikator per tahun
# --------------------------
tahun_opsi = sorted(prov_data["Tahun"].unique())
tahun_pilih = st.selectbox("Pilih Tahun", tahun_opsi)

row = prov_data[prov_data["Tahun"] == tahun_pilih].iloc[0]

indikator_cols = [
    "Skor_PPH", "Luas_Panen", "Produktivitas", "Tanah_Longsor",
    "Banjir", "Kekeringan", "Kebakaran", "OPD_Penggerek_Batang_Padi",
    "OPD_Wereng_Batang_Coklat", "OPD_Tikus", "OPD_Blas",
    "OPD_Hwar_Daun", "OPD_Tungro", "IKP"
]

st.write(f"### {prov} — Detail Indikator {tahun_pilih}")

cols = st.columns(3)
for i, col_name in enumerate(indikator_cols):
    with cols[i % 3]:
        st.markdown(
            f"""
            <div style="border:1px solid #ddd; border-radius:8px; padding:10px; margin-bottom:10px; background:#f9f9f9">
                <b>{col_name.replace('_',' ')}</b><br>
                <span style="font-size:18px; color:#2c3e50">{row[col_name]}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
