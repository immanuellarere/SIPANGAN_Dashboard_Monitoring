import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import branca.colormap as cm

from gtnnwr_wrapper import GTNNWRWrapper   # wrapper GTNNWR


# --------------------------
# Konfigurasi Halaman
# --------------------------
st.set_page_config(
    page_title="Kekuatan Data — Dashboard IKP",
    layout="wide"
)

st.title("SIPANGAN Dashboard Monitoring")


# --------------------------
# Load Dataset Lokal
# --------------------------
DATA_PATH = "datasec.xlsx"   # pastikan nama file sesuai

try:
    if DATA_PATH.endswith(".csv"):
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_excel(DATA_PATH, engine="openpyxl")

    # Preprocessing
    df = df.fillna(0)
    df = df.rename(columns=lambda x: x.strip().replace(" ", "_"))

except Exception as e:
    st.error(f"Gagal membaca dataset SEC 2025: {e}")
    st.stop()


# --------------------------
# Preview Data
# --------------------------
st.subheader("Data Preview")
st.dataframe(df.head())


# --------------------------
# Peta IKP per Provinsi (Full Width)
# --------------------------
st.write("---")
st.subheader("Peta Indonesia — IKP per Provinsi")

try:
    tahun_list = sorted(df["Tahun"].unique())
    tahun_peta = st.selectbox("Pilih Tahun untuk Peta", tahun_list)

    df_filtered = df[df["Tahun"] == tahun_peta]

    # Load GeoJSON provinsi Indonesia
    url = "https://raw.githubusercontent.com/ans-4175/peta-indonesia-geojson/master/indonesia-prov.geojson"
    gdf = gpd.read_file(url)

    # Samakan nama provinsi
    gdf["Provinsi"] = gdf["Propinsi"].str.title()
    df_filtered["Provinsi"] = df_filtered["Provinsi"].str.title()

    # Range IKP
    ikp_min = df_filtered["IKP"].min()
    ikp_max = df_filtered["IKP"].max()
    st.write(f"Range IKP ({tahun_peta}): {ikp_min:.2f} – {ikp_max:.2f}")

    bins = [0, 37.61, 48.27, 57.11, 65.96, 74.40, max(100, ikp_max + 1)]

    colormap = cm.StepColormap(
        colors=['#4B0000', '#FF3333', '#FF9999',
                '#CCFF99', '#66CC66', '#006600'],
        vmin=bins[0], vmax=bins[-1], index=bins,
        caption="Indeks Ketahanan Pangan (IKP)"
    )

    m = folium.Map(location=[-2.5, 118], zoom_start=5)

    # Gabungkan IKP ke GeoJSON
    gdf = gdf.merge(df_filtered[["Provinsi", "IKP"]], on="Provinsi", how="left")

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
        tooltip=folium.GeoJsonTooltip(fields=["Provinsi", "IKP"])
    ).add_to(m)

    colormap.add_to(m)
    st_folium(m, width=1000, height=600)

except Exception as e:
    st.error(f"Gagal memuat peta: {e}")


# --------------------------
# 🔹 Bagian GTNNWR
# --------------------------
st.write("---")
st.subheader("Analisis GTNNWR")

try:
    x_columns = [c for c in df.columns if c not in
                 ["Provinsi", "Tahun", "IKP", "Longitude", "Latitude", "id"]]

    if "gtnnwr_results" not in st.session_state:
        model = GTNNWRWrapper(x_columns, y_column="IKP")
        st.info("⏳ Melatih model GTNNWR... (butuh waktu sebentar)")
        results = model.fit(df)
        st.session_state["gtnnwr_results"] = results
        st.success("✅ Model selesai dilatih")
    else:
        results = st.session_state["gtnnwr_results"]

    if not results:
        st.warning("⚠️ Tidak ada hasil evaluasi dari GTNNWR.")
    else:
        st.write("### Hasil Evaluasi GTNNWR")
        st.json(results)

except Exception as e:
    st.error(f"Gagal menjalankan GTNNWR: {e}")


# --------------------------
# 🔹 Detail Provinsi + Simulasi
# --------------------------
st.write("---")
prov = st.selectbox("Pilih Provinsi untuk Detail", df["Provinsi"].unique())
prov_data = df[df["Provinsi"] == prov].iloc[0]

col3, col4 = st.columns([2, 1])

with col3:
    st.write(f"### {prov} — Indeks Ketahanan Pangan")
    for col in df.columns:
        if col not in ["Provinsi", "Tahun", "Longitude", "Latitude", "id"]:
            st.metric(col, prov_data[col])

with col4:
    st.subheader("Scenario & Simulation")
    variabel = st.selectbox("Pilih Variabel", [c for c in df.columns if c not in ["Provinsi", "IKP", "Tahun", "Longitude", "Latitude", "id"]])
    perubahan = st.number_input("Perubahan (%)", value=5, step=1)

    if st.button("Run Simulation"):
        # Jalankan GTNNWR
        x_columns = [c for c in df.columns if c not in ["Provinsi", "IKP", "Tahun", "Longitude", "Latitude", "id"]]
        model = GTNNWRWrapper(x_columns)
        results = model.fit(df)

        ikp_now = prov_data["IKP"]
        ikp_simulasi = ikp_now * (1 + perubahan / 100)

        st.metric("IKP saat ini", f"{ikp_now:.2f}")
        st.metric("IKP simulasi", f"{ikp_simulasi:.2f}")
        st.caption("Catatan: simulasi ini masih sederhana. Untuk akurasi penuh, hubungkan output GTNNWR.")

        # Plot hasil GTNNWR
        if results and "R2" in results:
            st.subheader("Prediksi GTNNWR")
            try:
                y_true = df["IKP"].values
                y_pred = model.model.predict(test_dataset=None)  # NOTE: sesuaikan jika wrapper punya predict
                fig, ax = plt.subplots()
                ax.scatter(y_true, y_pred, alpha=0.6)
                ax.set_xlabel("True IKP")
                ax.set_ylabel("Predicted IKP")
                st.pyplot(fig)
            except Exception:
                st.warning("Plot prediksi tidak tersedia.")


# --------------------------
# Export PDF (placeholder)
# --------------------------
st.download_button("📥 Export PDF", "Fitur export PDF akan ditambahkan", file_name="laporan.pdf")
