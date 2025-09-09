import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import branca.colormap as cm

from gtnnwr import GTNNWRWrapper   # file gtnnwr.py

# --------------------------
# Konfigurasi Halaman
# --------------------------
st.set_page_config(
    page_title="Kekuatan Data — Dashboard IKP",
    layout="wide"
)

st.title("Kekuatan Data — Dashboard IKP")
st.caption("Statistika & Teknologi untuk Indonesia Emas — Subtema: Krisis Pangan & Energi")

# --------------------------
# Upload Data
# --------------------------
uploaded_file = st.file_uploader("Upload Data SEC 2025", type=["csv", "xlsx"])

if uploaded_file:
    # Baca file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine="openpyxl")

    # Imputasi missing value dengan 0
    df = df.fillna(0)

    # Pastikan ada kolom Provinsi
    if "Provinsi" not in df.columns:
        if "ID" in df.columns:
            df = df.rename(columns={"ID": "Provinsi"})

    # Preview Data
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # --------------------------
    # Layout 2 kolom utama
    # --------------------------
    col1, col2 = st.columns([2, 1])

    # === Kolom Kiri: Peta ===
    with col1:
        st.subheader("Peta Indonesia — IKP per Provinsi")

        try:
            # Pilih Tahun
            tahun_list = sorted(df["Tahun"].unique())
            tahun_dipilih = st.selectbox("Pilih Tahun", tahun_list)

            df_filtered = df[df["Tahun"] == tahun_dipilih]

            # Load GeoJSON provinsi Indonesia
            url = "https://raw.githubusercontent.com/ans-4175/peta-indonesia-geojson/master/indonesia-prov.geojson"
            gdf = gpd.read_file(url)

            # Samakan nama provinsi
            gdf["Provinsi"] = gdf["Propinsi"].str.title()
            df_filtered["Provinsi"] = df_filtered["Provinsi"].str.title()

            # Range IKP
            ikp_min = df_filtered["IKP"].min()
            ikp_max = df_filtered["IKP"].max()
            st.write(f"Range IKP ({tahun_dipilih}): {ikp_min:.2f} – {ikp_max:.2f}")

            # Bins
            bins = [0, 37.61, 48.27, 57.11, 65.96, 74.40]
            if ikp_max > bins[-1]:
                bins.append(ikp_max + 1)
            else:
                bins.append(100)

            # Colormap
            colormap = cm.StepColormap(
                colors=['#4B0000', '#FF3333', '#FF9999', '#CCFF99', '#66CC66', '#006600'],
                vmin=bins[0],
                vmax=bins[-1],
                index=bins,
                caption="Indeks Ketahanan Pangan (IKP)"
            )

            m = folium.Map(location=[-2.5, 118], zoom_start=5)

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
            st_folium(m, width=700, height=500)

        except Exception as e:
            st.error(f"Gagal memuat peta: {e}")

    # === Kolom Kanan: Dashboard Kebijakan ===
    with col2:
        st.subheader("Policy & Action Dashboard")
        kebijakan = st.selectbox("Filter kebijakan", ["Semua", "Jangka Pendek", "Jangka Menengah", "Jangka Panjang"])
        if kebijakan == "Semua":
            st.markdown("""
            - Cadangan pangan pemerintah  
            - Pengaturan pola tanam & diversifikasi  
            - Investasi infrastruktur irigasi tahan iklim
            """)

    # --------------------------
    # Detail Provinsi + Koefisien
    # --------------------------
    st.write("---")
    prov = st.selectbox("Pilih Provinsi untuk Detail", df["Provinsi"].unique())
    prov_data = df[df["Provinsi"] == prov].iloc[0]

    st.write(f"### {prov} — Indeks Ketahanan Pangan")
    for col in df.columns:
        if col not in ["Provinsi", "Tahun"]:
            st.metric(col, prov_data[col])

    # === Jalankan GTNNWR untuk koefisien ===
    st.subheader("Koefisien Variabel (GTNNWR)")

    x_columns = [c for c in df.columns if c not in ["Provinsi", "IKP", "Tahun", "Longitude", "Latitude", "id"]]
    model = GTNNWRWrapper(x_columns)
    results = model.fit(df)

    coef_df = pd.DataFrame(list(results["coefficients"].items()), columns=["Variabel", "Koefisien"])
    st.dataframe(coef_df)

    # Plot hasil True vs Predicted
    if "true" in results and "pred" in results:
        st.subheader("True vs Predicted (GTNNWR)")
        fig, ax = plt.subplots()
        ax.scatter(results["true"], results["pred"], alpha=0.6)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)

    # --------------------------
    # Export PDF (placeholder)
    # --------------------------
    st.download_button("📥 Export PDF", "Fitur export PDF akan ditambahkan", file_name="laporan.pdf")

else:
    st.warning("Silakan upload file Data SEC 2025 (CSV/Excel).")
