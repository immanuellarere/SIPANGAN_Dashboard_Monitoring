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

    # Preprocessing
    df = df.fillna(0)
    df = df.rename(columns=lambda x: x.strip().replace(" ", "_"))

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
            tahun_list = sorted(df["Tahun"].unique())
            tahun_peta = st.selectbox("Pilih Tahun untuk Peta", tahun_list)

            df_filtered = df[df["Tahun"] == tahun_peta]

            # Load GeoJSON provinsi Indonesia
            url = "https://raw.githubusercontent.com/ans-4175/peta-indonesia-geojson/master/indonesia-prov.geojson"
            gdf = gpd.read_file(url)

            # Samakan nama provinsi
            gdf["Provinsi"] = gdf["Propinsi"].str.title()
            df_filtered["Provinsi"] = df_filtered["Provinsi"].str.title()

            # Cek range IKP
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
            st_folium(m, width=700, height=500)

        except Exception as e:
            st.error(f"Gagal memuat peta: {e}")

    # === Kolom Kanan: Dashboard Kebijakan ===
    with col2:
        st.subheader("Policy & Action Dashboard")
        kebijakan = st.selectbox(
            "Filter kebijakan",
            ["Semua", "Jangka Pendek", "Jangka Menengah", "Jangka Panjang"]
        )
        if kebijakan == "Semua":
            st.markdown("""
            - Cadangan pangan pemerintah  
            - Pengaturan pola tanam & diversifikasi  
            - Investasi infrastruktur irigasi tahan iklim
            """)

        st.subheader("Quick Tips")
        st.markdown("""
        - Pilih 2-3 provinsi untuk membandingkan indikator kunci.  
        - Gunakan Scenario untuk menguji perubahan Luas Panen, Produktivitas, atau Cadangan Pangan.
        """)

    # --------------------------
    # 🔹 Bagian GTNNWR (di bawah peta)
    # --------------------------
    st.write("---")
    st.subheader("Analisis Koefisien GTNNWR")

    try:
        # Tentukan variabel prediktor
        x_columns = [c for c in df.columns if c not in
                     ["Provinsi", "Tahun", "IKP", "Longitude", "Latitude", "id"]]

        # Latih model sekali (cache di session_state)
        if "gtnnwr_results" not in st.session_state:
            model = GTNNWRWrapper(x_columns, y_column="IKP")
            st.info("⏳ Melatih model GTNNWR... (butuh waktu sebentar)")
            results = model.fit(df)
            st.session_state["gtnnwr_results"] = results
            st.success("✅ Model selesai dilatih")
        else:
            results = st.session_state["gtnnwr_results"]

        # Ambil koefisien long
        coefs_long = results["coefs_long"]

        # Filter interaktif
        tahun_coef = st.selectbox("Pilih Tahun (Koefisien)", sorted(coefs_long["Tahun"].unique()))
        prov_coef = st.selectbox("Pilih Provinsi", sorted(coefs_long["Provinsi"].unique()))

        coefs_filtered = coefs_long[
            (coefs_long["Tahun"] == tahun_coef) &
            (coefs_long["Provinsi"] == prov_coef)
        ]

        # Tabel hasil
        st.write(f"### Koefisien Variabel — {prov_coef}, {tahun_coef}")
        st.dataframe(coefs_filtered[["Variabel", "Koefisien"]])

        # Bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(coefs_filtered["Variabel"], coefs_filtered["Koefisien"])
        ax.set_ylabel("Koefisien")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Gagal menjalankan GTNNWR: {e}")

else:
    st.warning("Silakan upload file Data SEC 2025 (CSV/Excel).")
