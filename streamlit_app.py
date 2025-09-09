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

    # Imputasi missing value dengan 0
    df = df.fillna(0)

    # Pastikan nama kolom standar
    df = df.rename(columns=lambda x: x.strip().replace(" ", "_"))

    # Preview Data
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # --------------------------
    # Jalankan GTNNWR
    # --------------------------
    st.subheader("Training Model GTNNWR")

    if st.button("Run GTNNWR Model"):
        try:
            # Tentukan X dan Y
            x_columns = [c for c in df.columns if c not in ["Provinsi", "Tahun", "IKP", "Longitude", "Latitude", "id"]]
            model = GTNNWRWrapper(x_columns, y_column="IKP")
            results = model.fit(df)

            st.success("Model berhasil dijalankan ✅")

            # --------------------------
            # Koefisien per provinsi & tahun
            # --------------------------
            st.subheader("Koefisien Variabel per Provinsi & Tahun")
            coefs_long = results["coefs_long"]

            # Filter interaktif
            tahun_dipilih = st.selectbox("Pilih Tahun", sorted(coefs_long["waktu"].unique()))
            prov_dipilih = st.multiselect("Pilih Provinsi", coefs_long["Provinsi"].unique(),
                                          default=coefs_long["Provinsi"].unique()[:3].tolist())

            coefs_filtered = coefs_long[
                (coefs_long["waktu"] == tahun_dipilih) &
                (coefs_long["Provinsi"].isin(prov_dipilih))
            ]

            st.dataframe(coefs_filtered)

            # Plot
            st.subheader("Visualisasi Koefisien")
            fig, ax = plt.subplots(figsize=(8, 5))
            for prov in prov_dipilih:
                subset = coefs_filtered[coefs_filtered["Provinsi"] == prov]
                ax.plot(subset["Variabel"], subset["Koefisien"], marker="o", label=prov)

            ax.set_title(f"Koefisien Variabel — Tahun {tahun_dipilih}")
            ax.set_ylabel("Koefisien")
            ax.set_xlabel("Variabel")
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Gagal menjalankan model: {e}")

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
            tahun_dipilih = st.selectbox("Pilih Tahun untuk Peta", tahun_list)

            # Filter data sesuai tahun
            df_filtered = df[df["Tahun"] == tahun_dipilih]

            # Load GeoJSON provinsi Indonesia
            url = "https://raw.githubusercontent.com/ans-4175/peta-indonesia-geojson/master/indonesia-prov.geojson"
            gdf = gpd.read_file(url)

            # Samakan format nama provinsi
            gdf["Provinsi"] = gdf["Propinsi"].str.title()
            df_filtered["Provinsi"] = df_filtered["Provinsi"].str.title()

            # Cek range IKP
            ikp_min = df_filtered["IKP"].min()
            ikp_max = df_filtered["IKP"].max()
            st.write(f"Range IKP ({tahun_dipilih}): {ikp_min:.2f} – {ikp_max:.2f}")

            # Definisi bins provinsi sesuai tabel
            bins = [0, 37.61, 48.27, 57.11, 65.96, 74.40]
            if ikp_max > bins[-1]:
                bins.append(ikp_max + 1)
            else:
                bins.append(100)

            # Colormap custom
            colormap = cm.StepColormap(
                colors=['#4B0000', '#FF3333', '#FF9999', '#CCFF99', '#66CC66', '#006600'],
                vmin=bins[0],
                vmax=bins[-1],
                index=bins,
                caption="Indeks Ketahanan Pangan (IKP)"
            )

            # Bikin map kosong
            m = folium.Map(location=[-2.5, 118], zoom_start=5)

            # Gabungkan data IKP ke GeoJSON
            gdf = gdf.merge(df_filtered[["Provinsi", "IKP"]], on="Provinsi", how="left")

            # Styling berdasarkan nilai IKP
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

            # Tambahkan layer GeoJSON
            folium.GeoJson(
                gdf,
                style_function=style_function,
                tooltip=folium.GeoJsonTooltip(fields=["Provinsi", "IKP"])
            ).add_to(m)

            # Tambahkan colormap (legend)
            colormap.add_to(m)

            # Tampilkan di Streamlit
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

        st.subheader("Quick Tips")
        st.markdown("""
        - Pilih 2-3 provinsi untuk membandingkan indikator kunci.  
        - Gunakan Scenario untuk menguji perubahan Luas Panen, Produktivitas, atau Cadangan Pangan.
        """)

else:
    st.warning("Silakan upload file Data SEC 2025 (CSV/Excel).")
