import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import branca.colormap as cm

from gtnnwr import GTNNWRWrapper

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
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine="openpyxl")

    # Imputasi missing value dengan 0
    df = df.fillna(0)

    # Preview Data
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # --------------------------
    # Layout 2 kolom utama
    # --------------------------
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Peta Indonesia — IKP per Provinsi")

        try:
            # Load GeoJSON provinsi Indonesia
            url = "https://raw.githubusercontent.com/ans-4175/peta-indonesia-geojson/master/indonesia-prov.geojson"
            gdf = gpd.read_file(url)

            # Samakan format nama provinsi
            gdf["Nama Provinsi"] = gdf["Propinsi"].str.title()   # field di geojson = "Propinsi"
            df["Nama Provinsi"] = df["Nama Provinsi"].str.title()

            # Bikin map choropleth
            import branca.colormap as cm

            import branca.colormap as cm

            # --- cek nilai IKP ---
            ikp_min = df["IKP"].min()
            ikp_max = df["IKP"].max()
            st.write(f"Range IKP: {ikp_min} – {ikp_max}")

            # --- definisi bins provinsi sesuai tabel ---
            bins = [0, 37.61, 48.27, 57.11, 65.96, 74.40]

            if ikp_max > bins[-1]:
                bins.append(ikp_max + 1)
            else:
                bins.append(100)

            # definisi colormap custom sesuai tabel
            colormap = cm.StepColormap(
                colors=['#4B0000', '#FF3333', '#FF9999', '#CCFF99', '#66CC66', '#006600'],
                vmin=bins[0],
                vmax=bins[-1],
                index=bins,
                caption="Indeks Ketahanan Pangan (IKP)"
            )

            # bikin map kosong
            m = folium.Map(location=[-2.5, 118], zoom_start=5)

            # gabungkan data IKP ke GeoJSON
            gdf = gdf.merge(df[["Nama Provinsi", "IKP"]], on="Nama Provinsi", how="left")

            # styling berdasarkan nilai IKP
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

            # tambahkan layer geojson
            folium.GeoJson(
                gdf,
                style_function=style_function,
                tooltip=folium.GeoJsonTooltip(fields=["Nama Provinsi", "IKP"])
            ).add_to(m)

            # tambahkan colormap (legend)
            colormap.add_to(m)

            # tampilkan di Streamlit
            st_folium(m, width=700, height=500)


        except Exception as e:
            st.error(f"Gagal memuat peta: {e}")

        # Comparative view
        st.write("#### Comparative View")
        prov_selected = st.multiselect("Pilih Provinsi", df["Nama Provinsi"].unique().tolist(), default=["Aceh", "Sumatera Utara"])
        st.write("Dipilih:", prov_selected)

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

    # --------------------------
    # Storytelling Ringkasan
    # --------------------------
    st.subheader("Storytelling — Ringkasan Interaktif")
    st.success("Aceh mengalami stabil/kenaikan IKP dalam 5 tahun terakhir, dipengaruhi oleh Kekeringan, Kebakaran, OPD Blas.")

    # --------------------------
    # Detail Provinsi + Simulasi
    # --------------------------
    st.write("---")
    prov = st.selectbox("Pilih Provinsi untuk Detail", df["Nama Provinsi"].unique())
    prov_data = df[df["Nama Provinsi"] == prov].iloc[0]

    col3, col4 = st.columns([2, 1])

    with col3:
        st.write(f"### {prov} — Indeks Ketahanan Pangan")
        # tampilkan semua kolom kecuali Nama Provinsi
        for col in df.columns:
            if col != "Nama Provinsi":
                st.metric(col, prov_data[col])

    with col4:
        st.subheader("Scenario & Simulation")
        variabel = st.selectbox("Pilih Variabel", [c for c in df.columns if c not in ["Nama Provinsi", "IKP"]])
        perubahan = st.number_input("Perubahan (%)", value=5, step=1)

        if st.button("Run Simulation"):
            # Jalankan GTNNWR
            x_columns = [c for c in df.columns if c not in ["Nama Provinsi", "IKP", "Tahun", "Longitude", "Latitude", "id"]]
            model = GTNNWRWrapper(x_columns)
            results = model.fit(df)

            ikp_now = prov_data["IKP"]
            ikp_simulasi = ikp_now * (1 + perubahan / 100)  # dummy simulasi

            st.metric("IKP saat ini", f"{ikp_now:.2f}")
            st.metric("IKP simulasi", f"{ikp_simulasi:.2f}")
            st.caption("Catatan: simulasi ini masih sederhana. Untuk akurasi penuh, hubungkan output GTNNWR.")

            # Plot hasil GTNNWR
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
