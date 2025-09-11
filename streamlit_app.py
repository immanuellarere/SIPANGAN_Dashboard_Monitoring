import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import branca.colormap as cm
import torch

from gnnwr.datasets import init_dataset_split
from gnnwr.models import GTNNWR

# --------------------------
# Halaman
# --------------------------
st.set_page_config(page_title="SIPANGAN Dashboard Monitoring", layout="wide")
st.title("📊 SIPANGAN Dashboard Monitoring")
st.caption("Inference GTNNWR dari .pt dengan pipeline yang sama seperti training")

DATA_PATH = "datasec.xlsx"
MODEL_PATH = "gtnnwr_model.pt"   # file .pt yang kamu simpan saat training

# --------------------------
# Load data
# --------------------------
try:
    df = pd.read_excel(DATA_PATH, engine="openpyxl") if DATA_PATH.endswith("xlsx") else pd.read_csv(DATA_PATH)
    df = df.fillna(0)
    df = df.rename(columns=lambda x: x.strip().replace(" ", "_"))
    df["id"] = range(len(df))
except Exception as e:
    st.error(f"❌ Gagal membaca dataset: {e}")
    st.stop()

prov_col = "Provinsi" if "Provinsi" in df.columns else ("Nama_Provinsi" if "Nama_Provinsi" in df.columns else None)
if prov_col is None:
    st.error("❌ Dataset harus punya kolom 'Provinsi' atau 'Nama_Provinsi'")
    st.stop()

st.write("---")
st.subheader("🔍 Data Preview")
st.dataframe(df.head())

# --------------------------
# Siapkan dataset (PENTING: sama persis seperti training)
# --------------------------
x_columns = [
    'Skor_PPH','Luas_Panen','Produktivitas','Produksi',
    'Tanah_Longsor','Banjir','Kekeringan','Kebakaran','Cuaca',
    'OPD_Penggerek_Batang_Padi','OPD_Wereng_Batang_Coklat',
    'OPD_Tikus','OPD_Blas','OPD_Hwar_Daun','OPD_Tungro'
]

# split seperti training
train_data = df[df["Tahun"] <= 2022].copy()
val_data   = df[df["Tahun"] == 2023].copy()
test_data  = df[df["Tahun"] == 2024].copy()

train_ds, val_ds, test_ds = init_dataset_split(
    train_data=train_data,
    val_data=val_data if len(val_data) else train_data,   # hindari scaler error jika kosong
    test_data=test_data if len(test_data) else train_data,
    x_column=x_columns,
    y_column=["IKP"],
    spatial_column=["Longitude","Latitude"],
    temp_column=["Tahun"],
    id_column=["id"],
    use_model="gtnnwr",
    batch_size=1024,
    shuffle=False
)

# --------------------------
# Rekonstruksi arsitektur & load bobot .pt
# --------------------------
@st.cache_resource
def build_and_load_model():
    # arsitektur sama seperti training
    optim_params = {
        "scheduler":"MultiStepLR",
        "scheduler_milestones":[1000, 2000, 3000, 4000],
        "scheduler_gamma":0.8,
    }
    wrapper = GTNNWR(
        train_ds, val_ds, test_ds,
        [[3],[512,256,64]],             # <== hidden layer persis training
        drop_out=0.5,
        optimizer="Adadelta",
        optimizer_params=optim_params,
        write_path="./gtnnwr_runs",
        model_name="GTNNWR_DSi"
    )
    wrapper.add_graph()  # <== WAJIB: membangun graph/fit scaler seperti training

    # load .pt lalu salin state_dict
    pretrained = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    try:
        wrapper._model.load_state_dict(pretrained.state_dict(), strict=True)
    except Exception:
        # jika tipe modul sama, ini akan berhasil; kalau pun beda, coba longgar
        wrapper._model.load_state_dict(pretrained.state_dict(), strict=False)
    wrapper._model.eval()
    return wrapper

try:
    gtnnwr = build_and_load_model()
    st.success("✅ Arsitektur dibangun & bobot .pt dimuat")
except Exception as e:
    st.error(f"❌ Gagal membangun/menyelaraskan model: {e}")
    st.stop()

# --------------------------
# Bentuk input persis seperti saat training → [N, 1, F]
# --------------------------
def dataset_to_input(ds):
    """
    Ambil fitur X dari dataset gnnwr dan kembalikan tensor float32
    dengan bentuk [N, 1, F], sesuai ekspektasi model.
    """
    if hasattr(ds, "x_data"):  # beberapa versi expose x_data yang sudah diskalakan
        x = torch.tensor(ds.x_data, dtype=torch.float32)
    else:
        # __getitem__ biasanya (x, y) → ambil x lalu flatten jadi [F]
        xs = [ds[i][0].flatten() for i in range(len(ds))]
        x = torch.stack(xs)
    if x.dim() == 2:
        x = x.unsqueeze(1)   # [N, 1, F] seperti training
    return x

try:
    # Gabungkan urutan train → val → test supaya sejajar dengan df
    x_train = dataset_to_input(train_ds)
    x_val   = dataset_to_input(val_ds)
    x_test  = dataset_to_input(test_ds)
    x_input = torch.cat([x_train, x_val, x_test], dim=0)

    st.write("📐 Shape input final ke model:", x_input.shape)  # harus [N_total, 1, F]
except Exception as e:
    st.error(f"❌ Gagal menyiapkan input: {e}")
    st.stop()

# --------------------------
# Prediksi
# --------------------------
st.write("---")
st.subheader("🤖 Analisis GTNNWR (.pt)")

try:
    with torch.no_grad():
        y_pred = gtnnwr._model(x_input)   # panggil model internal yang selaras
    # panjang gabungan train+val+test sama dengan jumlah baris train+val+test
    # pastikan urutannya kembali ke dataframe asli
    df_ordered = pd.concat([train_data, val_data, test_data], axis=0)
    df_ordered["IKP_Prediksi"] = y_pred.cpu().numpy().flatten()[:len(df_ordered)]

    # merge kembali supaya align ke df awal (jaga-jaga ada urutan berbeda)
    out = df.merge(df_ordered[["id","IKP_Prediksi"]], on="id", how="left")

    st.subheader("📑 Hasil Prediksi IKP")
    st.dataframe(out[["Tahun", prov_col, "IKP", "IKP_Prediksi"]])

    st.download_button(
        "📥 Download CSV Prediksi",
        out.to_csv(index=False).encode("utf-8"),
        file_name="prediksi_ikp.csv",
        mime="text/csv"
    )
except Exception as e:
    st.error(f"❌ Gagal menjalankan prediksi: {e}")
