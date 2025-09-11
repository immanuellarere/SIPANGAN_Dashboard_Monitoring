import streamlit as st
import pandas as pd
import torch

from gnnwr.datasets import init_dataset_split

# --------------------------
# Config
# --------------------------
st.set_page_config(page_title="SIPANGAN Dashboard Monitoring", layout="wide")
st.title("📊 SIPANGAN Dashboard Monitoring")
st.caption("Inference GTNNWR dengan pipeline yang sama (tanpa shape mismatch)")

DATA_PATH = "datasec.xlsx"
MODEL_PATH = "gtnnwr_model.pt"

# --------------------------
# Load Data
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
# Definisi Fitur
# --------------------------
x_columns = [
    'Skor_PPH','Luas_Panen','Produktivitas','Produksi',
    'Tanah_Longsor','Banjir','Kekeringan','Kebakaran','Cuaca',
    'OPD_Penggerek_Batang_Padi','OPD_Wereng_Batang_Coklat',
    'OPD_Tikus','OPD_Blas','OPD_Hwar_Daun','OPD_Tungro'
]

# Split sama dengan training
train_data = df[df["Tahun"] <= 2022].copy()
val_data   = df[df["Tahun"] == 2023].copy()
test_data  = df[df["Tahun"] == 2024].copy()

train_ds, val_ds, test_ds = init_dataset_split(
    train_data=train_data,
    val_data=val_data if len(val_data) else train_data,
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
# Load Model
# --------------------------
@st.cache_resource
def load_model():
    model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.eval()
    return model

try:
    model = load_model()
    st.success("✅ Model berhasil diload dari .pt")
except Exception as e:
    st.error(f"❌ Gagal load model: {e}")
    st.stop()

# --------------------------
# Siapkan Input dari Dataset Split
# --------------------------
def dataset_to_tensor(ds):
    if hasattr(ds, "x_data"):
        return torch.tensor(ds.x_data, dtype=torch.float32)
    else:
        xs = [ds[i][0].flatten() for i in range(len(ds))]
        return torch.stack(xs)

x_train = dataset_to_tensor(train_ds)
x_val   = dataset_to_tensor(val_ds)
x_test  = dataset_to_tensor(test_ds)

x_input = torch.cat([x_train, x_val, x_test], dim=0).unsqueeze(1)  # [N,1,152]

st.write("📐 Shape input final ke model:", x_input.shape)

# --------------------------
# Prediksi
# --------------------------
st.write("---")
st.subheader("🤖 Analisis GTNNWR (.pt)")

try:
    with torch.no_grad():
        y_pred = model(x_input)

    df_ordered = pd.concat([train_data, val_data, test_data]).sort_values("id")
    df_ordered["IKP_Prediksi"] = y_pred.cpu().numpy().flatten()[:len(df_ordered)]

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
