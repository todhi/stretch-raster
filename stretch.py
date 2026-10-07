import sys
import subprocess
import importlib

# ==========================================
#  1. MESIN AUTO-INSTALL DEPENDENCIES
# ==========================================
def install_requirements():
    requirements = {
        "numpy": "numpy",
        "rasterio": "rasterio",
        "PIL": "pillow",
        "skimage": "scikit-image",
        "scipy": "scipy",
        "customtkinter": "customtkinter"
    }
    
    install_needed = False
    for module_name in requirements.keys():
        try:
            importlib.import_module(module_name)
        except ImportError:
            install_needed = True
            break
            
    if install_needed:
        print("[SISTEM] Mendeteksi modul yang kurang. Memulai Auto-Install...")
        print("[SISTEM] Mohon tunggu sebentar, proses ini mungkin memakan waktu beberapa menit...\n")
        for module_name, pip_name in requirements.items():
            try:
                importlib.import_module(module_name)
            except ImportError:
                print(f" -> Menginstal {pip_name}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name, "-q"])
        print("\n[SISTEM] Instalasi selesai! Memulai aplikasi...")

install_requirements()

# ==========================================
#  2. IMPORT MODUL
# ==========================================
import os
import time
import threading
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy.ndimage import binary_dilation
from PIL import Image, ImageTk
from skimage import exposure

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

# ==========================================
#  TEMA & PALET WARNA (PROFESSIONAL + WINDOWS BLUE)
# ==========================================
ctk.set_appearance_mode("System")  
ctk.set_default_color_theme("blue")

C_APP_BG     = ("#F0F3F4", "#181A1B")  
C_CARD_BG    = ("#FFFFFF", "#242729")  
C_BORDER     = ("#D5DBDB", "#3A3F42")  
C_TEXT       = ("#1A1A1A", "#E8E8E8")  
C_TEXT_MUTED = ("#5D6D7E", "#A0AAB2")  

C_BTN_TEAL   = ("#067C76", "#0D9B94")  
C_BTN_TEAL_H = ("#045C57", "#0A7A74")  
C_BTN_SLATE  = ("#34495E", "#4B6584")  
C_BTN_SLATE_H= ("#2C3E50", "#34495E")  
C_BTN_CANCEL = ("#C0392B", "#D32F2F")  
C_BTN_CANCEL_H=("#962D22", "#B71C1C")  
C_BTN_BROWSE = ("#E5E8E8", "#3A3F42")  
C_BTN_BROW_H = ("#D5DBDB", "#4D5458")
C_BTN_BROW_T = ("#212F3D", "#FDFEFE")  

C_PROG_BLUE  = ("#0078D7", "#0063B1")  
C_PROG_BG    = ("#D5DBDB", "#3A3F42")  

# ==========================================
#  3. FUNGSI PENGOLAHAN CITRA
# ==========================================
def linear_stretch_preserve_dtype(masked_band, lower=2.0, upper=98.0):
    band = masked_band
    if band.mask.all():
        return np.zeros(band.shape, dtype=band.dtype)
    valid = band.compressed().astype(np.float64)
    p_low, p_high = np.percentile(valid, (lower, upper))
    if not np.isfinite(p_low) or not np.isfinite(p_high) or p_high <= p_low:
        filled = band.filled(np.min(valid) if valid.size else 0)
        return filled.astype(band.dtype)
    norm = (band.filled(p_low) - p_low) / (p_high - p_low)
    norm = np.clip(norm, 0, 1)
    if np.issubdtype(band.dtype, np.integer):
        info = np.iinfo(band.dtype)
        out = (norm * (info.max - info.min) + info.min).round().astype(band.dtype)
        out[band.mask] = info.min
    else:
        out = norm.astype(band.dtype)
        out[band.mask] = 0.0
    return out

def hist_equalize_preserve_dtype(masked_band):
    if masked_band.mask.all():
        return np.zeros(masked_band.shape, dtype=masked_band.dtype)
    tmp = linear_stretch_preserve_dtype(masked_band, lower=0.5, upper=99.5)
    if np.issubdtype(tmp.dtype, np.integer):
        info = np.iinfo(tmp.dtype)
        norm = (tmp.astype(np.float64) - info.min) / (info.max - info.min)
    else:
        norm = tmp.astype(np.float64)
    mask = ~masked_band.mask
    
    eq = np.zeros_like(norm)
    eq[mask] = exposure.equalize_hist(norm[mask])
    
    if np.issubdtype(tmp.dtype, np.integer):
        info = np.iinfo(tmp.dtype)
        out = (eq * (info.max - info.min) + info.min).round().astype(tmp.dtype)
        out[~mask] = info.min
    else:
        out = eq.astype(tmp.dtype)
        out[~mask] = 0.0
    return out

def clahe_preserve_dtype(masked_band, clip_limit=0.01, tile_grid_size=8):
    if masked_band.mask.all():
        return np.zeros(masked_band.shape, dtype=masked_band.dtype)
    band = masked_band
    valid = band.compressed().astype(np.float64)
    vmin, vmax = np.percentile(valid, (1, 99))
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = np.clip((band.filled(vmin) - vmin) / (vmax - vmin), 0, 1)
    eq_2d = exposure.equalize_adapthist(
        norm, clip_limit=float(clip_limit), nbins=256, kernel_size=tile_grid_size
    )
    mask = ~band.mask
    if np.issubdtype(band.dtype, np.integer):
        info = np.iinfo(band.dtype)
        out = (eq_2d * (info.max - info.min) + info.min).round().astype(band.dtype)
        out[~mask] = info.min
    else:
        out = eq_2d.astype(band.dtype)
        out[~mask] = 0.0
    return out

def apply_remove_frame_array(data, dtype=None):
    """
    Logika inti pemisahan bingkai luar hitam (NoData=0) vs data di dalam citra (minimal 1).
    Melindungi data perairan dan daratan agar tidak bolong.
    """
    if dtype is None:
        dtype = str(data.dtype)
    else:
        dtype = str(dtype)

    out_dtype = np.uint16 if dtype in ['float32', 'float64'] else data.dtype
    zeros_mask = (data == 0.0) | (~np.isfinite(data))

    if np.all(zeros_mask):
        return np.zeros(data.shape, dtype=out_dtype)

    # Petakan data valid ke nilai minimal 1 (agar air/bayangan tidak bolong)
    if dtype == 'uint16':
        data_out = np.round(1.0 + (data.astype(np.float32) / 65535.0) * 65534.0).astype(np.uint16)
    elif dtype in ['float32', 'float64']:
        data_out = np.round(1.0 + np.clip(data, 0.0, 1.0) * 65534.0).astype(np.uint16)
    elif dtype == 'uint8':
        data_out = np.round(1.0 + (data.astype(np.float32) / 255.0) * 254.0).astype(np.uint8)
    else:
        data_out = np.maximum(data, 1).astype(out_dtype)

    if np.any(zeros_mask):
        # Buat zona buffer selebar 1 piksel dari data yang bernilai (> 0)
        valid_region = (data > 0.0) & np.isfinite(data)
        expanded_valid = binary_dilation(valid_region, structure=np.ones((3, 3)))

        # A. BINGKAI LUAR SEJATI: Area nol di luar zona buffer -> Set 0 (Transparan)
        true_outer_border = zeros_mask & (~expanded_valid)
        data_out[true_outer_border] = 0

        # B. DATA DI DALAM CITRA: Area nol di dalam darat/laut -> Set minimal 1 (Solid)
        internal_zeros = zeros_mask & expanded_valid
        data_out[internal_zeros] = np.maximum(data_out[internal_zeros], 1)

    return data_out


def remove_raster_frame(
    input_path: str, output_path: str, tile_size: int = 512,
    per_file_progress_cb=None, status_cb=None, eta_cb=None, cancel_check_cb=None
):
    """
    Menghilangkan bingkai hitam luar citra satelit (menjadikannya NoData/Transparan)
    sekaligus melindungi data perairan dan daratan agar tidak bolong.
    """
    t0 = time.perf_counter()
    if status_cb:
        status_cb(f"Membuka: {os.path.basename(input_path)}")

    with rasterio.open(input_path) as src:
        dtype = src.dtypes[0]
        out_dtype = 'uint16' if dtype in ['float32', 'float64'] else dtype

        # 1. ATUR PROFIL OUTPUT DENGAN NODATA = 0
        profile = src.profile.copy()
        profile.update(
            dtype=out_dtype,
            nodata=0,              # Nilai 0 dijadikan NoData (100% Transparan di GIS)
            tiled=True,
            blockxsize=tile_size,
            blockysize=tile_size,
            compress='lzw',
            BIGTIFF="IF_SAFER"
        )

        count = src.count
        processed_bands = 0
        band_times = []

        with rasterio.open(output_path, 'w', **profile) as dst:
            for b in range(1, count + 1):
                if cancel_check_cb and cancel_check_cb():
                    raise RuntimeError("Proses dibatalkan oleh pengguna.")

                tb0 = time.perf_counter()
                if status_cb:
                    status_cb(f"Menghilangkan frame hitam band {b}/{count}...")

                windows = [w for _, w in src.block_windows(b)]
                total_win = max(len(windows), 1)

                # Membaca data per blok 2D agar hemat memori
                for w_idx, window in enumerate(windows, start=1):
                    if cancel_check_cb and cancel_check_cb():
                        raise RuntimeError("Proses dibatalkan oleh pengguna.")

                    data = src.read(b, window=window)
                    data_out = apply_remove_frame_array(data, dtype=dtype)
                    dst.write(data_out, b, window=window)

                    if per_file_progress_cb and (w_idx % 10 == 0 or w_idx == total_win):
                        frac = ((b - 1) + (w_idx / total_win)) / count
                        per_file_progress_cb(frac)

                dt = max(time.perf_counter() - tb0, 1e-6)
                band_times.append(dt)
                processed_bands += 1

                if eta_cb:
                    eta_cb((count - processed_bands) * (sum(band_times) / len(band_times)))

    print(f"✓ Selesai! Bingkai hitam berhasil dihilangkan: {output_path}")
    return time.perf_counter() - t0


def process_one_file(
    in_path, out_path, method, lower, upper, clip_limit, tile_grid_size,
    remove_frame=False, block_tile_size=512,
    per_file_progress_cb=None, status_cb=None, eta_cb=None, cancel_check_cb=None
):
    if method == "Hanya Hilangkan Frame":
        return remove_raster_frame(
            in_path, out_path, tile_size=block_tile_size,
            per_file_progress_cb=per_file_progress_cb,
            status_cb=status_cb,
            eta_cb=eta_cb,
            cancel_check_cb=cancel_check_cb
        )

    t0 = time.perf_counter()
    if status_cb: status_cb(f"Membuka: {os.path.basename(in_path)}")
    
    with rasterio.open(in_path) as src:
        img = src.read(masked=True)  
        in_profile = src.profile.copy()
        in_nodata = src.nodata
        in_dtype = src.dtypes[0]

    count = img.shape[0]
    processed, times = 0, []
    out_bands = []

    for i in range(count):
        if cancel_check_cb and cancel_check_cb():
            raise RuntimeError("Proses dibatalkan oleh pengguna.")

        band = img[i]
        mode_label = f"{method} + NoFrame" if remove_frame else method
        if status_cb: status_cb(f"Memproses band {i+1}/{count} [{mode_label}]")
        tb0 = time.perf_counter()

        if remove_frame:
            # Pisahkan bingkai hitam luar dari piksel 0 internal sebelum stretch
            raw_data = band.filled(0)
            zeros_mask = (raw_data == 0.0) | np.ma.getmaskarray(band) | (~np.isfinite(raw_data))
            valid_region = (raw_data > 0.0) & (~np.ma.getmaskarray(band)) & np.isfinite(raw_data)
            if np.any(valid_region) and np.any(zeros_mask):
                expanded_valid = binary_dilation(valid_region, structure=np.ones((3, 3)))
                true_outer_border = zeros_mask & (~expanded_valid)
                internal_zeros = zeros_mask & expanded_valid
            elif not np.any(valid_region):
                true_outer_border = np.ones(raw_data.shape, dtype=bool)
                internal_zeros = np.zeros(raw_data.shape, dtype=bool)
            else:
                true_outer_border = np.zeros(raw_data.shape, dtype=bool)
                internal_zeros = np.zeros(raw_data.shape, dtype=bool)

            work_band = np.ma.MaskedArray(raw_data, mask=true_outer_border)
        else:
            work_band = band

        if method == "Linear Stretch":
            out = linear_stretch_preserve_dtype(work_band, lower, upper)
        elif method == "Equalization":
            out = hist_equalize_preserve_dtype(work_band)
        elif method == "CLAHE":
            out = clahe_preserve_dtype(work_band, clip_limit=clip_limit, tile_grid_size=tile_grid_size)

        if remove_frame:
            # Petakan hasil pencerahan agar data valid minimal 1 dan bingkai luar = 0 (NoData)
            out = apply_remove_frame_array(out, dtype=in_dtype)
            out[true_outer_border] = 0
            out[internal_zeros] = np.maximum(out[internal_zeros], 1)

        out_bands.append(out)
        dt = max(time.perf_counter() - tb0, 1e-6)
        times.append(dt); processed += 1

        if per_file_progress_cb:
            per_file_progress_cb(processed / count)
        
        # Callback ETA hanya dikirim jika fungsi meminta (untuk single file)
        if eta_cb:
            eta_cb((count - processed) * (sum(times) / len(times)))

    stacked = np.stack(out_bands)
    profile = in_profile.copy()
    profile.update(count=stacked.shape[0], dtype=stacked.dtype)

    if remove_frame:
        profile.update(
            nodata=0,
            tiled=True,
            blockxsize=block_tile_size,
            blockysize=block_tile_size,
            compress='lzw',
            BIGTIFF="IF_SAFER"
        )
    elif in_nodata is not None:
        profile.update(nodata=in_nodata)

    if status_cb: status_cb("Menyimpan hasil GeoTIFF...")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(stacked)

    return time.perf_counter() - t0

# ==========================================
#  4. HELPER PREVIEW & THUMBNAIL
# ==========================================
_preview_cache = {}

def raster_to_rgb_array(path, max_size=550):
    global _preview_cache
    if path in _preview_cache:
        return _preview_cache[path]

    with rasterio.open(path) as src:
        H_full, W_full = src.height, src.width
        count = src.count
        has_nodata = src.nodata is not None
        scale = min(max_size / max(H_full, W_full, 1), 1.0)
        new_h, new_w = max(1, int(H_full * scale)), max(1, int(W_full * scale))

        if count >= 3:
            bands = src.read([1, 2, 3], out_shape=(3, new_h, new_w), resampling=Resampling.bilinear, masked=True)
            r = np.ma.MaskedArray(bands[0].data.copy(), mask=bands[0].mask.copy())
            g = np.ma.MaskedArray(bands[1].data.copy(), mask=bands[1].mask.copy())
            b = np.ma.MaskedArray(bands[2].data.copy(), mask=bands[2].mask.copy())
        else:
            raw = src.read(1, out_shape=(new_h, new_w), resampling=Resampling.bilinear, masked=True)
            r = np.ma.MaskedArray(raw.data.copy(), mask=np.ma.getmaskarray(raw).copy())
            g, b = r.copy(), r.copy()

    def norm_band(band):
        mask_arr = np.ma.getmaskarray(band)
        h, w = band.shape[0], band.shape[1]
        n_valid = int((~mask_arr).sum())
        if n_valid == 0: return np.zeros((h, w), dtype=np.uint8)

        filled = band.filled(0).astype(np.float64)
        valid_vals = filled[~mask_arr]
        vmin, vmax = valid_vals.min(), valid_vals.max()
        if vmax <= vmin: vmax = vmin + 1.0

        out = np.clip((filled - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
        out[mask_arr] = 0
        return out.reshape(h, w)

    rgb = np.stack([norm_band(r), norm_band(g), norm_band(b)], axis=2)

    # Tampilkan pola checkerboard transparan pada area NoData agar terlihat jelas di Preview
    if has_nodata:
        nodata_mask = np.ma.getmaskarray(r) & np.ma.getmaskarray(g) & np.ma.getmaskarray(b)
        if np.any(nodata_mask):
            yy, xx = np.indices((new_h, new_w))
            checker = (((yy // 10) + (xx // 10)) % 2 == 0)
            bg_val = np.where(checker, 55, 35).astype(np.uint8)
            for c in range(3):
                ch = rgb[:, :, c]
                ch[nodata_mask] = bg_val[nodata_mask]

    _preview_cache[path] = rgb
    return rgb

# ==========================================
#  5. JENDELA PREVIEW MODERN
# ==========================================
class PreviewWindow(ctk.CTkToplevel):
    def __init__(self, parent, in_path, out_path):
        super().__init__(parent)
        self.title("Preview — Before vs After")
        self.grab_set()
        self.configure(fg_color=C_APP_BG)

        self.in_path = in_path
        self.out_path = out_path

        hdr = ctk.CTkFrame(self, fg_color=C_BTN_TEAL, corner_radius=0)
        hdr.pack(fill="x")
        ctk.CTkLabel(hdr, text="Preview Citra — Sebelum vs Sesudah", text_color="#FFFFFF", font=ctk.CTkFont(size=14, weight="bold")).pack(side="left", padx=15, pady=8)

        self.load_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.load_frame.pack(fill="both", expand=True, padx=40, pady=40)
        
        self.lbl_status = ctk.CTkLabel(self.load_frame, text="Memuat dan me-render citra, mohon tunggu...", text_color=C_TEXT, font=ctk.CTkFont(size=12))
        self.lbl_status.pack(pady=(0,10))
        
        self.pbar = ctk.CTkProgressBar(self.load_frame, mode="determinate", width=300, progress_color=C_PROG_BLUE, fg_color=C_PROG_BG)
        self.pbar.pack()
        self.pbar.set(0)

        self.geometry("450x180")
        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"+{(sw-450)//2}+{(sh-180)//2}")

        threading.Thread(target=self._load_images, daemon=True).start()

    def update_pbar_smooth(self, target_val):
        current = self.pbar.get()
        if target_val <= current:
            self.pbar.set(target_val)
            return
            
        steps = 15
        diff = target_val - current
        
        def animate(step_count):
            if step_count <= steps:
                try:
                    self.pbar.set(current + (diff * (step_count / steps)))
                    self.after(15, animate, step_count + 1)
                except Exception:
                    pass
        animate(1)

    def _load_images(self):
        try:
            self.after(0, lambda: self.update_pbar_smooth(0.33))
            arr_before = raster_to_rgb_array(self.in_path, max_size=500)
            
            self.after(0, lambda: self.update_pbar_smooth(0.66))
            arr_after = raster_to_rgb_array(self.out_path, max_size=500)
            
            self.after(0, lambda: self.update_pbar_smooth(1.0))
        except Exception as e:
            self.after(0, lambda: self.lbl_status.configure(text=f"Error: {str(e)}", text_color=C_BTN_CANCEL))
            return
            
        self.after(300, lambda: self._build_ui(arr_before, arr_after))

    def _build_ui(self, arr_before, arr_after):
        self.load_frame.destroy()
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=15, pady=15)

        img_b = ImageTk.PhotoImage(Image.fromarray(arr_before))
        img_a = ImageTk.PhotoImage(Image.fromarray(arr_after))
        w, h = max(arr_before.shape[1], arr_after.shape[1]), max(arr_before.shape[0], arr_after.shape[0])

        canvas_bg = "#111111" 

        frm_b = ctk.CTkFrame(main, fg_color=C_CARD_BG, border_color=C_BORDER, border_width=1, corner_radius=6)
        frm_b.grid(row=0, column=0, padx=10, pady=10)
        ctk.CTkLabel(frm_b, text="SEBELUM (Asli)", text_color=C_TEXT, font=ctk.CTkFont(weight="bold")).pack(fill="x", pady=(5, 5))
        cv_b = tk.Canvas(frm_b, width=w, height=h, bg=canvas_bg, highlightthickness=0)
        cv_b.pack(padx=10, pady=(0, 10))
        cv_b.create_image(w//2, h//2, anchor="center", image=img_b)
        cv_b.image = img_b

        frm_a = ctk.CTkFrame(main, fg_color=C_CARD_BG, border_color=C_BORDER, border_width=1, corner_radius=6)
        frm_a.grid(row=0, column=1, padx=10, pady=10)
        ctk.CTkLabel(frm_a, text="SESUDAH (Diproses)", text_color=C_TEXT, font=ctk.CTkFont(weight="bold")).pack(fill="x", pady=(5, 5))
        cv_a = tk.Canvas(frm_a, width=w, height=h, bg=canvas_bg, highlightthickness=0)
        cv_a.pack(padx=10, pady=(0, 10))
        cv_a.create_image(w//2, h//2, anchor="center", image=img_a)
        cv_a.image = img_a

        ftr = ctk.CTkFrame(self, fg_color="transparent")
        ftr.pack(fill="x", padx=25, pady=(0, 15))
        ctk.CTkLabel(ftr, text=f"Output: {os.path.basename(self.out_path)}", text_color=C_TEXT_MUTED).pack(side="left")
        ctk.CTkButton(ftr, text="Tutup Preview", command=self.destroy, width=120, fg_color=C_BTN_BROWSE, hover_color=C_BTN_BROW_H, text_color=C_BTN_BROW_T, font=ctk.CTkFont(weight="bold")).pack(side="right")

        total_w = (w * 2) + 90
        total_h = h + 130
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{total_w}x{total_h}+{(sw-total_w)//2}+{(sh-total_h)//2}")


# ==========================================
#  6. GUI UTAMA MODERN
# ==========================================
class RasterHistGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Stretch v2.0")
        self.root.geometry("690x705")
        self.root.resizable(False, False)
        
        self.root.configure(fg_color=C_APP_BG)

        self.mode_var = tk.StringVar(value="single")
        self.method_var = tk.StringVar(value="Linear Stretch")
        self.remove_frame_var = tk.BooleanVar(value=True)
        
        # Variabel Stopwatch & ETA
        self.is_processing = False
        self.start_time = 0
        self.current_eta_str = "-"
        
        self.build_ui()

    def build_ui(self):
        f1 = ctk.CTkFrame(self.root, fg_color=C_CARD_BG, border_color=C_BORDER, border_width=1, corner_radius=8)
        f1.pack(fill="x", padx=20, pady=(20, 10))
        f1.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(f1, text="Pengaturan Algoritma", font=ctk.CTkFont(size=14, weight="bold"), text_color=C_TEXT).grid(row=0, column=0, sticky="w", padx=15, pady=(10, 0))
        
        row1 = ctk.CTkFrame(f1, fg_color="transparent")
        row1.grid(row=1, column=0, sticky="w", padx=15, pady=10)
        
        ctk.CTkLabel(row1, text="Tipe Data:", text_color=C_TEXT).pack(side="left", padx=(0,10))
        
        ctk.CTkRadioButton(row1, text="Single File", variable=self.mode_var, value="single", text_color=C_TEXT, fg_color=C_BTN_TEAL[0], hover_color=C_BTN_TEAL_H[0], command=self.update_mode_ui).pack(side="left", padx=10)
        ctk.CTkRadioButton(row1, text="Batch Folder", variable=self.mode_var, value="batch", text_color=C_TEXT, fg_color=C_BTN_TEAL[0], hover_color=C_BTN_TEAL_H[0], command=self.update_mode_ui).pack(side="left", padx=10)
        
        ctk.CTkLabel(row1, text="Metode:", text_color=C_TEXT).pack(side="left", padx=(20, 10))
        self.combo_method = ctk.CTkOptionMenu(
            row1, variable=self.method_var,
            values=["Linear Stretch", "Equalization", "CLAHE", "Hanya Hilangkan Frame"],
            width=185,
            fg_color=C_BTN_SLATE, button_color=C_BTN_SLATE, button_hover_color=C_BTN_SLATE_H,
            command=self.on_method_change
        )
        self.combo_method.pack(side="left")

        row2 = ctk.CTkFrame(f1, fg_color="transparent")
        row2.grid(row=2, column=0, sticky="w", padx=15, pady=(0, 10))
        
        ctk.CTkLabel(row2, text="Linear (%):", text_color=C_TEXT).pack(side="left", padx=(0,5))
        self.entry_lower = ctk.CTkEntry(row2, width=50)
        self.entry_lower.insert(0, "2.0")
        self.entry_lower.pack(side="left")
        
        ctk.CTkLabel(row2, text="-", text_color=C_TEXT).pack(side="left", padx=5)
        self.entry_upper = ctk.CTkEntry(row2, width=50)
        self.entry_upper.insert(0, "98.0")
        self.entry_upper.pack(side="left")

        ctk.CTkLabel(row2, text="CLAHE (Clip/Tile):", text_color=C_TEXT).pack(side="left", padx=(30,5))
        self.entry_clip = ctk.CTkEntry(row2, width=50)
        self.entry_clip.insert(0, "0.01")
        self.entry_clip.pack(side="left")
        
        ctk.CTkLabel(row2, text="/", text_color=C_TEXT).pack(side="left", padx=5)
        self.entry_tile = ctk.CTkEntry(row2, width=50)
        self.entry_tile.insert(0, "8")
        self.entry_tile.pack(side="left")

        row3 = ctk.CTkFrame(f1, fg_color="transparent")
        row3.grid(row=3, column=0, sticky="w", padx=15, pady=(0, 15))

        self.check_remove_frame = ctk.CTkCheckBox(
            row3,
            text="Hilangkan Frame Hitam",
            variable=self.remove_frame_var,
            text_color=C_TEXT,
            fg_color=C_BTN_TEAL[0],
            hover_color=C_BTN_TEAL_H[0]
        )
        self.check_remove_frame.pack(side="left")

        ctk.CTkLabel(row3, text="Ukuran Blok Tile:", text_color=C_TEXT).pack(side="left", padx=(25, 5))
        self.entry_block_tile = ctk.CTkEntry(row3, width=55)
        self.entry_block_tile.insert(0, "512")
        self.entry_block_tile.pack(side="left")

        f2 = ctk.CTkFrame(self.root, fg_color=C_CARD_BG, border_color=C_BORDER, border_width=1, corner_radius=8)
        f2.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(f2, text="Lokasi Data", font=ctk.CTkFont(size=14, weight="bold"), text_color=C_TEXT).pack(anchor="w", padx=15, pady=(10, 5))

        self.path_in = tk.StringVar()
        self.path_out = tk.StringVar()
        self.suffix_var = tk.StringVar(value="_stretch")
        self.suffix_var.trace_add("write", self._on_suffix_edited)

        r_in_f = ctk.CTkFrame(f2, fg_color="transparent")
        r_in_f.pack(fill="x", padx=15, pady=5)
        self.lbl_in = ctk.CTkLabel(r_in_f, text="Input File:", width=90, anchor="w", text_color=C_TEXT)
        self.lbl_in.pack(side="left")
        ctk.CTkEntry(r_in_f, textvariable=self.path_in).pack(side="left", fill="x", expand=True, padx=10)
        ctk.CTkButton(r_in_f, text="Browse", width=80, fg_color=C_BTN_BROWSE, hover_color=C_BTN_BROW_H, text_color=C_BTN_BROW_T, command=self.browse_in).pack(side="left")

        r_out_f = ctk.CTkFrame(f2, fg_color="transparent")
        r_out_f.pack(fill="x", padx=15, pady=5)
        self.lbl_out = ctk.CTkLabel(r_out_f, text="Output File:", width=90, anchor="w", text_color=C_TEXT)
        self.lbl_out.pack(side="left")
        ctk.CTkEntry(r_out_f, textvariable=self.path_out).pack(side="left", fill="x", expand=True, padx=10)
        ctk.CTkButton(r_out_f, text="Browse", width=80, fg_color=C_BTN_BROWSE, hover_color=C_BTN_BROW_H, text_color=C_BTN_BROW_T, command=self.browse_out).pack(side="left")

        r_suf = ctk.CTkFrame(f2, fg_color="transparent")
        r_suf.pack(fill="x", padx=15, pady=(5, 15))
        ctk.CTkLabel(r_suf, text="Akhiran File (Suffix):", text_color=C_TEXT).pack(side="left")
        self.entry_suffix = ctk.CTkEntry(r_suf, width=120, textvariable=self.suffix_var)
        self.entry_suffix.pack(side="left", padx=10)
        
        self.check_subfolder = ctk.CTkCheckBox(r_suf, text="Proses Subfolder", text_color=C_TEXT, fg_color=C_BTN_TEAL[0], hover_color=C_BTN_TEAL_H[0])
        self.check_subfolder.pack(side="right")
        self.check_subfolder.configure(state="disabled") 

        f3 = ctk.CTkFrame(self.root, fg_color=C_CARD_BG, border_color=C_BORDER, border_width=1, corner_radius=8)
        f3.pack(fill="x", padx=20, pady=10)

        r_btn = ctk.CTkFrame(f3, fg_color="transparent")
        r_btn.pack(fill="x", padx=15, pady=15)
        
        self.btn_start = ctk.CTkButton(r_btn, text="MULAI PROSES", height=40, fg_color=C_BTN_TEAL, hover_color=C_BTN_TEAL_H, text_color="#FFFFFF", font=ctk.CTkFont(weight="bold"), command=self.start_processing)
        self.btn_start.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        self.btn_cancel = ctk.CTkButton(r_btn, text="BATAL", height=40, width=100, fg_color=C_BTN_CANCEL, hover_color=C_BTN_CANCEL_H, text_color="#FFFFFF", font=ctk.CTkFont(weight="bold"), state="disabled", command=self.cancel_processing)
        self.btn_cancel.pack(side="left", padx=5)

        self.btn_preview = ctk.CTkButton(r_btn, text="🔍 Preview Hasil", height=40, fg_color=C_BTN_SLATE, hover_color=C_BTN_SLATE_H, text_color="#FFFFFF", font=ctk.CTkFont(weight="bold"), state="disabled", command=self._open_preview)
        self.btn_preview.pack(side="left", padx=(5, 0))

        self.lbl_status = ctk.CTkLabel(f3, text="Sistem Siap.", text_color=C_TEXT_MUTED)
        self.lbl_status.pack(anchor="w", padx=15)
        
        # PERBAIKAN: Label Timer & ETA digabungkan
        self.lbl_timer = ctk.CTkLabel(f3, text="Waktu Berjalan: 00:00 | Sisa Waktu (Est): -", font=ctk.CTkFont(size=11), text_color=C_TEXT_MUTED)
        self.lbl_timer.pack(anchor="e", padx=15, pady=(0, 2))

        ctk.CTkLabel(f3, text="Progress File:", text_color=C_TEXT).pack(anchor="w", padx=15)
        self.pbar_file = ctk.CTkProgressBar(f3, progress_color=C_PROG_BLUE, fg_color=C_PROG_BG)
        self.pbar_file.pack(fill="x", padx=15, pady=(0, 10))
        self.pbar_file.set(0)

        ctk.CTkLabel(f3, text="Progress Total:", text_color=C_TEXT).pack(anchor="w", padx=15)
        self.pbar_total = ctk.CTkProgressBar(f3, progress_color=C_PROG_BLUE, fg_color=C_PROG_BG)
        self.pbar_total.pack(fill="x", padx=15, pady=(0, 15))
        self.pbar_total.set(0)

        self.cancel_flag = False
        self._last_in_path = None
        self._last_out_path = None

    def on_method_change(self, selected_method):
        if selected_method == "Hanya Hilangkan Frame":
            self.remove_frame_var.set(True)
            self.check_remove_frame.configure(state="disabled")
            for w in (self.entry_lower, self.entry_upper, self.entry_clip, self.entry_tile):
                w.configure(state="disabled")
        else:
            self.check_remove_frame.configure(state="normal")
            for w in (self.entry_lower, self.entry_upper, self.entry_clip, self.entry_tile):
                w.configure(state="normal")

    def _on_suffix_edited(self, *_):
        if self.mode_var.get() == "single":
            in_p = self.path_in.get().strip()
            if in_p:
                base, _ = os.path.splitext(in_p)
                self.path_out.set(f"{base}{self.suffix_var.get().strip()}.tif")

    # ==========================================
    #  FUNGSI UI & ANIMASI WAKTU
    # ==========================================
    def update_pbar_smooth(self, pbar, target_val):
        current = pbar.get()
        if target_val <= 0 or target_val <= current:
            pbar.set(target_val)
            return
            
        steps = 20 
        diff = target_val - current
        
        def animate(step_count):
            if step_count <= steps:
                try:
                    pbar.set(current + (diff * (step_count / steps)))
                    self.root.after(10, animate, step_count + 1)
                except Exception:
                    pass
        animate(1)

    def start_timer(self):
        """Memulai stopwatch waktu berjalan"""
        self.is_processing = True
        self.start_time = time.time()
        self.current_eta_str = "-"
        self._loop_timer()

    def stop_timer(self):
        """Menghentikan stopwatch"""
        self.is_processing = False

    def _loop_timer(self):
        """Looping yang berjalan setiap detik untuk meng-update UI Waktu"""
        if not self.is_processing:
            return
            
        elapsed = int(time.time() - self.start_time)
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        
        # Format string Waktu Berjalan
        if h > 0:
            elapsed_str = f"{h:02d}:{m:02d}:{s:02d}"
        else:
            elapsed_str = f"{m:02d}:{s:02d}"
            
        self.lbl_timer.configure(text=f"Waktu Berjalan: {elapsed_str} | Sisa Waktu (Est): {self.current_eta_str}")
        
        # Panggil ulang diri sendiri setiap 1 detik
        self.root.after(1000, self._loop_timer)

    def _upd_eta(self, sec):
        """Fungsi helper untuk mengatur string ETA dari thread"""
        if sec < 0:
            self.current_eta_str = "-"
            return
        s = int(sec)
        m, r = divmod(s, 60)
        h, m = divmod(m, 60)
        
        if h > 0:
            self.current_eta_str = f"{h:02d}:{m:02d}:{r:02d}"
        else:
            self.current_eta_str = f"{m:02d}:{r:02d}"

    def _upd_status(self, txt):
        self.root.after(0, lambda: self.lbl_status.configure(text=txt))

    # ==========================================
    #  FUNGSI BROWSER
    # ==========================================
    def update_mode_ui(self):
        mode = self.mode_var.get()
        self.path_in.set("")
        self.path_out.set("")
        
        if mode == "single":
            self.lbl_in.configure(text="Input File:")
            self.lbl_out.configure(text="Output File:")
            self.check_subfolder.configure(state="disabled")
            self.btn_preview.configure(state="normal" if self._last_out_path else "disabled")
        else:
            self.lbl_in.configure(text="Input Folder:")
            self.lbl_out.configure(text="Output Folder:")
            self.check_subfolder.configure(state="normal")
            self.btn_preview.configure(state="disabled") 

    def browse_in(self):
        mode = self.mode_var.get()
        if mode == "single":
            p = filedialog.askopenfilename(title="Pilih File", filetypes=[("Raster", "*.tif *.img *.jp2 *.vrt")])
            if p:
                self.path_in.set(p)
                base, _ = os.path.splitext(p)
                self.path_out.set(f"{base}{self.entry_suffix.get()}.tif")
        else:
            d = filedialog.askdirectory(title="Pilih Folder Input")
            if d:
                self.path_in.set(d)
                self.path_out.set(d) 

    def browse_out(self):
        mode = self.mode_var.get()
        if mode == "single":
            p = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("GeoTIFF", "*.tif")])
            if p: self.path_out.set(p)
        else:
            d = filedialog.askdirectory(title="Pilih Folder Output")
            if d: self.path_out.set(d)

    def cancel_processing(self):
        self.cancel_flag = True
        self.lbl_status.configure(text="Membatalkan proses...")

    def _open_preview(self):
        if self.mode_var.get() == "batch":
            messagebox.showinfo("Preview", "Fitur preview hanya untuk mode Single File.")
            return
            
        if self._last_in_path and self._last_out_path and os.path.exists(self._last_out_path):
            self.preview_win = PreviewWindow(self.root, self._last_in_path, self._last_out_path)
        else:
            messagebox.showwarning("Oops!", "Lakukan proses citra tunggal terlebih dahulu.")

    # ==========================================
    #  FUNGSI EKSEKUSI
    # ==========================================
    def start_processing(self):
        mode = self.mode_var.get()
        method = self.method_var.get()
        remove_frame = bool(self.remove_frame_var.get()) or (method == "Hanya Hilangkan Frame")
        try:
            lower = float(self.entry_lower.get())
            upper = float(self.entry_upper.get())
            clip_limit = float(self.entry_clip.get())
            tile_grid = int(self.entry_tile.get())
            block_tile_size = int(self.entry_block_tile.get())
            if block_tile_size <= 0 or block_tile_size % 16 != 0:
                messagebox.showerror("Error", "Ukuran Blok Tile harus kelipatan 16 (misal: 256, 512, 1024)!")
                return
        except ValueError:
            messagebox.showerror("Error", "Parameter angka tidak valid!")
            return

        in_path = self.path_in.get().strip()
        out_path = self.path_out.get().strip()

        if not in_path:
            messagebox.showerror("Error", "Lokasi Input tidak boleh kosong!")
            return

        if mode == "single":
            if not os.path.exists(in_path):
                messagebox.showerror("Error", "File input tidak ditemukan!")
                return
            self._run_single(in_path, out_path, method, lower, upper, clip_limit, tile_grid, remove_frame, block_tile_size)
        else:
            if not os.path.isdir(in_path):
                messagebox.showerror("Error", "Folder input tidak ditemukan!")
                return
            self._run_batch(in_path, out_path, method, lower, upper, clip_limit, tile_grid, remove_frame, block_tile_size)

    def _run_single(self, in_p, out_p, method, lower, upper, clip_limit, tile_grid, remove_frame, block_tile_size):
        self.btn_start.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.btn_preview.configure(state="disabled")
        self.pbar_file.set(0)
        self.pbar_total.set(0)
        self.cancel_flag = False
        
        # Mulai Timer
        self.start_timer()

        def worker():
            try:
                self.root.after(0, lambda: self.pbar_total.set(0))
                process_one_file(
                    in_p, out_p, method, lower, upper, clip_limit, tile_grid,
                    remove_frame=remove_frame,
                    block_tile_size=block_tile_size,
                    per_file_progress_cb=lambda val: self.root.after(0, lambda: self.update_pbar_smooth(self.pbar_file, val)),
                    status_cb=self._upd_status,
                    eta_cb=self._upd_eta, # Untuk file tunggal, hitung estimasi berdasarkan band
                    cancel_check_cb=lambda: self.cancel_flag
                )
                self.root.after(0, lambda: self.update_pbar_smooth(self.pbar_total, 1.0))
                self._upd_status("Proses Selesai! Klik tombol Preview Hasil.")
                self.current_eta_str = "Selesai"
                
                self._last_in_path, self._last_out_path = in_p, out_p
                _preview_cache.pop(out_p, None)  
                
                self.root.after(0, lambda: messagebox.showinfo("Sukses", "Citra berhasil diproses!"))
            except Exception as e:
                self._upd_status("Proses gagal atau dibatalkan.")
                self.current_eta_str = "-"
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.stop_timer() # Hentikan Timer
                def _restore():
                    self.btn_start.configure(state="normal")
                    self.btn_cancel.configure(state="disabled")
                    if self._last_out_path and os.path.exists(self._last_out_path):
                        self.btn_preview.configure(state="normal")
                self.root.after(0, _restore)

        threading.Thread(target=worker, daemon=True).start()

    def _run_batch(self, in_d, out_d, method, lower, upper, clip_limit, tile_grid, remove_frame, block_tile_size):
        self.btn_start.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.btn_preview.configure(state="disabled")
        self.pbar_file.set(0)
        self.pbar_total.set(0)
        self.cancel_flag = False

        exts = (".tif", ".tiff", ".img", ".jp2", ".vrt", ".bil", ".hdr", ".ntf", ".pix")
        files = []
        recurse = self.check_subfolder.get()

        if recurse:
            for rootd, _, fnames in os.walk(in_d):
                for fn in fnames:
                    if fn.lower().endswith(exts):
                        files.append(os.path.join(rootd, fn))
        else:
            for fn in os.listdir(in_d):
                p = os.path.join(in_d, fn)
                if os.path.isfile(p) and fn.lower().endswith(exts):
                    files.append(p)

        if not files:
            messagebox.showinfo("Tidak ada file", "Tidak ditemukan file raster dengan format yang didukung di folder tersebut.")
            self.btn_start.configure(state="normal")
            self.btn_cancel.configure(state="disabled")
            return

        # Mulai Timer
        self.start_timer()

        def worker():
            t0_batch = time.perf_counter()
            n = len(files)
            suffix = self.entry_suffix.get().strip()
            
            for idx, in_p in enumerate(files, start=1):
                if self.cancel_flag:
                    self._upd_status("Dibatalkan oleh pengguna.")
                    self.current_eta_str = "-"
                    break
                
                base = os.path.splitext(os.path.basename(in_p))[0]
                out_p = os.path.join(out_d, f"{base}{suffix}.tif")
                
                self.root.after(0, lambda: self.pbar_file.set(0))

                try:
                    # PERBAIKAN ETA: Untuk Batch, kita hitung ETA berdasarkan file, bukan band (agar lebih akurat).
                    # Jadi kita matikan ETA dari process_one_file (eta_cb=None).
                    process_one_file(
                        in_p, out_p, method, lower, upper, clip_limit, tile_grid,
                        remove_frame=remove_frame,
                        block_tile_size=block_tile_size,
                        per_file_progress_cb=lambda val: self.root.after(0, lambda: self.update_pbar_smooth(self.pbar_file, val)),
                        status_cb=lambda s: self._upd_status(f"[{idx}/{n}] {s}"),
                        eta_cb=None,
                        cancel_check_cb=lambda: self.cancel_flag
                    )
                except Exception as e:
                    self._upd_status(f"Error pada {os.path.basename(in_p)}: {e}")

                # Hitung ETA Batch (Total Waktu yang Berjalan / Jumlah File yang Selesai * Sisa File)
                elapsed_batch = time.perf_counter() - t0_batch
                files_left = n - idx
                if idx > 0 and files_left > 0:
                    time_per_file = elapsed_batch / idx
                    eta_sec = time_per_file * files_left
                    self._upd_eta(eta_sec)
                else:
                    self.current_eta_str = "Selesai"

                # Update Progress Keseluruhan
                self.root.after(0, lambda v=idx/n: self.update_pbar_smooth(self.pbar_total, v))

            self.stop_timer() # Hentikan Timer

            t1_batch = time.perf_counter()
            if not self.cancel_flag:
                self._upd_status(f"Selesai! {n} file diproses. Total waktu: {t1_batch - t0_batch:.1f} detik")
                self.root.after(0, lambda: messagebox.showinfo("Selesai", f"Batch selesai!\n{n} file telah diproses dan disimpan di:\n{out_d}"))
            
            def _restore():
                self.btn_start.configure(state="normal")
                self.btn_cancel.configure(state="disabled")
            self.root.after(0, _restore)

        threading.Thread(target=worker, daemon=True).start()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = RasterHistGUI()
    app.run()