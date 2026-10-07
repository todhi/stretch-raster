# Stretch v2.0

> **A modern, high-performance desktop tool for geospatial raster contrast enhancement and intelligent black border removal.**

Stretch v2.0 is built with Python, [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter), and [Rasterio](https://github.com/rasterio/rasterio). It provides remote sensing analysts and GIS professionals with an intuitive interface to enhance satellite imagery and aerial orthophotos (GeoTIFF) while safeguarding geospatial metadata, projection, and data integrity.

---

## Key Features

### 1. Contrast Enhancement Algorithms
- **Linear Percentile Stretch**: Dynamically clips outliers using customizable lower and upper percentiles (default: `2.0%` – `98.0%`) to expand dynamic range without blowing out highlights or shadows.
- **Histogram Equalization**: Redistributes pixel values across the entire radiometric range using cumulative distribution matching.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Local adaptive histogram enhancement that improves contrast in subtle terrains while avoiding over-amplification of noise via clip-limit thresholds.

### 2. Intelligent Black Frame Removal (`NoData = 0`)
- **True Outer Border vs. Internal Data Separation**: Automatically detects image boundaries and marks extraneous outer black margins as `NoData = 0` (transparent in QGIS, ArcGIS, and Global Mapper).
- **Waterbody & Shadow Preservation**: Applies morphological binary dilation buffers (`3x3`) around valid data pixels, guaranteeing that internal zero-value pixels (such as deep water bodies, cast building shadows, or dark volcanic soils) are preserved at $\ge 1$ and never hollowed out.
- **Standalone Mode**: Select *"Hanya Hilangkan Frame"* to clean borders without altering the original pixel radiometry.

### 3. Dual Processing Modes
- **Single File Mode**: Process individual raster files with real-time filename suffix updates and instant side-by-side preview.
- **Batch Folder Mode**: Recursively or flatly process directories of satellite rasters (`.tif`, `.tiff`, `.img`, `.jp2`, etc.) with automated queue management.

### 4. Interactive Before vs. After Preview
- Side-by-side visual comparison window.
- Renders transparent `NoData` areas with a standard checkered grid pattern, allowing immediate verification of border removal before opening files in GIS software.

### 5. Multi-Threaded Engine & Real-Time Telemetry
- Non-blocking asynchronous worker threads.
- Smooth progress bars, elapsed stopwatch timer, and intelligent estimated time of arrival (ETA).
- Responsive cancellation support.

---

## Supported Formats

Supports any geospatial format readable by GDAL/Rasterio, including:
- **GeoTIFF** (`.tif`, `.tiff`)
- **Erdas Imagine** (`.img`)
- **JPEG 2000** (`.jp2`)
- **Virtual Raster** (`.vrt`)
- **Band Interleaved / Raw Binary** (`.bil`, `.hdr`, `.ntf`, `.pix`)

Output rasters are saved as tiled GeoTIFFs with LZW compression (`compress='lzw'`, `tiled=True`, `BIGTIFF='IF_SAFER'`).

---

## Installation & Requirements

### Prerequisites
- Python 3.8 or higher.

### Quick Start (Windows)
Simply double-click the included batch launcher:
```cmd
Jalankan_Stretch.bat
```
*Note: The launcher automatically checks for missing Python libraries and installs them on first launch.*

### Manual Installation
Clone this repository and install the dependencies:
```bash
git clone https://github.com/<your-username>/stretch.git
cd stretch
pip install -r requirements.txt
python main.py
```

---

## Project Structure

```text
stretch/
├── stretch.py                   # Core application engine and CustomTkinter GUI
├── main.py                      # Application entry point
├── Jalankan_Stretch.bat         # 1-Click Windows launcher
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore specifications
├── LICENSE                      # MIT License
├── CONTRIBUTING.md              # Contribution guidelines
└── README.md                    # Project documentation
```

---

## Dependencies

- [`numpy`](https://numpy.org/) - Numerical matrix manipulation
- [`rasterio`](https://rasterio.readthedocs.io/) - Geospatial raster I/O
- [`pillow`](https://python-pillow.org/) - Image handling
- [`scikit-image`](https://scikit-image.org/) - Exposure & CLAHE algorithms
- [`scipy`](https://scipy.org/) - Morphological operations (`binary_dilation`)
- [`customtkinter`](https://github.com/TomSchimansky/CustomTkinter) - Modern dark/light theme desktop UI

---

## License

This project is licensed under the [MIT License](LICENSE).
