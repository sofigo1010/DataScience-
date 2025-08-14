
import os
from pathlib import Path
import openeo

# =============================
# Paso 1: Conexión y login OIDC
# =============================
BACKEND = "https://openeo.dataspace.copernicus.eu"
conn = openeo.connect(BACKEND)
conn = conn.authenticate_oidc()  # device flow: sigue el enlace/código que aparece en la terminal
print("[OK] Autenticado en", BACKEND)

# ===============================================
# Paso 2: Descarga de .tif por lago y por fechas
# ===============================================
# AOIs (BBox en EPSG:4326)
lago_atitlan = {
    "west": -91.326256,
    "east": -91.07151,
    "south": 14.5948,
    "north": 14.750979,
}

lago_amatitlan = {
    "west": -90.638065,
    "east": -90.512924,
    "south": 14.412347,
    "north": 14.493799,
}

# Fechas con nubosidad < 20%
FECHAS = [
    "2025-02-07","2025-02-10","2025-02-25","2025-02-27","2025-03-02","2025-03-04",
    "2025-03-07","2025-03-09","2025-03-12","2025-03-14","2025-03-19","2025-03-22",
    "2025-03-24","2025-03-26","2025-04-03","2025-04-11","2025-04-13","2025-04-15",
    "2025-04-16","2025-04-18","2025-04-28","2025-05-03","2025-05-13","2025-05-28",
    "2025-07-10","2025-07-17","2025-07-20","2025-07-24","2025-08-01",
]

# Directorio de salida
OUT_DIR = Path("data/s2_lakes_tifs")
(OUT_DIR / "atitlan").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "amatitlan").mkdir(parents=True, exist_ok=True)

# Bandas clave (incluye B05 para NDCI y SCL para futuras máscaras). 
# Nota: S2 tiene resoluciones mixtas (10/20/60m). Reescalamos todo a 10m.
BANDS = ["B02", "B03", "B04", "B05", "B08", "B8A", "SCL"]
TARGET_RES_M = 10


def download_lake(lake_name: str, bbox: dict, dates: list[str]):
    lake_dir = OUT_DIR / lake_name
    for d in dates:
        out_path = lake_dir / f"{lake_name}_S2_{d}.tif"
        if out_path.exists():
            print(f"[SKIP] Ya existe {out_path}")
            continue
        try:
            print(f"[LOAD] {lake_name} {d} …")
            cube = conn.load_collection(
                "SENTINEL2_L2A",
                spatial_extent=bbox,
                temporal_extent=[d, d],
                bands=BANDS,
                max_cloud_cover=20,
            )
            # Armoniza resolución a 10m para combinar bandas sin conflictos
            cube = cube.resample_spatial(resolution=TARGET_RES_M, method="near")
            # Si hay más de una observación en el día, tomamos el máximo por tiempo (quick mosaic)
            cube = cube.max_time()

            # Descarga GeoTIFF multibanda
            cube.download(str(out_path))
            print(f"[OK] Guardado: {out_path}")
        except Exception as e:
            print(f"[WARN] {lake_name} {d}: {e}. Posible falta de datos/huella parcial. Continuando…")


if __name__ == "__main__":
    download_lake("atitlan", lago_atitlan, FECHAS)
    download_lake("amatitlan", lago_amatitlan, FECHAS)
    print("\n[DONE] Paso 2 completado: GeoTIFFs guardados en:", OUT_DIR.resolve())
