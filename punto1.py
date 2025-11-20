import pandas as pd
from pathlib import Path
import numpy as np

# === 1) Carpeta donde están TODOS los .txt ===
CARPETA_DATOS = Path(
    r"D:\ALMACENAMIENTO\UTN Tecnicatura\2do AÑO\2do Cuatrimestre\Introducción al Análisis de Datos\TP final\datos"
)

# Ahora busco TODOS los .txt de la carpeta
archivos = sorted(CARPETA_DATOS.glob("*.txt"))

print("Archivos encontrados:")
for a in archivos:
    print(" -", a.name)

dfs = []

# columnas mínimas que necesito para el análisis
columnas_necesarias = {"ANO4", "TRIMESTRE", "AGLOMERADO", "P47T"}

for archivo in archivos:
    print(f"\nLeyendo {archivo.name}...")
    df = pd.read_csv(
        archivo,
        sep=";",
        encoding="latin1",
        low_memory=False
    )

    # chequeo columnas necesarias
    if not columnas_necesarias.issubset(df.columns):
        print(f"  -> Se omite {archivo.name} (no tiene columnas necesarias).")
        continue

    # convertir P47T a número (fundamental)
    df["P47T"] = pd.to_numeric(df["P47T"], errors="coerce")

    df["archivo_origen"] = archivo.name
    dfs.append(df)


if not dfs:
    raise ValueError("No se cargó ningún archivo válido. Revisá que los .txt tengan las columnas esperadas.")

# Uno todo en un solo DataFrame
eph = pd.concat(dfs, ignore_index=True)
print("\nForma total EPH:", eph.shape)
print("Columnas (primeras 30):", eph.columns.tolist()[:30])

# === 2) Filtro por aglomerado ===

AGLOS = {20: "Río Gallegos", 27: "Gran San Juan"}

eph = eph[eph["AGLOMERADO"].isin(AGLOS.keys())].copy()
eph["aglomerado_nombre"] = eph["AGLOMERADO"].map(AGLOS)

# Población de 14 años o más (criterio clásico laboral)
# eph = eph[eph["CH06"] >= 14].copy()

print("\nLuego de filtrar aglomerados")
print("Forma:", eph.shape)
print(eph[["ANO4", "TRIMESTRE", "AGLOMERADO", "aglomerado_nombre"]].head())

# === 3) Variables de ingresos ===

if "P47T" not in eph.columns:
    raise ValueError("No encuentro la columna P47T (ingreso total individual).")

# ingreso nominal (después lo vamos a deflactar para inflación)
eph = eph.copy()
eph["ingreso"] = eph["P47T"].replace(-9, np.nan)  # -9 = no respuesta

# Elijo ponderador adecuado
if "PONDII" in eph.columns:
    ponder_col = "PONDII"
else:
    ponder_col = "PONDERA"  # fallback razonable

eph["ponderador"] = eph[ponder_col]

# Me quedo sólo con quienes tienen ingreso válido (>0)
eph_valid = eph[(eph["ingreso"].notna()) & (eph["ingreso"] > 0)].copy()
print("\nCon ingreso válido > 0:", eph_valid.shape)

# === 4) Funciones de estadística ponderada ===

def weighted_mean(x, w):
    return (x * w).sum() / w.sum()

def weighted_quantile(values, weights, quantile):
    """values y weights numpy arrays, quantile entre 0 y 1"""
    sorter = np.argsort(values)
    v = values[sorter]
    w = weights[sorter]
    cumsum = np.cumsum(w)
    cutoff = quantile * cumsum[-1]
    return v[cumsum >= cutoff][0]

# === 5) Cálculo de medidas de tendencia central y posición ===

resultados = []

for (anio, aglo), grupo in eph_valid.groupby(["ANO4", "aglomerado_nombre"]):
    vals = grupo["ingreso"].to_numpy()
    pesos = grupo["ponderador"].to_numpy()

    media = weighted_mean(vals, pesos)
    mediana = weighted_quantile(vals, pesos, 0.5)
    p10 = weighted_quantile(vals, pesos, 0.10)
    p25 = weighted_quantile(vals, pesos, 0.25)
    p75 = weighted_quantile(vals, pesos, 0.75)
    p90 = weighted_quantile(vals, pesos, 0.90)

    resultados.append({
        "anio": anio,
        "aglomerado": aglo,
        "media_ingreso": media,
        "mediana_ingreso": mediana,
        "p10_ingreso": p10,
        "p25_ingreso": p25,
        "p75_ingreso": p75,
        "p90_ingreso": p90,
    })

tabla_univariada = pd.DataFrame(resultados).sort_values(["aglomerado", "anio"])

print("\n=== Tabla univariada (primeras filas) ===")
print(tabla_univariada.head(20))

# Si querés, exportamos a CSV para meter en Excel/Word:
tabla_univariada.to_csv("punto1_ingresos_univariado.csv", index=False)
print("\nSe guardó 'punto1_ingresos_univariado.csv'")

# === 6) IPC por año (base 100 en 2016) ===
ipc = {
    2016: 100.000000,
    2017: 112.985075,
    2018: 151.569817,
    2019: 231.605092,
    2020: 325.364358,
    2021: 478.615058,
    2022: 828.603683,
    2023: 1939.297233,
    2024: 6263.886042,
    2025: 8741.757380
}

ipc_df = pd.DataFrame(list(ipc.items()), columns=["anio", "ipc"])

# unimos IPC con tabla univariada
tabla_univariada = tabla_univariada.merge(ipc_df, on="anio", how="left")

# ingreso real = nominal * (IPC_base / IPC_año)
IPC_BASE = 100

tabla_univariada["media_real"] = tabla_univariada["media_ingreso"] * (IPC_BASE / tabla_univariada["ipc"])
tabla_univariada["mediana_real"] = tabla_univariada["mediana_ingreso"] * (IPC_BASE / tabla_univariada["ipc"])
tabla_univariada["p10_real"] = tabla_univariada["p10_ingreso"] * (IPC_BASE / tabla_univariada["ipc"])
tabla_univariada["p25_real"] = tabla_univariada["p25_ingreso"] * (IPC_BASE / tabla_univariada["ipc"])
tabla_univariada["p75_real"] = tabla_univariada["p75_ingreso"] * (IPC_BASE / tabla_univariada["ipc"])
tabla_univariada["p90_real"] = tabla_univariada["p90_ingreso"] * (IPC_BASE / tabla_univariada["ipc"])

print("\n=== Tabla univariada REAL (primeras filas) ===")
print(tabla_univariada.head(20))

# Exportar también la tabla con ingresos reales
tabla_univariada.to_csv("punto1_ingresos_univariado_reales.csv", index=False)
print("\nSe guardó 'punto1_ingresos_univariado_reales.csv'")

# gráficos

import matplotlib.pyplot as plt

# === Gráfico SOLO de ingresos reales ===
plt.figure(figsize=(12,6))

for aglo in tabla_univariada["aglomerado"].unique():
    sub = tabla_univariada[tabla_univariada["aglomerado"] == aglo]

    plt.plot(sub["anio"], sub["media_real"], marker="o", linestyle="-", label=f"Media real - {aglo}")
    plt.plot(sub["anio"], sub["mediana_real"], marker="s", linestyle="--", label=f"Mediana real - {aglo}")

plt.title("Evolución del ingreso real (deflactado por IPC) — Río Gallegos y Gran San Juan")
plt.xlabel("Año")
plt.ylabel("Ingreso real (base 2016 = 100)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Gráfico de percentiles reales (P10, P50, P90) ===

plt.figure(figsize=(12,6))

for aglo in tabla_univariada["aglomerado"].unique():
    sub = tabla_univariada[tabla_univariada["aglomerado"] == aglo]
    
    plt.plot(sub["anio"], sub["p10_real"], marker="o", linestyle="-", label=f"P10 real - {aglo}")
    plt.plot(sub["anio"], sub["mediana_real"], marker="o", linestyle="--", label=f"P50/Mediana real - {aglo}")
    plt.plot(sub["anio"], sub["p90_real"], marker="o", linestyle="-.", label=f"P90 real - {aglo}")

plt.title("Evolución de percentiles reales del ingreso — Río Gallegos y Gran San Juan")
plt.xlabel("Año")
plt.ylabel("Ingreso real (base 2016 = 100)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
