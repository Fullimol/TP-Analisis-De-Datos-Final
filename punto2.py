# ==========================================================
# PUNTO 2 - ANALISIS MULTIVARIADO - TP ANALISIS DE DATOS
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------------------
# 1. CARGA DE LOS DATOS (LEER TODOS LOS TXT)
# -----------------------------------------

carpeta = Path(
    r"D:\ALMACENAMIENTO\UTN Tecnicatura\2do AÑO\2do Cuatrimestre\Introducción al Análisis de Datos\TP final\datos"
)

# cargo TODOS los .txt de la carpeta
archivos = sorted(list(carpeta.glob("*.txt")))

df_list = []

# columnas mínimas necesarias para el análisis
columnas_necesarias = {"ANO4", "TRIMESTRE", "AGLOMERADO", "P47T", "CH06"}

for arch in archivos:
    print("Leyendo:", arch.name)
    temp = pd.read_csv(
        arch,
        sep=";",
        encoding="latin1",
        low_memory=False
    )

    # evitar cargar archivos 'hogar' u otros formatos
    if not columnas_necesarias.issubset(temp.columns):
        print(f"  -> Se omite {arch.name} (no tiene columnas necesarias)")
        continue

    # asegurar que P47T sea numérico
    temp["P47T"] = pd.to_numeric(temp["P47T"], errors="coerce")

    df_list.append(temp)

# concatenar todos los trimestres de todos los años
df = pd.concat(df_list, ignore_index=True)

print("\nDataFrame cargado. Forma:", df.shape)
print(df.head())

# -----------------------------------------
# 2. FILTROS: AGLOMERADOS + EDAD >=14 + INGRESO > 0
# -----------------------------------------
# (NO filtramos TRIMESTRE: usamos todos los trimestres)

aglos = {20: "Río Gallegos", 27: "Gran San Juan"}

df = df[df["AGLOMERADO"].isin(aglos.keys())].copy()
df["aglomerado_nombre"] = df["AGLOMERADO"].map(aglos)

# población de 14 años o más
df = df[df["CH06"] >= 14]

# ingreso válido > 0
df = df[df["P47T"] > 0]

print("\nLuego de filtros básicos. Forma:", df.shape)

# -----------------------------------------
# 3. IPC Y INGRESO REAL
# -----------------------------------------

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

df["ipc"] = df["ANO4"].map(ipc)
df["ingreso_real"] = df["P47T"] * (100 / df["ipc"])

# -----------------------------------------
# 4. VARIABLES MULTIVARIADAS
# -----------------------------------------

# sexo
df["sexo"] = df["CH04"].map({1: "Varón", 2: "Mujer"})

# nivel educativo
niveles = {
    1: "Primaria incompleta",
    2: "Primaria completa",
    3: "Secundaria incompleta",
    4: "Secundaria completa",
    5: "Sup/Univ incompleto",
    6: "Sup/Univ completo"
}
df["nivel_educ"] = df["NIVEL_ED"].map(niveles)
df = df[~df["nivel_educ"].isna()]

# grupos de edad
df["grupo_edad"] = pd.cut(
    df["CH06"],
    bins=[14, 24, 44, 64, 100],
    labels=["14–24", "25–44", "45–64", "65+"]
)

# categoría ocupacional (PP04B_COD = posición en la ocupación)
cat_map = {
    1: "Patrón/empleador",
    2: "Cuenta propia",
    3: "Obrero/empleado",
    4: "Trabajador familiar"
}
df["cat_ocup_label"] = df["PP04B_COD"].map(cat_map)

# -----------------------------------------
# 5. TABLAS MULTIVARIADAS
# -----------------------------------------

tabla_sexo = df.groupby(
    ["ANO4", "aglomerado_nombre", "sexo"], as_index=False
)["ingreso_real"].mean()

tabla_educ = df.groupby(
    ["ANO4", "aglomerado_nombre", "nivel_educ"], as_index=False
)["ingreso_real"].mean()

tabla_edad = df.groupby(
    ["ANO4", "aglomerado_nombre", "grupo_edad"], as_index=False
)["ingreso_real"].mean()

tabla_cat = df.groupby(
    ["ANO4", "aglomerado_nombre", "cat_ocup_label"], as_index=False
)["ingreso_real"].mean()

# -----------------------------------------
# 7. GRÁFICOS MULTIVARIADOS
# -----------------------------------------

# 7.1 Sexo
plt.figure(figsize=(12, 6))
for sexo in tabla_sexo["sexo"].unique():
    for aglo in tabla_sexo["aglomerado_nombre"].unique():
        sub = tabla_sexo[
            (tabla_sexo["sexo"] == sexo) &
            (tabla_sexo["aglomerado_nombre"] == aglo)
        ]
        if len(sub) > 0:
            plt.plot(sub["ANO4"], sub["ingreso_real"], marker="o", label=f"{sexo} – {aglo}")

plt.title("Ingreso real promedio por sexo y aglomerado (todos los trimestres)")
plt.xlabel("Año")
plt.ylabel("Ingreso real")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 7.2 Nivel educativo
plt.figure(figsize=(12, 6))
for nivel in tabla_educ["nivel_educ"].unique():
    for aglo in tabla_educ["aglomerado_nombre"].unique():
        sub = tabla_educ[
            (tabla_educ["nivel_educ"] == nivel) &
            (tabla_educ["aglomerado_nombre"] == aglo)
        ]
        if len(sub) > 0:
            plt.plot(sub["ANO4"], sub["ingreso_real"], marker="o", label=f"{nivel} – {aglo}")

plt.title("Ingreso real por nivel educativo y aglomerado (todos los trimestres)")
plt.xlabel("Año")
plt.ylabel("Ingreso real")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 7.3 Grupo de edad
plt.figure(figsize=(12, 6))
for edad in tabla_edad["grupo_edad"].unique():
    for aglo in tabla_edad["aglomerado_nombre"].unique():
        sub = tabla_edad[
            (tabla_edad["grupo_edad"] == edad) &
            (tabla_edad["aglomerado_nombre"] == aglo)
        ]
        if len(sub) > 0:
            plt.plot(sub["ANO4"], sub["ingreso_real"], marker="o", label=f"{edad} – {aglo}")

plt.title("Ingreso real por grupos de edad y aglomerado (todos los trimestres)")
plt.xlabel("Año")
plt.ylabel("Ingreso real")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 7.4 Categoría ocupacional (posición en la ocupación)
plt.figure(figsize=(12, 6))
for cat in tabla_cat["cat_ocup_label"].dropna().unique():
    for aglo in tabla_cat["aglomerado_nombre"].unique():
        sub = tabla_cat[
            (tabla_cat["cat_ocup_label"] == cat) &
            (tabla_cat["aglomerado_nombre"] == aglo)
        ]
        if len(sub) > 0:
            plt.plot(sub["ANO4"], sub["ingreso_real"], marker="o", label=f"{cat} – {aglo}")

plt.title("Ingreso real por categoría ocupacional (PP04B_COD) y aglomerado (todos los trimestres)")
plt.xlabel("Año")
plt.ylabel("Ingreso real")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("Punto 2 finalizado correctamente.")
