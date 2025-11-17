# ==========================================================
# PUNTO 2 - ANALISIS MULTIVARIADO - TP ANALISIS DE DATOS
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------------------
# 1. CARGA DE LOS DATOS (MISMO PROCEDIMIENTO DEL PUNTO 1)
# -----------------------------------------

carpeta = r"D:\ALMACENAMIENTO\UTN Tecnicatura\2do AÑO\2do Cuatrimestre\Introducción al Análisis de Datos\TP final\datos"
archivos = [a for a in os.listdir(carpeta) if a.lower().startswith("usu_individual")]

df_list = []

for arch in archivos:
    print("Leyendo:", arch)
    temp = pd.read_csv(
        os.path.join(carpeta, arch),
        sep=";", encoding="latin1", low_memory=False
    )
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)

# -----------------------------------------
# 2. FILTROS: TRIMESTRE 2 + AGLOMERADOS + EDAD >=14
# -----------------------------------------

# Mantener solo 2º trimestre (coherente con el punto 1)
df = df[df["TRIMESTRE"] == 2]

# Equivalencia de nombres
aglos = {
    20: "Río Gallegos",
    27: "Gran San Juan"
}

df = df[df["AGLOMERADO"].isin(aglos.keys())].copy()
df["aglomerado_nombre"] = df["AGLOMERADO"].map(aglos)

# edad >= 14
df = df[df["CH06"] >= 14]

# ingreso nominal válido
df = df[df["P47T"] > 0]

# -----------------------------------------
# 3. IPC (MISMO IPC DEL PUNTO 1) Y CÁLCULO DE INGRESO REAL
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
# 4. CREAR VARIABLES MULTIVARIADAS
# -----------------------------------------

# SEXO (CH04)
df["sexo"] = df["CH04"].map({1: "Varón", 2: "Mujer"})

# NIVEL EDUCATIVO (NIVEL_ED)
niveles = {
    1: "Primaria incompleta",
    2: "Primaria completa",
    3: "Secundaria incompleta",
    4: "Secundaria completa",
    5: "Superior/Univ incompleto",
    6: "Superior/Univ completo"
}
df["nivel_educ"] = df["NIVEL_ED"].map(niveles)
df = df[~df["nivel_educ"].isna()]

# GRUPOS DE EDAD
df["grupo_edad"] = pd.cut(
    df["CH06"],
    bins=[14, 24, 44, 64, 100],
    labels=["14–24", "25–44", "45–64", "65+"]
)

# CATEGORÍA OCUPACIONAL (PP04D_COD)
cat_map = {
    1: "Patrón/empleador",
    2: "Cuenta propia",
    3: "Obrero/empleado",
    4: "Trabajador familiar"
}
df["cat_ocup_label"] = df["PP04D_COD"].map(cat_map)


# -----------------------------------------
# 5. TABLAS MULTIVARIADAS PRINCIPALES
# -----------------------------------------

# --- 5.1 Ingreso real por sexo ---
tabla_sexo = df.groupby(
    ["ANO4", "aglomerado_nombre", "sexo"]
)["ingreso_real"].mean().reset_index()

# --- 5.2 Ingreso real por nivel educativo ---
tabla_educ = df.groupby(
    ["ANO4", "aglomerado_nombre", "nivel_educ"]
)["ingreso_real"].mean().reset_index()

# --- 5.3 Ingreso real por grupo de edad ---
tabla_edad = df.groupby(
    ["ANO4", "aglomerado_nombre", "grupo_edad"]
)["ingreso_real"].mean().reset_index()

# --- 5.4 Ingreso real por categoría ocupacional ---
tabla_cat = df.groupby(
    ["ANO4", "aglomerado_nombre", "cat_ocup_label"]
)["ingreso_real"].mean().reset_index()

# -----------------------------------------
# 6. EXPORTAR TABLAS A CSV
# -----------------------------------------

# salida = r"D:\ALMACENAMIENTO\UTN Tecnicatura\2do AÑO\2do Cuatrimestre\Introducción al Análisis de Datos\TP final\csv_punto2"
# os.makedirs(salida, exist_ok=True)

# tabla_sexo.to_csv(os.path.join(salida, "punto2_ingreso_real_por_sexo.csv"), index=False)
# tabla_educ.to_csv(os.path.join(salida, "punto2_ingreso_real_por_nivel_educ.csv"), index=False)
# tabla_edad.to_csv(os.path.join(salida, "punto2_ingreso_real_por_edad.csv"), index=False)
# tabla_cat.to_csv(os.path.join(salida, "punto2_ingreso_real_por_categoria_ocupacional.csv"), index=False)

# -----------------------------------------
# 7. GRAFICOS MULTIVARIADOS
# -----------------------------------------

# grafico sexo por aglomerado
plt.figure(figsize=(12,6))

for sexo in tabla_sexo["sexo"].unique():
    for aglo in tabla_sexo["aglomerado_nombre"].unique():
        sub = tabla_sexo[
            (tabla_sexo["sexo"] == sexo) &
            (tabla_sexo["aglomerado_nombre"] == aglo)
        ]
        plt.plot(sub["ANO4"], sub["ingreso_real"], marker="o", label=f"{sexo} – {aglo}")

plt.title("Ingreso real promedio por sexo y aglomerado (2º trimestre)")
plt.xlabel("Año")
plt.ylabel("Ingreso real")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# grafico nivel educativo por aglomerado
plt.figure(figsize=(12,6))

for nivel in tabla_educ["nivel_educ"].unique():
    for aglo in tabla_educ["aglomerado_nombre"].unique():
        sub = tabla_educ[
            (tabla_educ["nivel_educ"] == nivel) &
            (tabla_educ["aglomerado_nombre"] == aglo)
        ]
        plt.plot(sub["ANO4"], sub["ingreso_real"], marker="o", label=f"{nivel} – {aglo}")

plt.title("Ingreso real por nivel educativo y aglomerado (2º trimestre)")
plt.xlabel("Año")
plt.ylabel("Ingreso real")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# grafico edad por aglomerado
plt.figure(figsize=(12,6))

for edad in tabla_edad["grupo_edad"].unique():
    for aglo in tabla_edad["aglomerado_nombre"].unique():
        sub = tabla_edad[
            (tabla_edad["grupo_edad"] == edad) &
            (tabla_edad["aglomerado_nombre"] == aglo)
        ]
        plt.plot(sub["ANO4"], sub["ingreso_real"], marker="o", label=f"{edad} – {aglo}")

plt.title("Ingreso real por grupos de edad y aglomerado (2º trimestre)")
plt.xlabel("Año")
plt.ylabel("Ingreso real")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# grafico categoria ocupacional por aglomerado (aunque solo existe una categoria por los filtros)
plt.figure(figsize=(12,6))

for cat in tabla_cat["cat_ocup_label"].dropna().unique():
    for aglo in tabla_cat["aglomerado_nombre"].unique():
        sub = tabla_cat[
            (tabla_cat["cat_ocup_label"] == cat) &
            (tabla_cat["aglomerado_nombre"] == aglo)
        ]
        if len(sub) > 0:
            plt.plot(sub["ANO4"], sub["ingreso_real"], marker="o", label=f"{cat} – {aglo}")

plt.title("Ingreso real por categoría ocupacional y aglomerado (2º trimestre)")
plt.xlabel("Año")
plt.ylabel("Ingreso real")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




print("Punto 2 finalizado correctamente.")

