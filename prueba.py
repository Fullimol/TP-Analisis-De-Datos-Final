# ============================================================
# 1. PREPARAR ENTORNO EN GOOGLE COLAB
# ============================================================

# !pip install pandas numpy matplotlib seaborn plotly geopandas scikit-learn statsmodels requests

# ============================================================
# 2. DESCARGAR MICRODATOS EPH (2016–2025)
# ============================================================

import pandas as pd
import zipfile, io, requests

# Crear lista de URLs de todos los trimestres 2016–2025
lista_urls = []
anios = range(2016, 2026)
trimestres = [1, 2, 3, 4]

for a in anios:
    for t in trimestres:
        sufijo = f"{str(a)[2:]}{t}"
        url = f"https://www.indec.gob.ar/ftp/cuadros/menusuperior/eph/EPH_usu_individual_T{sufijo}.zip"
        lista_urls.append(url)

# Descargar y concatenar todo
dfs = []

for url in lista_urls:
    try:
        content = requests.get(url).content
        z = zipfile.ZipFile(io.BytesIO(content))
        fname = z.namelist()[0]
        df_temp = pd.read_csv(z.open(fname), sep=';', low_memory=False)
        dfs.append(df_temp)
        print("Descargado:", url)
    except:
        print("Error:", url)

data = pd.concat(dfs, ignore_index=True)



# ============================================================
# 3. FILTRADO POR AGLOMERADOS (NUESTRO CASO: 27 y 20)
# ============================================================

agl_sj = 27   # San Juan
agl_rg = 20   # Río Gallegos

df = data[data['AGLOMERADO'].isin([agl_sj, agl_rg])]



# ============================================================
# 4. CREACIÓN DE TASAS (ACTIVIDAD, EMPLEO, DESOCUPACIÓN)
# ============================================================

df['activo'] = df['ESTADO'] != 3
df['empleado'] = df['ESTADO'] == 1
df['desocupado'] = df['ESTADO'] == 2

tasas = df.groupby(['ANO4', 'TRIMESTRE', 'AGLOMERADO']).agg(
    tasa_actividad=('activo', 'mean'),
    tasa_empleo=('empleado', 'mean'),
    tasa_desocup=('desocupado', 'mean')
).reset_index()



# ============================================================
# 5. AJUSTE DE INGRESOS POR INFLACIÓN (IPC)
# ============================================================

# Cargar archivo IPC previamente descargado del INDEC
ipc = pd.read_csv("ipc_2016_2025.csv")

ipc['factor'] = ipc['IPC'] / ipc['IPC'].iloc[-1]  # base 2025 = 100

# Unir IPC con EPH
df = df.merge(ipc, on=['ANO4','MES'], how='left')
df['ingreso_real'] = df['P47T'] / df['factor']



# ============================================================
# 6. ANÁLISIS UNIVARIADO
# ============================================================

# Estadísticos descriptivos
print(df['ingreso_real'].describe(percentiles=[.25, .5, .75, .9, .95]))

# Gráfico de evolución de la tasa de empleo
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(tasas['tasa_empleo'])
plt.title("Evolución de la tasa de empleo (2016–2025)")
plt.grid()
plt.show()



# ============================================================
# 7. ANÁLISIS MULTIVARIADO (REGRESIÓN)
# ============================================================

import statsmodels.formula.api as smf

modelo = smf.ols(
    "ingreso_real ~ C(NIVEL_ED) + EDAD + C(SEXO) + empleado",
    data=df
).fit()

print(modelo.summary())



# ============================================================
# 8. VISUALIZACIÓN DE DATOS (BOXPLOTS, HISTOGRAMAS)
# ============================================================

import seaborn as sns

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='AGLOMERADO', y='ingreso_real')
plt.title("Distribución del ingreso real por aglomerado")
plt.show()



# ============================================================
# 9. IMPUTACIÓN DE NO RESPUESTA EN INGRESOS
# ============================================================

from sklearn.ensemble import RandomForestRegressor

train = df[df['P47T'].notna()]
test  = df[df['P47T'].isna()]

X_train = train[['EDAD','CH07','NIVEL_ED']]
y_train = train['ingreso_real']

model = RandomForestRegressor()
model.fit(X_train, y_train)

test['ingreso_imputado'] = model.predict(test[['EDAD','CH07','NIVEL_ED']])



# ============================================================
# 10. VISUALIZACIÓN GEORREFERENCIADA (SAN JUAN 27 – RÍO GALLEGOS 20)
# ============================================================

import geopandas as gpd

# Descargar shapefile INDEC
url_shp = "https://www.indec.gob.ar/ftp/cuadros/menusuperior/eph/mapas/aglomerados_eph.zip"
content = requests.get(url_shp).content

z = zipfile.ZipFile(io.BytesIO(content))
z.extractall("shp_eph")

mapa = gpd.read_file("shp_eph/aglomerados_eph.shp")

# Filtrar solo San Juan y Río Gallegos
mapa_sel = mapa[mapa["AGLOMERADO"].isin([20, 27])]

# Unir con tasas del último trimestre
ultimo = tasas.sort_values(["ANO4", "TRIMESTRE"]).groupby("AGLOMERADO").tail(1)
geo = mapa_sel.merge(ultimo, on="AGLOMERADO", how="left")

# Graficar
geo.plot(
    column="tasa_empleo",
    cmap="Blues",
    edgecolor="black",
    legend=True,
    figsize=(8,6)
)
plt.title("Tasa de empleo – San Juan (27) y Río Gallegos (20)")
plt.show()
