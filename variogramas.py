#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 16:12:33 2021

@author: lucas
"""
import pandas as pd
from pykrige.ok import OrdinaryKriging
from glob import glob
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
#%%
titles = glob("datos_eventoscorregidos/*")#, "Evento 18-19 a]gosto 2021"]
titles = [i for i in [titles[1],titles[2],titles[0]]]
datos = [pd.read_excel(title,sheet_name="mapa") for title in titles]
datos[0].drop("Unnamed: 6",axis=1,inplace=True)
for j in range(len(datos)):
    for cord in ["lon","lat"]:
        datos[j].loc[:,cord] = list(map(lambda x: float(str(x).replace(",",".")),datos[j].loc[:,cord]))
        datos[j].loc[:,cord] = list(map(lambda x: -x if x>0 else x,datos[j].loc[:,cord]))
    datos[j].loc[:,"lon"]    = list(map(lambda x: np.nan if x>-69 else x,datos[j].loc[:,"lon"]))
    datos[j].dropna(inplace=True)
    datos[j].reset_index(inplace=True)
gdatos = []
for j in range(len(datos)):
    gdatos.append(gpd.GeoDataFrame(datos[j], geometry=gpd.points_from_xy(datos[j].lon,datos[j].lat), crs="epsg:4326"))
    gdatos[j] = gdatos[j].to_crs("+proj=utm +zone=19 +ellps=WGS84 +datum=WGS84 +units=m +no_defs +south")
datos = gdatos
del gdatos

#%%
lags,semivar = [],[]
for i in range(len(datos)):
    OK = OrdinaryKriging(x=datos[i].geometry.x.values*1e-3,
                         y=datos[i].geometry.y.values*1e-3,
                         z=datos[i]["pp"].values,
                         enable_plotting=False,
                         verbose=False,
                         nlags=50,
                         variogram_model="exponential")
    lags.append(OK.lags)
    semivar.append(OK.semivariance)

#%%

fig,ax = plt.subplots(1,3,sharex=True,sharey=False,figsize=(14,3))
titulos = ["Evento 20/05/2021", "Evento 24/06/2021", "Evento 18/08/2021"]
ax = ax.ravel()

colors = plt.cm.tab10(np.linspace(0,0.8,3))
for i in range(3):
    ax[i].scatter(lags[i],(semivar[i]),marker="o",edgecolor="k",color=colors[i])
    ax[i].set_xlim(0,76.2)
    ax[i].set_xticks(np.arange(0,75,5))
    ax[i].set_title(titulos[i])
    ax[i].set_xlabel("Lags distancia (km)")

ax[0].set_ylabel("Semivariancia ($mm^2$)")
# krig = []
# z,ss = [],[]
# models = ["exponential","spherical","gaussian","power"]
# for mod in models:
#     OK = OrdinaryKriging(
#         x=datos[:, 0],
#         y=datos[:, 1],
#         z=datos[:, 2],
#         variogram_model=mod,
#         verbose=False,
#         enable_plotting=False,
#         nlags=20
#     )
#     krig.append(OK)
#     a, b = OK.execute("grid", x*1.0, y*1.0)
#     z.append(a)
#     ss.append(b)
# max_dist = np.sqrt((x.max()-x.min())**2+(y.max()-y.min())**2)
# lags,semivar = krig[0].lags,krig[0].semivariance
