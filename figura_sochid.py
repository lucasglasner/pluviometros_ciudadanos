#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 17:30:41 2021

@author: lucas
"""

import numpy as np 
import pandas as pd
from scipy.stats import linregress
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles
import matplotlib as mpl
import imageio
import geopandas as gpd
import seaborn as sns
import glob
from shapely.geometry import box

#%%
def scale_bar(ax, length=None, location=(0.5, 0.05), linewidth=3,col="w"):
    """
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.
    """
    #Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    #Make tmc horizontally centred on the middle of the map,
    #vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    #Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    #Calculate a scale bar length if none has been given
    #(Theres probably a more pythonic way of rounding the number but this works)
    if not length: 
        length = (x1 - x0) / 5000 #in km
        ndim = int(np.floor(np.log10(length))) #number of digits in number
        length = round(length, -ndim) #round to 1sf
        #Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']: return int(x)        
            else: return scale_number(x - 10 ** ndim)
        length = scale_number(length) 

    #Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx - length * 500, sbx + length * 500]
    #Plot the scalebar
    ax.plot(bar_xs, [sby, sby], transform=tmc, color=col, linewidth=linewidth)
    #Plot the scalebar label
    ax.text(sbx, sby, str(length) + ' km', transform=tmc,color=col,
            horizontalalignment='center', verticalalignment='bottom')


#%%
#Leer datos, reemplazar "," por "." checkear coordenadas
titles = glob.glob("datos_eventoscorregidos/*")#, "Evento 18-19 a]gosto 2021"]
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
datos = gdatos
del gdatos
#%%
#Agregar elevacion
topo = xr.open_dataset("DEM_RM.nc")

for j in range(len(datos)):
    datos[j]["altura"] = np.empty(len(datos[j]))
    for i in range(len(datos[j])):
        datos[j].loc[i,"altura"] = topo["Band1"].sel(lon=datos[j].loc[i,"lon"],
                                                     lat=datos[j].loc[i,"lat"],
                                                     method="nearest").item()
    datos[j].where(~np.isnan(datos[j]["pp"])).dropna(inplace=True)
# del topo

#%%
#Separar datos en pluviometros y estaciones normales
mask  = [datos[j]["grupo"] == "vismet" for j in range(len(datos))]
cuencas = gpd.read_file("datos_gis/Cuencas_BNA.shp")
datos   = [datos[j].to_crs({"init":"epsg:4326"}) for j in range(len(datos))]
cuencas = cuencas.to_crs({"init":"epsg:4326"})[cuencas["COD_CUEN"]=="057"]


#%%
#Request google image
# request = cartopy.io.img_tiles.GoogleTiles(style="satellite")
request = cartopy.io.img_tiles.Stamen(style="terrain-background")
leyenda = plt.imread('plots/leyenda2.png')
flecha  = plt.imread("plots/flechanorte.png")
sat     = [plt.imread(glob.glob("congreso_sochid/correccion/*.png")[i]) for i in range(3)]

# logoPC  = plt.imread("plots/logo_pc.png")
# logodgf = plt.imread("plots/dgf.png")
#%%
#Plotear puntos con pp en un mapa

fig = plt.figure(figsize=(9,9),num=0)
ax = [[],[],[]]
ax_sats = []
#plots pr vs z
ax[0].append(fig.add_axes([0,0.7,0.4,0.3]))
ax[0].append(fig.add_axes([0,0.35,0.4,0.3]))
ax[0].append(fig.add_axes([0,0,0.4,0.3]))


pos = [ax[0][i].get_position() for i in range(3)]
ax_sats.append(fig.add_axes([pos[0].xmax-0.17,
                             0.7-0.029,
                             0.17,
                             0.17]))
ax_sats.append(fig.add_axes([pos[1].xmax-0.17,
                             0.35-0.029,
                             0.17,
                             0.17]))
ax_sats.append(fig.add_axes([pos[2].xmax-0.17,
                             0-0.029,
                             0.17,
                             0.17]))

#boxplots pr
ax[1].append(fig.add_axes([0.4,0.7,0.05,0.3]))
ax[1].append(fig.add_axes([0.4,0.35,0.05,0.3]))
ax[1].append(fig.add_axes([0.4,0,0.05,0.3]))

#mapas
ax[2].append(fig.add_axes([0.5,0.7,0.3,0.3],projection=ccrs.Mercator()))
ax[2].append(fig.add_axes([0.5,0.35,0.3,0.3],projection=ccrs.Mercator()))
ax[2].append(fig.add_axes([0.5,0,0.3,0.3],projection=ccrs.Mercator()))

ax[0][1].set_ylabel("Precipitación Acumulada (mm)",fontsize=15)
ax[0][2].set_xlabel("Elevación (m.s.n.m)",fontsize=15)
#params
extent = [-70.9, -33.75, -70.33,-33.23]
roi = box(extent[0],extent[1],extent[2],extent[3])
center = -70.77,-33.45
ratio = 1.2
tlat = 0.67
tlon  = tlat*ratio
titulos = ["Evento 20/05/2021", "Evento 24/06/2021", "Evento 18/08/2021"]

ax_flecha = fig.add_axes([ax[2][0].get_position().xmax*1.025,
                          ax[2][0].get_position().ymax-0.08,
                          0.08,
                          0.08])
ax_flecha.imshow(flecha)
ax_flecha.axis("off")
ax_flecha.scatter([],[],color="white",edgecolor="red",label="Estaciones\nMeteorologicas")
ax_flecha.scatter([],[],color="white",edgecolor="black",label="Pluviómetros\nciudadanos")
ax_flecha.legend(loc=(-0.7,-5),fontsize=13,frameon=False)
#loop de 3


for i in range(3):
    ax[i][0].text(0.5,1.3,"ABC"[i],transform=ax[i][0].transAxes,fontsize=20)
    datos_stgo = gpd.clip(datos[i],roi).dropna()
    e_meteo    = datos_stgo["grupo"] == "vismet"
    #pr vs z
    m = linregress(datos_stgo["altura"],datos_stgo["pp"])
    x = np.arange(300,1200,1)
    lg_label = r"$R^2$: "+"{:.2f}".format(m.rvalue**2)
    ax[0][i].sharex(ax[0][0])
    ax[0][i].plot(x,m.slope*x+m.intercept,color="tab:red",ls="--",label=lg_label)
    ax[0][i].legend(loc=(0,0.9),frameon=False)
    ax[0][i].scatter(datos_stgo["altura"][~e_meteo],datos_stgo["pp"][~e_meteo],
                     edgecolor="k",color="mediumblue",label="Pluviómetros Ciudadanos")
        
    ax[0][i].scatter(datos_stgo["altura"][e_meteo],datos_stgo["pp"][e_meteo],
                     edgecolor="k",color="darkorange",label="Estaciones Meteorológicas")
    ax[0][i].set_xlim(2e2,1.4e3)
    ax[0][i].grid(True,ls="--")
    
    ax_sats[i].imshow(sat[i])
    ax_sats[i].axis("off")
    #boxplots
    ax[1][i].sharey(ax[0][i])
    ax[1][i].axis("off")
    
    ax[1][i].boxplot(datos_stgo["pp"],patch_artist=True,widths=0.4,
                     medianprops=dict(color="red"),
                     boxprops=dict(facecolor="aliceblue"))



    #mapas
    ax[2][i].set_extent([center[0]-tlon,center[0]+tlon,center[1]-tlat,center[1]+tlat])
    ax[2][i].add_image(request,11)
    bins    = np.arange(0,datos_stgo["pp"].max()+5,5)
    bins   = np.linspace(0,datos_stgo["pp"].max(),7)
    binned = (pd.cut(datos[i]["pp"],bins,right=False))
    grouped = datos[i].groupby(binned)    
    
    colores = mpl.cm.viridis(np.linspace(0,1,len(bins)))
    sizes = sorted(colores.mean(axis=1)*0+np.exp(colores.mean(axis=1)*11)/11)
    # # sizes = sorted(colores.mean(axis=1)*0+(colores.mean(axis=1)*6)**4)
    labels  = np.empty(len(bins)-1,dtype="U100")
    for j in range(len(bins)-2):
        labels[j] = "["+"{:.0f}".format(bins[j])+","+"{:.0f}".format(bins[j+1])+")"
    labels[len(bins)-2] = "["+"{:.0f}".format(bins[len(bins)-2])+",)"

    for j, (name,group) in enumerate(grouped):
        # mask = group.alias == "VisMet"
        mask = group["grupo"] == "vismet"
        ax[2][i].scatter(group[mask].lon,group[mask].lat,s=sizes[j],alpha=0.8,
                          color=colores[j],edgecolor="red",transform=ccrs.PlateCarree())
        ax[2][i].scatter(group[~mask].lon,group[~mask].lat,s=sizes[j],alpha=0.8,label=labels[j],
                          color=colores[j],edgecolor="k",transform=ccrs.PlateCarree())
    ax[2][i].legend(loc=(1.01,0),frameon=False)
    scale_bar(ax=ax[2][i],length=20,location=(0.1,0.01),col="k")
    ax[2][i].set_adjustable('datalim')
    ax[2][i].set_title(titulos[i],fontsize=15,loc="left")
    # ax[2][i].set_aspect('auto')
    text = "N1: "+"{:.0f}".format((datos[i]["grupo"]=="vismet").sum())
    text = text +"\nN2: "+"{:.0f}".format((datos[i]["grupo"]!="vismet").sum())
    ax[2][i].text(0.73,0.84,text,transform=ax[2][i].transAxes,fontsize=13)
    x,y=cuencas.geometry.values[0].exterior.coords.xy
    ax[2][i].plot(x,y,transform=ccrs.PlateCarree(),color="k")
    # cuencas.boundary.plot(ax=ax[2][i],color="k",transform=ccrs.PlateCarree())
    # gpd.GeoSeries(roi).boundary.plot(ax=ax[2][i],transform=ccrs.PlateCarree())    

ax[2][2].text(0,-0.25,"N1: Número de Estaciones\nmeteorológicas\nN2: Número de observadores",
              transform=ax[2][2].transAxes,fontsize=13)
handles, labels = ax[0][0].get_legend_handles_labels()
# sort both labels and handles by labels
# labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
labels,handles = labels[::-1],handles[::-1]
ax[0][0].legend(handles, labels,frameon=False,loc=(0,0.9))
plt.savefig("congreso_sochid/correccion/figura4.pdf",dpi=150,bbox_inches="tight")
# ax[2][2].text(1.1,0.6,"N1: Numero de\n      Estaciones\n      meteorológicas\nN2: Numero de\n      observadores",
#               transform=ax[2][2].transAxes,fontsize=13)

#%%





# fig = plt.figure(figsize=(10,10),num=0)
# ax1  = fig.add_subplot(331,projection=ccrs.Mercator())
# ax2  = fig.add_subplot(334,projection=ccrs.Mercator())
# ax3  = fig.add_subplot(337,projection=ccrs.Mercator())
# ax4  = fig.add_subplot(332)
# ax5  = fig.add_subplot(335)
# ax6  = fig.add_subplot(338)
# # ax5.sharex(ax4);ax5.sharex(ax4)
# # ax5.sharey(ax4);ax6.sharey(ax4)
# ax7  = fig.add_subplot(333,projection=ccrs.PlateCarree())
# ax8  = fig.add_subplot(336,projection=ccrs.PlateCarree())
# ax9  = fig.add_subplot(339,projection=ccrs.PlateCarree())
# center = -70.65,-33.45

# ratio = 1.2
# tlat = 0.68
# tlon  = tlat*ratio
# pmax = np.max([np.max(datos[j]["pp"]) for j in range(len(datos))])
# extent = [-70.9, -33.75, -70.33,-33.23]
# roi = box(extent[0],extent[1],extent[2],extent[3])
# titulos = ["Evento 20/Mayo/2021", "Evento 23/06/2021", "Evento 26/06/2021"]
# tiempos = [19*4,53*4,56*4]
# for j in range(9):
#     ax = eval("ax"+str(j+1))
#     if j in [0,1,2]:
#         ax.set_extent([center[0]-tlon,center[0]+tlon,center[1]-tlat,center[1]+tlat])
#         ax.gridlines(linestyle=":")

#         # bins    = np.arange(0,datos["pp"].max()+10,5)
#         bins    = np.arange(0,41+5,5)
#         binned = (pd.cut(datos[j]["pp"],bins,right=False))
#         grouped = datos[j].groupby(binned)

#         colores = mpl.cm.YlGnBu(np.linspace(0,1,len(bins)))
#         sizes = sorted(colores.mean(axis=1)*0+colores.mean(axis=1)**3*300)
#         labels  = np.empty(len(bins)-1,dtype="U100")
#         for i in range(len(bins)-2):
#             labels[i] = "["+"{:.0f}".format(bins[i])+","+"{:.0f}".format(bins[i+1])+")"
#         labels[len(bins)-2] = "["+"{:.0f}".format(bins[len(bins)-2])+",)"

#         for i, (name,group) in enumerate(grouped):
#             # mask = group.alias == "VisMet"
#             mask = group["grupo"] == "vismet"
#             ax.scatter(group[mask].lon,group[mask].lat,s=sizes[i],hatch="xxxx",alpha=0.8,
#                         color=colores[i],edgecolor="k",transform=ccrs.PlateCarree())
#             ax.scatter(group[~mask].lon,group[~mask].lat,s=sizes[i],alpha=0.8,label=labels[i],
#                         color=colores[i],edgecolor="k",transform=ccrs.PlateCarree())

#         scale_bar(ax=ax,length=10,location=(0.1,0.05))
#         # ax.add_image(request,11)
#         # ax.legend(loc=(1.1,0),title="Precipitación (mm)",frameon=False)
#         ax.set_title(titulos[j],fontsize=10)
#         ax.set_aspect('auto')
#         gpd.GeoSeries(roi).boundary.plot(ax=ax,transform=ccrs.PlateCarree())
#     elif j in [3,4,5]:
#         from scipy.stats import linregress
#         # pos = ax.get_position().get_points().flatten()
#         datos_stgo = gpd.clip(datos[j-3],roi).dropna()
#         # datos_stgo = datos_stgo.where(datos_stgo["altura"]<1300).dropna()
#         # print((datos[j-3].grupo=="vismet").sum())
#         # datos_stgo = datos.where(datos["lat"]>extent[2]).where(datos["lat"]<extent[3])
#         # datos_stgo = datos_stgo.where(datos["lon"]>extent[0]).where(datos["lon"]<extent[1]).dropna()
#         # datos_stgo = gpd.clip(datos[j-3],cuencas[cuencas["COD_CUEN"]=="057"].loc[[175,179,181,185]])
#         m = linregress(datos_stgo["altura"],datos_stgo["pp"])
#         mask = datos_stgo["grupo"] == "vismet"
#         ax.scatter(datos_stgo["altura"][mask],datos_stgo["pp"][mask],edgecolor="k",color="gold")
#         ax.scatter(datos_stgo["altura"][~mask],datos_stgo["pp"][~mask],edgecolor="k",color="royalblue")
#         x = np.linspace(datos_stgo["altura"].min()-50,datos_stgo["altura"].max()+50,100)
#         lb = "$R^2$: "+"{:0.2f}".format(m.rvalue**2)+"\npvalue: "+"{:0.1e}".format(m.pvalue)
#         ax.plot(x,x*m.slope+m.intercept,color="tab:red",ls="--",label=lb)
#         ax.legend(frameon=False)
#         # ax.set_title(title)
#         # ax.set_ylabel("Precipitación\nAcumulada (mm)")
#         # ax.set_xlabel("Elevación (m.s.n.m)")
#         ax.set_xticks(np.arange(300,1600,300))
#         # ax.set_yticks(np.arange(0,datos[j-3]["pp"].max(),5))
#         # ax.set_ylim([0,datos[j-3]["pp"].max()*1.1])
#     else:
#         ax.coastlines()
#         # mapa=ax.pcolormesh(LON,LAT,era5.tcw[tiempos[j-6],0,:,:].squeeze(),transform=ccrs.PlateCarree(),cmap="Blues")
#         # ax.contour(LON,LAT,era5.msl[tiempos[j-6],0,:,:],colors="k",levels=20,alpha=0.5)
#         # ax.quiver(LON,LAT,era5["p71.162"][tiempos[j-6],0,:,:].values,era5["p72.162"][tiempos[j-6],0,:,:].values,regrid_shape=15)
# # fig.colorbar(mapa,ax=ax9)
