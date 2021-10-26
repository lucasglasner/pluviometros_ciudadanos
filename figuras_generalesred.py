# -*- coding: utf-8 -*-


import xarray as xr
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import cartopy.io.img_tiles
import geopandas as gpd
import matplotlib as mpl
from shapely.geometry import box
#%%

import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
def plot_inset(xy,clon,clat,output=None):
    fig = plt.figure()
    proj = ccrs.Orthographic(central_longitude=clon,central_latitude=clat)
    ax = fig.add_subplot(1, 1, 1,projection=proj)
    ax.stock_img()
    ax.coastlines()
    # ax.set_extent([clon*0.6,clon*1.5,clat*0.6,clat*1.5])
    ax.add_patch(mpatches.Rectangle(xy=xy, width=3, height=3,
                                    fill=False,color="red",
                                    transform=ccrs.PlateCarree()))
    ax.gridlines(linestyle="--")
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)
    if output==None:
        pass
    else:
        plt.savefig(output,dpi=150,bbox_inches="tight")
        
def scale_bar(ax, length=None, location=(0.5, 0.05), linewidth=3,col="w",fs=10):
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
            horizontalalignment='center', verticalalignment='bottom',fontsize=fs)
        
#%%

#Leer datos, reemplazar "," por "." checkear coordenadas
title = "datos_eventoscorregidos/DATOS Evento 24-26 Jun 2021_FIGURA 3"
datos = pd.read_excel(title+".xlsx",sheet_name="mapa")
for cord in ["lon","lat"]:
    datos.loc[:,cord] = list(map(lambda x: float(str(x).replace(",",".")),datos.loc[:,cord]))
    datos.loc[:,cord] = list(map(lambda x: -x if x>0 else x,datos.loc[:,cord]))

#Agregar elevacion
topo = xr.open_dataset("DEM_RM.nc")
datos["altura"] = np.empty(len(datos))
for i in range(len(datos)):
    datos.loc[i,"altura"] = topo["Band1"].sel(lon=datos.loc[i,"lon"],
                                              lat=datos.loc[i,"lat"],
                                              method="nearest").item()
# del topo
datos = gpd.GeoDataFrame(datos, geometry=gpd.points_from_xy(datos.lon, datos.lat))
# datos = datos[~np.isnan(datos["pp"]).values]
#Separar datos en pluviometros y estaciones normales
mask  = np.logical_or(datos["grupo"] == "Estación",datos["grupo"] == "vismet")
comunas = gpd.read_file("datos_gis/division_comunal.shp")
comunas = comunas.to_crs(ccrs.Mercator().proj4_init)
cuencas = gpd.read_file("datos_gis/Cuencas_BNA.shp").to_crs(ccrs.Mercator().proj4_init)
datos.crs = ccrs.Orthographic().proj4_init
datos = datos.to_crs(cuencas.crs)
cuencas = cuencas.to_crs(epsg="4326")
# topo = xr.open_dataset("RM_topo.nc")
topo = topo.Band1.interp(lat=np.arange(-35,-32,0.01),lon=np.arange(-72,-69,0.01))
topo.where(topo<0,topo,0)
#%%
#Request google image
# request = cartopy.io.img_tiles.GoogleTiles(style="satellite")

#%%
extent = [-70.9, -33.75, -70.33,-33.23]
roi = box(extent[0],extent[1],extent[2],extent[3])
fig = plt.figure(figsize=(10,10),num=0)
ax  = fig.add_subplot(111,projection=ccrs.Mercator())
center = -70.65,-33.45

ratio = 1.2
tlat = 0.72
tlon  = tlat*ratio

ax.set_extent([center[0]-tlon,center[0]+tlon,center[1]-tlat,center[1]+tlat])
ax.gridlines(linestyle=":")
scale_bar(ax=ax,length=20,location=(0.1,0.05), fs=18)
# ax.add_image(request,11)
colors   = plt.cm.terrain(np.linspace(0.15,1,100))
colormap = mpl.colors.LinearSegmentedColormap.from_list("my_colormap",colors)

mapa = topo.plot.contourf(ax=ax,transform=ccrs.PlateCarree(),cmap=colormap,
                          levels=np.arange(0,5e3+100,100),vmin=0,vmax=5e3,
                          add_colorbar=False)
# topo.plot.contour(ax=ax,transform=ccrs.PlateCarree(),colors="grey",linewidths=0.1,
#                   levels=np.arange(0,5e3+250,250),
#                   add_colorbar=False)
cuencas[cuencas["NOM_CUEN"] == "Rio Maipo"].boundary.plot(ax=ax,color="k",lw=3,
                                                          label="Cuenca Río Maipo",
                                                          transform=ccrs.PlateCarree())

gpd.GeoSeries(roi).boundary.plot(ax=ax,transform=ccrs.PlateCarree(),color="r",lw=3)
ax.scatter(datos[~mask].lon,datos[~mask].lat,s=70,transform=ccrs.PlateCarree(),
           edgecolor="k",color="mediumblue", label="Pluviómetros Ciudadanos")

ax.scatter(datos[mask].lon,datos[mask].lat,s=70,transform=ccrs.PlateCarree(),
           edgecolor="k",color="darkorange", label="Estaciones Meteorológicas")

ax.legend(loc=(0,1),ncol=1,frameon=False,fontsize=18)
cax = ax.get_position()
cax = [cax.xmax*1.1,cax.ymin,0.03,cax.ymax-cax.ymin]
cax = fig.add_axes(cax)
cb=fig.colorbar(mapa,pad=0.15,cax=cax)
cb.set_label(label="Elevación (m.s.n.m)",size=18)
cb.ax.tick_params(labelsize=15)

plt.savefig("plots/red_pluviometros.pdf",dpi=150,bbox_inches="tight")
# ax.legend(loc=(1.1,0),title="Precipitación (mm)",frameon=False)
# ax.set_title(title)
# ax.set_aspect('auto')


# ax1 = fig.add_axes([ax.get_position().xmax+0.05,0.75,0.18,0.18])
# ax1.axis("off")
# ax1.imshow(logoPC)


# ax2 = fig.add_axes([ax.get_position().xmax+0.05,0.65,0.18,0.18])
# ax2.axis("off")
# ax2.imshow(logodgf)

# ax3 = fig.add_axes([ax.get_position().xmax+0.05,0.4,0.13,0.13])
# ax3.axis("off")
# ax3.imshow(leyenda)

# fig.show()
