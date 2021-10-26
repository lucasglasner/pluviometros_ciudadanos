import numpy as np 
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles
import matplotlib as mpl
import imageio
import geopandas as gpd
import seaborn as sns
from shapely.geometry import box

#%%
def scale_bar(ax, length=None, location=(0.5, 0.05), linewidth=3,col="w", fs=10):
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
# title = "Resultados evento 20-21 mayo 2021"
# title = "Informe Evento 260621"
title = "Mediciones evento 11-12 sep 2021"
datos = pd.read_excel(title+".xlsx",sheet_name="mapa")
for cord in ["lon","lat"]:
    datos.loc[:,cord] = list(map(lambda x: float(str(x).replace(",",".")),datos.loc[:,cord]))
    datos.loc[:,cord] = list(map(lambda x: -x if x>0 else x,datos.loc[:,cord]))

#Agregar elevacion
topo = xr.open_dataset("RM_topo.nc")
datos["altura"] = np.empty(len(datos))
for i in range(len(datos)):
    datos.loc[i,"altura"] = topo["Band1"].sel(lon=datos.loc[i,"lon"],
                                              lat=datos.loc[i,"lat"],
                                              method="nearest").item()
del topo

datos = datos[~np.isnan(datos["pp"]).values]
#Separar datos en pluviometros y estaciones normales
mask  = np.logical_or(datos["grupo"] == "Estación",datos["grupo"] == "vismet")
datos = gpd.GeoDataFrame(datos, geometry=gpd.points_from_xy(datos.lon, datos.lat), crs="epsg:4326")
cuencas = gpd.read_file("datos_gis/Cuencas_BNA.shp")
datos   = datos.to_crs({"init":"epsg:4326"})
cuencas = cuencas.to_crs({"init":"epsg:4326"})
extent = [-70.9, -33.75, -70.33,-33.23]
roi = box(extent[0],extent[1],extent[2],extent[3])
datos_stgo = gpd.clip(datos,roi)
#%%
#Request google image
# request = cartopy.io.img_tiles.Stamen(style="terrain-background")
request = cartopy.io.img_tiles.GoogleTiles(style="satellite")
leyenda = plt.imread('plots/leyenda2.png')
logoPC  = plt.imread("plots/logo_pc.png")
logodgf = plt.imread("plots/dgf.png")
#%%

#Plotear puntos con pp en un mapa
fig = plt.figure(figsize=(10,10),num=0)
ax  = fig.add_subplot(111,projection=ccrs.Mercator())
center = -70.75,-33.45
# center = -70.85,-33.45
ratio = 1.2
tlat = 0.60
tlon  = tlat*ratio

ax.set_extent([center[0]-tlon,center[0]+tlon,center[1]-tlat,center[1]+tlat])
ax.gridlines(linestyle=":")

# bins    = np.arange(0,datos["pp"].max()+10,10)
bins    = np.arange(0,40,5)
bins   = np.hstack((bins[:-1],float("inf")))
binned = (pd.cut(datos["pp"],bins,right=False))
grouped = datos.groupby(binned)

colores = mpl.cm.YlGnBu(np.linspace(0,1,len(bins)))
sizes = sorted(colores.mean(axis=1)*0+colores.mean(axis=1)**3*1000)
# sizes = [51,82,117,142,175,214,262,318,385,460,558,647,711,795,892]
labels  = np.empty(len(bins)-1,dtype="U100")
for i in range(len(bins)-2):
    labels[i] = "["+"{:.0f}".format(bins[i])+","+"{:.0f}".format(bins[i+1])+")"
labels[len(bins)-2] = "["+"{:.0f}".format(bins[len(bins)-2])+",)"

for i, (name,group) in enumerate(grouped):
    # mask = group.alias == "VisMet"
    mask = group["grupo"] == "vismet"
    ax.scatter(group[mask].lon,group[mask].lat,s=sizes[i],hatch="xxxx",alpha=0.8,
               color=colores[i],edgecolor="k",transform=ccrs.PlateCarree())
    ax.scatter(group[~mask].lon,group[~mask].lat,s=sizes[i],alpha=0.8,label=labels[i],
               color=colores[i],edgecolor="k",transform=ccrs.PlateCarree())

scale_bar(ax=ax,length=20,location=(0.1,0.03), fs=18, col="w")
ax.add_image(request,11)
ax.legend(loc=(1.1,0),title="Precipitación (mm)",frameon=False)
ax.set_title(title,fontsize=18)
ax.set_aspect('auto')
# cuencas[cuencas["COD_CUEN"]=="057"].boundary.plot(ax=ax,transform=ccrs.PlateCarree(),color="k")


ax1 = fig.add_axes([ax.get_position().xmax+0.05,0.75,0.18,0.18])
ax1.axis("off")
ax1.imshow(logoPC)


ax2 = fig.add_axes([ax.get_position().xmax+0.05,0.65,0.18,0.18])
ax2.axis("off")
ax2.imshow(logodgf)

ax3 = fig.add_axes([ax.get_position().xmax+0.05,0.4,0.13,0.13])
ax3.axis("off")
ax3.imshow(leyenda)

plt.savefig("plots/tormenta1_"+title+".pdf",dpi=150,bbox_inches="tight")
# plt.close()

#%%

#Plot zoom en stgo
fig = plt.figure(figsize=(10,10),num=1)
ax  = fig.add_subplot(111,projection=ccrs.Mercator())
center = -70.65,-33.45


ratio = 1.2
tlat  = 0.18
tlon  = tlat*ratio
extent = [center[0]-tlon,center[0]+tlon,center[1]-tlat,center[1]+tlat]
ax.set_extent(extent)
ax.gridlines(linestyle=":")

# bins    = np.arange(0,datos["pp"].max()+10,10)
bins    = np.arange(0,40,5)
bins   = np.hstack((bins[:-1],float("inf")))
binned = (pd.cut(datos["pp"],bins,right=False))
grouped = datos.groupby(binned)

colores = mpl.cm.YlGnBu(np.linspace(0,1,len(bins)))
sizes = sorted(colores.mean(axis=1)*0+colores.mean(axis=1)**3*1300)

labels  = np.empty(len(bins)-1,dtype="U100")
for i in range(len(bins)-2):
    labels[i] = "["+"{:.0f}".format(bins[i])+","+"{:.0f}".format(bins[i+1])+")"
labels[len(bins)-2] = "["+"{:.0f}".format(bins[len(bins)-2])+",)"

for i, (name,group) in enumerate(grouped):
    # mask = np.logical_or(group["grupo"] == "Estación",group.alias == "VisMet")
    mask = group["grupo"] == "vismet"
    ax.scatter(group[mask].lon,group[mask].lat,s=sizes[i],hatch="xxxx",alpha=0.8,
                color=colores[i],edgecolor="k",transform=ccrs.PlateCarree())
    ax.scatter(group[~mask].lon,group[~mask].lat,s=sizes[i],alpha=0.8,label=labels[i],
                color=colores[i],edgecolor="k",transform=ccrs.PlateCarree())

scale_bar(ax=ax,length=10,location=(0.1,0.03), fs=18, col="w")
ax.add_image(request,11)
lg = ax.legend(loc=(1.1,0),title="Precipitación (mm)",frameon=False)
ax.set_title(title,fontsize=18)
ax.set_aspect('auto')

ax1 = fig.add_axes([ax.get_position().xmax+0.05,0.75,0.18,0.18])
ax1.axis("off")
ax1.imshow(logoPC)


ax2 = fig.add_axes([ax.get_position().xmax+0.05,0.65,0.18,0.18])
ax2.axis("off")
ax2.imshow(logodgf)

ax3 = fig.add_axes([ax.get_position().xmax+0.05,0.4,0.13,0.13])
ax3.axis("off")
ax3.imshow(leyenda)

# fig.show()

plt.savefig("plots/tormenta2_"+title+".pdf",dpi=150,bbox_inches="tight")
# plt.close()
#%%

#influencia orografia
from scipy.stats import linregress
# pos = ax.get_position().get_points().flatten()
extent = [-70.9, -33.75, -70.33,-33.23]
roi = box(extent[0],extent[1],extent[2],extent[3])
datos_stgo = gpd.clip(datos,roi)
# datos_stgo = datos.where(datos["lat"]>extent[2]).where(datos["lat"]<extent[3])
# datos_stgo = datos_stgo.where(datos["lon"]>extent[0]).where(datos["lon"]<extent[1]).dropna()
# datos_stgo = gpd.clip(datos,cuencas[cuencas["COD_CUEN"]=="057"].loc[[175,179,181,185]])
m = linregress(datos_stgo["altura"],datos_stgo["pp"])


fig = plt.figure(num=2,figsize=(10,5))
ax = []
ax.append(fig.add_axes([0,0,.6,.8]))
# ax = fig.add_subplot(111)
# ax = [ax]
ax.append(fig.add_axes([.6,0,0.1,.8]))
mask = datos_stgo["grupo"] == "vismet"
ax[0].scatter(datos_stgo["altura"][mask],datos_stgo["pp"][mask],edgecolor="k",color="gold",label="Estaciones Convencionales")
ax[0].scatter(datos_stgo["altura"][~mask],datos_stgo["pp"][~mask],edgecolor="k",color="royalblue",label="Pluviometros Ciudadanos")
x = np.linspace(datos_stgo["altura"].min()-50,datos_stgo["altura"].max()+50,100)
lb = "$R^2$: "+"{:0.2f}".format(m.rvalue**2)+" ; pvalue: "+"{:0.2e}".format(m.pvalue)
ax[0].plot(x,x*m.slope+m.intercept,color="tab:red",ls="--",label=lb)
ax[0].legend()
# ax[0].set_title(title)
ax[0].set_ylabel("Precipitación\nAcumulada (mm)")
ax[0].set_xlabel("Elevación (m.s.n.m)")
ax[0].set_xticks(np.arange(200,1600,200))
ax[0].set_yticks(np.arange(0,datos["pp"].max(),5))
ax[0].set_ylim([0,datos["pp"].max()*1.1])

ax[1].boxplot(datos_stgo["pp"],showmeans=True,meanline=True,patch_artist=True,
              boxprops=dict(facecolor="royalblue"),
              medianprops=dict(color="k"),
              meanprops=dict(color="k",ls=":"))
# sns.histplot(data=datos_stgo,y="pp",ax=ax[1],stat="density",bins=8,alpha=0.6,color="royalblue")
# sns.kdeplot(data=datos_stgo,y="pp",ax=ax[1])
ax[1].set_yticks(ax[0].get_yticks())
ax[1].set_ylim([0,datos["pp"].max()*1.1])
ax[1].axis("off")
# ax[1].grid(axis="y",ls=":")
ax[1].set_yticklabels([])
ax[1].set_ylabel("")
ax[1].set_xlabel("Densidad de\nProbabilidad")
fig.show()
plt.savefig("plots/z_"+title+".pdf",dpi=150,bbox_inches="tight")
#%%

# plt.close()

#distribucion de pp
fig = plt.figure(num=3)
# sns.kdeplot(datos["pp"],color="k")
plt.hist(datos["pp"],bins=np.arange(bins.min(),bins.max()+2.5,2.5),edgecolor="k",color="tab:blue")
plt.xticks(bins)
# plt.title(title)
plt.xlabel("Precipitación acumulada (mm)")
plt.ylabel("N°Estaciones")
fig.show()
plt.savefig("plots/dist_"+title+".pdf",dpi=150,bbox_inches="tight")

# plt.close()
#%%
# fig = plt.figure(figsize=(5,5))
# ax  = fig.add_subplot(111)
# c1 = mpl.patches.Circle((0.3, 0.8), 0.05,facecolor="w",edgecolor="k")
# c2 = mpl.patches.Circle((0.3,0.6),0.05,facecolor="w",hatch="xxx",edgecolor="k")
# ax.add_patch(c1)
# ax.add_patch(c2)
# ax.text(0.7,0.8,"Pluviómetros\nCiudadanos",ha="center",va="center",fontsize=20)
# ax.text(0.7,0.6,"Estaciones\nConvencionales",ha="center",va="center",fontsize=20)
# ax.axis("off")
# fig.show()
# plt.savefig("plots/leyenda2.pdf",dpi=150,bbox_inches="tight")

# Create the DataFrame from your randomised data and bin it using groupby.
# df = pd.DataFrame(data=dict(x=x, y=y, a2=a2))
# bins = np.linspace(df.a2.min(), df.a2.max(), M)
# grouped = df.groupby(np.digitize(df.a2, bins))

# # Create some sizes and some labels.
# sizes = [50*(i+1.) for i in range(M)]
# labels = ['Tiny', 'Small', 'Medium', 'Large', 'Huge']

# for i, (name, group) in enumerate(grouped):
#     plt.scatter(group.x, group.y, s=sizes[i], alpha=0.5, label=labels[i])

# plt.legend()
# plt.show()