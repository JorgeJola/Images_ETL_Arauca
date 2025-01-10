from flask import Blueprint, Flask, render_template, redirect, request,flash, url_for,send_file
import geopandas as gpd
import tempfile
import rasterio
import os
import pandas as pd
import numpy as np
from skimage import exposure
from rasterio.features import shapes
import shapely
from shapely.geometry import shape
from skimage.segmentation import slic
from sklearn.preprocessing import StandardScaler
from rasterio.mask import mask
import joblib
import zipfile
import folium
import psutil

main = Blueprint('main', __name__)

UPLOAD_FOLDER = tempfile.mkdtemp()
RESULT_FOLDER = tempfile.mkdtemp()
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@main.route('/', methods=["GET", "POST"])
def image_classification():
    if request.method == "POST":
        if 'raster' not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files['raster']
        municipality = request.form.get("municipality")

        if file.filename == '':
            flash("No selected file")
            return redirect(request.url)

        if file:
            # Guardar archivo subido
            raster_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(raster_path)
            processed_raster_path = process_raster(raster_path, municipality)
            segmented_raster_path = segment_raster(processed_raster_path, municipality)
            #polygons_path = generate_shapefile(segmented_raster_path, municipality)
            #polygons_bands = extract_bands(polygons_path, processed_raster_path, municipality)
            #polygons_classif = apply_model(polygons_path, polygons_bands, municipality)
            #view_shapefile(polygons_classif)
            return redirect(url_for('main.download_file', filename=os.path.basename(segmented_raster_path)))
        
    return render_template('image_classification.html', success=False)

def monitor_memory():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / 1e6} MB")

def process_raster(input_path, municipality):
    output_path = os.path.join(RESULT_FOLDER, f"Cleaned_Raster_{municipality}.tif")

    with rasterio.open(input_path) as multiband_raster:
        # Copiar metadatos y ajustar para el archivo de salida
        new_metadata = multiband_raster.meta
        new_metadata.update(count=6)

        with rasterio.open(output_path, 'w', **new_metadata) as dst:
            for i in range(1, 7):  # Procesar bandas 1 a 6
                band = multiband_raster.read(i)  # Leer banda individualmente
                dst.write(band, indexes=i)  # Escribir banda individualmente
                monitor_memory()

    return output_path

def segment_raster(input_path, municipality):
    with rasterio.open(input_path) as src:
        nbands = src.count
        width = src.width
        height = src.height
        band_data = []

        # Leer las bandas en bloques y apilar
        for i in range(1, nbands + 1):
            band = src.read(i, window=rasterio.windows.Window(0, 0, width, height))
            band_data.append(band)
        band_data = np.dstack(band_data)

        # Ajustar intensidades y realizar segmentación
        img = exposure.rescale_intensity(band_data)
        segments = slic(img, n_segments=10, compactness=0.03)

        # Guardar el resultado de la segmentación
        output_path = os.path.join(RESULT_FOLDER, f"Segmented_Raster_{municipality}.tif")

        with rasterio.open(input_path) as src:
            profile = src.profile
            profile.update(dtype=rasterio.float32, count=1)

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(segments.astype(rasterio.float32), 1)

    return output_path
municipality_shapefiles = {
    'Arauca': 'AraucaLC.shp',
    'Arauquita': 'ArauquitaLC.shp',
    'Cravo Norte': 'CravoLC.shp',
    'Fortul': 'FortulLC.shp',
    'Puerto Rondon': 'PuertoLC.shp',
    'Saravena': 'SaravenaLC.shp',
    'Tame': 'TameLC.shp'
}
def generate_shapefile(segmented_raster_path, municipality):
    # Abrir el raster segmentado
    with rasterio.open(segmented_raster_path) as segmented_raster:
        band = segmented_raster.read(1)  # Leer la primera banda
        transform = segmented_raster.transform  # Obtener la transformación geográfica

        # Generar geometrías y valores a partir del raster
        shapes_generator = shapes(band, transform=transform)
        geometries = []
        values = []

        for geom, value in shapes_generator:
            geometries.append(shape(geom))  # Convertir geometría a objeto Shapely
            values.append(value)  # Guardar valor asociado

    # Crear un GeoDataFrame
    gdf = gpd.GeoDataFrame({'geometry': geometries, 'value': values})
    gdf.set_crs(segmented_raster.crs, allow_override=True, inplace=True)

    # Leer el archivo de área de interés (AOI) para el municipio
    aoi = gpd.read_file(f'Image_Classification_Arauca/static/aois/{municipality_shapefiles.get(municipality)}')

    # Guardar el GeoDataFrame como shapefile
    polygons = gpd.clip(gdf, aoi)
    output_path = os.path.join(RESULT_FOLDER, f"Polygons_{municipality}.shp")

    # Guardar los archivos .shp, .shx, .dbf, etc.
    polygons.to_file(f"{output_path}")

    return output_path

municipality_dem = {
    'Arauca': 'Arauca_elevation_slope.tif',
    'Arauquita': 'Arauquita_elevation_slope.tif',
    'Cravo Norte': 'Cravo_elevation_slope.tif',
    'Fortul': 'Fortul_elevation_slope.tif',
    'Puerto Rondon': 'Puerto_elevation_slope.tif',
    'Saravena': 'Saravena_elevation_slope.tif',
    'Tame': 'Tame_elevation_slope.tif'
}
def extract_bands(polygons_path, input_path,municipality):
    dem_raster=rasterio.open(f'Image_Classification_Arauca/static/aois/{municipality_dem.get(municipality)}')
    newbands_raster=rasterio.open(input_path)
    polygons=gpd.read_file(polygons_path)

    polygons = polygons.to_crs(dem_raster.crs)

    means_spectra_polygon = []
    means_dem_polygon = []

    for _, polygon in polygons.iterrows():
        # Extract the polygon geometry
        geom = [polygon['geometry']]
        # Mask the raster with the polygon
        out_image1, out_transform1 = mask(newbands_raster, geom, crop=True)
        out_image2, out_transform2 = mask(dem_raster, geom, crop=True)
        # Calculate the mean for each band in the masked area
        means1 = np.nanmean(out_image1, axis=(1, 2)) 
        means2 = np.nanmean(out_image2, axis=(1, 2)) 
        # Store the means and associated polygon id (or other attributes)
        means_spectra_polygon.append(dict(polygon_id=polygon['value'], means=means1))
        means_dem_polygon.append(dict(polygon_id=polygon['value'], means=means2))

    means_df1 = pd.DataFrame(means_spectra_polygon)
    means_df2 = pd.DataFrame(means_dem_polygon)

    for i, band_mean in enumerate(means_df1['means']):
        # Create a new column for each band
        for j, mean in enumerate(band_mean):
            if j==0:
                band_name = f"Blue_mean"
            elif j==1:
                band_name = f"Green_mean"
            elif j==2:
                band_name = f"Red_mean"
            elif j==3:
                band_name = f"NIR_mean"
            elif j==4:
                band_name = f"WIR1_mean"
            elif j==5:
                band_name = f"WIR2_mean"
            polygons.at[i, band_name] = mean

    for i, band_mean in enumerate(means_df2['means']):
        # Create a new column for each band
        for j, mean in enumerate(band_mean):
            if j==0:
                band_name = f"Elevation_"
            elif j==1:
                band_name = f"Slope_mean"
            polygons.at[i, band_name] = mean

    gdf_X=polygons[['Blue_mean', 'Green_mean', 'Red_mean', 'NIR_mean',
        'WIR1_mean', 'WIR2_mean', 'Elevation_', 'Slope_mean']]
    gdf_X=gdf_X.set_axis(['Blue','Green','Red','NIR','WIR1','WIR2','Elevation','Slope'],axis=1)
    gdf_X[['Blue','Green','Red','NIR','WIR1','WIR2']]=gdf_X[['Blue','Green','Red','NIR','WIR1','WIR2']]*0.0000275-0.2
    gdf_X['ndvi']=(gdf_X['NIR']-gdf_X['Red'])/(gdf_X['NIR']+gdf_X['Red'])
    #gdf_X['ndbi']=(gdf_X['WIR1']-gdf_X['NIR'])/(gdf_X['WIR1']+gdf_X['NIR'])
    gdf_X['rvi']= gdf_X['WIR1']/gdf_X['NIR']
    #gdf_X['dvi']=gdf_X['WIR1']-gdf_X['NIR']
    gdf_X['evi']=2.5 * (gdf_X['NIR'] - gdf_X['Red']) / (gdf_X['NIR'] + (gdf_X['Red'] * 6) - (gdf_X['Blue'] * 7.5) + 1)
    scaler=StandardScaler()
    gdf_X_standarized=pd.DataFrame(scaler.fit_transform(gdf_X),columns=gdf_X.columns)
    gdf_X_standarized['geometry']=polygons.geometry
    gdf_X_standarized=gpd.GeoDataFrame(gdf_X_standarized)

    output_path = os.path.join(RESULT_FOLDER, f"gdf_X_{municipality}.shp")
    gdf_X_standarized.to_file(f"{output_path}")

    #zip_filename = f"{shapefile_base}.zip"
    #with zipfile.ZipFile(zip_filename, 'w') as zipf:
    #    for ext in ['.shp', '.shx', '.dbf', '.prj']:
    #        file_path = f"{shapefile_base}{ext}"
    #        if os.path.exists(file_path):
    #            zipf.write(file_path, os.path.basename(file_path))
    return output_path

def apply_model(polygons_path,polygons_band,municipality):
    model=joblib.load("Image_Classification_Arauca/static/models/model.joblib")
    polygons=gpd.read_file(polygons_path)
    gdf_X=gpd.read_file(polygons_band)
    gdf_final=gpd.GeoDataFrame()
    gdf_X = gdf_X.drop(columns='geometry')
    gdf_X=gdf_X[['Blue','Green','Red','NIR' ,'WIR1','WIR2','ndvi','rvi','evi','Elevation','Slope']]
    gdf_final['class']=model.predict(gdf_X)
    gdf_final['geometry']=polygons.geometry
    gdf_final=gpd.GeoDataFrame(gdf_final)
    shapefile_base = os.path.join(RESULT_FOLDER, f"Class_{municipality}")
    gdf_final.to_file(f"{shapefile_base}.shp")
    
    # Crear un archivo ZIP que contiene todos los archivos necesarios para el shapefile
    zip_filename = f"{shapefile_base}.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for ext in ['.shp', '.shx', '.dbf', '.prj']:
            file_path = f"{shapefile_base}{ext}"
            if os.path.exists(file_path):
                zipf.write(file_path, os.path.basename(file_path))
    
    return zip_filename

def view_shapefile(polygons_classif):
    # Leer el shapefile generado
    gdf = gpd.read_file(polygons_classif)

    # Crear un mapa centrado en la media de las coordenadas de los polígonos
    m = folium.Map(location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()], zoom_start=10)

    # Colores por clase
    class_colors = colors = {  
    'Urban Zones': '#761800',
    'Industry and Comerciall': '#934741',
    'Mining': '#4616d4',
    'Pastures': '#e8d610',
    'Pastures': '#cddc97',
    'Agricultural Areas': '#dbc382',
    'Forest': '#3a6a00',
    'Shrublands and Grassland': '#cafb4d',
    'Little vegetation areas': '#bfc5b9',
    'Continental Wetlands': '#6b5c8c',
    'Continental Waters': '#0127ff'
}

    # Añadir los polígonos al mapa, coloreando por el atributo 'class'
    folium.GeoJson(
        gdf,
        style_function=lambda feature: {
            'fillColor': class_colors.get(feature['properties']['class'], 'gray'),  # Color por clase
            'color': 'black',  # Color del borde del polígono
            'weight': 2,  # Grosor del borde
            'fillOpacity': 0.6  # Opacidad del relleno
        }
    ).add_to(m)

    # Leyenda simple
    legend_html = '''
    <div style="position: fixed; 
                top: 50px; left: 50px; width: 250px; height: 300px; 
                background-color: white; border: 2px solid black; 
                z-index: 9999; font-size: 14px; padding: 10px;">
        <b>Class Legend</b><br>
    '''
    for class_value, color in class_colors.items():
        legend_html += f'<i style="background: {color}; width: 20px; height: 20px; display: inline-block;"></i> {class_value}<br>'

    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))

    # Guardar el mapa en un archivo HTML
    map_html_path = "Image_Classification_Arauca/static/new_map.html"
    m.save(map_html_path)
    
@main.route('/download/<filename>')
def download_file(filename):
    # Construir la ruta del archivo .zip
    file_path = os.path.join(RESULT_FOLDER, filename)
    
    # Enviar el archivo .zip como una descarga
    return send_file(file_path, as_attachment=True)