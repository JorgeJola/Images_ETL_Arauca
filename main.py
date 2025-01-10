from flask import Blueprint, Flask, render_template, redirect, request,flash, url_for,send_file
import tempfile
import rasterio
import os


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
            process_raster_path = process_raster(raster_path, municipality)
            return redirect(url_for('main.download_file', filename=os.path.basename(process_raster_path)))
        
    return render_template('image_classification.html', success=False)

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

    return output_path


def download_file(filename):
    file_path = os.path.join(RESULT_FOLDER, filename)

    return send_file(file_path, as_attachment=True)