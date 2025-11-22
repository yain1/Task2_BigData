import requests
import tarfile
from io import BytesIO
from scipy.io import mmread

# URL del archivo .tar.gz de la matriz
url = "https://suitesparse-collection-website.herokuapp.com/MM/Williams/mc2depi.tar.gz"

print("Descargando archivo...")

# ----- DESCARGA CON BARRA DE PROGRESO -----
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", 0))
    downloaded = 0
    chunk_size = 8192
    data_chunks = []

    for chunk in r.iter_content(chunk_size=chunk_size):
        if chunk:
            data_chunks.append(chunk)
            downloaded += len(chunk)
            # Barra de progreso básica
            percent = downloaded * 100 / total if total else 0
            print(f"\rDescargado: {downloaded/1e6:.1f}/{total/1e6:.1f} MB ({percent:.1f}%)", end="")

print("\nDescarga completa.\n")

# Guardar en disco el archivo descargado
with open("mc2depi.tar.gz", "wb") as f:
    f.write(b"".join(data_chunks))

print("Archivo guardado como mc2depi.tar.gz")

# Convertir a bytes para abrir como tar.gz desde memoria
tar_bytes = BytesIO(b"".join(data_chunks))

print("Extrayendo archivo .mtx del tar.gz...")

# ----- ABRIR TAR.GZ Y BUSCAR .MTX -----
with tarfile.open(fileobj=tar_bytes) as tar:
    mtx_member = None
    for member in tar.getmembers():
        if member.name.endswith(".mtx"):
            mtx_member = member
            break

    if mtx_member is None:
        raise RuntimeError("No se encontró ningún archivo .mtx dentro del tar.gz.")

    mtx_file = tar.extractfile(mtx_member)

print(f"Archivo .mtx encontrado: {mtx_member.name}")
print("Cargando la matriz con scipy...")

# ----- LEER LA MATRIZ -----
A = mmread(mtx_file)

# ----- MOSTRAR INFORMACIÓN -----
print("\nMatriz cargada exitosamente:")
print("Shape:", A.shape)
print("Elementos no cero:", A.nnz)
print("Tipo:", type(A))
