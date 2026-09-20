"""Limpia imagenes y sus etiquetas asociadas de un dataset YOLO.

Por defecto solo muestra lo que encontraria. Para borrar hay que ejecutar:
	python LimpiarDatos.py --ejecutar

Requiere Pillow: python -m pip install Pillow
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

try:
	from PIL import Image
except ImportError as error:
	raise SystemExit(
		"Falta Pillow. Instala la dependencia con: python -m pip install Pillow"
	) from error


EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def encontrar_data(ruta_indicada: Path | None) -> Path:
	"""Encuentra una carpeta que contenga Images y Labels."""
	script_dir = Path(__file__).resolve().parent
	candidatos = []

	if ruta_indicada is not None:
		candidatos.append(ruta_indicada.expanduser().resolve())
	candidatos.extend(
		[
			script_dir / "Data",
			script_dir / "Data" / "Data",
			Path.cwd() / "Data",
		]
	)

	for candidato in candidatos:
		if (candidato / "Images").is_dir() and (candidato / "Labels").is_dir():
			return candidato

	lugares = "\n".join(f"  - {ruta}" for ruta in candidatos)
	raise FileNotFoundError(
		"No se encontraron las carpetas Images y Labels. Se buscaron en:\n" + lugares
	)


def porcentaje_pixeles_grises(imagen: Image.Image) -> float:
	"""Calcula que proporcion de pixeles tiene sus canales practicamente iguales."""
	rgb = imagen.convert("RGB").resize((256, 256))
	pixeles = list(rgb.getdata())
	grises = sum(
		max(rojo, verde, azul) - min(rojo, verde, azul) <= 3
		for rojo, verde, azul in pixeles
	)
	return grises / len(pixeles)


def analizar_imagen(ruta: Path) -> list[str]:
	with Image.open(ruta) as imagen:
		razones = []
		if porcentaje_pixeles_grises(imagen) >= 0.995:
			razones.append("blanco y negro/grises")
		return razones


def iterar_imagenes(carpeta: Path) -> Iterable[Path]:
	return sorted(
		ruta
		for ruta in carpeta.iterdir()
		if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_IMAGEN
	)


def procesar(data_dir: Path, ejecutar: bool) -> tuple[int, int, int]:
	imagenes_eliminadas = 0
	etiquetas_eliminadas = 0
	errores = 0

	for imagen in iterar_imagenes(data_dir / "Images"):
		try:
			razones = analizar_imagen(imagen)
		except (OSError, ValueError) as error:
			errores += 1
			print(f"ERROR  {imagen.name}: {error}")
			continue

		if not razones:
			continue

		etiqueta = data_dir / "Labels" / f"{imagen.stem}.txt"
		accion = "BORRAR" if ejecutar else "SIMULAR"
		print(f"{accion}  {imagen.name} <- {', '.join(razones)}")

		if ejecutar:
			imagen.unlink()
			imagenes_eliminadas += 1
			if etiqueta.exists():
				etiqueta.unlink()
				etiquetas_eliminadas += 1
			else:
				print(f"AVISO   No existe label para {imagen.stem}")

	return imagenes_eliminadas, etiquetas_eliminadas, errores


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--data-dir",
		type=Path,
		help="Ruta que contiene Images y Labels (se detecta automaticamente si se omite).",
	)
	parser.add_argument(
		"--ejecutar",
		action="store_true",
		help="Borra las imagenes detectadas y sus labels; sin esta opcion solo informa.",
	)
	argumentos = parser.parse_args()
	data_dir = encontrar_data(argumentos.data_dir)
	imagenes, etiquetas, errores = procesar(data_dir, argumentos.ejecutar)

	if argumentos.ejecutar:
		print(f"\nEliminadas: {imagenes} imagenes y {etiquetas} labels.")
	else:
		print("\nSimulacion terminada. Usa --ejecutar para aplicar los borrados.")
	if errores:
		print(f"Archivos con error: {errores}")


if __name__ == "__main__":
	main()
