import requests
import re
import json
from langchain.schema import Document

# URL de un texto plano del Código Penal (o puedes usar un archivo local)
# Usaremos un raw de GitHub o similar si tienes el txt, aquí simulo la carga de uno extenso.
URL_CODIGO_PENAL = "https://raw.githubusercontent.com/tu-usuario/codigo-penal-bolivia/main/codigo_penal.txt" 

def limpiar_texto(texto):
    """Limpieza básica según Figura 3.13 de la tesis (DOMParser)"""
    texto = re.sub(r'\s+', ' ', texto) # Quitar espacios extra
    return texto.strip()

def procesar_codigo_penal_texto(texto_completo):
    """
    Transforma el texto bruto en una lista de objetos Document.
    Estrategia: Usar Regex para encontrar 'ARTICULO X.-'
    """
    # Patrón para identificar inicios de artículos (Ej: "ARTICULO 10.-")
    patron = r'(ARTICULO\s+\d+[\.\-º°]+)'
    fragmentos = re.split(patron, texto_completo, flags=re.IGNORECASE)
    
    documentos = []
    
    # El split deja el título separado del contenido, hay que unirlos
    # fragmentos[0] es intro, fragmentos[1] es titulo Art 1, fragmentos[2] es contenido Art 1...
    for i in range(1, len(fragmentos), 2):
        titulo_art = fragmentos[i].strip()
        contenido = fragmentos[i+1].strip() if i+1 < len(fragmentos) else ""
        
        texto_full = f"{titulo_art} {contenido}"
        
        # Extraer número para metadatos
        numero = re.search(r'\d+', titulo_art).group()
        
        meta = {
            "fuente": "Código Penal Boliviano (Ley 1768)",
            "articulo": numero,
            "titulo": "Norma Penal", # Podríamos extraer el título específico si el txt lo tiene claro
            "contenido": contenido[:100] + "..." # Snippet para mostrar
        }
        
        doc = Document(page_content=texto_full, metadata=meta)
        documentos.append(doc)
        
    print(f"✅ Procesados {len(documentos)} artículos exitosamente.")
    return documentos

def ejecutar_etl():
    print("--- INICIANDO PROCESO ETL (Tesis Módulo 3) ---")
    
    # 1. EXTRACCIÓN
    # Si no tienes URL, crea un archivo 'codigo_penal.txt' en la misma carpeta y usa:
    # with open('codigo_penal.txt', 'r', encoding='utf-8') as f: texto = f.read()
    
    # Simulación de texto real (Copia y pega el código penal real en un txt para mejores resultados)
    texto_simulado = """
    ARTICULO 1.- (EN CUANTO AL ESPACIO). Este Código se aplicará:
    1. A los delitos cometidos en el territorio de Bolivia o en los lugares sometidos a su jurisdicción.
    2. A los delitos cometidos en el extranjero, que produzcan sus efectos en el territorio de la República.
    
    ARTICULO 2.- (SENTENCIA EXTRANJERA). En los casos previstos en el Artículo anterior, cuando el agente haya sido juzgado en el extranjero...
    
    ARTICULO 251.- (HOMICIDIO). El que matare a otro, será sancionado con presidio de cinco a veinte años. Si la víctima del delito resultare ser niña, niño o adolescente, la pena será de diez a veinticinco años.
    
    ARTICULO 331.- (ROBO). El que se apoderare de una cosa mueble ajena con fuerza en las cosas o con violencia o intimidación en las personas, será sancionado con privación de libertad de uno a cinco años.
    """
    
    print("1. Extracción completada.")
    
    # 2. TRANSFORMACIÓN
    docs = procesar_codigo_penal_texto(texto_simulado)
    
    # 3. CARGA (Guardar en disco para que el chatbot lo use siempre)
    # Aquí es donde 'persistimos' los datos para no cargarlos cada vez
    return docs

if __name__ == "__main__":
    docs = ejecutar_etl()
    # Aquí podríamos guardar 'docs' en un archivo pickle o json para que mvp_google.py lo lea