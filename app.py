import streamlit as st
import re
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# --- CONFIGURACIÓN ---
# ¡Recuerda! Configura tu API Key aquí o en variables de entorno
# os.environ["GOOGLE_API_KEY"] = "TU_CLAVE_API_AQUI"

# Nombre de la carpeta donde guardaremos la "memoria" del bot
CARPETA_VECTOR_DB = "faiss_index_penal"

# --- 1. MÓDULO ETL (Extract, Transform, Load) ---
def ejecutar_etl_completo():
    print("🔄 Iniciando lectura masiva del Código Penal...")
    
    try:
        with open("codigo_penal.txt", "r", encoding="utf-8") as f:
            texto_completo = f.read()
    except FileNotFoundError:
        print("⚠️ No encontré 'codigo_penal.txt'.")
        return []

    # Regex mejorado: Busca ARTICULO seguido de numero
    patron = r'(ARTICULO\s+\d+[\.\-º°]+)'
    fragmentos = re.split(patron, texto_completo, flags=re.IGNORECASE)
    
    documentos = []
    
    # Debug: Ver qué estamos leyendo
    print("🔍 Analizando fragmentos...")

    for i in range(1, len(fragmentos), 2):
        titulo_raw = fragmentos[i].strip() 
        contenido_raw = fragmentos[i+1].strip() if i+1 < len(fragmentos) else ""
        contenido_limpio = re.sub(r'\s+', ' ', contenido_raw)
        
        # --- FILTROS DE CALIDAD (NUEVO) ---
        
        # 1. Filtro de Longitud: Si la descripción es muy corta (< 50 caracteres), 
        # probablemente es un índice o un error de copiado. Lo ignoramos.
        if len(contenido_limpio) < 50:
            continue
            
        # 2. Filtro de "Derogado": Si el artículo fue borrado, no nos sirve.
        if "derogado" in contenido_limpio.lower() or "abrogado" in contenido_limpio.lower():
            continue

        numero_match = re.search(r'\d+', titulo_raw)
        numero = numero_match.group() if numero_match else "S/N"
        
        # --- ENRIQUECIMIENTO SEMÁNTICO (NUEVO) ---
        # Le decimos explícitamente a la IA qué es esto para mejorar la búsqueda
        texto_final = f"LEY PENAL BOLIVIANA. DELITO: {titulo_raw}. DEFINICIÓN Y PENA: {contenido_limpio}"
        
        meta = {
            "articulo": numero,
            "titulo": titulo_raw,
            "contenido": contenido_limpio,
            "fuente": "Código Penal (Texto Completo)"
        }
        
        documentos.append(Document(page_content=texto_final, metadata=meta))
    
    print(f"✅ ¡Ingestión exitosa! Se procesaron {len(documentos)} artículos de calidad (Filtros aplicados).")
    return documentos

# --- 2. GESTIÓN DE PERSISTENCIA (Base Vectorial) ---
@st.cache_resource(show_spinner=False)
def cargar_motor_inteligente():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # ¿Existe la base de datos en disco?
    if os.path.exists(CARPETA_VECTOR_DB):
        print("📂 Cargando base de datos desde disco...")
        vector_db = FAISS.load_local(CARPETA_VECTOR_DB, embeddings, allow_dangerous_deserialization=True)
    else:
        print("⚙️ Creando nueva base de datos vectorial...")
        # 1. Ejecutar ETL
        docs = ejecutar_etl_completo()
        # 2. Vectorizar
        vector_db = FAISS.from_documents(docs, embeddings)
        # 3. Guardar en disco
        vector_db.save_local(CARPETA_VECTOR_DB)
        print(f"¡Indexación completada! Se guardaron {len(docs)} leyes.")
        
    return vector_db

def consultar_gemini(query, vector_db):
    # Recuperación (RAG)
    docs_relacionados = vector_db.similarity_search(query, k=3)
    contexto = "\n\n".join([f"- Art. {d.metadata['articulo']} ({d.metadata['titulo']}): {d.page_content}" for d in docs_relacionados])
    
    # Prompt de Ingeniería (Rol Experto)
    template = """
    Actúa como un Abogado Penalista Senior de Bolivia.
    
    LEYES APLICABLES AL CASO (Contexto Real):
    {contexto}
    
    CONSULTA DEL CLIENTE:
    "{pregunta}"
    
    TU RESPUESTA DEBE:
    1. Ser directa y empática.
    2. Identificar el posible delito basándote SOLO en las leyes de arriba.
    3. Explicar la pena posible (años de cárcel).
    4. Mencionar el número de artículo.
    
    Respuesta legal:
    """
    
    prompt = PromptTemplate(template=template, input_variables=["contexto", "pregunta"])
    
    # Modelo Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    chain = prompt | llm
    
    respuesta = chain.invoke({"contexto": contexto, "pregunta": query})
    return respuesta.content, docs_relacionados

# --- 3. INTERFAZ GRÁFICA ---
def main():
    st.set_page_config(page_title="LegalBot Pro Bolivia", page_icon="⚖️", layout="centered")
    
    st.title("⚖️ Asistente Penal Bolivia")
    st.caption("Powered by **Google Gemini** + **FAISS Vector DB**")

    # Carga del motor (Solo se ejecuta al inicio o si no hay cache)
    with st.spinner("Inicializando motor jurídico..."):
        db = cargar_motor_inteligente()

    # Chat
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []

    for msg in st.session_state.mensajes:
        with st.chat_message(msg["rol"]):
            st.markdown(msg["contenido"])
            # Si el mensaje tiene fuentes guardadas, las mostramos (opcional, para persistencia visual)
            if "fuentes" in msg:
                 with st.expander("📚 Ver Artículos del Código Penal citados (Texto Completo)"):
                    for doc in msg["fuentes"]:
                        st.markdown(f"### {doc['titulo']}")
                        st.info(doc['contenido'])
                        st.divider()

    pregunta = st.chat_input("Escribe tu caso (Ej: Vendí una casa que estaba embargada...)")

    if pregunta:
        # Usuario
        st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        # Asistente
        with st.chat_message("assistant"):
            with st.spinner("Analizando jurisprudencia..."):
                resp_texto, fuentes = consultar_gemini(pregunta, db)
                st.markdown(resp_texto)
                
                # Guardamos las fuentes en un formato simple para el historial
                fuentes_serializadas = [{"titulo": d.metadata['titulo'], "contenido": d.metadata['contenido']} for d in fuentes]

                # Mostrar evidencia colapsable
                with st.expander("📚 Ver Artículos del Código Penal citados (Texto Completo)"):
                    for doc in fuentes:
                        st.markdown(f"### {doc.metadata['titulo']}")
                        # Usamos 'info' para resaltar el texto legal completo
                        st.info(doc.metadata['contenido'])
                        st.divider() 
        
        st.session_state.mensajes.append({
            "rol": "assistant", 
            "contenido": resp_texto,
            "fuentes": fuentes_serializadas
        })

if __name__ == "__main__":
    main()