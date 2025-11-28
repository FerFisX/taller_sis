import streamlit as st
import re
import os
import time
import sqlite3
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# --- CONFIGURACIÓN GENERAL ---
# Asegurate de tener tu API KEY en el archivo .env o en las variables de entorno del sistema
# os.environ["GOOGLE_API_KEY"] = "TU_API_KEY"

CARPETA_VECTOR_DB = "faiss_index_penal"
DB_NAME = "legal_data.db"

# --- CAPA DE BASE DE DATOS (SQLite) ---

def init_db():
    """Inicializa la base de datos y crea las tablas si no existen"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Tabla de Usuarios
    c.execute('''CREATE TABLE IF NOT EXISTS usuarios
                 (username TEXT PRIMARY KEY, password TEXT, rol TEXT, nombre TEXT, ciudad TEXT)''')
    
    # Tabla de Historial
    c.execute('''CREATE TABLE IF NOT EXISTS historial
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, usuario TEXT, fecha TEXT, area TEXT, consulta TEXT, respuesta TEXT)''')
    
    # Usuarios por defecto (Semilla)
    # Solo insertamos si la tabla está vacía
    c.execute('SELECT count(*) FROM usuarios')
    if c.fetchone()[0] == 0:
        usuarios_iniciales = [
            ('ciudadano', '123', 'ciudadano', 'Juan Perez', 'La Paz'),
            ('abogado', '123', 'abogado', 'Dr. Mendez', 'Santa Cruz'),
            ('admin', 'admin', 'admin', 'Administrador UCB', 'N/A')
        ]
        c.executemany('INSERT INTO usuarios VALUES (?,?,?,?,?)', usuarios_iniciales)
        conn.commit()
    
    conn.close()

def verificar_credenciales(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT * FROM usuarios WHERE username = ? AND password = ?', (username, password))
    user = c.fetchone()
    conn.close()
    if user:
        return {"username": user[0], "rol": user[2], "nombre": user[3], "ciudad": user[4]}
    return None

def guardar_historial_db(usuario, consulta, respuesta, area):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    c.execute('INSERT INTO historial (usuario, fecha, area, consulta, respuesta) VALUES (?,?,?,?,?)',
              (usuario, fecha, area, consulta, respuesta))
    conn.commit()
    conn.close()

def leer_historial_db(usuario):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT fecha, area, consulta, respuesta FROM historial WHERE usuario = ? ORDER BY id DESC', (usuario,))
    data = c.fetchall()
    conn.close()
    return data

def leer_todos_historiales():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT usuario, fecha, area, consulta, respuesta FROM historial ORDER BY id DESC')
    data = c.fetchall()
    conn.close()
    return data

# --- MOTOR ETL Y RAG ---

def ejecutar_etl_completo():
    """Lee el txt, limpia y estructura los datos"""
    try:
        with open("codigo_penal.txt", "r", encoding="utf-8") as f:
            texto_completo = f.read()
    except FileNotFoundError:
        st.error("Error: Falta el archivo codigo_penal.txt")
        return []

    patron = r'(ARTICULO\s+\d+[\.\-º°]+)'
    fragmentos = re.split(patron, texto_completo, flags=re.IGNORECASE)
    
    documentos = []
    for i in range(1, len(fragmentos), 2):
        titulo_raw = fragmentos[i].strip()
        contenido_raw = fragmentos[i+1].strip() if i+1 < len(fragmentos) else ""
        contenido_limpio = re.sub(r'\s+', ' ', contenido_raw)
        
        if len(contenido_limpio) < 50:
            continue
        
        texto_final = f"LEY BOLIVIANA: {titulo_raw}. DETALLE: {contenido_limpio}"
        
        numero_match = re.search(r'\d+', titulo_raw)
        numero = numero_match.group() if numero_match else "S/N"

        meta = {
            "articulo": numero,
            "titulo": titulo_raw,
            "contenido": contenido_limpio,
            "fuente": "Codigo Penal"
        }
        documentos.append(Document(page_content=texto_final, metadata=meta))
    
    return documentos

@st.cache_resource(show_spinner=False)
def cargar_motor():
    # Usamos embeddings de Google
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    if os.path.exists(CARPETA_VECTOR_DB):
        return FAISS.load_local(CARPETA_VECTOR_DB, embeddings, allow_dangerous_deserialization=True)
    else:
        docs = ejecutar_etl_completo()
        if not docs: return None
        vector_db = FAISS.from_documents(docs, embeddings)
        vector_db.save_local(CARPETA_VECTOR_DB)
        return vector_db

def consultar_ia(query, vector_db, filtros):
    if vector_db is None: return "Error: Base de datos no disponible.", []
    
    # Retrieval
    docs = vector_db.similarity_search(query, k=4)
    contexto = "\n\n".join([f"--- ARTICULO {d.metadata['articulo']} ---\n{d.page_content}" for d in docs])
    
    # Prompt sin emojis y formal
    template = f """
    Actua como un Asistente Legal Boliviano experto.
    
    CONTEXTO LEGAL (Leyes vigentes):
    {contexto}
    
    DATOS DEL CIUDADANO:
    - Consulta: "{query}"
    - Area de interes: {filtros['area']}
    - Ciudad: {filtros['ciudad']}
    
    INSTRUCCIONES:
    1. Analiza si la consulta coincide con los articulos legales proporcionados.
    2. Si coincide, explica el delito y la pena de forma clara y formal.
    3. Si la consulta es sobre un tema y las leyes proporcionadas no aplican, indicalo cortesmente.
    4. NO inventes leyes. Usa solo el contexto provisto.
    
    Respuesta:
    """
    
    prompt = PromptTemplate(template=template, input_variables=["contexto", "query"])
    
    # Usamos el modelo que indicaste que funciona
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    
    chain = prompt | llm
    res = chain.invoke({"contexto": contexto, "query": query})
    return res.content, docs

# --- VISTAS DE LA INTERFAZ ---

def vista_login():
    st.markdown("<h2 style='text-align: center;'>Plataforma Legal UCB</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Sistema de Orientacion Legal Inteligente</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        with st.container(border=True):
            st.subheader("Iniciar Sesion")
            usuario = st.text_input("Usuario")
            password = st.text_input("Contrasena", type="password")
            
            if st.button("Ingresar", use_container_width=True):
                user_data = verificar_credenciales(usuario, password)
                if user_data:
                    st.session_state["usuario_actual"] = usuario
                    st.session_state["rol"] = user_data["rol"]
                    st.session_state["datos"] = user_data
                    st.success("Acceso correcto")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Credenciales incorrectas")
            
            st.caption("Usuarios de prueba: ciudadano/123, abogado/123, admin/admin")

def vista_ciudadano(db):
    user_data = st.session_state["datos"]
    st.title(f"Bienvenido, {user_data['nombre']}")
    
    # Barra lateral
    with st.sidebar:
        st.header("Configuracion")
        area_filtro = st.selectbox("Area Legal", ["Penal", "Civil", "Familiar", "Laboral"])
        ciudad_filtro = st.selectbox("Ciudad", ["La Paz", "Santa Cruz", "Cochabamba"], index=0 if user_data['ciudad'] == "La Paz" else 1)
        st.divider()
        if st.button("Cerrar Sesion"):
            st.session_state.clear()
            st.rerun()

    # Pestañas limpias (sin emojis)
    tab1, tab2, tab3 = st.tabs(["Asistente Legal", "Mi Historial", "Buscar Abogado"])

    # PESTAÑA 1: CHAT
    with tab1:
        st.subheader("Consulta Legal")
        
        if "chat_history" not in st.session_state: st.session_state.chat_history = []
        
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["rol"]):
                st.markdown(msg["content"])
                if "fuentes" in msg:
                    with st.expander("Referencias Legales"):
                        for f in msg["fuentes"]:
                            st.markdown(f"**{f['titulo']}**")
                            st.info(f['contenido'])

        pregunta = st.chat_input("Escriba su consulta legal aqui...")
        
        if pregunta:
            st.session_state.chat_history.append({"rol": "user", "content": pregunta})
            with st.chat_message("user"):
                st.markdown(pregunta)
            
            with st.chat_message("assistant"):
                with st.spinner("Analizando jurisprudencia..."):
                    filtros = {"area": area_filtro, "ciudad": ciudad_filtro}
                    respuesta, docs = consultar_ia(pregunta, db, filtros)
                    
                    st.markdown(respuesta)
                    
                    fuentes_serializables = [{"titulo": d.metadata['titulo'], "contenido": d.metadata['contenido']} for d in docs]
                    
                    with st.expander("Referencias Legales"):
                        for doc in docs:
                            st.markdown(f"**{doc.metadata['titulo']}**")
                            st.write(doc.metadata['contenido'])
                            st.divider()

            st.session_state.chat_history.append({
                "rol": "assistant", 
                "content": respuesta,
                "fuentes": fuentes_serializables
            })
            # Guardar en SQLite
            guardar_historial_db(st.session_state["usuario_actual"], pregunta, respuesta, area_filtro)

    # PESTAÑA 2: HISTORIAL
    with tab2:
        st.subheader("Registro de Consultas")
        historial = leer_historial_db(st.session_state["usuario_actual"])
        
        if not historial:
            st.info("No hay consultas registradas.")
        else:
            for fecha, area, consulta, respuesta in historial:
                with st.container(border=True):
                    st.markdown(f"**Fecha:** {fecha} | **Area:** {area}")
                    st.markdown(f"**Consulta:** {consulta}")
                    st.markdown(f"**Respuesta:** {respuesta[:200]}...")

    # PESTAÑA 3: ABOGADOS (Datos simulados en lista, pero sin emojis)
    with tab3:
        st.subheader(f"Directorio de Abogados - {ciudad_filtro}")
        
        # Lista estática por ahora (se podría mover a SQLite también)
        abogados_db = [
            {"nombre": "Dra. Laura Calle", "area": "Penal", "ciudad": "La Paz", "cel": "77700001", "rating": 4.8},
            {"nombre": "Dr. Carlos Torrez", "area": "Civil", "ciudad": "La Paz", "cel": "77700002", "rating": 4.5},
            {"nombre": "Bufete Justicia", "area": "Penal", "ciudad": "Santa Cruz", "cel": "60000001", "rating": 4.9},
        ]
        
        recomendados = [abg for abg in abogados_db if abg['ciudad'] == ciudad_filtro and abg['area'] == area_filtro]
        
        if recomendados:
            for abg in recomendados:
                with st.container(border=True):
                    c1, c2 = st.columns([3,1])
                    with c1:
                        st.markdown(f"**{abg['nombre']}**")
                        st.write(f"Especialidad: {abg['area']}")
                    with c2:
                        st.write(f"Calif: {abg['rating']}/5.0")
                        st.write(f"Tel: {abg['cel']}")
                    st.button("Contactar", key=f"btn_{abg['nombre']}")
        else:
            st.warning(f"No hay abogados registrados en {area_filtro} para {ciudad_filtro}.")

def vista_admin():
    st.title("Panel de Administracion")
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT count(*) FROM usuarios")
    num_usuarios = c.fetchone()[0]
    c.execute("SELECT count(*) FROM historial")
    num_consultas = c.fetchone()[0]
    conn.close()

    col1, col2 = st.columns(2)
    col1.metric("Usuarios Registrados", num_usuarios)
    col2.metric("Consultas Realizadas", num_consultas)
    
    st.subheader("Auditoria de Consultas")
    data_historial = leer_todos_historiales()
    if data_historial:
        st.table(data_historial) # Tabla simple en lugar de dataframe complejo
    else:
        st.info("Sin registros.")

    if st.button("Cerrar Sesion"):
        st.session_state.clear()
        st.rerun()

# --- MAIN ---
def main():
    st.set_page_config(page_title="Plataforma Legal UCB", layout="wide")
    
    # Inicializar DB al arrancar
    init_db()
    
    db = cargar_motor()

    if "usuario_actual" not in st.session_state:
        vista_login()
    else:
        rol = st.session_state.get("rol")
        if rol == "ciudadano":
            vista_ciudadano(db)
        elif rol == "admin":
            vista_admin()
        else:
            vista_ciudadano(db) # Fallback para abogados

if __name__ == "__main__":
    main()
