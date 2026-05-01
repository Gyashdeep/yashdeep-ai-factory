import streamlit as st
import pandas as pd
import json
import pdfplumber
from groq import Groq
from sqlalchemy import create_engine, text
from pydantic import BaseModel, Field
from typing import Dict

# --- 1. SECURE CONFIGURATION ---
# This looks for keys in the Streamlit "Vault" (Secrets)
# No hardcoded keys here = No GitHub security warnings!
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    ACCESS_KEY = st.secrets["ACCESS_KEY"]
except KeyError:
    st.error("🔑 API Keys not found in Secrets! Please add GROQ_API_KEY and ACCESS_KEY to Streamlit Settings.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# --- 2. DATA MODELS ---
class AIProposal(BaseModel):
    rename_map: Dict[str, str]
    column_types: Dict[str, str]
    confidence_score: float = Field(ge=0, le=1.0)
    reasoning: str

# --- 3. THE ENGINE ---
class DataFactory:
    def __init__(self, db_name="factory_warehouse.db"):
        self.engine = create_engine(f'sqlite:///{db_name}')
        self._init_db()
        if "bench_queue" not in st.session_state:
            st.session_state.bench_queue = []

    def _init_db(self):
        with self.engine.connect() as conn:
            conn.execute(text("CREATE TABLE IF NOT EXISTS learning_log (id INTEGER PRIMARY KEY AUTOINCREMENT, cols TEXT, map TEXT)"))
            conn.commit()

    def audit_data(self, df: pd.DataFrame):
        # Sample data for AI context
        sample = df.head(5).to_json(orient='records')
        prompt = (
            f"Audit this data. Map columns to standard, clean names. "
            f"Return JSON with 'rename_map' (old:new), 'column_types', "
            f"'confidence_score' (0-1), and 'reasoning'. Data: {sample}"
        )
        
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Expert Data Architect."}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)

    def save_to_warehouse(self, df, mapping, table_name):
        clean_df = df.copy()
        clean_df.rename(columns=mapping['rename_map'], inplace=True)
        # Clean table name for SQL safety
        safe_name = "".join(c if c.isalnum() else "_" for c in table_name).lower()
        clean_df.to_sql(safe_name, self.engine, if_exists='replace', index=False)

# --- 4. WEB INTERFACE ---
st.set_page_config(page_title="AI Data Factory", layout="wide", page_icon="🛡️")

# Authentication Layer
if "auth" not in st.session_state:
    st.title("🔐 Secure Factory Login")
    pwd = st.text_input("Enter Access Key to Enter", type="password")
    if st.button("Unlock System"):
        if pwd == ACCESS_KEY:
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Access Denied. Check your key.")
    st.stop()

factory = DataFactory()
st.title("Enterprise AI Factory")
st.caption("Advanced Data Ingestion & Transformation Pipeline")

tab1, tab2, tab3 = st.tabs(["🚀 Ingestion", "🧑‍💻 Human Bench", "📀 Gold Warehouse"])

with tab1:
    st.header("Step 1: Upload Raw Data")
    uploaded_files = st.file_uploader("Upload CSV, Excel, or PDF Resumes", accept_multiple_files=True)
    
    if uploaded_files:
        for f in uploaded_files:
            df = None
            try:
                # Handle PDF
                if f.name.lower().endswith('.pdf'):
                    with pdfplumber.open(f) as pdf:
                        content = "\n".join([p.extract_text() or "" for p in pdf.pages])
                    df = pd.DataFrame([{"filename": f.name, "content": content}])
                # Handle Excel
                elif f.name.lower().endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(f)
                # Handle CSV with encoding safety
                else:
                    for enc in ['utf-8', 'latin1', 'cp1252']:
                        try:
                            f.seek(0)
                            df = pd.read_csv(f, encoding=enc)
                            break
                        except: continue
                
                if df is not None:
                    with st.expander(f"Preview: {f.name}"):
                        st.dataframe(df.head(3))
                    
                    if st.button(f"Analyze {f.name}", key=f"btn_{f.name}"):
                        with st.spinner("AI Auditing Schema..."):
                            res = factory.audit_data(df)
                            prop = AIProposal(**res)
                            
                            if prop.confidence_score >= 0.90:
                                factory.save_to_warehouse(df, prop.dict(), f.name)
                                st.success(f"✅ {f.name} Automatically Promoted to Warehouse!")
                            else:
                                st.session_state.bench_queue.append({"name": f.name, "df": df, "prop": prop.dict()})
                                st.warning(f"⚠️ {f.name} sent to Human Bench for review.")
            except Exception as e:
                st.error(f"Error processing {f.name}: {e}")

with tab2:
    st.header(f"QA Bench ({len(st.session_state.bench_queue)} tasks)")
    if not st.session_state.bench_queue:
        st.info("No tasks currently waiting for review.")
    
    for i, task in enumerate(st.session_state.bench_queue):
        with st.expander(f"Review Required: {task['name']}"):
            st.write(f"**AI Reasoning:** {task['prop']['reasoning']}")
            
            # Allow user to edit the mapping
            map_df = pd.DataFrame(task['prop']['rename_map'].items(), columns=["Source Column", "Target Column"])
            edited_map = st.data_editor(map_df, key=f"ed_{i}")
            
            if st.button("Confirm & Promote", key=f"prom_{i}"):
                final_map = dict(zip(edited_map["Source Column"], edited_map["Target Column"]))
                task['prop']['rename_map'] = final_map
                factory.save_to_warehouse(task['df'], task['prop'], task['name'])
                st.session_state.bench_queue.pop(i)
                st.success("Data Promoted!")
                st.rerun()

with tab3:
    st.header("📀 Gold Warehouse")
    try:
        with factory.engine.connect() as conn:
            tables = [row[0] for row in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name != 'learning_log'"))]
        
        if not tables:
            st.info("The warehouse is empty. Upload and promote data to see it here.")
        else:
            selected = st.selectbox("Select Data Asset:", tables)
            data = pd.read_sql(f'SELECT * FROM "{selected}"', factory.engine)
            st.dataframe(data, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 Download as CSV", data.to_csv(index=False), f"{selected}.csv", "text/csv")
            with col2:
                if st.button("🗑️ Delete Asset", key="del_asset"):
                    with factory.engine.connect() as conn:
                        conn.execute(text(f'DROP TABLE "{selected}"'))
                        conn.commit()
                    st.rerun()
    except Exception as e:
        st.info("Initializing database system...")
