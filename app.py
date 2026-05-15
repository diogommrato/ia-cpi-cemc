import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader


APP_TITLE = "Assistente RAG do TFC"
DEFAULT_PDF_PATH = "data/TFC_RAG_demo_higienizado.pdf"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

SYSTEM_PROMPT = """
És um assistente RAG académico, em português de Portugal, alimentado exclusivamente por excertos do TFC disponibilizado.

Regras obrigatórias:
1. Responde apenas com base nos excertos fornecidos no contexto.
2. Não uses conhecimento externo, inferências sem apoio textual ou informação da internet.
3. Se a resposta não estiver sustentada nos excertos, diz claramente: "Essa informação não consta da base documental disponível."
4. Mantém uma linguagem académica, simples e elegante.
5. Não apresentes a resposta como doutrina oficial.
6. Quando for útil, identifica as páginas que sustentam a resposta.
7. Assume que todo o output é assistido por IA e deve ser validado pelo autor.
""".strip()


def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
        return str(value) if value is not None else default
    except Exception:
        return os.getenv(name, default)


def require_password() -> bool:
    app_password = get_secret("APP_PASSWORD", "")
    if not app_password:
        return True

    if st.session_state.get("authenticated"):
        return True

    st.markdown("### Acesso protegido")
    candidate = st.text_input("Palavra-passe", type="password")
    if st.button("Entrar"):
        if candidate == app_password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Palavra-passe incorreta.")
    return False


@st.cache_data(show_spinner=False)
def load_pdf_chunks(pdf_path: str, words_per_chunk: int = 260, overlap: int = 55) -> List[Dict]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Ficheiro não encontrado: {pdf_path}")

    reader = PdfReader(str(path))
    chunks: List[Dict] = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if len(text) < 40:
            continue

        words = text.split()
        step = max(1, words_per_chunk - overlap)
        for start in range(0, len(words), step):
            part_words = words[start:start + words_per_chunk]
            if len(part_words) < 35:
                continue
            chunks.append({
                "page": page_index,
                "text": " ".join(part_words)
            })

    return chunks


@st.cache_data(show_spinner="A indexar a base documental...")
def build_embeddings(chunks: List[Dict], api_key: str, embedding_model: str) -> np.ndarray:
    client = OpenAI(api_key=api_key)
    texts = [c["text"] for c in chunks]
    vectors = []

    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=embedding_model, input=batch)
        vectors.extend([item.embedding for item in response.data])

    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def embed_query(query: str, api_key: str, embedding_model: str) -> np.ndarray:
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model=embedding_model, input=[query])
    vector = np.array(response.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def retrieve(query: str, chunks: List[Dict], embeddings: np.ndarray, api_key: str, embedding_model: str, top_k: int = 6) -> List[Tuple[float, Dict]]:
    q = embed_query(query, api_key, embedding_model)
    scores = embeddings @ q
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(float(scores[i]), chunks[int(i)]) for i in top_idx]


def build_context(results: List[Tuple[float, Dict]]) -> str:
    parts = []
    for rank, (score, chunk) in enumerate(results, start=1):
        parts.append(
            f"[Excerto {rank} | página {chunk['page']} | relevância {score:.3f}]\n{chunk['text']}"
        )
    return "\n\n".join(parts)


def answer_question(question: str, context: str, api_key: str, model: str) -> str:
    client = OpenAI(api_key=api_key)
    user_input = f"""
Pergunta da audiência:
{question}

Excertos recuperados da base documental:
{context}

Tarefa:
Responde à pergunta, usando apenas os excertos recuperados. Quando a resposta não estiver sustentada, recusa com a fórmula definida nas regras.
""".strip()

    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=user_input,
        temperature=0.2,
    )
    return response.output_text.strip()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📘", layout="wide")

    st.title("📘 Assistente RAG do TFC")
    st.caption("Demonstração experimental, alimentada exclusivamente pelo TFC carregado como base documental.")

    if not require_password():
        st.stop()

    api_key = get_secret("OPENAI_API_KEY", "")
    if not api_key:
        st.error("Falta configurar OPENAI_API_KEY nos secrets do Streamlit.")
        st.stop()

    model = get_secret("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    embedding_model = get_secret("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    pdf_path = get_secret("PDF_PATH", DEFAULT_PDF_PATH)

    with st.sidebar:
        st.header("Base documental")
        st.write("Ficheiro:", f"`{pdf_path}`")
        st.write("Modelo:", f"`{model}`")
        st.write("Embeddings:", f"`{embedding_model}`")
        st.divider()
        st.info(
            "A ferramenta não consulta a internet. A resposta resulta da recuperação de excertos do TFC e deve ser validada pelo autor."
        )
        st.markdown("**Perguntas de teste**")
        examples = [
            "Qual é o problema central da investigação?",
            "Qual é o contributo original do framework?",
            "O que é o binómio CGCD/DCI?",
            "Em que fases do CPI a IA tem maior utilidade?",
            "Quais são os principais riscos da integração de IA no CPI?",
            "Qual é o impacto da IA na Marinha Portuguesa?",
        ]
        for e in examples:
            if st.button(e, use_container_width=True):
                st.session_state["question"] = e

    try:
        chunks = load_pdf_chunks(pdf_path)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if not chunks:
        st.error("Não foi possível extrair texto suficiente do PDF.")
        st.stop()

    embeddings = build_embeddings(chunks, api_key, embedding_model)

    question = st.text_area(
        "Coloque uma pergunta sobre o trabalho",
        value=st.session_state.get("question", ""),
        height=110,
        placeholder="Exemplo: Qual é o contributo original da investigação?",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        ask = st.button("Responder", type="primary", use_container_width=True)
    with col2:
        st.markdown("**Aviso:** resposta assistida por IA, baseada na base documental carregada.")

    if ask and question.strip():
        with st.spinner("A recuperar excertos e a gerar resposta..."):
            results = retrieve(question.strip(), chunks, embeddings, api_key, embedding_model, top_k=6)
            context = build_context(results)
            answer = answer_question(question.strip(), context, api_key, model)

        st.subheader("Resposta assistida por IA")
        st.write(answer)

        pages = sorted(set(chunk["page"] for _, chunk in results))
        st.caption(f"Páginas recuperadas: {', '.join(map(str, pages))}")

        with st.expander("Ver excertos recuperados"):
            for idx, (score, chunk) in enumerate(results, start=1):
                st.markdown(f"**Excerto {idx}, página {chunk['page']}, relevância {score:.3f}**")
                st.write(chunk["text"])
                st.divider()

    st.divider()
    st.caption(
        "Protótipo demonstrativo. Não constitui doutrina oficial. A validação final pertence ao autor do trabalho."
    )


if __name__ == "__main__":
    main()
