# Assistente RAG do TFC

Protótipo Streamlit para demonstrar um sistema LLM/RAG alimentado exclusivamente por uma versão higienizada do TFC.

## Estrutura

```
app.py
requirements.txt
data/TFC_RAG_demo_higienizado.pdf
.streamlit/secrets.toml.example
scripts/create_qr.py
```

## Como testar localmente

1. Instalar dependências:

```bash
pip install -r requirements.txt
```

2. Criar o ficheiro `.streamlit/secrets.toml`, com base em `.streamlit/secrets.toml.example`.

3. Colocar a chave da OpenAI em `OPENAI_API_KEY`.

4. Executar:

```bash
streamlit run app.py
```

## Como publicar no Streamlit Community Cloud

1. Criar um repositório GitHub.

2. Fazer upload destes ficheiros para o repositório.

3. Confirmar que `.streamlit/secrets.toml` não foi enviado para o GitHub.

4. Aceder ao Streamlit Community Cloud.

5. Criar uma nova app, selecionando o repositório, a branch e o ficheiro `app.py`.

6. Em Advanced settings, colar o conteúdo do ficheiro `.streamlit/secrets.toml` no campo Secrets.

7. Publicar a app.

## Gerar QR code

Depois de teres o link da app:

```bash
python scripts/create_qr.py https://o-teu-link.streamlit.app
```

O ficheiro `qr_assistente_tfc.png` pode ser inserido no slide final da apresentação.

## Nota de segurança

A versão incluída em `data/TFC_RAG_demo_higienizado.pdf` contém apenas o corpo principal e as referências bibliográficas do TFC. Os apêndices foram removidos nesta versão demonstrativa.
