# frontend/app.py
import streamlit as st
import requests
import os
import urllib.parse
from typing import List

# safe API base (your approach was fine)
API_BASE_DEFAULT = "http://127.0.0.1:8000"

# 1) allow override from environment (useful for local dev / docker)
api_base = os.environ.get("API_BASE", None)

if api_base:
    API_BASE = api_base
else:
    # 2) try Streamlit secrets safely; do NOT use "in" or other checks that trigger parsing
    try:
        # Import the exception class (safe even if secrets missing)
        from streamlit.runtime.secrets import StreamlitSecretNotFoundError
    except Exception:
        # older/newer streamlit internals might differ; fall back to base Exception in the next block
        StreamlitSecretNotFoundError = Exception

    try:
        # accessing st.secrets[...] raises StreamlitSecretNotFoundError if no secrets file exists
        API_BASE = st.secrets["API_BASE"]
    except (KeyError, StreamlitSecretNotFoundError, Exception):
        # KeyError -> secrets exist but key missing
        # StreamlitSecretNotFoundError -> no secrets file at all
        # Generic Exception -> catch other streamlit internals differences
        API_BASE = API_BASE_DEFAULT

st.set_page_config(page_title="PDF Vector Search", layout="wide")
st.title("PDF Vector Search — upload & query")

# ----- Upload form -----
st.header("1) Upload PDFs (multiple)")
with st.form("upload_form"):
    uploaded_files = st.file_uploader(
        "Choose PDF files", accept_multiple_files=True, type=["pdf"]
    )
    ingest_btn = st.form_submit_button("Upload and Ingest")

if ingest_btn:
    if not uploaded_files:
        st.warning("Please choose at least one PDF.")
    else:
        # build files payload for requests; resetting file pointer isn't necessary for Streamlit UploadedFile
        files_payload = [("files", (f.name, f, "application/pdf")) for f in uploaded_files]

        with st.spinner("Uploading and ingesting PDFs... (this may take a while)"):
            try:
                resp = requests.post(
                    f"{API_BASE.rstrip('/')}/upload",
                    files=files_payload,
                    timeout=600,
                )
                # raise_for_status will raise HTTPError for 4xx/5xx
                resp.raise_for_status()
                try:
                    j = resp.json()
                    st.success(f"Ingested: {j.get('ingested_files')}")
                    st.write("Chunks indexed:", j.get("total_chunks_indexed"))
                except ValueError:
                    st.error("Upload succeeded but response JSON was invalid.")
            except requests.exceptions.RequestException as err:
                st.error(f"Upload failed: {err}")

st.markdown("---")

# ----- Search -----
st.header("2) Search across uploaded PDFs")
with st.form("search_form"):
    q = st.text_input("Enter your query", "")
    k = st.slider("Results (top k)", min_value=1, max_value=50, value=10)
    search_btn = st.form_submit_button("Search")

if search_btn:
    if not q.strip():
        st.warning("Type a query first.")
    else:
        with st.spinner("Searching..."):
            try:
                r = requests.get(
                    f"{API_BASE.rstrip('/')}/search",
                    params={"q": q, "top_k": k},
                    timeout=60,
                )
                r.raise_for_status()
                try:
                    data = r.json()
                except ValueError:
                    st.error("Search succeeded but returned invalid JSON.")
                    data = {}

                hits = data.get("hits", [])
                first = data.get("first_occurrence_by_pdf", {})

                st.subheader("Results")
                # show results
                for idx, h in enumerate(hits, start=1):
                    pdf_name = h.get("pdf_name", "unknown.pdf")
                    page_num = h.get("page_num", "?")
                    start_line = h.get("start_line", "?")
                    end_line = h.get("end_line", "?")
                    snippet = h.get("snippet", "")

                    st.markdown(
                        f"**{idx}.** {pdf_name} — page {page_num} lines {start_line}-{end_line}"
                    )
                    st.write(snippet)

                    col1, col2 = st.columns([1, 6])
                    with col1:
                        # Build a safe download filename key (unique key per hit)
                        dl_key = f"dl-{idx}-{urllib.parse.quote_plus(pdf_name)}"
                        save_key = f"save-{idx}-{urllib.parse.quote_plus(pdf_name)}"
                        # Button to trigger download fetch
                        if st.button(f"Download {pdf_name}", key=dl_key):
                            try:
                                download_url = f"{API_BASE.rstrip('/')}/download/{urllib.parse.quote(pdf_name)}"
                                dr = requests.get(download_url, stream=True, timeout=60)
                                dr.raise_for_status()
                                # read bytes
                                content = dr.content
                                # Store in session state
                                st.session_state[f"download_data_{idx}"] = content
                                st.session_state[f"download_filename_{idx}"] = pdf_name
                            except requests.exceptions.RequestException as e:
                                st.error(f"Download failed: {e}")
                        # Render download button if data is available
                        if f"download_data_{idx}" in st.session_state:
                            st.download_button(
                                label=f"Save {pdf_name}",
                                data=st.session_state[f"download_data_{idx}"],
                                file_name=st.session_state[f"download_filename_{idx}"],
                                key=save_key,
                            )

                st.markdown("### First occurrence per PDF (earliest page/line among hits)")
                for pdf, v in first.items():
                    st.markdown(f"- **{pdf}** — page {v['page_num']} lines {v['start_line']}-{v['end_line']}")
                    st.write(v.get("snippet", ""))

            except requests.exceptions.RequestException as e:
                st.error(f"Search failed: {e}")


# # frontend/app.py
# import streamlit as st
# import requests
# import os
# from typing import List

# # API_BASE = st.secrets.get("API_BASE", "http://127.0.0.1:8000")  # set via Streamlit secrets or env

# API_BASE = "http://127.0.0.1:8000"
# if hasattr(st, "secrets") and "API_BASE" in st.secrets:
#     API_BASE = st.secrets["API_BASE"]

# st.set_page_config(page_title="PDF Vector Search", layout="wide")

# st.title("PDF Vector Search — upload & query")

# st.header("1) Upload PDFs (multiple)")
# uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type=["pdf"])

# if st.button("Upload and Ingest"):
#     if not uploaded_files:
#         st.warning("Please choose at least one PDF.")
#     else:
#         with st.spinner("Uploading and ingesting PDFs... (this may take a while)"):
#             files_payload = [("files", (f.name, f, "application/pdf")) for f in uploaded_files]
#             # use requests to POST multipart
#             resp = requests.post(f"{API_BASE}/upload", files=files_payload, timeout=600)
#             if resp.status_code == 200:
#                 st.success(f"Ingested: {resp.json().get('ingested_files')}")
#                 st.write("Chunks indexed:", resp.json().get("total_chunks_indexed"))
#             else:
#                 st.error(f"Upload failed: {resp.status_code} - {resp.text}")

# st.markdown("---")
# st.header("2) Search across uploaded PDFs")
# q = st.text_input("Enter your query", "")
# k = st.slider("Results (top k)", min_value=1, max_value=50, value=10)

# if st.button("Search"):
#     if not q.strip():
#         st.warning("Type a query first.")
#     else:
#         with st.spinner("Searching..."):
#             try:
#                 r = requests.get(f"{API_BASE}/search", params={"q": q, "top_k": k}, timeout=60)
#                 r.raise_for_status()
#                 data = r.json()
#                 hits = data.get("hits", [])
#                 first = data.get("first_occurrence_by_pdf", {})
#                 st.subheader("Results")
#                 for idx, h in enumerate(hits, start=1):
#                     st.markdown(f"**{idx}.** {h.get('pdf_name')} — page {h.get('page_num')} lines {h.get('start_line')}-{h.get('end_line')}")
#                     st.write(h.get("snippet"))
#                     col1, col2 = st.columns([1, 6])
#                     with col1:
#                         if st.button(f"Download {h.get('pdf_name')}", key=f"dl-{idx}"):
#                             download_url = f"{API_BASE}/download/{h.get('pdf_name')}"
#                             st.write(f"[Download {h.get('pdf_name')}]({download_url})")
#                 st.markdown("### First occurrence per PDF (earliest page/line among hits)")
#                 for pdf, v in first.items():
#                     st.markdown(f"- **{pdf}** — page {v['page_num']} lines {v['start_line']}-{v['end_line']}")
#                     st.write(v['snippet'])

#             except Exception as e:
#                 st.error(f"Search failed: {e}")
