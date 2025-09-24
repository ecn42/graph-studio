import os
import re
import time
import shutil
from typing import Optional, List
from io import StringIO
import sqlite3

import streamlit as st
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC

CVM_URL = "https://cvmweb.cvm.gov.br/swb/default.asp?sg_sistema=fundosreg"

# Absolute XPaths provided (search page)
XPATH_CNPJ_INPUT = "/html/body/form/table/tbody/tr[2]/td[1]/input"
XPATH_TIPO_SELECT = "/html/body/form/table/tbody/tr[4]/td/select"
XPATH_SEARCH_BTN = "/html/body/form/table/tbody/tr[7]/td/input"

# Extra navigation steps after search
XPATH_LINK_1 = "/html/body/form/table/tbody/tr/td[2]/a"
XPATH_LINK_2 = "/html/body/form/table[2]/tbody/tr[3]/td/li/a"

# Target info on the final page
XPATH_TARGET_TABLE = "/html/body/form/table/tbody/tr[7]/td/table"
XPATH_COMP_SELECT = "/html/body/form/table/tbody/tr[1]/td/select"
XPATH_FUNDO_NAME = "/html/body/form/table/tbody/tr[2]/td[1]/span"

DB_PATH = "fundos.db"


def find_chromium_binary() -> str:
    env_path = os.environ.get("CHROMIUM_BINARY")
    if env_path and os.path.exists(env_path):
        return env_path

    candidates = [
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/snap/bin/chromium",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            if p == "/snap/bin/chromium":
                inner_candidates = [
                    "/snap/chromium/current/usr/lib/chromium/chrome",
                    "/snap/chromium/current/usr/lib/chromium-browser/chrome",
                ]
                for ic in inner_candidates:
                    if os.path.exists(ic):
                        return ic
            return p

    raise FileNotFoundError(
        "Chromium binary not found. Install it (e.g., `sudo snap install chromium`) "
        "or set CHROMIUM_BINARY to its path."
    )


def create_driver(headless: bool = True) -> webdriver.Chrome:
    opts = Options()
    opts.binary_location = find_chromium_binary()

    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1400,900")
    opts.add_argument("--lang=pt-BR")
    opts.add_argument("--no-first-run")
    opts.add_argument("--no-default-browser-check")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--disable-software-rasterizer")
    opts.add_argument("--remote-allow-origins=*")
    opts.add_argument("--user-data-dir=/tmp/selenium-chromium-profile")

    driver = webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(60)
    driver.implicitly_wait(0)
    return driver


def clean_cnpj(raw: str) -> str:
    return re.sub(r"\D", "", raw or "")


def highlight_and_capture(driver: webdriver.Chrome, element=None) -> Optional[bytes]:
    if not driver:
        return None
    if element is not None:
        try:
            driver.execute_script(
                "arguments[0].scrollIntoView({behavior:'instant', "
                "block:'center', inline:'center'});",
                element,
            )
            prev_outline = driver.execute_script(
                "const e=arguments[0]; const prev=e.style.outline; "
                "e.style.outline='3px solid #ff2d55'; "
                "e.style.outlineOffset='2px'; return prev;",
                element,
            )
            time.sleep(0.2)
            png = driver.get_screenshot_as_png()
        finally:
            try:
                driver.execute_script(
                    "arguments[0].style.outline=arguments[1];",
                    element,
                    prev_outline,
                )
            except Exception:
                pass
        return png
    return driver.get_screenshot_as_png()


def wait_ready(driver: webdriver.Chrome, timeout: int = 30) -> None:
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )


def list_frames(driver: webdriver.Chrome) -> List[str]:
    frames = driver.find_elements(By.TAG_NAME, "frame")
    iframes = driver.find_elements(By.TAG_NAME, "iframe")
    labels = []
    for i, f in enumerate(frames):
        name = f.get_attribute("name") or f.get_attribute("id") or f"{i}"
        labels.append(f"frame[{i}] name/id={name}")
    for i, f in enumerate(iframes):
        name = f.get_attribute("name") or f.get_attribute("id") or f"{i}"
        labels.append(f"iframe[{i}] name/id={name}")
    return labels


def switch_to_frame_containing_xpath(
    driver: webdriver.Chrome, xpath: str, total_timeout: float = 15.0
) -> None:
    deadline = time.time() + total_timeout

    def try_here() -> bool:
        return len(driver.find_elements(By.XPATH, xpath)) > 0

    while time.time() < deadline:
        driver.switch_to.default_content()
        if try_here():
            return

        frames_lvl1 = driver.find_elements(By.TAG_NAME, "frame") + driver.find_elements(
            By.TAG_NAME, "iframe"
        )
        for f1 in frames_lvl1:
            driver.switch_to.default_content()
            driver.switch_to.frame(f1)
            if try_here():
                return

            frames_lvl2 = driver.find_elements(By.TAG_NAME, "frame") + driver.find_elements(
                By.TAG_NAME, "iframe"
            )
            for f2 in frames_lvl2:
                driver.switch_to.default_content()
                driver.switch_to.frame(f1)
                driver.switch_to.frame(f2)
                if try_here():
                    return

        time.sleep(0.3)

    raise TimeoutError(f"Could not find target xpath within any frame: {xpath}")


def select_fd_investimento(sel: Select) -> None:
    target = "FDOS DE INVESTIMENTO"
    try:
        sel.select_by_visible_text(target)
        return
    except Exception:
        pass
    for o in sel.options:
        if target in (o.text or ""):
            sel.select_by_visible_text(o.text)
            return
    raise ValueError(f"Option '{target}' not found in the dropdown.")


def click_xpath_any_frame(
    driver: webdriver.Chrome,
    xpath: str,
    pre_click_screenshot_cb=None,
    post_click_wait: float = 1.0,
    total_timeout: float = 20.0,
):
    before_handles = driver.window_handles[:]
    switch_to_frame_containing_xpath(driver, xpath, total_timeout)
    el = WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.XPATH, xpath)))
    if pre_click_screenshot_cb:
        try:
            pre_click_screenshot_cb(el)
        except Exception:
            pass
    try:
        el.click()
    except Exception:
        driver.execute_script("arguments[0].click();", el)
    time.sleep(post_click_wait)
    after_handles = driver.window_handles[:]
    if len(after_handles) > len(before_handles):
        new_handle = list(set(after_handles) - set(before_handles))[0]
        driver.switch_to.window(new_handle)
        try:
            wait_ready(driver, timeout=30)
        except Exception:
            pass
    return el


def extract_table_df(driver: webdriver.Chrome, xpath: str) -> pd.DataFrame:
    switch_to_frame_containing_xpath(driver, xpath, 25.0)
    table_el = WebDriverWait(driver, 25).until(
        EC.presence_of_element_located((By.XPATH, xpath))
    )
    html = table_el.get_attribute("outerHTML") or ""
    dfs = pd.read_html(StringIO(html))
    if not dfs:
        raise ValueError("No tables parsed from the target HTML.")
    return dfs[0]


def get_text_any_frame(driver: webdriver.Chrome, xpath: str, timeout: float = 20.0) -> str:
    switch_to_frame_containing_xpath(driver, xpath, timeout)
    el = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.XPATH, xpath)))
    return (el.text or "").strip()


def get_select_value_any_frame(
    driver: webdriver.Chrome, xpath: str, timeout: float = 20.0
) -> str:
    switch_to_frame_containing_xpath(driver, xpath, timeout)
    sel_el = WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.XPATH, xpath))
    )
    sel = Select(sel_el)
    try:
        return (sel.first_selected_option.text or "").strip()
    except Exception:
        options = sel.options
        if options:
            return (options[0].text or "").strip()
        return ""


def transform_table(
    df: pd.DataFrame, nome_fundo: str, competencia: str, cnpj_digits: str
) -> pd.DataFrame:
    # 1) Drop the first 4 rows
    t = df.iloc[4:].reset_index(drop=True).copy()

    # 2) Keep only the first and last columns
    if t.shape[1] < 2:
        raise ValueError("Unexpected table shape: need at least two columns.")
    first_col = t.columns[0]
    last_col = t.columns[-1]
    t = t[[first_col, last_col]].copy()

    # Rename columns for downstream logic
    t.columns = ["descricao", t.columns[1]]

    # 3) Last column processing: normalize digits to float percent/100
    def normalize_to_digits(s: str) -> str:
        s = str(s).strip()
        if not s:
            return ""
        neg = s.startswith("-")
        digits = "".join(ch for ch in s if ch.isdigit())
        if not digits:
            return "-0" if neg else "0"
        dotted = digits[0] + ("." + digits[1:] if len(digits) > 1 else "")
        return "-" + dotted if neg else dotted

    val_col = t.columns[1]
    t[val_col] = (
        t[val_col].map(normalize_to_digits).replace({"": "0", "-": "-0"}).astype(float) / 100.0
    )

    # Helpers
    def extract_token_after(desc: str, marker: str) -> str:
        idx = desc.find(marker)
        if idx == -1:
            return ""
        after = desc[idx + len(marker) :].strip()
        token = after.split()[0] if after else ""
        token = re.sub(r"[^\w\.-]+$", "", token)
        return token

    def between(desc: str, start_marker: str, end_marker: str) -> str:
        s_idx = desc.find(start_marker)
        if s_idx == -1:
            return ""
        s_idx += len(start_marker)
        sub = desc[s_idx:]
        e_idx = sub.find(end_marker)
        if e_idx == -1:
            return sub.strip()
        return sub[:e_idx].strip()

    def earliest_token_after_markers(desc: str, markers: List[str]) -> str:
        positions = [(desc.find(m), m) for m in markers if desc.find(m) != -1]
        if not positions:
            return ""
        _, marker = min(positions, key=lambda x: x[0])
        return extract_token_after(desc, marker)

    # 4) Create ativo and ticker
    def classify_and_extract(desc: str) -> tuple[str, str]:
        s = str(desc).strip()
        if not s:
            return "", ""

        if s.startswith("Ações e outros TVM cedidos em empréstimo"):
            return "Ações Emprestadas", earliest_token_after_markers(
                s, ["Descrição:", "Cod. Ativo:"]
            )

        if s.startswith("Ações"):
            return "Ações", earliest_token_after_markers(s, ["Descrição:", "Cod. Ativo:"])

        if s.startswith("Cotas de Fundos "):
            prefix = "Cotas de Fundos "
            remainder = s[len(prefix) :].strip()
            return "Cotas de Fundos", remainder

        if s.startswith("Debêntures"):
            return "Debêntures", extract_token_after(s, "Cod. Ativo:")

        if s.startswith("Depósitos a prazo e outros títulos de IF"):
            ativo_val = between(s, "Tipo de Ativo:", "CNPJ")
            ticker_val = between(s, "Denominação Social do emissor:", "Venc.:")
            return ativo_val, ticker_val

        if s.startswith("Títulos Públicos"):
            return "Títulos Públicos", extract_token_after(s, "Cod. SELIC:")

        return "", ""

    pairs = t["descricao"].apply(classify_and_extract)
    t["ativo"] = pairs.map(lambda x: x[0])
    t["ticker"] = pairs.map(lambda x: x[1])

    # Rename value column to Percentual
    t = t.rename(columns={val_col: "Percentual"})

    # Separate meta columns
    t["nome_fundo"] = nome_fundo or ""
    t["cnpj"] = cnpj_digits or ""
    t["competencia"] = competencia or ""

    # Reorder columns
    t = t[
        ["nome_fundo", "cnpj", "competencia", "descricao", "ativo", "ticker", "Percentual"]
    ]

    return t


# ------------- SQLite helpers (store only transformed rows) -------------


def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fundos_rows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cnpj TEXT NOT NULL,
            competencia TEXT NOT NULL,
            nome_fundo TEXT,
            descricao TEXT,
            ativo TEXT,
            ticker TEXT,
            percentual REAL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        """.strip()
    )
    # An index to quickly test existence of a dataset (cnpj, competencia)
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ix_rows_unique_dataset_once
        ON fundos_rows (cnpj, competencia, id);
        """.strip()
    )
    # A separate existence index for (cnpj, competencia) to speed up checks
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_rows_dataset
        ON fundos_rows (cnpj, competencia);
        """.strip()
    )
    conn.commit()
    return conn


def dataset_exists_rows(conn: sqlite3.Connection, cnpj: str, competencia: str) -> bool:
    cur = conn.execute(
        """
        SELECT 1
        FROM fundos_rows
        WHERE cnpj = ? AND competencia = ?
        LIMIT 1
        """.strip(),
        (cnpj, competencia),
    )
    return cur.fetchone() is not None


def insert_transformed_rows(
    conn: sqlite3.Connection, df_transformed: pd.DataFrame
) -> int:
    # df_transformed columns:
    # nome_fundo, cnpj, competencia, descricao, ativo, ticker, Percentual
    rows = df_transformed.copy()
    rows = rows.fillna({"descricao": "", "ativo": "", "ticker": ""})
    rows["Percentual"] = pd.to_numeric(rows["Percentual"], errors="coerce")

    to_insert = [
        (
            str(r["cnpj"]),
            str(r["competencia"]),
            str(r["nome_fundo"]),
            str(r["descricao"]),
            str(r["ativo"]),
            str(r["ticker"]),
            None if pd.isna(r["Percentual"]) else float(r["Percentual"]),
        )
        for _, r in rows.iterrows()
    ]
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO fundos_rows
            (cnpj, competencia, nome_fundo, descricao, ativo, ticker, percentual)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """.strip(),
        to_insert,
    )
    conn.commit()
    return cur.rowcount


# ------------------------ Main scraping flow ------------------------


def run_sequence(cnpj: str, headless: bool, placeholders: dict) -> Optional[dict]:
    img_ph = placeholders["img"]
    log_ph = placeholders["log"]
    prog_ph = placeholders["prog"]

    def log(msg: str) -> None:
        log_ph.write(msg)

    driver = None
    try:
        prog_ph.progress(0)
        driver = create_driver(headless=headless)

        log("Step 1/13: Opening CVM page…")
        driver.get(CVM_URL)
        wait_ready(driver, timeout=45)
        img_ph.image(
            highlight_and_capture(driver),
            caption="Loaded CVM page (top-level)",
            use_container_width=True,
        )
        prog_ph.progress(15)

        log("Step 2/13: Switching to CNPJ input frame…")
        switch_to_frame_containing_xpath(driver, XPATH_CNPJ_INPUT, 20.0)
        cnpj_input = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.XPATH, XPATH_CNPJ_INPUT))
        )
        img_ph.image(
            highlight_and_capture(driver, cnpj_input),
            caption="Located CNPJ input (inside frame)",
            use_container_width=True,
        )
        prog_ph.progress(25)

        log("Step 3/13: Typing CNPJ…")
        cnpj_input.clear()
        cnpj_input.send_keys(cnpj)
        img_ph.image(
            highlight_and_capture(driver, cnpj_input),
            caption="CNPJ filled",
            use_container_width=True,
        )
        prog_ph.progress(35)

        log("Step 4/13: Locating dropdown…")
        try:
            tipo_select_el = WebDriverWait(driver, 8).until(
                EC.element_to_be_clickable((By.XPATH, XPATH_TIPO_SELECT))
            )
        except Exception:
            switch_to_frame_containing_xpath(driver, XPATH_TIPO_SELECT, 10.0)
            tipo_select_el = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, XPATH_TIPO_SELECT))
            )
        img_ph.image(
            highlight_and_capture(driver, tipo_select_el),
            caption="Dropdown located",
            use_container_width=True,
        )
        prog_ph.progress(45)

        log("Step 5/13: Selecting 'FDOS DE INVESTIMENTO'…")
        select_fd_investimento(Select(tipo_select_el))
        img_ph.image(
            highlight_and_capture(driver, tipo_select_el),
            caption="Dropdown after selecting FDOS DE INVESTIMENTO",
            use_container_width=True,
        )
        prog_ph.progress(55)

        log("Step 6/13: Clicking the search button…")
        try:
            search_btn = WebDriverWait(driver, 8).until(
                EC.element_to_be_clickable((By.XPATH, XPATH_SEARCH_BTN))
            )
        except Exception:
            switch_to_frame_containing_xpath(driver, XPATH_SEARCH_BTN, 10.0)
            search_btn = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, XPATH_SEARCH_BTN))
            )
        img_ph.image(
            highlight_and_capture(driver, search_btn),
            caption="Search button (pre-click)",
            use_container_width=True,
        )
        search_btn.click()
        prog_ph.progress(70)

        log("Step 7/13: Waiting for results…")
        time.sleep(2.0)
        try:
            wait_ready(driver, timeout=30)
        except Exception:
            pass
        img_ph.image(
            highlight_and_capture(driver),
            caption="Post-search view",
            use_container_width=True,
        )
        prog_ph.progress(78)

        log("Step 8/13: Clicking first link …")

        def pre1(el):
            img_ph.image(
                highlight_and_capture(driver, el),
                caption="First link (pre-click)",
                use_container_width=True,
            )

        click_xpath_any_frame(driver, XPATH_LINK_1, pre_click_screenshot_cb=pre1)
        try:
            wait_ready(driver, timeout=30)
        except Exception:
            pass
        img_ph.image(
            highlight_and_capture(driver),
            caption="After first link",
            use_container_width=True,
        )
        prog_ph.progress(84)

        log("Step 9/13: Clicking second link …")

        def pre2(el):
            img_ph.image(
                highlight_and_capture(driver, el),
                caption="Second link (pre-click)",
                use_container_width=True,
            )

        click_xpath_any_frame(driver, XPATH_LINK_2, pre_click_screenshot_cb=pre2)
        try:
            wait_ready(driver, timeout=30)
        except Exception:
            pass
        img_ph.image(
            highlight_and_capture(driver),
            caption="After second link",
            use_container_width=True,
        )
        prog_ph.progress(88)

        log("Step 10/13: Extracting target table …")
        df = extract_table_df(driver, XPATH_TARGET_TABLE)
        img_ph.image(
            highlight_and_capture(driver),
            caption="Table view (page context)",
            use_container_width=True,
        )
        prog_ph.progress(92)

        log("Step 11/13: Getting 'Competência' …")
        competencia = get_select_value_any_frame(driver, XPATH_COMP_SELECT, 25.0)
        prog_ph.progress(94)

        log("Step 12/13: Getting 'Nome do Fundo' …")
        nome_fundo = get_text_any_frame(driver, XPATH_FUNDO_NAME, 25.0)
        prog_ph.progress(96)

        # Transform per your rules and show only the modified dataset
        log("Transforming table and deriving ativo/ticker …")
        df_transformed = transform_table(df, nome_fundo, competencia, cnpj)

        st.subheader("Tabela transformada")
        st.caption(
            f"Competência: {competencia or '—'} | Nome do Fundo: {nome_fundo or '—'} | CNPJ: {cnpj}"
        )
        st.dataframe(df_transformed, use_container_width=True)

        safe_comp = re.sub(r"[^0-9A-Za-z._-]+", "_", competencia or "competencia")
        safe_nome = re.sub(r"[^0-9A-Za-z._-]+", "_", nome_fundo or "fundo")
        safe_cnpj = re.sub(r"[^0-9A-Za-z._-]+", "_", cnpj or "cnpj")

        tr_csv = df_transformed.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar CSV (transformado)",
            data=tr_csv,
            file_name=f"cvm_tabela_transformado_{safe_comp}_{safe_nome}_{safe_cnpj}.csv",
            mime="text/csv",
        )
        # Gerar CSV em CP1252 (Windows-1252), comum no Brasil, e separador ';'
        tr_csv = df_transformed.to_csv(index=False, sep=";", encoding="cp1252").encode(
            "cp1252"
        )

        st.download_button(
            "Baixar CSV (transformado - CP1252)",
            data=tr_csv,
            file_name=f"cvm_tabela_transformado_{safe_comp}_{safe_nome}_{safe_cnpj}.csv",
            mime="text/csv; charset=windows-1252",
        )

        prog_ph.progress(100)
        log("Done.")

        # Keep in session to manage DB step outside driver lifecycle
        result = {
            "df": df_transformed,
            "nome_fundo": nome_fundo,
            "competencia": competencia,
            "cnpj": cnpj,
        }
        st.session_state["pending_dataset"] = result
        return result

    except Exception as e:
        try:
            img = highlight_and_capture(driver) if driver else None
            if img:
                img_ph.image(
                    img,
                    caption="Error state (last screenshot)",
                    use_container_width=True,
                )
        except Exception:
            pass
        st.error(f"Failed: {type(e).__name__}: {e}")
        return None
    finally:
        try:
            if driver:
                driver.quit()
        except Exception:
            pass


def main():
    st.set_page_config(page_title="CVM Fundos (Selenium + Chromium)")
    st.title("CVM Fundos — Selenium via Chromium")
    st.caption(
        "Inputs a CNPJ, navigates CVM, and outputs the transformed table with "
        "separate meta columns (nome_fundo, cnpj, competencia) and ativo/ticker rules. "
        "Optionally stores the transformed rows into a local SQLite database."
    )

    with st.sidebar:
        headless = st.checkbox(
            "Headless mode (recommended on WSL without GUI)",
            value=True,
        )
        st.markdown("On Windows 11 with WSLg, uncheck to see the real browser window.")

    cnpj_input = st.text_input(
        "CNPJ",
        placeholder="00.000.000/0000-00 or only digits",
        help="We'll sanitize to digits (14 expected).",
    )

    run_btn = st.button("Run search", type="primary")

    img_ph = st.empty()
    log_ph = st.empty()
    prog_ph = st.progress(0)

    if run_btn:
        cnpj_digits = clean_cnpj(cnpj_input)
        if len(cnpj_digits) != 14:
            st.warning(f"CNPJ should have 14 digits after cleaning. Got {len(cnpj_digits)}.")
        placeholders = {"img": img_ph, "log": log_ph, "prog": prog_ph}
        result = run_sequence(cnpj_digits, headless=headless, placeholders=placeholders)
        if result:
            st.session_state["pending_dataset"] = result

    # SQLite section: check and optionally insert the pulled (transformed) dataset
    pending = st.session_state.get("pending_dataset")
    if pending:
        st.subheader("Banco de dados (SQLite)")
        st.caption(f"Arquivo do banco: {DB_PATH}")

        try:
            conn = init_db(DB_PATH)
            exists = dataset_exists_rows(conn, pending["cnpj"], pending["competencia"])
            if exists:
                st.success(
                    "Já armazenado no banco ao menos uma linha para este conjunto: "
                    f"CNPJ {pending['cnpj']} | Competência {pending['competencia']}. "
                    "Os dados existentes não serão sobrescritos."
                )
            else:
                st.warning(
                    "Este fundo/competência ainda não está no banco: "
                    f"CNPJ {pending['cnpj']} | Competência {pending['competencia']}."
                )
                if st.button("Adicionar esta competência ao banco", key="add_db_button"):
                    inserted = insert_transformed_rows(conn, pending["df"])
                    st.success(
                        f"Competência adicionada ao banco com sucesso. "
                        f"Linhas inseridas: {inserted}."
                    )
        except Exception as e:
            st.error(f"Erro no banco de dados: {type(e).__name__}: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()