"""Provide the earlier spreadsheet analysis panel."""
from hashlib import sha256
from io import BytesIO
import pandas as pd
import streamlit as st
from app.spreadsheet.analysis import read_table, calculate, OPERATIONS, detect_header_row
from app.spreadsheet.planner import create_excel_model, plan_question, api_error_message, local_tax_plan

def render_excel_panel():
    with st.expander("Excel / CSV Analyst", expanded=False):
        st.caption("Separate provider: OpenRouter free models. File data and calculations stay local; AI receives column names/types and your question.")
        st.caption("Supports sum, average, minimum, maximum and non-empty count, optionally by group. Existing Excel formulas are not recalculated; saved values may be stale or missing.")
        upload = st.file_uploader("Upload a table (up to 10 MB)", type=["csv", "xlsx"], key="excel_upload")
        if upload is None:
            return
        data = upload.getvalue()
        if len(data) > 10 * 1024 * 1024:
            st.error("File is too large. Maximum: 10 MB.")
            return
        sheet = 0
        if upload.name.lower().endswith(".xlsx"):
            try:
                from zipfile import ZipFile
                with ZipFile(BytesIO(data)) as archive:
                    if sum(item.file_size for item in archive.infolist()) > 50 * 1024 * 1024:
                        raise ValueError
                with pd.ExcelFile(BytesIO(data)) as workbook:
                    sheet = st.selectbox("Worksheet", workbook.sheet_names)
            except Exception:
                st.error("Cannot open this workbook. Use a valid, unencrypted XLSX file.")
                return
        try:
            detected_header = detect_header_row(data, upload.name, sheet)
        except Exception:
            st.error("Cannot detect the table header. Check the workbook format.")
            return
        st.caption(f"Detected header row: {detected_header}. Check the preview below.")
        header_row = int(st.number_input("Header row (column names)", min_value=1, max_value=100, value=detected_header,
                                         key="excel_header_auto_v2_" + sha256(data).hexdigest()[:12] + str(sheet)))
        st.caption("If you see Unnamed columns, select the row containing actual column names. For example, use 3 when the first two rows contain a title or blank space.")
        try:
            frame = read_table(data, upload.name, sheet, header_row)
        except ValueError as error:
            st.error(str(error))
            return
        except Exception:
            st.error("Cannot read this table. Use a valid UTF-8 CSV or XLSX with 1–50,000 rows and at most 100 columns.")
            return
        st.caption(f"{len(frame):,} rows · {len(frame.columns)} columns. Preview shows the first 10 rows.")
        st.dataframe(frame.head(10), hide_index=True)
        fingerprint = sha256(data).hexdigest() + str(sheet) + str(header_row)
        if any(str(c).startswith("Unnamed:") for c in frame.columns):
            st.warning("Some column names are missing. Check the header row before asking AI to calculate.")
        with st.form("excel_local_" + fingerprint):
            st.write("Calculate locally — no API needed")
            operation = st.selectbox("Operation", OPERATIONS)
            column = st.selectbox("Value column", frame.columns.tolist(), key="excel_value_" + fingerprint)
            grouping = st.selectbox("Group by (optional)", [None, *frame.columns.tolist()], format_func=lambda x: x or "No grouping", key="excel_group_" + fingerprint)
            local_submit = st.form_submit_button("Calculate locally")
        with st.form("excel_ai_" + fingerprint):
            question = st.text_input("Ask OpenRouter", placeholder="What is the total Sales by Department?", max_chars=1000)
            ai_submit = st.form_submit_button("Analyze with OpenRouter")
        if local_submit or ai_submit:
            try:
                if ai_submit:
                    plan = local_tax_plan(frame, question)
                    model = create_excel_model() if plan is None else None
                    with st.spinner("OpenRouter is selecting a calculation…"):
                        try:
                            if plan is None:
                                plan = plan_question(model, frame, question)
                        except ValueError:
                            raise
                        except Exception as error:
                            raise ValueError(api_error_message(error)) from None
                else:
                    plan = {"operation": operation, "column": column, "group_by": grouping}
                result = calculate(frame, plan)
                st.session_state.excel_result = (fingerprint, plan, result)
            except ValueError as error:
                st.session_state.pop("excel_result", None)
                st.error(str(error))
        saved = st.session_state.get("excel_result")
        if saved and saved[0] == fingerprint:
            st.write("Calculation used:", saved[1])
            st.caption("All loaded rows included. Blank values are skipped; count counts non-empty cells.")
            st.dataframe(saved[2].style.format({"result": "{:,.2f}"}), hide_index=True)
            st.download_button("Download result CSV", saved[2].to_csv(index=False), "calculation_result.csv", "text/csv")
