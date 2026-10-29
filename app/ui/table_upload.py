"""Read a sidebar table upload for analysis through the main chat."""
from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile
import pandas as pd
import streamlit as st
from app.spreadsheet.analysis import detect_header_row, read_table

def render_table_upload(user_id):
    st.subheader("Excel / CSV")
    upload = st.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"], key="chat_table_" + user_id)
    if upload is None:
        return None
    data = upload.getvalue()
    try:
        if len(data) > 10 * 1024 * 1024:
            raise ValueError("Maximum file size: 10 MB.")
        sheet = 0
        if upload.name.lower().endswith(".xlsx"):
            with ZipFile(BytesIO(data)) as archive:
                if sum(item.file_size for item in archive.infolist()) > 50 * 1024 * 1024:
                    raise ValueError("Expanded workbook is too large.")
            with pd.ExcelFile(BytesIO(data)) as workbook:
                sheet = st.selectbox("Worksheet", workbook.sheet_names)
        detected = detect_header_row(data, upload.name, sheet)
        header = int(st.number_input("Header row", 1, 100, value=detected,
                                    key="chat_header_" + sha256(data).hexdigest()[:12] + str(sheet)))
        frame = read_table(data, upload.name, sheet, header)
        st.caption(f"{len(frame):,} rows loaded. Ask calculations in chat.")
        with st.expander("Table preview"):
            st.dataframe(frame.head(5), hide_index=True)
        return frame
    except Exception:
        st.error("Cannot read this table. Check worksheet, header row and file limits.")
        return None
