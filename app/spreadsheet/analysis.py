"""Read tables and perform supported calculations with Pandas."""
from io import BytesIO
import math
from zipfile import ZipFile, BadZipFile
import pandas as pd

MAX_ROWS = 50000
OPERATIONS = ("sum", "mean", "min", "max", "count")

def read_table(data: bytes, filename: str, sheet: str | int = 0, header_row: int = 1) -> pd.DataFrame:
    if not data or len(data) > 10 * 1024 * 1024:
        raise ValueError("Upload a non-empty file up to 10 MB.")
    if not isinstance(header_row, int) or not 1 <= header_row <= 100:
        raise ValueError("Header row must be between 1 and 100.")
    try:
        if filename.lower().endswith(".csv"):
            frame = pd.read_csv(BytesIO(data), nrows=MAX_ROWS + 1, header=header_row - 1)
        elif filename.lower().endswith(".xlsx"):
            with ZipFile(BytesIO(data)) as archive:
                if sum(item.file_size for item in archive.infolist()) > 50 * 1024 * 1024:
                    raise ValueError("The expanded workbook is too large.")
            frame = pd.read_excel(BytesIO(data), sheet_name=sheet, nrows=MAX_ROWS + 1, header=header_row - 1)
        else:
            raise ValueError("Use .csv or .xlsx files.")
    except (BadZipFile, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, KeyError) as error:
        raise ValueError("Cannot read this table. Use a valid UTF-8 CSV or XLSX file.") from error
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if frame.empty or len(frame) > MAX_ROWS or len(frame.columns) > 100:
        raise ValueError("Use a table with 1–50,000 rows and at most 100 columns.")
    frame.columns = [str(column) for column in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("Column names must be unique.")
    return frame

def calculate(frame: pd.DataFrame, plan: dict) -> pd.DataFrame:
    """Calculate across the loaded table using a supported operation."""
    if not isinstance(plan, dict):
        raise ValueError("Invalid calculation plan.")
    columns = plan.get("columns")
    if columns is not None:
        if not isinstance(columns, list) or not columns or any(not isinstance(c, str) for c in columns):
            raise ValueError("Select one or more existing column names.")
        if len(columns) != len(set(columns)) or len(columns) > 100:
            raise ValueError("Select unique column names.")
        if plan.get("group_by"):
            raise ValueError("Multiple-column calculations do not support grouping yet.")
        return pd.concat([
            calculate(frame, {"operation": plan.get("operation"), "column": c})
            for c in columns
        ], ignore_index=True)
    operation = plan.get("operation")
    column = plan.get("column")
    group_by = plan.get("group_by") or None
    if operation not in OPERATIONS:
        raise ValueError("Supported operations: sum, mean, min, max and count.")
    if not isinstance(column, str) or (group_by is not None and not isinstance(group_by, str)):
        raise ValueError("Column names must be text.")
    if column not in frame.columns or (group_by and group_by not in frame.columns):
        raise ValueError("Select existing column names.")
    values = frame[column]
    if operation != "count":
        numeric = pd.to_numeric(values, errors="coerce")
        if (values.notna() & numeric.isna()).any() or numeric.notna().sum() == 0:
            raise ValueError("This column contains non-numeric data or has no numeric values.")
        if not numeric.dropna().map(math.isfinite).all():
            raise ValueError("This column contains infinite values.")
        values = numeric
    if group_by:
        if frame[group_by].nunique(dropna=False) > 200:
            raise ValueError("Choose a grouping column with at most 200 groups.")
        result = values.groupby(frame[group_by], dropna=False).agg(operation).reset_index(name="result")
    else:
        result = pd.DataFrame({"column": [column], "operation": [operation], "result": [values.agg(operation)]})
    return result


def detect_header_row(data: bytes, filename: str, sheet: str | int = 0) -> int:
    """Look for column names followed by data, rather than a report title."""
    if len(data) > 10 * 1024 * 1024:
        raise ValueError("File exceeds 10 MB.")
    if filename.lower().endswith(".xlsx"):
        with ZipFile(BytesIO(data)) as archive:
            if sum(item.file_size for item in archive.infolist()) > 50 * 1024 * 1024:
                raise ValueError("The expanded workbook is too large.")
        sample = pd.read_excel(BytesIO(data), sheet_name=sheet, header=None, nrows=100)
    else:
        # Start with the first CSV row unless a different header was selected.
        return 1
    best_row, best_score = 1, 0
    for index in range(max(0, len(sample) - 1)):
        cells = sample.iloc[index].dropna()
        texts = [value for value in cells if isinstance(value, str) and value.strip()]
        if len(texts) < 2 or len(texts) != len(cells):
            continue
        following = sample.iloc[index + 1].dropna()
        if len(following) < max(2, len(cells) // 2):
            continue
        score = len(texts)
        if score > best_score:
            best_row, best_score = index + 1, score
    return best_row
