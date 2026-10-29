import json
import unittest
from unittest.mock import Mock
from langchain_core.messages import AIMessage
from app.spreadsheet.analysis import read_table, calculate
from app.spreadsheet.planner import plan_question

class SpreadsheetTests(unittest.TestCase):
    def setUp(self):
        self.frame = read_table(b"Department,Sales\nA,100\nA,200\nB,50\nB,\n", "sample.csv")

    def test_total_all_rows(self):
        result = calculate(self.frame, {"operation": "sum", "column": "Sales"})
        self.assertEqual(result.iloc[0]["result"], 350)

    def test_groups_and_blanks(self):
        result = calculate(self.frame, {"operation": "sum", "column": "Sales", "group_by": "Department"})
        self.assertEqual(dict(zip(result["Department"], result["result"])), {"A": 300, "B": 50})
        self.assertEqual(calculate(self.frame, {"operation": "count", "column": "Sales"}).iloc[0]["result"], 3)

    def test_invalid_operation_and_column(self):
        for plan in [{"operation": "exec", "column": "Sales"}, {"operation": "sum", "column": "Missing"}]:
            with self.assertRaises(ValueError):
                calculate(self.frame, plan)

    def test_non_numeric_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            calculate(self.frame, {"operation": "sum", "column": "Department"})

    def test_planner_sends_metadata_only(self):
        model = Mock()
        model.invoke.return_value = AIMessage(content='{"operation":"sum","column":"Sales","group_by":null}')
        plan = plan_question(model, self.frame, "Total Sales?")
        self.assertEqual(plan["column"], "Sales")
        payload = json.loads(model.invoke.call_args.args[0][1].content)
        self.assertEqual(set(payload), {"columns", "question"})
        self.assertNotIn("100", json.dumps(payload))

    def test_unsupported_is_rejected(self):
        model = Mock()
        model.invoke.return_value = AIMessage(content='{"error":"unsupported"}')
        with self.assertRaises(ValueError):
            plan_question(model, self.frame, "Sales minus costs?")


    def test_title_row_import(self):
        frame = read_table(b"Tax Report,,\n,,\nItem,CGST,SGST\nA,10,10\nB,20,20\n", "tax.csv", header_row=3)
        self.assertEqual(list(frame.columns), ["Item", "CGST", "SGST"])
        from app.spreadsheet.planner import local_tax_plan
        result = calculate(frame, local_tax_plan(frame, "calculate all tax"))
        self.assertEqual(result["result"].tolist(), [30, 30])

    def test_total_tax_not_double_counted(self):
        from app.spreadsheet.planner import local_tax_plan
        frame = read_table(b"CGST,SGST,Total Tax\n10,10,20\n", "tax.csv")
        plan = local_tax_plan(frame, "calculate all tax")
        self.assertEqual(calculate(frame, plan).iloc[0]["result"], 20)

    def test_tax_rates_not_summed(self):
        from app.spreadsheet.planner import local_tax_plan
        frame = read_table(b"Tax Rate\n18\n", "tax.csv")
        with self.assertRaises(ValueError):
            local_tax_plan(frame, "calculate all tax")


    def test_auto_header_actual_workbook(self):
        from pathlib import Path
        from app.spreadsheet.analysis import detect_header_row
        from app.spreadsheet.planner import local_tax_plan
        path = Path(__file__).resolve().parent / "fixtures" / "Supermarket-Sales-Sample-Data.xlsx"
        if not path.exists():
            self.skipTest("Local sample workbook unavailable")
        data = path.read_bytes()
        header = detect_header_row(data, path.name)
        self.assertEqual(header, 8)
        frame = read_table(data, path.name, header_row=header)
        result = calculate(frame, local_tax_plan(frame, "calculate all tax"))
        self.assertAlmostEqual(float(result.iloc[0]["result"]), 93.984)

if __name__ == "__main__":
    unittest.main()
