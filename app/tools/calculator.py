"""Calculate arithmetic expressions with an AST parser instead of eval."""
import ast
import math
import operator
from collections.abc import Callable

from langchain_core.tools import tool

MAX_EXPRESSION_LENGTH = 200
MAX_ABS_VALUE = 1e300  # reject absurdly large results
MAX_RESULT_DIGITS = 300  # for powers: reject if result would exceed 10**300

# Allow only these arithmetic operations.
_BINARY_OPERATORS: dict[type[ast.operator], Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

Number = int | float


class CalculatorError(ValueError):
    """A problem with the expression that is safe to show to the user."""


def evaluate_expression(expression: str) -> Number:
    """Evaluate a math expression such as '(125 * 37) + 450 / 9'."""
    expression = expression.strip()
    if not expression:
        raise CalculatorError("Expression cannot be empty.")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalculatorError(f"Expression is too long (max {MAX_EXPRESSION_LENGTH} characters).")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        raise CalculatorError(
            "Invalid expression. Use numbers, + - * / **, and parentheses."
        ) from None
    try:
        return _evaluate_node(tree.body)
    except RecursionError:
        raise CalculatorError("Expression is too deeply nested.") from None


def _evaluate_node(node: ast.AST) -> Number:
    if isinstance(node, ast.Constant):
        # Accept plain numbers only, excluding booleans, strings and complex values.
        if type(node.value) in (int, float):
            return _check_size(node.value)
        raise CalculatorError("Only numbers are allowed.")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        value = _evaluate_node(node.operand)
        return _check_size(_UNARY_OPERATORS[type(node.op)](value))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        return _apply_binary(type(node.op), left, right)
    raise CalculatorError("Unsupported expression. Use only numbers, + - * / **, and parentheses.")


def _apply_binary(op_type: type[ast.operator], left: Number, right: Number) -> Number:
    if op_type is ast.Pow:
        _check_power(left, right)
    try:
        result = _BINARY_OPERATORS[op_type](left, right)
    except ZeroDivisionError:
        raise CalculatorError("Division by zero is not possible.") from None
    except OverflowError:
        raise CalculatorError("The result is too large.") from None
    if isinstance(result, complex):  # e.g. (-8) ** 0.5
        raise CalculatorError("The result is not a real number.")
    return _check_size(result)


def _check_power(base: Number, exponent: Number) -> None:
    """Stop huge powers (like 9**9**9) before Python spends time computing them."""
    if base != 0 and exponent * math.log10(abs(base)) > MAX_RESULT_DIGITS:
        raise CalculatorError("The result is too large.")


def _check_size(value: Number) -> Number:
    if isinstance(value, float) and not math.isfinite(value):
        raise CalculatorError("The result is too large.")
    if abs(value) > MAX_ABS_VALUE:
        raise CalculatorError("The result is too large.")
    return value


def format_number(value: Number) -> str:
    """4675.0 -> '4675'; 0.1 + 0.2 -> '0.3' (hides float rounding noise)."""
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the exact numeric result.

    Use this tool for ANY calculation instead of computing in your head.
    The input `expression` is a plain math string, for example "(125 * 37) + 450 / 9".
    Supported: + - * / ** (power), parentheses and decimals such as 2.5.
    Write powers as ** (not ^). Do not include words, units or variables.
    """
    try:
        return format_number(evaluate_expression(expression))
    except CalculatorError as error:
        return f"Calculator error: {error}"
