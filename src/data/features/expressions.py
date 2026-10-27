"""A bounded AST interpreter, NOT Python eval or a generated-code sandbox.

Supported arithmetic and temporal operators have one implementation shared by
research and inference. Negative shifts, global/time-series rank, centered
windows, arbitrary attributes, comprehensions and unknown expressions fail.
"""
from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


class ExpressionError(ValueError):
    pass


ENGINE_VERSION = "bounded-ast-1"
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
           ast.BitAnd: operator.and_, ast.BitOr: operator.or_}
_COMPARE = {ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Lt: operator.lt,
            ast.LtE: operator.le, ast.Eq: operator.eq, ast.NotEq: operator.ne}


def panel_groups(frame: pd.DataFrame) -> list[np.ndarray]:
    """Positions for independent instrument histories; validate causal ordering."""
    identity = next((name for name in ("instrument_id", "instrument", "ticker") if name in frame.columns), None)
    if identity:
        if frame[identity].isna().any():
            raise ExpressionError("missing instrument identity")
        groups = list(frame.groupby(identity, sort=False).indices.values())
    elif isinstance(frame.index, pd.MultiIndex):
        identity = next((name for name in ("instrument_id", "instrument", "ticker") if name in frame.index.names), None)
        if identity is None:
            raise ExpressionError("MultiIndex must name its instrument level")
        groups = list(frame.groupby(level=identity, sort=False).indices.values())
    else:
        groups = [np.arange(len(frame))]
    for positions in groups:
        index = frame.iloc[positions].index
        if isinstance(index, pd.MultiIndex):
            name = next((n for n in ("datetime", "date", "timestamp") if n in index.names), None)
            if name is None:
                raise ExpressionError("MultiIndex must name its date level")
            index = index.get_level_values(name)
        if not index.is_unique or not index.is_monotonic_increasing:
            raise ExpressionError("each instrument must have unique chronological observations")
    return groups


@dataclass
class _Window:
    series: pd.Series
    kind: str
    options: dict


class _Interpreter:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def window(self, value, *, zero=False) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not float(value).is_integer() or not (0 if zero else 1) <= value <= 4096:
            raise ExpressionError("window must be a bounded nonnegative integer" if zero else "window must be a bounded positive integer")
        return int(value)

    def series(self, value) -> pd.Series:
        if isinstance(value, pd.Series):
            return value
        if isinstance(value, (float, int, bool)):
            return pd.Series(value, index=self.frame.index)
        raise ExpressionError("expected a series or number")

    def visit(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
            if not np.isfinite(node.value) or abs(node.value) > 1e12:
                raise ExpressionError("constant outside bounds")
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in self.frame.columns or node.id.startswith("_"):
                raise ExpressionError(f"unknown input field: {node.id}")
            return pd.to_numeric(self.frame[node.id], errors="raise")
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "df" and isinstance(node.slice, ast.Constant):
            return self.visit(ast.Name(id=str(node.slice.value)))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = self.visit(node.left), self.visit(node.right)
            if isinstance(node.op, ast.Pow) and (not np.isscalar(right) or abs(right) > 8):
                raise ExpressionError("power exponent must be a bounded scalar")
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                return _BINARY[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Invert)):
            value = self.visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else +value if isinstance(node.op, ast.UAdd) else ~value
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _COMPARE:
            return _COMPARE[type(node.ops[0])](self.visit(node.left), self.visit(node.comparators[0]))
        if isinstance(node, ast.Call):
            if any(k.arg is None for k in node.keywords):
                raise ExpressionError("argument expansion is not permitted")
            args = [self.visit(arg) for arg in node.args]
            kwargs = {k.arg: self.visit(k.value) for k in node.keywords}
            if isinstance(node.func, ast.Name):
                return self.function(node.func.id, args, kwargs)
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id in ("np", "numpy"):
                    return self.function(node.func.attr, args, kwargs)
                return self.method(self.visit(node.func.value), node.func.attr, args, kwargs)
        raise ExpressionError(f"unsupported expression syntax: {type(node).__name__}")

    def function(self, name, args, kwargs):
        if kwargs:
            raise ExpressionError("function keyword arguments are not supported")
        if name in ("Abs", "abs", "Log", "log", "sqrt", "Sign", "sign") and len(args) == 1:
            return {"Abs": np.abs, "abs": np.abs, "Log": np.log, "log": np.log,
                    "sqrt": np.sqrt, "Sign": np.sign, "sign": np.sign}[name](args[0])
        if name in ("Greater", "maximum", "Less", "minimum") and len(args) == 2:
            return (np.maximum if name in ("Greater", "maximum") else np.minimum)(*args)
        if name in ("If", "where") and len(args) == 3:
            return pd.Series(np.where(*args), index=self.frame.index)
        if name in ("Ref", "Delta", "Mean", "Sum", "Std", "Var", "Max", "Min", "EMA", "Rank") and len(args) == 2:
            series, n = self.series(args[0]), self.window(args[1], zero=name in ("Ref", "Delta"))
            if name == "Ref":
                return series.shift(n)
            if name == "Delta":
                return series.diff(n)
            if name == "EMA":
                return series.ewm(span=n, adjust=False, min_periods=n).mean()
            window = series.rolling(n, min_periods=n)
            if name == "Rank":
                return window.rank(pct=True)
            return getattr(window, {"Mean": "mean", "Sum": "sum", "Std": "std", "Var": "var", "Max": "max", "Min": "min"}[name])()
        if name in ("Corr", "Cov") and len(args) == 3:
            window = self.series(args[0]).rolling(self.window(args[2]))
            return getattr(window, "corr" if name == "Corr" else "cov")(self.series(args[1]))
        raise ExpressionError(f"unsupported operator or arity: {name}")

    def method(self, value, name, args, kwargs):
        if isinstance(value, _Window):
            allowed = {"mean", "std", "var", "sum", "min", "max", "corr", "cov"} if value.kind == "rolling" else {"mean"}
            if name not in allowed or kwargs or (args and name not in ("corr", "cov")):
                raise ExpressionError("unsupported window aggregation")
            window = getattr(value.series, value.kind)(**value.options)
            return getattr(window, name)(*args)
        if not isinstance(value, pd.Series):
            raise ExpressionError("method target must be a series")
        if name in ("shift", "diff", "pct_change"):
            if len(args) > 1 or set(kwargs) - {"periods"} or (args and kwargs):
                raise ExpressionError("invalid temporal arguments")
            n = self.window(args[0] if args else kwargs.get("periods", 1), zero=True)
            return value.pct_change(n, fill_method=None) if name == "pct_change" else getattr(value, name)(n)
        if name == "rolling":
            if len(args) > 1 or set(kwargs) - {"window", "min_periods"}:
                raise ExpressionError("only trailing rolling windows are permitted")
            n = self.window(args[0] if args else kwargs.get("window"))
            minimum = self.window(kwargs.get("min_periods", n))
            if minimum > n:
                raise ExpressionError("min_periods exceeds window")
            return _Window(value, "rolling", {"window": n, "min_periods": minimum})
        if name == "ewm":
            if args or set(kwargs) - {"span", "alpha", "adjust", "min_periods"} or ("span" in kwargs) == ("alpha" in kwargs):
                raise ExpressionError("ewm requires exactly span or alpha")
            options = dict(kwargs)
            if "span" in options:
                options["span"] = self.window(options["span"])
            if "alpha" in options and not 0 < options["alpha"] <= 1:
                raise ExpressionError("invalid EMA alpha")
            if "min_periods" in options:
                options["min_periods"] = self.window(options["min_periods"], zero=True)
            options.setdefault("adjust", False)
            return _Window(value, "ewm", options)
        if name in ("abs", "cumsum") and not args and not kwargs:
            return getattr(value, name)()
        if name == "clip" and not args and set(kwargs) <= {"lower", "upper"}:
            return value.clip(**kwargs)
        raise ExpressionError(f"unsupported series method: {name}")


def evaluate_expression(expression: str, frame: pd.DataFrame) -> pd.Series:
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 8192:
        raise ExpressionError("empty or oversized factor expression")
    try:
        tree = ast.parse(re.sub(r"\$([a-zA-Z][a-zA-Z0-9_]*)", r"\1", expression), mode="eval")
        if len(list(ast.walk(tree))) > 512:
            raise ExpressionError("expression exceeds operation budget")
        result = pd.Series(np.nan, index=frame.index, dtype=float)
        for positions in panel_groups(frame):
            part = frame.iloc[positions]
            evaluator = _Interpreter(part)
            value = evaluator.series(evaluator.visit(tree.body))
            result.iloc[positions] = np.asarray(value, dtype=float)
        return result.replace([np.inf, -np.inf], np.nan)
    except ExpressionError:
        raise
    except (ValueError, TypeError, SyntaxError, RecursionError, KeyError, OverflowError) as exc:
        raise ExpressionError(f"invalid factor expression: {type(exc).__name__}") from None
