import numpy as np
import pandas as pd
import pytest

from src.core.data_pipeline import _compute_factor, instrument_index
from src.core.factor_engine import FactorEngine
from src.data.features.expressions import ExpressionError, evaluate_expression
from src.utils.schemas import FactorDefinition


def panel():
    dates = pd.date_range("2020-01-01", periods=8, tz="UTC")
    index = pd.MultiIndex.from_product([dates, ["id:a", "id:b"]], names=["datetime", "instrument"])
    return pd.DataFrame({"close": np.array([[10 + n, 200 - n * 3] for n in range(8)]).ravel(),
                         "open": 10., "high": 250., "low": 5., "volume": 100.}, index=index)


def test_exact_qlib_expression_ignores_misleading_name():
    frame = panel()
    expr = "Ref($close, 2)/$close-1"
    expected = frame.close.groupby(level="instrument").shift(2) / frame.close - 1
    actual = _compute_factor(frame, FactorDefinition(name="momentum_99", expression=expr))
    pd.testing.assert_series_equal(actual, expected.rename("momentum_99"))
    assert actual.loc[(slice(None), "id:a")].dropna().lt(0).all()


def test_safe_pandas_and_qlib_forms_identical():
    frame = panel()
    left = evaluate_expression("(df['close'] / df['close'].shift(2) - 1) / df['volume'].rolling(3).mean()", frame)
    right = evaluate_expression("($close / Ref($close, 2) - 1) / Mean($volume, 3)", frame)
    pd.testing.assert_series_equal(left, right)


@pytest.mark.parametrize("expression", ["unknown_momentum", "close.shift(-1)", "Ref($close,-1)",
    "close.rolling(3, center=True).mean()", "close.mean()", "close.rank()", "close.__class__",
    "__import__('os').system('true')", "close.apply(lambda x: x)", "np.load('file')", "close ** 99999"])
def test_unsupported_or_future_expressions_fail(expression):
    with pytest.raises(ExpressionError):
        evaluate_expression(expression, panel())


def test_future_perturbation_cannot_change_past_features():
    frame = panel()
    cutoff = frame.index.get_level_values("datetime")[9]
    old = evaluate_expression("Mean($close,3) + $volume.cumsum()", frame)
    altered = frame.copy()
    altered.loc[altered.index.get_level_values("datetime") > cutoff, "close"] = 1e9
    new = evaluate_expression("Mean($close,3) + $volume.cumsum()", altered)
    pd.testing.assert_series_equal(old.loc[:cutoff], new.loc[:cutoff])


def test_duplicate_dates_multi_symbol_preserve_alignment(mock_config):
    frame = panel().reset_index("instrument").rename(columns={"instrument": "ticker"})
    result = FactorEngine(mock_config).compute_factors(frame, names=["obv"])
    assert result.index.equals(frame.index)
    assert result.obv.iloc[2] == 100
    assert result.obv.iloc[3] == -100
    canonical = instrument_index(frame)
    assert canonical.index.is_unique and len(canonical) == len(frame)


def test_unordered_history_fails():
    with pytest.raises(ExpressionError, match="chronological"):
        evaluate_expression("Mean($close,3)", panel().iloc[::-1])


def test_static_validator_shares_exact_engine_without_observations():
    from src.data.features.expressions import validate_expression
    from src.strategies.specification import baseline_specs
    fields = ('open','high','low','close','volume','pe','roic','free_cash_flow','benchmark_close','funding_rate','open_interest','btc_dominance')
    for spec in baseline_specs().values():
        for feature in spec.features:
            assert validate_expression(feature.expression, fields) is None
    for expression in ('Ref(close,-1)', 'close.shift(-2)', 'close.mean()', 'close.rolling(20,center=True).mean()',
                       '__import__("os")', 'unknown_field / close', 'close ** close', 'close.rolling(0).mean()',
                       'close.ewm(alpha=0).mean()', '0/0'):
        with pytest.raises(ExpressionError):
            validate_expression(expression, fields)
