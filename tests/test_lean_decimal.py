"""Exact decimal construction tests independent of optional CLR installation."""

from decimal import Decimal as D
from decimal import localcontext

import pytest

from src.execution.lean_decimal import decimal_bits
from src.execution.ledger import LedgerError


@pytest.mark.parametrize(
    "text",
    [
        "0",
        "-0",
        "1",
        "-1.25",
        "4294967295",
        "18446744073709551615",
        "79228162514264337593543950335",
        "100000.000000000000000000001",
        "0.0000000000000000000000000001",
        "1.00000000000000000000000000000",
        "1E+28",
    ],
)
def test_exact_96_bit_roundtrip(text):
    original = D(text)
    low, mid, high, sign, scale = decimal_bits(original)
    assert all(-(2**31) <= word < 2**31 for word in (low, mid, high))
    coefficient = sum(
        (word & 0xFFFFFFFF) << shift for word, shift in zip((low, mid, high), (0, 32, 64))
    )
    with localcontext() as ctx:
        ctx.prec = 100
        restored = D(coefficient).scaleb(-scale) * (-1 if sign else 1)
    assert restored == original


@pytest.mark.parametrize(
    "text",
    [
        "79228162514264337593543950336",
        "1E+29",
        "1E+1000000",
        "1E-29",
        "NaN",
        "Infinity",
        "-Infinity",
    ],
)
def test_unrepresentable_value_never_rounds(text):
    with pytest.raises(LedgerError):
        decimal_bits(D(text))


def test_original_native_ticks_round_forward_without_float():
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from src.execution.lean_runtime import LeanAlgorithmPort

    port = object.__new__(LeanAlgorithmPort)
    port._clock_reader = SimpleNamespace(read=lambda *args: 638712864000000001)
    assert port._clock() == datetime(2025, 1, 1, 0, 0, 0, 1, tzinfo=timezone.utc)
    port._clock_reader = SimpleNamespace(read=lambda *args: 638712863999999999)
    assert port._clock() == datetime(2025, 1, 1, tzinfo=timezone.utc)


def test_direct_float_monetary_transport_is_still_rejected():
    from src.execution.lean_runtime import LeanAlgorithmPort

    port = object.__new__(LeanAlgorithmPort)
    with pytest.raises(LedgerError, match="actual decimal"):
        port._number(100000.0)
