"""Native enum names differ from Python display aliases in the actual LEAN image."""

import pytest

from src.execution.lean_runtime import enum_name
from src.execution.ledger import LedgerError


@pytest.mark.parametrize("name", ["Crypto", "Equity", "Filled", "PartiallyFilled", "TrailingStop"])
def test_native_enum_name_preserved(name):
    class NativeEnum:
        def __str__(self):
            return name.upper()

        def ToString(self):
            return name

    assert enum_name(NativeEnum()) == name
    assert enum_name(name) == name


@pytest.mark.parametrize("value", ["", None, 1])
def test_invalid_native_enum_name_rejected(value):
    class Invalid:
        def ToString(self):
            return value

    with pytest.raises(LedgerError):
        enum_name(Invalid())
