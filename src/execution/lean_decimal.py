"""Exact CLR decimal transport without Python.NET's decimal-to-float decoder."""

from decimal import Decimal

from src.execution.ledger import LedgerError


def decimal_bits(value: Decimal) -> tuple[int, int, int, bool, int]:
    """Return System.Decimal constructor words, rejecting any necessary rounding."""
    if not isinstance(value, Decimal) or not value.is_finite():
        raise LedgerError("LEAN decimal requires a finite exact Decimal")
    sign, digits, exponent = value.as_tuple()
    coefficient = int("".join(map(str, digits)))
    if coefficient == 0:
        return 0, 0, 0, bool(sign), 0
    while exponent < 0 and coefficient % 10 == 0:
        coefficient //= 10
        exponent += 1
    if exponent > 0:
        if len(str(coefficient)) + exponent > 29:
            raise LedgerError("quantity/price is not exactly representable by LEAN System.Decimal")
        coefficient *= 10**exponent
    scale = max(0, -exponent)
    if scale > 28 or coefficient >= 2**96:
        raise LedgerError("quantity/price is not exactly representable by LEAN System.Decimal")
    words = [(coefficient >> shift) & 0xFFFFFFFF for shift in (0, 32, 64)]
    return *(word if word < 2**31 else word - 2**32 for word in words), bool(sign), scale


class NativeDecimalReader:
    """Compile CLR-only property access plus invariant formatting; never read a float."""

    def __init__(self):
        self._readers = {}

    def read(self, source, path):
        import clr
        from System import (
            Array,
            Func,
            IFormatProvider,
            Object,
            String,
            Type,
        )
        from System import (
            Decimal as NativeDecimal,
        )
        from System.Globalization import CultureInfo
        from System.Linq.Expressions import Expression, ParameterExpression

        native_type = source.GetType()
        key = (native_type, path)
        reader = self._readers.get(key)
        if reader is None:
            argument = Expression.Parameter(clr.GetClrType(Object), "source")
            field = Expression.Convert(argument, native_type)
            for name in path.split("."):
                field = Expression.Property(field, name)
            if field.Type != clr.GetClrType(NativeDecimal):
                raise LedgerError("LEAN monetary property is not System.Decimal")
            provider_type = clr.GetClrType(IFormatProvider)
            method = field.Type.GetMethod("ToString", Array[Type]([provider_type]))
            body = Expression.Call(
                field, method, Expression.Constant(CultureInfo.InvariantCulture, provider_type)
            )
            reader = Expression.Lambda[Func[Object, String]](
                body, Array[ParameterExpression]([argument])
            ).Compile()
            self._readers[key] = reader
        return Decimal(reader(source))


class NativeClockReader:
    """Read original CLR ticks before Python.NET truncates to Python microseconds."""

    def __init__(self):
        self._readers = {}

    def read(self, source=None, path=None, zone=None):
        import clr
        from NodaTime import DateTimeZone
        from QuantConnect import Extensions
        from System import Array, Boolean, DateTime, Func, Object, String, Type
        from System.Linq.Expressions import Expression, ParameterExpression

        native_type = source.GetType() if source is not None else clr.GetClrType(Object)
        key = (native_type, path, str(zone.Id) if zone is not None else None)
        reader = self._readers.get(key)
        if reader is None:
            argument = Expression.Parameter(clr.GetClrType(Object), "source")
            if source is None:
                field = Expression.Property(None, clr.GetClrType(DateTime), "UtcNow")
            else:
                field = Expression.Convert(argument, native_type)
                for name in path.split("."):
                    field = Expression.Property(field, name)
            if field.Type != clr.GetClrType(DateTime):
                raise LedgerError("LEAN clock property is not System.DateTime")
            if zone is not None:
                method = clr.GetClrType(Extensions).GetMethod(
                    "ConvertToUtc",
                    Array[Type](
                        [
                            clr.GetClrType(DateTime),
                            clr.GetClrType(DateTimeZone),
                            clr.GetClrType(Boolean),
                        ]
                    ),
                )
                field = Expression.Call(
                    method,
                    field,
                    Expression.Constant(zone, clr.GetClrType(DateTimeZone)),
                    Expression.Constant(False),
                )
            ticks = Expression.Property(field, "Ticks")
            body = Expression.Call(ticks, ticks.Type.GetMethod("ToString", Array[Type]([])))
            reader = Expression.Lambda[Func[Object, String]](
                body, Array[ParameterExpression]([argument])
            ).Compile()
            self._readers[key] = reader
        return int(reader(source))
