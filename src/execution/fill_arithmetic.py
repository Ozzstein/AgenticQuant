"""Preserve historical arithmetic and exact native settlement amounts."""

from src.domain.portfolio import exact_decimal_sum


class FillArithmetic:
    """Native sourced cash/asset debits are exact finite Decimal additions.

    Existing fills without the extension retain their original operation order
    and Decimal rounding, so archived historical economics do not change.
    """

    def __init__(self, fill):
        self.exact = fill.settlement is not None or bool(fill.inventory_fees)

    def add(self, left, right):
        return exact_decimal_sum([left, right]) if self.exact else left + right

    def subtract(self, left, right):
        return exact_decimal_sum([left, right.copy_negate()]) if self.exact else left - right

    def negate(self, value):
        return value.copy_negate() if self.exact else -value
