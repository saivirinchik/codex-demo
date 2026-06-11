def calculate_invoice(
    subtotal: float,
    discount_percent: float,
) -> dict[str, float]:
    """Calculate an invoice total after applying a percentage discount."""

    # Intentional bug:
    # For a 10% discount, this multiplies subtotal by 10 instead of 0.10.
    discount_amount = subtotal * discount_percent

    total = subtotal - discount_amount

    return {
        "subtotal": round(subtotal, 2),
        "discount_percent": round(discount_percent, 2),
        "discount_amount": round(discount_amount, 2),
        "total": round(total, 2),
    }
