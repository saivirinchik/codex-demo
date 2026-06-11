from app.invoice_service import calculate_invoice


def test_invoice_without_discount() -> None:
    result = calculate_invoice(
        subtotal=100.00,
        discount_percent=0,
    )

    assert result["discount_amount"] == 0.00
    assert result["total"] == 100.00
