from fastapi import FastAPI

from app.invoice_service import calculate_invoice
from app.schemas import InvoiceRequest, InvoiceResponse

app = FastAPI(title="Invoice API")


@app.post("/invoices/calculate", response_model=InvoiceResponse)
def calculate_invoice_endpoint(invoice: InvoiceRequest) -> InvoiceResponse:
    result = calculate_invoice(
        subtotal=invoice.subtotal,
        discount_percent=invoice.discount_percent,
    )

    return InvoiceResponse(**result)
