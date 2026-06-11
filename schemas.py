from pydantic import BaseModel, Field


class InvoiceRequest(BaseModel):
    subtotal: float = Field(gt=0)
    discount_percent: float = Field(default=0, ge=0, le=100)


class InvoiceResponse(BaseModel):
    subtotal: float
    discount_percent: float
    discount_amount: float
    total: float
