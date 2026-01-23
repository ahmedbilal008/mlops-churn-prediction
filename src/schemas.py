from pydantic import BaseModel, Field, field_validator


class CustomerData(BaseModel):
    """Customer data schema with validation rules."""
    
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(gt=0)
    TotalCharges: float = Field(ge=0)
    
    @field_validator('MonthlyCharges')
    @classmethod
    def validate_monthly_charges(cls, v):
        if v <= 0:
            raise ValueError('MonthlyCharges must be positive')
        return v
    
    @field_validator('TotalCharges')
    @classmethod
    def validate_total_charges(cls, v):
        if v < 0:
            raise ValueError('TotalCharges cannot be negative')
        return v


class PredictionRequest(BaseModel):
    """Schema for prediction requests."""
    
    customer_data: dict
    
    
class PredictionResponse(BaseModel):
    """Schema for prediction responses."""
    
    churn_risk: str
    probability: float
