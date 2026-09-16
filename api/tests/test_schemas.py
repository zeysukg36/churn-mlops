import pytest
from pydantic import ValidationError
from models import ChurnPredictionRequest

def test_valid_payload_accepted(valid_payload):
    request = ChurnPredictionRequest(**valid_payload)
    assert request.gender == "Female"

def test_wrong_case_rejected(valid_payload):
    # "Fiber optic" yerine küçük harfle — bir önceki mesajda uyardığımız senaryo
    bad_payload = {**valid_payload, "InternetService": "fiber optic"}
    with pytest.raises(ValidationError):
        ChurnPredictionRequest(**bad_payload)

def test_negative_tenure_rejected(valid_payload):
    bad_payload = {**valid_payload, "tenure": -5}
    with pytest.raises(ValidationError):
        ChurnPredictionRequest(**bad_payload)
