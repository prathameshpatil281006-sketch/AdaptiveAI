import pytest
from app.use_cases.base import BaseUseCase
from app.use_cases.house_price.use_case import HousePriceUseCase

def test_house_price_use_case_inherits_from_base():
    use_case = HousePriceUseCase()
    assert isinstance(use_case, BaseUseCase)

def test_house_price_use_case_properties():
    use_case = HousePriceUseCase()
    
    assert use_case.name == "house_price"
    assert use_case.display_name == "House Price Prediction"
    assert use_case.problem_type == "regression"
    assert use_case.target_column == "price"

def test_house_price_use_case_feature_names():
    use_case = HousePriceUseCase()
    features = use_case.get_feature_names()

    expected_features = ["area_sqft", "bedrooms", "bathrooms", "parking",
                         "property_age", "floor_number", "total_floors",
                         "location", "furnishing_status", "property_type"]
    assert features == expected_features

def test_house_price_use_case_schema_version():
    use_case = HousePriceUseCase()
    assert use_case.schema_version == 2
    assert use_case.get_numeric_feature_names() == [
        "area_sqft", "bedrooms", "bathrooms", "parking",
        "property_age", "floor_number", "total_floors",
    ]
    assert use_case.get_categorical_feature_names() == [
        "location", "furnishing_status", "property_type",
    ]
    assert use_case.get_monitored_feature_names() == use_case.get_numeric_feature_names()

def test_house_price_use_case_model_candidates():
    use_case = HousePriceUseCase()
    candidates = use_case.get_model_candidates()
    
    assert isinstance(candidates, list)
    assert len(candidates) == 3
    
    for candidate in candidates:
        assert "model_type" in candidate
        assert "hyperparameters" in candidate
        
    model_types = [candidate["model_type"] for candidate in candidates]
    assert "LinearRegression" in model_types
    assert "RandomForestRegressor" in model_types
    assert "GradientBoostingRegressor" in model_types
