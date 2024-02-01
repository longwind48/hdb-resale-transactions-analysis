# test_feat_eng.py

import pandas as pd
import pytest
from src.feat_eng import clean_data, get_lease_remaining_in_years, prepare_features_for_inference

# Sample data for testing
sample_df = pd.DataFrame(
    {
        "_id": {0: 1, 1: 2, 2: 3},
        "month": {0: "2000-01", 1: "2000-01", 2: "2000-01"},
        "town": {0: "ANG MO KIO", 1: "ANG MO KIO", 2: "ANG MO KIO"},
        "flat_type": {0: "3 ROOM", 1: "3 ROOM", 2: "3 ROOM"},
        "block": {0: "170", 1: "174", 2: "216"},
        "street_name": {0: "ANG MO KIO AVE 4", 1: "ANG MO KIO AVE 4", 2: "ANG MO KIO AVE 1"},
        "storey_range": {0: "07 TO 09", 1: "04 TO 06", 2: "07 TO 09"},
        "floor_area_sqm": {0: "69", 1: "61", 2: "73"},
        "flat_model": {0: "Improved", 1: "Improved", 2: "New Generation"},
        "lease_commence_date": {0: "1986", 1: "1986", 2: "1976"},
        "resale_price": {0: "147000", 1: "144000", 2: "159000"},
        "remaining_lease": {0: None, 1: None, 2: None},
    }
)

sample_payload = {
    "town": "SENGKANG",
    "flat_type": "4 ROOM",
    "storey_range": "04 TO 06",
    "floor_area_sqm": 93,
    "flat_model": "Model A",
    "remaining_lease": 95,
}


def test_get_lease_remaining_in_years():
    assert get_lease_remaining_in_years("99 years") == 99.0
    assert get_lease_remaining_in_years("99 years 6 months") == 99.5
    assert get_lease_remaining_in_years(99) == 99.0
    assert get_lease_remaining_in_years("invalid lease info") is None
    assert get_lease_remaining_in_years(None) is None


def test_clean_data():
    # Assuming clean_data is supposed to filter data from 1990 to 2023
    cleaned_data = clean_data(sample_df)
    assert cleaned_data["month"].dt.year.min() >= 1990
    assert cleaned_data["month"].dt.year.max() < 2024


def test_prepare_features_for_inference():
    transformed_df = prepare_features_for_inference(sample_payload)

    # Assert that the dataframe is correctly formatted
    assert isinstance(transformed_df, pd.DataFrame)
    assert transformed_df.shape[0] == 1  # Should have 1 row
    assert "town" in transformed_df.columns
    # Add more assertions as per your requirements


# Add more tests for other functions

if __name__ == "__main__":
    pytest.main()
