import pandas as pd
import pytest

from src.preprocessing import add_features, clean_data


def make_raw():
    """A tiny raw table with known problems: a duplicate, a bad age, messy labels."""
    return pd.DataFrame({
        "Customer_ID": ["C1", "C2", "C3", "C3"],
        "Age": [30, 999, 45, 45],
        "Gender": [" male", "F", "Other", "Other"],
        "City": [" mumbai ", "DELHI", "Pune", "Pune"],
        "Discount_Applied": ["Yes", "no", "Yes", "Yes"],
    })


def test_removes_duplicate_rows():
    assert len(clean_data(make_raw())) == 3


def test_impossible_age_becomes_missing():
    df = clean_data(make_raw())
    assert df["Age"].isna().sum() == 1


def test_gender_labels_are_standardized():
    df = clean_data(make_raw())
    assert set(df["Gender"]) == {"Male", "Female", "Other"}


def test_city_labels_are_standardized():
    df = clean_data(make_raw())
    assert set(df["City"]) == {"Mumbai", "Delhi", "Pune"}


def test_discount_is_encoded_as_numbers():
    df = clean_data(make_raw())
    assert list(df["Discount_Applied"]) == [1, 0, 1]


def test_unknown_discount_value_raises_error():
    raw = make_raw()
    raw.loc[0, "Discount_Applied"] = "Maybe"
    with pytest.raises(ValueError):
        clean_data(raw)


def test_missing_column_raises_error():
    with pytest.raises(ValueError):
        clean_data(pd.DataFrame({"Age": [30]}))


def test_add_features_calculates_correct_values():
    df = pd.DataFrame({"Total_Spend": [1000], "Items_Purchased": [10], "Number_of_Orders": [5]})
    result = add_features(df)
    assert result.loc[0, "Avg_Order_Value"] == 200
    assert result.loc[0, "Items_Per_Order"] == 2


def test_add_features_rejects_zero_orders():
    df = pd.DataFrame({"Total_Spend": [1000], "Items_Purchased": [10], "Number_of_Orders": [0]})
    with pytest.raises(ValueError):
        add_features(df)


def test_add_features_does_not_change_the_input():
    df = pd.DataFrame({"Total_Spend": [1000], "Items_Purchased": [10], "Number_of_Orders": [5]})
    add_features(df)
    assert "Avg_Order_Value" not in df.columns