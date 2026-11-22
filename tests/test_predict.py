import pytest

from src.predict import predict_segment, validate_customer

GOOD = {
    "Total_Spend": 100000,
    "Items_Purchased": 70,
    "Number_of_Orders": 25,
    "Days_Since_Last_Purchase": 10,
    "Average_Rating": 4.5,
    "Discount_Applied": "No",
}


def test_valid_customer_passes():
    clean = validate_customer(GOOD)
    assert clean["Total_Spend"] == 100000.0
    assert clean["Discount_Applied"] == 0


def test_rating_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="Average_Rating"):
        validate_customer({**GOOD, "Average_Rating": 9})


def test_missing_field_is_rejected():
    bad = {key: value for key, value in GOOD.items() if key != "Total_Spend"}
    with pytest.raises(ValueError, match="Total_Spend"):
        validate_customer(bad)


def test_text_instead_of_number_is_rejected():
    with pytest.raises(ValueError, match="not a number"):
        validate_customer({**GOOD, "Total_Spend": "abc"})


def test_bad_discount_value_is_rejected():
    with pytest.raises(ValueError, match="Discount_Applied"):
        validate_customer({**GOOD, "Discount_Applied": "Maybe"})


def test_items_cannot_be_fewer_than_orders():
    with pytest.raises(ValueError, match="cannot be less"):
        validate_customer({**GOOD, "Items_Purchased": 5, "Number_of_Orders": 25})


def test_all_problems_are_reported_together():
    with pytest.raises(ValueError) as error:
        validate_customer({**GOOD, "Average_Rating": 9, "Total_Spend": -5})
    message = str(error.value)
    assert "Average_Rating" in message
    assert "Total_Spend" in message


def test_prediction_has_expected_fields():
    result = predict_segment(GOOD)
    assert {"cluster", "name", "summary", "strategy"} <= set(result)
    assert result["cluster"] in (0, 1, 2, 3)


def test_high_value_customer_lands_in_cluster_2():
    assert predict_segment(GOOD)["cluster"] == 2