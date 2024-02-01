import os
from typing import Optional

import pandas as pd
from joblib import dump, load
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

LABEL_ENCODERS_PATH = "models/label_encoders"


def get_lease_remaining_in_years(lease_info: str or int) -> float:
    """Convert remaining lease information to a number of years.

    This function takes a lease duration expressed either as a string in the format
    "X years Y months" or as an integer representing the number of years, and converts
    it into a float representing the total number of years. If the input is a string
    but does not contain valid numbers, it returns None. If the input is neither a
    string nor an integer, it also returns None.

    Args:
        lease_info (str or int): The lease duration, either as a string with years and months
                                 or as an integer representing years.

    Returns:
        float or None: The total number of years of the lease, or None if input is invalid.

    """
    if isinstance(lease_info, str):
        try:
            yearmonth = [int(s) for s in lease_info.split() if s.isdigit()]
            if len(yearmonth) == 2:  # Format: "X years Y months"
                return yearmonth[0] + (yearmonth[1] / 12)
            elif len(yearmonth) == 1:  # Format: "X years"
                return float(yearmonth[0])
            else:
                return None
        except ValueError:
            return None
    elif isinstance(lease_info, int):
        return float(lease_info)
    else:
        return None


def clean_data(df: pd.DataFrame, df_cpi: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Clean the resale hdb data, return only 1990 to 2023 data.

    Args:
        df: resale hdb data

    Returns:
        df: cleaned resale hdb data
    """
    logger.info("Cleaning data...")
    # Convert columns to correct type
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
    df["resale_price"] = df["resale_price"].astype(float)
    df["floor_area_sqm"] = df["floor_area_sqm"].astype(float)

    # Clean flat type
    df["flat_type"] = df["flat_type"].str.replace("MULTI-GENERATION", "MULTI GENERATION")
    # Rename flat model duplicates
    replace_values = {
        "NEW GENERATION": "New Generation",
        "SIMPLIFIED": "Simplified",
        "STANDARD": "Standard",
        "MODEL A-MAISONETTE": "Maisonette",
        "MULTI GENERATION": "Multi Generation",
        "IMPROVED-MAISONETTE": "Executive Maisonette",
        "Improved-Maisonette": "Executive Maisonette",
        "Premium Maisonette": "Executive Maisonette",
        "2-ROOM": "2-room",
        "MODEL A": "Model A",
        "MAISONETTE": "Maisonette",
        "Model A-Maisonette": "Maisonette",
        "IMPROVED": "Improved",
        "TERRACE": "Terrace",
        "PREMIUM APARTMENT": "Premium Apartment",
        "Premium Apartment Loft": "Premium Apartment",
        "APARTMENT": "Apartment",
        "Type S1": "Type S1S2",
        "Type S2": "Type S1S2",
    }
    df = df.replace({"flat_model": replace_values})

    # only include 1990 to 2023 data
    df = df[df["month"] < "2024-01-01"]

    return df


def clean_label(df: pd.DataFrame, df_cpi: pd.DataFrame) -> pd.DataFrame:
    """Adjust label for inflation, return only 1990 to 2023 data."""
    # Adjust resale price for inflation
    df_cpi["month"] = df_cpi["month"].apply(lambda x: x.strip())
    df_cpi["month"] = pd.to_datetime(df_cpi["month"], format="%Y %b")
    df = df.merge(df_cpi, on="month", how="left")
    df["cpi"] = df["cpi"].astype(float)
    df["real_price"] = (df["resale_price"] / df["cpi"]) * 100
    return df


def get_lease_remaining_in_years(lease_info: str or int) -> float:
    """Convert remaining lease information to a number of years.

    This function takes a lease duration expressed either as a string in the format
    "X years Y months" or as an integer representing the number of years, and converts
    it into a float representing the total number of years. If the input is a string
    but does not contain valid numbers, it returns None. If the input is neither a
    string nor an integer, it also returns None.

    Args:
        lease_info (str or int): The lease duration, either as a string with years and months
                                 or as an integer representing years.

    Returns:
        float or None: The total number of years of the lease, or None if input is invalid.

    """
    if isinstance(lease_info, str):
        try:
            yearmonth = [int(s) for s in lease_info.split() if s.isdigit()]
            if len(yearmonth) == 2:  # Format: "X years Y months"
                return yearmonth[0] + (yearmonth[1] / 12)
            elif len(yearmonth) == 1:  # Format: "X years"
                return float(yearmonth[0])
            else:
                return None
        except ValueError:
            return None
    elif isinstance(lease_info, int):
        return float(lease_info)
    else:
        return None


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding features...")
    df["year"] = pd.DatetimeIndex(df["month"]).year  # extract out year

    # reduce number of class of town to regions
    d_region = {
        "ANG MO KIO": "North East",
        "BEDOK": "East",
        "BISHAN": "Central",
        "BUKIT BATOK": "West",
        "BUKIT MERAH": "Central",
        "BUKIT PANJANG": "West",
        "BUKIT TIMAH": "Central",
        "CENTRAL AREA": "Central",
        "CHOA CHU KANG": "West",
        "CLEMENTI": "West",
        "GEYLANG": "Central",
        "HOUGANG": "North East",
        "JURONG EAST": "West",
        "JURONG WEST": "West",
        "KALLANG/WHAMPOA": "Central",
        "MARINE PARADE": "Central",
        "PASIR RIS": "East",
        "PUNGGOL": "North East",
        "QUEENSTOWN": "Central",
        "SEMBAWANG": "North",
        "SENGKANG": "North East",
        "SERANGOON": "North East",
        "TAMPINES": "East",
        "TOA PAYOH": "Central",
        "WOODLANDS": "North",
        "YISHUN": "North",
    }
    df["region"] = df["town"].map(d_region)

    df["remaining_lease"] = df["remaining_lease"].apply(lambda x: get_lease_remaining_in_years(x))

    # Select relevant columns
    df = df[
        [
            "town",
            "flat_type",
            "storey_range",
            "floor_area_sqm",
            "flat_model",
            "remaining_lease",
            "year",
            "real_price",
        ]
    ]
    # remove flat types with very few cases
    df = df[~df["flat_type"].isin(["MULTI GENERATION", "1 ROOM"])]

    # Re-categorize flat model to reduce num classes
    replace_values = {
        "Executive Maisonette": "Maisonette",
        "Terrace": "Special",
        "Adjoined flat": "Special",
        "Type S1S2": "Special",
        "DBSS": "Special",
        "Model A2": "Model A",
        "Premium Apartment": "Apartment",
        "Improved": "Standard",
        "Simplified": "Model A",
        "2-room": "Standard",
    }
    df = df.replace({"flat_model": replace_values})

    df = df.reset_index(drop=True)

    # Train Test Split
    y = df["real_price"]
    X = df.drop(["real_price", "year"], axis=1)

    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=True, random_state=0
    )

    # Identifying non-numeric columns
    non_numeric_columns = X_train.select_dtypes(include=["object"]).columns
    label_encoders = {}

    # Check if label encoder directory is empty
    if not os.listdir(LABEL_ENCODERS_PATH):
        logger.info("Initializing label encoders and tranforming...")
        # Initialize a dictionary to store the label encoders for each column

        # Fit label encoders on the training set and transform both training and test sets
        for col in non_numeric_columns:
            label_encoders[col] = LabelEncoder()
            X_train[col] = label_encoders[col].fit_transform(X_train[col])
            X_test[col] = label_encoders[col].transform(
                X_test[col]
            )  # Transform the test set using the same encoder

        # Assuming label_encoders is the dictionary of your trained label encoders
        for col, encoder in label_encoders.items():
            dump(encoder, f"{LABEL_ENCODERS_PATH}/label_encoder_{col}.joblib")
    else:
        logger.info("Loading label encoders and transforming...")
        for col in non_numeric_columns:
            label_encoders[col] = load(f"{LABEL_ENCODERS_PATH}/label_encoder_{col}.joblib")
            X_train[col] = label_encoders[col].transform(X_train[col])
            X_test[col] = label_encoders[col].transform(X_test[col])

    return X_train, X_test, y_train, y_test


def prepare_features_for_inference(payload):
    df = pd.DataFrame([payload])
    # Re-categorize flat model to reduce num classes
    replace_values = {
        "Executive Maisonette": "Maisonette",
        "Terrace": "Special",
        "Adjoined flat": "Special",
        "Type S1S2": "Special",
        "DBSS": "Special",
        "Model A2": "Model A",
        "Premium Apartment": "Apartment",
        "Improved": "Standard",
        "Simplified": "Model A",
        "2-room": "Standard",
    }
    df = df.replace({"flat_model": replace_values})
    non_numeric_columns = ["town", "flat_type", "storey_range", "flat_model"]
    label_encoders = {}
    for col in non_numeric_columns:
        print(f"Loading label encoder for {col}")
        label_encoders[col] = load(f"{LABEL_ENCODERS_PATH}/label_encoder_{col}.joblib")
        df[col] = label_encoders[col].transform(df[col])

    return df
