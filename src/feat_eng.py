import pandas as pd


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


def clean_data(df: pd.DataFrame, df_cpi: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the resale hdb data and merge with cpi data, return only 1990 to 2023 data.

    Args:
        df: resale hdb data
        df_cpi: cpi data

    Returns:
        df: cleaned resale hdb data
    """
    # Convert columns to correct type
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
    df["resale_price"] = df["resale_price"].astype(float)

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

    # Adjust resale price for inflation
    df_cpi["month"] = df_cpi["month"].apply(lambda x: x.strip())
    df_cpi["month"] = pd.to_datetime(df_cpi["month"], format="%Y %b")
    df = df.merge(df_cpi, on="month", how="left")
    df["cpi"] = df["cpi"].astype(float)
    df["real_price"] = (df["resale_price"] / df["cpi"]) * 100

    # only include 1990 to 2023 data
    df = df[df["month"] < "2024-01-01"]

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
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
            "lease_commence_date",
            "year",
            "region",
            "real_price",
        ]
    ]
    # label encode storeys
    df = df.sort_values(by="storey_range")
    df["storey_range"] = df["storey_range"].astype("category").cat.codes  # label encode

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

    # Label encode flat type
    replace_values = {"2 ROOM": 0, "3 ROOM": 1, "4 ROOM": 2, "5 ROOM": 3, "EXECUTIVE": 4}
    df = df.replace({"flat_type": replace_values})

    df = df.reset_index(drop=True)

    ## dummy encoding
    df = pd.get_dummies(df, columns=["region"], prefix=["region"], drop_first=True)  # central is baseline
    df = pd.get_dummies(df, columns=["flat_model"], prefix=["model"])
    df = df.drop("model_Standard", axis=1)  # remove standard, setting it as the baseline

    return df
