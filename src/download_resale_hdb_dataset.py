"""Python CLI script to download 34 years of resale HDB data from data.gov.sg and save it in
the specified format to the given destination. Resale HDB data is available from 1990 to present (2024).

example command:
poetry run python src/download_resale_hdb_dataset.py --output-format parquet --destination data/raw --log-level INFO

"""
import os
import sys

import pandas as pd
import requests
import typer
from loguru import logger
from pydantic import BaseModel, field_validator

app = typer.Typer()


class ResaleHDBDataRequest(BaseModel):
    """A Pydantic model for validating output format and destination path for downloading Resale HDB data.

    Attributes:
        output_format: The format in which to save the data ('csv' or 'parquet').
        destination: The path to the destination folder where the data will be saved.
    """

    output_format: str
    destination: str

    @field_validator("output_format")
    def validate_output_format(cls, v):
        if v.lower() not in ["csv", "parquet"]:
            raise ValueError("Output format must be 'csv' or 'parquet'")
        return v


def search_for_resale_hdb_datasets():
    """Search for resale HDB datasets from data.gov.sg.

    Returns:
        List[dict]: A list of datasets that contain resale HDB data.
    """
    datasets_to_query = []
    page = 1
    while True:
        response = requests.get(
            f"https://api-production.data.gov.sg/v2/public/api/datasets?page={page}"
        )
        datasets = response.json()["data"]["datasets"]
        if not datasets:  # If datasets is empty, break the loop
            break
        for dataset in datasets:
            if "resale flat prices" in dataset["name"].lower():
                datasets_to_query.append(dataset)
        page += 1  # Increment the page number

    return datasets_to_query


@app.command()
def download_resale_hdb_data(
    output_format: str = typer.Option(default="csv", help="Output format: 'csv' or 'parquet'"),
    destination: str = typer.Option(..., help="Destination folder path"),
    log_level: str = typer.Option(
        "INFO", help="Set the logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')"
    ),
):
    """Download resale HDB data from data.gov.sg and save it in the specified format to the given destination.

    Args:
        output_format (str): The format in which to save the data ('csv' or 'parquet').
        destination (str): The path to the destination folder where the data will be saved.
    """
    # Configure Loguru logger
    logger.remove()  # Remove default logger configuration
    logger.add(
        sys.stderr, level=log_level.upper()
    )  # Add new configuration with specified log level

    # Validate input arguments
    request = ResaleHDBDataRequest(output_format=output_format, destination=destination)

    # Fetch dataset information
    logger.info("Searching for resale HDB datasets...")
    datasets_to_query = search_for_resale_hdb_datasets()
    if not datasets_to_query:
        logger.error("No datasets found.")
        raise typer.Exit()

    # Download and compile data
    df_list = []
    for dataset in datasets_to_query:
        dataset_id = dataset["datasetId"]
        logger.info(f"Fetching data for dataset ID: {dataset_id}")
        df = query_dataset(dataset_id)
        df_list.append(df)

    if not df_list:
        logger.error("No data fetched.")
        raise typer.Exit()

    df = pd.concat(df_list, ignore_index=True)

    # Save the compiled data
    if not os.path.exists(destination):
        os.makedirs(destination)

    output_path = os.path.join(destination, f"resale_hdb_data.{request.output_format}")
    if request.output_format.lower() == "csv":
        df.to_csv(output_path, index=False)
    elif request.output_format.lower() == "parquet":
        df.to_parquet(output_path, index=False)
    logger.info(f"Data saved to {output_path}")


def query_dataset(dataset_id: str):
    """Query a specific dataset from data.gov.sg using its dataset ID.

    Args:
        dataset_id (str): The unique identifier of the dataset to query.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the queried dataset.
    """
    offset = 0
    df_list = []  # List to store chunks of dataframes
    total = None
    while total is None or offset < total:
        response = requests.get(
            f"https://data.gov.sg/api/action/datastore_search?resource_id={dataset_id}",
            params={"limit": 10000, "offset": offset},
        )
        if response.json()["success"] == True:
            logger.debug(
                f"Retrieved {offset} rows of {total} rows for dataset_id: {dataset_id[:4]}..."
            )

            result = response.json()["result"]
            records = result["records"]
            df_list.append(pd.DataFrame(records))  # Convert records to DataFrame and append to list
            offset += 10000

            if total is None:
                total = result["total"]  # Get the total number of records
        else:
            break
    return pd.concat(df_list, ignore_index=True)  # Concatenate all dataframes in the list


if __name__ == "__main__":
    app()
