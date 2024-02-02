# Resale HDB Analysis

A python-based project that analyse Singapore public housing for my own benefit lol.
Follow the instructions in getting started section to deploy a machine learning model locally via Docker.

## Notebooks
All analysis and experiments are in the notebooks directory.

| Notebook Name | Year Created | Description |
|---------------|--------------|-------------|
| [1_data_preprocessing.ipynb](https://github.com/longwind48/hdb-resale-transactions-analysis/blob/master/notebooks/1_data_preprocessing.ipynb) | 2019 | Preprocessing data purposes. Uses deprecated OneMap API. |
| [2_regression.ipynb](https://github.com/longwind48/hdb-resale-transactions-analysis/blob/master/notebooks/2_regression.ipynb) | 2019 | Build simple model to predict HDB resale flats prices in 2018. |
| [3_multiclass_classification.ipynb](https://github.com/longwind48/hdb-resale-transactions-analysis/blob/master/notebooks/3_multiclass_classification.ipynb) | 2019 | To predict flat type for flats from 2015 onwards. |
| [4_diff_n_diff_plot.ipynb](https://github.com/longwind48/hdb-resale-transactions-analysis/blob/master/notebooks/4_diff_n_diff_plot.ipynb) | 2019 | Plot diff-in-diff model. |
| [5_diff_n_diff_model.ipynb](https://github.com/longwind48/hdb-resale-transactions-analysis/blob/master/notebooks/5_diff_n_diff_model.ipynb) | 2019 | Conduct diff-in-diff analysis to measure downtown stations' impact on property prices. |
| [prime_vs_remote_hdb_2024.ipynb](https://github.com/longwind48/hdb-resale-transactions-analysis/blob/master/notebooks/prime_vs_remote_hdb_2024.ipynb) | 2024 | Data analysis to decide between prime-location HDB or more remote HDB. |
| [train_baseline_model_300124.ipynb](https://github.com/longwind48/hdb-resale-transactions-analysis/blob/master/notebooks/train_baseline_model_300124.ipynb) | 2024 | Create baseline model to predict resale HDB prices. |


## Getting Started
These instructions will get your project up and running on your local machine for development and testing purposes.

### Prerequisites
- Python 3.x
- Poetry
- Docker (for Docker deployment)

### Commands
1. Initialize the Python Environment:
    Use Poetry to install dependencies and activate the virtual environment.

    ```bash
    poetry install
    poetry shell
    ```

2. Download 34 Years of HDB Resale Data:
    This command downloads the HDB resale data and saves it in the specified format and location.

    ```bash
    poetry run python src/download_resale_hdb_dataset.py --output-format parquet --destination data/raw --log-level INFO
    ```

3. Train the baseline model to predict housing prices 
    To train the model using the downloaded dataset, run the following command. This uses the configurations specified in the config/sweep_rf.yaml file.

    ```bash
    poetry run python -m src.train --wand-config-path config/sweep_rf.yaml --log-level INFO
    ```

4. Deployment
    - Local Deployment
      To deploy the application locally for development and testing:

    ```bash
    uvicorn src.api:app --reload
    ```

    - Local Deployment Using Docker
    For deploying with Docker, ensure Docker is installed and running on your system. Then execute the following command:

    ```bash
    docker-compose up -d --build app
    ```

## Contributing
If you would like to contribute to this project, please fork the repository and send a pull request with your proposed changes. Make sure to follow the project's code style and add unit tests for any new or changed functionality.
