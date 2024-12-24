"""Preprocessing.

Available Preprocessing Apps
- fleur | https://apps.apple.com/it/app/fleur-gestione-spese-e-budget/id1621020173
"""

# ⚙️ Ruff Settings
# ruff: noqa: PTH118 PTH120 PTH123

# Libraries
import os
from io import StringIO

import pandas as pd

AVAILABLE_PREPROCESSING_APPS = {
    # no operations needed for dummy
    'dummy': {
        'columns_to_drop': ['E'],
    },
    'fleur': {
        'columns_to_drop': ['E'],
    },
    '1money': {
        'delete_rows': {'start': 0, 'end': -5},
        'columns_to_drop': ['TAG', 'VALUTA 2', 'IMPORTO 2', 'VALUTA'],
        'columns_to_rename': {
            'DATA': 'Date',
            'TIPOLOGIA': 'Transaction Type',
            'AL CONTO / ALLA CATEGORIA': 'Category',
            'IMPORTO': 'Amount',
            'NOTE': 'Notes',
            'DAL CONTO': 'Account',
        },
        'values_to_rename': {'Transaction Type': {'Entrata': 'Reddito'}},
        'date_format': '%d/%m/%y',
    },
    'inbank': {},
}


# MAIN FUNCTIONS


def preprocess_csv(csv_path: str, app: str = 'dummy', app_custom_dict=None):
    """
    Import a CSV file and perform some preprocessing operations based on the specified application.

    Parameters
    ----------
    csv_path : str
        The path to the input CSV file.
    app : str, optional
        The application based on which to perform preprocessing, by default 'fleur'.

    Returns
    -------
    list of pandas.DataFrame
        If `app` is supported, returns a list of DataFrames,
        each containing the data for a particular year.
        Otherwise, returns None.
    """
    # * EXPENSES *

    # 1. Import

    # Fleur custom check (double dataset)
    csv_transfers_path = None
    if app == 'fleur':
        csv_expenses_path, csv_transfers_path = _split_fleur_csv(csv_path)
    else:
        csv_expenses_path = csv_path

    # Import expenses dataset
    dataset_expenses = pd.read_csv(csv_expenses_path, on_bad_lines='warn')

    # 2. Custom Preprocessing
    dataset_expenses = _apply_custom_preprocessing(
        dataset=dataset_expenses, app=app, app_custom_dict=app_custom_dict
    )

    # 3. General Preprocessing
    dataset_expenses = _apply_general_preprocessing(dataset=dataset_expenses)

    # 4. Save Datasets by Year
    available_years = dataset_expenses['Year'].unique().tolist()
    print(f'Found {len(available_years)} years: {available_years}')

    datasets_expenses_list = []
    for year in available_years:
        temp_df = dataset_expenses.loc[dataset_expenses['Year'] == year].reset_index(drop=True)
        datasets_expenses_list.append(temp_df)

    # * TRANSFERS *

    # 1. Import
    if csv_transfers_path is not None:
        dataset_transfers = pd.read_csv(csv_transfers_path, on_bad_lines='warn')

        # 2. General Preprocessing
        dataset_transfers = _apply_general_preprocessing(dataset=dataset_transfers)

        # 3. Save Datasets by Year
        datasets_transfers_list = []
        for year in available_years:
            temp_df = dataset_transfers.loc[dataset_transfers['Year'] == year].reset_index(
                drop=True
            )
            datasets_transfers_list.append(temp_df)
    else:
        datasets_transfers_list = []

    # * SAVE *
    # create dataframe dict
    return {
        'expenses': datasets_expenses_list,
        'transfers': datasets_transfers_list,
    }


# SUPPORT FUNCTIONS


def _split_fleur_csv(csv_path):
    """Split Fleur CSV file into two separate datasets.

    The Fleur .csv has 2 datasets inside
    This functions splits them and saves them separately.
    """
    # Read the entire file
    with open(csv_path, encoding='utf-8') as file:
        data = file.read()

    # Split the data where the row is empty
    datasets = data.split('\n\n\n\n')

    # Read each dataset into a DataFrame
    dfs = [pd.read_csv(StringIO(ds)) for ds in datasets]

    # Simple preprocessing
    dfs[1]['Date'] = pd.to_datetime(dfs[1]['Date'], format='%Y%m%d')
    dfs[1]['Date'] = dfs[1]['Date'].dt.strftime('%Y/%m/%d')

    # Get the directory of the original CSV file
    dir_path = os.path.dirname(csv_path)

    # Save each DataFrame to a new CSV file
    fleur_expenses_path = os.path.join(dir_path, 'fleur_expenses.csv')
    fleur_transfers_path = os.path.join(dir_path, 'fleur_transfers.csv')
    dfs[0].to_csv(fleur_expenses_path, index=False)
    dfs[1].to_csv(fleur_transfers_path, index=False)

    return fleur_expenses_path, fleur_transfers_path


def _apply_custom_preprocessing(dataset, app, app_custom_dict):
    if app in AVAILABLE_PREPROCESSING_APPS:
        print(f"Starting preprocessing for app '{app}'")
        settings = AVAILABLE_PREPROCESSING_APPS[app]

    elif app == 'custom':
        if app_custom_dict is not None:
            print('Starting custom preprocessing')
            settings = app_custom_dict
        else:
            print("Sorry, you have to provide an 'app_custom_dict' for your custom application.")
            return []

    else:
        print(
            f'Sorry, application {app} is not supported. '
            f"Please use app='custom' and provide your own 'app_custom_dict'"
        )
        return []

    # 1. Setting: 'delete_row'
    if 'delete_rows' in settings:
        dataset = dataset[settings['delete_rows']['start'] : settings['delete_rows']['end']]

    # 2. Settings: 'columns_to_drop'
    if 'columns_to_drop' in settings:
        dataset = dataset.drop(settings['columns_to_drop'], axis=1)

    # 3. Settings: 'columns_to_rename'
    if 'columns_to_rename' in settings:
        dataset = dataset.rename(columns=settings['columns_to_rename'])

    # 4. Settings: 'values_to_rename'
    if 'values_to_rename' in settings:
        for col in settings['values_to_rename']:
            for key, value in settings['values_to_rename'][col].items():
                dataset[col] = dataset[col].replace(key, value)

    # 5. Settings: 'date_format'
    if 'date_format' in settings:
        dataset['Date'] = pd.to_datetime(dataset['Date'], format=settings['date_format'])

    return dataset


def _apply_general_preprocessing(dataset):
    # 1. add information about year and month
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset['Year'] = dataset['Date'].dt.year
    dataset['Month'] = dataset['Date'].dt.to_period('M')

    # 2. convert column based on type

    # str for dates, months, ...
    dataset['Date'] = dataset['Date'].astype(str)
    dataset['Month'] = dataset['Month'].astype(str)

    # numeric for 'Amount' and 'E'
    dataset['Amount'] = pd.to_numeric(dataset['Amount'], errors='coerce')

    return dataset
