"""
Preprocessing

Available Preprocessing Apps
- fleur | https://apps.apple.com/it/app/fleur-gestione-spese-e-budget/id1621020173
"""

# Custom Pylint rules for the file
# pylint: disable=C0301
# C0301:line-too-long


# Libraries
import pandas as pd

AVAILABLE_PREPROCESSING_APPS = {

    # no operations needed for dummy
    'dummy': {
        'columns_to_drop': ['E'],
    },

    'fleur': {
        'delete_rows': {
            'start': 0,
            'end': -1
        },
        'columns_to_drop': ['E'],
    },

    '1money': {
        'delete_rows': {
            'start': 0,
            'end': -5
        },
        'columns_to_drop': ['TAG','VALUTA 2','IMPORTO 2','VALUTA'],
        'columns_to_rename': {
            'DATA': 'Date',
            'TIPOLOGIA': 'Transaction Type',
            'AL CONTO / ALLA CATEGORIA': 'Category',
            'IMPORTO': 'Amount',
            'NOTE': 'Notes',
            'DAL CONTO': 'Account'
        },
        'values_to_rename': {
            'Transaction Type': {
                'Entrata': 'Reddito'
            }
        },
        'date_format': '%d/%m/%y'
    },

    'inbank': {}

}


def preprocess_csv(csv_path: str, app: str='dummy', app_custom_dict=None):
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

    # * GENERAL *

    # 1. import original dataset
    dataset = pd.read_csv(csv_path, on_bad_lines='warn')

    # * APPLY CUSTOM RULES *

    if app in AVAILABLE_PREPROCESSING_APPS:
        print(f"Starting preprocessing for app '{app}'")
        settings = AVAILABLE_PREPROCESSING_APPS[app]

    elif app == 'custom':
        if app_custom_dict is not None:
            print("Starting custom preprocessing")
            settings = app_custom_dict
        else:
            print("Sorry, you have to provide an 'app_custom_dict' for your custom application.")
            return []

    else:
        print(f"Sorry, application {app} is not supported. Please use app='custom' and provide your own 'app_custom_dict'")
        return []

    # A. Setting: 'delete_row'
    if 'delete_rows' in settings:
        dataset = dataset[settings['delete_rows']['start']:settings['delete_rows']['end']]

    # B. Settings: 'columns_to_drop'
    if 'columns_to_drop' in settings:
        dataset = dataset.drop(settings['columns_to_drop'], axis=1)

    # C. Settings: 'columns_to_rename'
    if 'columns_to_rename' in settings:
        dataset = dataset.rename(columns=settings['columns_to_rename'])

    # D. Settings: 'values_to_rename'
    if 'values_to_rename' in settings:
        for col in settings['values_to_rename']:
            for key,value in settings['values_to_rename'][col].items():
                dataset[col] = dataset[col].replace(key,value)

    # E. Settings: 'date_format'
    if 'date_format' in settings:
        dataset["Date"] = pd.to_datetime(dataset["Date"], format=settings['date_format'])

    # * GENERAL *

    # 2. add information about year and month
    dataset["Date"] = pd.to_datetime(dataset["Date"])
    dataset["Year"] = dataset["Date"].dt.year
    dataset["Month"] = dataset["Date"].dt.to_period('M')

    # 3. convert column based on type

    # str for dates, months, ...
    dataset["Date"] = dataset["Date"].astype(str)
    dataset["Month"] = dataset["Month"].astype(str)

    # numeric for 'Amount' and 'E'
    dataset['Amount'] = pd.to_numeric(dataset['Amount'], errors='coerce')

    # 4. display info on founded years in data
    available_years = dataset['Year'].unique().tolist()
    print(f'Found {len(available_years)} years: {available_years}')

    # 5. save all years datasets in a list and return it
    datasets_list = []
    for year in available_years:
        temp_df = dataset.loc[dataset['Year'] == year].reset_index(drop=True)
        datasets_list.append(temp_df)
    return datasets_list
