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


AVAILABLE_PREPROCESSING_APPS = ['dummy', 'fleur']


def preprocess_csv(csv_path: str, app: str='dummy'):
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

    # import original dataset
    dataset = pd.read_csv(csv_path)

    # * CUSTOM *

    if app == 'dummy':
        print("Dummy dataset selected")

    elif app == 'fleur':

        # delete last row (misleading)
        dataset = dataset[:-1]

        # add information about year and month
        dataset["Date"] = pd.to_datetime(dataset["Date"])
        dataset["Year"] = dataset["Date"].dt.year
        dataset["Month"] = dataset["Date"].dt.to_period('M')

        # convert column based on type

        # str for dates, months, ...
        dataset["Date"] = dataset["Date"].astype(str)
        dataset["Month"] = dataset["Month"].astype(str)

        # numeric for 'Amount' and 'E'
        dataset['Amount'] = pd.to_numeric(dataset['Amount'], errors='coerce')
        dataset['E'] = pd.to_numeric(dataset['E'], errors='coerce')

    else:
        print(f"Sorry, application {app} is not supported. Please choose between: {AVAILABLE_PREPROCESSING_APPS}")
        return []

    # display info on founded years in data
    available_years = dataset['Year'].unique().tolist()
    print(f'Found {len(available_years)} years: {available_years}')

    # save all years datasets in a list and return it
    datasets_list = []
    for year in available_years:
        temp_df = dataset.loc[dataset['Year'] == year].reset_index(drop=True)
        datasets_list.append(temp_df)
    return datasets_list
