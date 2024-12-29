"""Networth Report Generator."""

# ⚙️ Ruff Settings
# ruff: noqa: PTH100 PTH103 PTH110 PTH118 PTH120 PTH123 C408

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objs as go

# global variables

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


class NetworthReportGenerator:
    """
    A class to read, process, and visualize a portfolio's composition over time.

    This class:
    1. Reads data from a CSV file.
    2. Cleans, renames, and converts columns into the proper formats.
    3. Computes relevant metrics (taxes, totals, etc.).
    4. Creates stacked bar charts for absolute values and percentages.

    Parameters
    ----------
    data_path : Union[str, Path]
        The path to the CSV file containing the portfolio data.

    Attributes
    ----------
    data_path : Path
        Resolved path to the data file.
    dataframe : pd.DataFrame
        The DataFrame holding all processed portfolio data.
    month_map : dict
        Mapping of Italian month names to month numbers.
    MIN_VALUE_FOR_TEXT : int
        Minimum value required to display bar text in the absolute chart.
    MIN_PERCENT_FOR_TEXT : float
        Minimum percentage required to display bar text in the percentage chart.
    """

    # A class-level mapping from Italian month names to month numbers
    month_map = {  # noqa: RUF012
        'gennaio': 1,
        'febbraio': 2,
        'marzo': 3,
        'aprile': 4,
        'maggio': 5,
        'giugno': 6,
        'luglio': 7,
        'agosto': 8,
        'settembre': 9,
        'ottobre': 10,
        'novembre': 11,
        'dicembre': 12,
    }

    # Thresholds for conditional text display
    MIN_VALUE_FOR_TEXT = 3000
    MIN_PERCENT_FOR_TEXT = 3.0

    def __init__(self, data_path: Union[str, Path]) -> None:
        self.data_path = Path(data_path).resolve()
        self.dataframe: Optional[pd.DataFrame] = None

        self._read_and_clean_data()
        self._compute_metrics()
        print('NetworthReportGenerator initialized.')

    @staticmethod
    def _parse_italian_date(date_str: str, month_map: dict) -> pd.Timestamp:
        """
        Parse an Italian date string in the format 'day month year' into a pandas Timestamp.

        Parameters
        ----------
        date_str : str
            The date string to parse, e.g., '10 gennaio 2023'.
        month_map : dict
            A dictionary mapping the Italian month name to its month number.

        Returns
        -------
        pd.Timestamp
            The parsed date, or pd.NaT if parsing fails.
        """
        expected_parts = 3
        try:
            date_str = str(date_str).strip().lower()
            parts = date_str.split()
            if len(parts) != expected_parts:
                return pd.NaT
            day = int(parts[0])
            month_str = parts[1]
            year = int(parts[2])
            month = month_map.get(month_str)
            if month is None:
                return pd.NaT
            return pd.Timestamp(year=year, month=month, day=day)
        except:  # noqa
            return pd.NaT

    @staticmethod
    def _parse_currency(value: str) -> float:
        """
        Parse a currency value in (Italian) string format to a float.

        Parameters
        ----------
        value : str
            The currency value as a string, e.g., '€ 1.234,56'.

        Returns
        -------
        float
            The numeric representation of the currency value.
        """
        try:
            if pd.isna(value):
                return np.nan
            value = str(value)
            # Remove euro symbol, spaces, thousand separators, etc.
            value = value.replace('€', '').replace(' ', '').replace('.', '').replace(',', '.')
            return float(value)
        except:  # noqa
            return np.nan

    def _read_and_clean_data(self) -> pd.DataFrame:
        """
        Read, clean, and transform the CSV file into a pandas DataFrame.

        This method performs the following steps:
        1. Reads the raw CSV, skipping rows until it finds the 'Mese' column header.
        2. Loads the dataframe, dropping unnecessary columns and NaNs.
        3. Renames columns to a standardized list.
        4. Converts the 'Mese' column to datetime.
        5. Converts currency-like columns to floats.

        Returns
        -------
        pd.DataFrame
            The cleaned and preprocessed DataFrame.

        Raises
        ------
        ValueError
            If no column containing 'Mese' is found in the CSV headers.
        """
        with self.data_path.open(encoding='utf-8') as f:
            lines = f.readlines()

        # Find the header line where 'Mese' appears
        header_line_number = None
        for i, line in enumerate(lines):
            if 'Mese' in line:
                header_line_number = i
                break
        if header_line_number is None:
            exc_desc = "No header line containing 'Mese' found in the CSV."
            raise ValueError(exc_desc)

        # Read the CSV, skipping lines up to the appropriate header row
        dataframe = pd.read_csv(
            self.data_path,
            skiprows=header_line_number + 1,  # skip the lines up to the actual header
            encoding='utf-8',
        )

        # Remove unnecessary columns (assuming columns 2:-8 are relevant)
        dataframe = dataframe.iloc[:, 2:-8].dropna()

        # Rename columns
        cols_name = [
            'Mese',
            'Liquidità',
            'Fondo Pensione (Lordo)',
            'Fondo Pensione (Netto)',
            'Obbligazioni (Lordo)',
            'Obbligazioni (Netto)',
            'BIT:SWDA (ETF)',
            'BIT:EIMI (ETF)',
            'BIT:SGLD (ETC)',
            'Azioni ed ETF (Lordo)',
            'Azioni ed ETF (Netto)',
        ]
        dataframe.columns = cols_name

        # Parse the 'Mese' column into datetime
        mese_columns = [col for col in dataframe.columns if 'Mese' in col]
        if not mese_columns:
            exc_desc = "No column containing 'Mese' found."
            raise ValueError(exc_desc)
        mese_col = mese_columns[0]
        dataframe[mese_col] = dataframe[mese_col].apply(
            lambda x: self._parse_italian_date(x, self.month_map)
        )

        # Parse currency columns
        for col in cols_name[1:]:
            dataframe[col] = dataframe[col].apply(self._parse_currency)

        # Store the cleaned DataFrame in the class instance
        self.dataframe = dataframe.copy()

        return self.dataframe

    def _compute_metrics(self) -> pd.DataFrame:
        """
        Compute additional columns (taxes, totals, percentages, etc.) for the dataframe.

        This method:
        1. Re-converts 'Mese' to a year-month format (if not already).
        2. Calculates tax for Fondo Pensione, Obbligazioni, Azioni ed ETF.
        3. Calculates gross and net totals per month.
        4. Computes percentage distribution of each component in the gross total.

        Returns
        -------
        pd.DataFrame
            The DataFrame with additional computed columns.
        """
        if self.dataframe is None:
            exc_desc = "Dataframe is not loaded. Call 'read_and_clean_data' first."
            raise ValueError(exc_desc)

        dataframe = self.dataframe.copy()

        # Ensure 'Mese' is only year-month
        dataframe['Mese'] = pd.to_datetime(dataframe['Mese'], format='%Y-%m')
        dataframe['Anno_Mese_str'] = dataframe['Mese'].dt.strftime('%Y-%m')

        # Compute taxes
        dataframe['Fondo Pensione Tax'] = (
            dataframe['Fondo Pensione (Lordo)'] - dataframe['Fondo Pensione (Netto)']
        )
        dataframe['Obbligazioni Tax'] = (
            dataframe['Obbligazioni (Lordo)'] - dataframe['Obbligazioni (Netto)']
        )
        dataframe['Azioni ed ETF Tax'] = (
            dataframe['Azioni ed ETF (Lordo)'] - dataframe['Azioni ed ETF (Netto)']
        )

        # Gross and net totals
        dataframe['Gross_Total'] = (
            dataframe['Liquidità']
            + dataframe['Fondo Pensione (Netto)']
            + dataframe['Fondo Pensione Tax']
            + dataframe['Obbligazioni (Netto)']
            + dataframe['Obbligazioni Tax']
            + dataframe['Azioni ed ETF (Netto)']
            + dataframe['Azioni ed ETF Tax']
        )
        dataframe['Net_Total'] = (
            dataframe['Liquidità']
            + dataframe['Fondo Pensione (Netto)']
            + dataframe['Obbligazioni (Netto)']
            + dataframe['Azioni ed ETF (Netto)']
        )

        # Compute percentages
        components = {
            'Liquidità': 'Liquidità',
            'Fondo Pensione (Netto)': 'Fondo Pensione (Netto)',
            'Fondo Pensione Tax': 'Fondo Pensione Tax',
            'Obbligazioni (Netto)': 'Obbligazioni (Netto)',
            'Obbligazioni Tax': 'Obbligazioni Tax',
            'Azioni ed ETF (Netto)': 'Azioni ed ETF (Netto)',
            'Azioni ed ETF Tax': 'Azioni ed ETF Tax',
        }
        for column in components.values():
            percentage_column = f'{column}_Percentage'
            dataframe[percentage_column] = (dataframe[column] / dataframe['Gross_Total']) * 100

        # Update self.dataframe with the new columns
        self.dataframe = dataframe
        return dataframe

    def _add_bar_trace(self, fig: go.Figure, name: str, x, y, color: str) -> None:
        """
        Add a bar trace to a figure with conditional text (for the absolute chart).

        Parameters
        ----------
        fig : go.Figure
            The Plotly figure to which the trace will be added.
        name : str
            Legend name for the trace.
        x : array-like
            The x-axis values (months).
        y : array-like
            The y-axis values (amounts).
        color : str
            The color of the bars, in an rgba format.
        """
        text = [f'{val:.0f} €' if val > self.MIN_VALUE_FOR_TEXT else '' for val in y]
        fig.add_trace(
            go.Bar(
                name=name,
                x=x,
                y=y,
                marker_color=color,
                text=text,
                textposition='inside',
                insidetextanchor='middle',
                textfont={'size': 9},
            )
        )

    def _add_bar_trace_percentage(self, fig: go.Figure, name: str, x, y, color: str) -> None:
        """
        Add a bar trace to a figure with conditional text (for the percentage chart).

        Parameters
        ----------
        fig : go.Figure
            The Plotly figure to which the trace will be added.
        name : str
            Legend name for the trace.
        x : array-like
            The x-axis values (months).
        y : array-like
            The y-axis values (percentages).
        color : str
            The color of the bars, in an rgba format.
        """
        text = [f'{val:.1f}%' if val > self.MIN_PERCENT_FOR_TEXT else '' for val in y]
        fig.add_trace(
            go.Bar(
                name=name,
                x=x,
                y=y,
                marker_color=color,
                text=text,
                textposition='inside',
                insidetextanchor='middle',
                textfont={'size': 10},
            )
        )

    def _save_list_of_plotly_figs(self, path, fig_list, title='My Report'):
        """Save a list of Plotly figures to a single HTML file.

        This function accepts a path to an HTML file and a list of Plotly figures.
        It then generates and saves these figures to the specified HTML file.
        If the file already exists, it is overwritten.

        Parameters
        ----------
        path : str
            The path to the HTML file where the figures should be saved.
            This should include the filename and the '.html' extension.
        fig_list : list[plotly.graph_objs._figure.Figure]
            A list of Plotly figures to be saved to the HTML file.

        Raises
        ------
        Exception
            If there is an error during the saving process,
            an exception is raised with a description of the error.

        Notes
        -----
        If the HTML file specified by the 'path' parameter already exists, it will be overwritten.
        """
        try:
            icon_path = str(os.path.join(PACKAGE_DIR, 'images', 'chart.svg'))
            gif_path = str(os.path.join(PACKAGE_DIR, 'images', 'finance-cards.webp'))

            # Check if the gif_path exists
            if not os.path.exists(gif_path):
                exc_desc = f'The file {gif_path} does not exist.'
                raise FileNotFoundError(exc_desc)  # noqa: TRY301

            with open(path, 'w', encoding='utf-8') as output_file:
                output_file.write(
                    f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>{title.replace('<br><i>',' ').replace('</i>','')}</title>
                        <link rel="icon" type="image/svg" href="{icon_path}">
                        <style>
                            .centered {{
                                text-align: center;
                                width: 1980px;
                                margin: 0 auto;
                            }}
                            h1 {{
                                font-family: 'Open Sans', sans-serif;
                            }}
                        </style>
                    </head>
                    <body>
                    <div class='centered'>
                        <br><h1>{title}</h1>
                        <!--<img src={icon_path} alt="Finance Cards" width="125">-->
                    """
                )

            with open(path, 'a', encoding='utf-8') as output_file:
                output_file.write(fig_list[0].to_html(full_html=False, include_plotlyjs='cdn'))

            with open(path, 'a', encoding='utf-8') as output_file:
                for fig in fig_list[1:]:
                    output_file.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

            with open(path, 'a', encoding='utf-8') as output_file:
                output_file.write('</div>')  # Close the div after all figures have been written.

            print(f"The plots were correctly saved in '{path}'")
        except Exception as exc:  # noqa: BLE001
            print(f'There were errors in saving the plot. Details: {exc}')

    def create_absolute_figure(self) -> go.Figure:
        """
        Create a stacked bar chart fig. showing the absolute values (in €) of each component.

        Returns
        -------
        go.Figure
            A Plotly stacked bar chart figure with absolute portfolio composition.

        Raises
        ------
        ValueError
            If the dataframe is not set or does not contain expected columns.
        """
        if self.dataframe is None or 'Gross_Total' not in self.dataframe.columns:
            exc_desc = "Metrics have not been computed. Call 'compute_metrics' first."
            raise ValueError(exc_desc)

        dataframe = self.dataframe
        fig = go.Figure()

        # Add bar traces
        self._add_bar_trace(
            fig,
            'Liquidità',
            dataframe['Anno_Mese_str'],
            dataframe['Liquidità'],
            'rgba(50, 171, 96, 0.7)',
        )
        self._add_bar_trace(
            fig,
            'Fondo Pensione Netto',
            dataframe['Anno_Mese_str'],
            dataframe['Fondo Pensione (Netto)'],
            'rgba(26, 118, 255, 0.7)',
        )
        self._add_bar_trace(
            fig,
            'Fondo Pensione Tax',
            dataframe['Anno_Mese_str'],
            dataframe['Fondo Pensione Tax'],
            'rgba(26, 118, 255, 0.3)',
        )
        self._add_bar_trace(
            fig,
            'Obbligazioni Netto',
            dataframe['Anno_Mese_str'],
            dataframe['Obbligazioni (Netto)'],
            'rgba(247, 238, 127, 0.7)',
        )
        self._add_bar_trace(
            fig,
            'Obbligazioni Tax',
            dataframe['Anno_Mese_str'],
            dataframe['Obbligazioni Tax'],
            'rgba(247, 238, 127, 0.3)',
        )
        self._add_bar_trace(
            fig,
            'Azioni ed ETF Netto',
            dataframe['Anno_Mese_str'],
            dataframe['Azioni ed ETF (Netto)'],
            'rgba(219, 64, 82, 0.7)',
        )
        self._add_bar_trace(
            fig,
            'Azioni ed ETF Tax',
            dataframe['Anno_Mese_str'],
            dataframe['Azioni ed ETF Tax'],
            'rgba(219, 64, 82, 0.3)',
        )

        # Define offsets for the net and gross total labels
        offset = dataframe['Gross_Total'].max() * 0.02
        dataframe['Net_Total_Text_Y'] = dataframe['Gross_Total'] + offset
        dataframe['Gross_Total_Text_Y'] = dataframe['Gross_Total'] + offset * 2

        # Add net total labels
        fig.add_trace(
            go.Scatter(
                x=dataframe['Anno_Mese_str'],
                y=dataframe['Net_Total_Text_Y'],
                mode='text',
                text=dataframe['Net_Total'].apply(lambda val: f'({val:,.0f} €)'),
                textposition='top center',
                textfont={'size': 12, 'color': 'black', 'family': 'Arial'},
                showlegend=False,
            )
        )

        # Add gross total labels
        fig.add_trace(
            go.Scatter(
                x=dataframe['Anno_Mese_str'],
                y=dataframe['Gross_Total_Text_Y'],
                mode='text',
                text=dataframe['Gross_Total'].apply(lambda val: f'{val:,.0f} €'),
                textposition='top center',
                textfont={'size': 12, 'color': 'black', 'family': 'Arial'},
                showlegend=False,
            )
        )

        # Update layout
        fig.update_layout(
            barmode='stack',
            title='Portfolio Composition Over Time (Absolute Values)',
            xaxis_title='Month',
            yaxis_title='Amount (€)',
            legend_title='Categories',
            # Switch to vertical legend
            legend=dict(
                orientation='v',
                x=0.06,  # horizontal center
                xanchor='center',
                y=1.085,  # "top" of the plot area
                yanchor='top',
            ),
            template='plotly_white',
            xaxis={'tickangle': -45, 'type': 'category'},
            margin={'b': 150, 't': 150},
            width=1980,
            height=1080,
        )

        # Ensure bars are stacked from left to right by month
        fig.update_xaxes(categoryorder='category ascending')

        # Adjust the y-axis range to accommodate labels
        y_max = dataframe['Gross_Total_Text_Y'].max() * 1.05
        fig.update_yaxes(range=[0, y_max])

        return fig

    def create_percentage_figure(self) -> go.Figure:
        """
        Create a 100% stacked bar chart fig. showing the percentage distribution of each component.

        Returns
        -------
        go.Figure
            A Plotly stacked bar chart figure with percentage composition.

        Raises
        ------
        ValueError
            If the dataframe is not set or does not contain the required percentage columns.
        """
        if self.dataframe is None or 'Liquidità_Percentage' not in self.dataframe.columns:
            exc_desc = "Metrics have not been computed. Call 'compute_metrics' first."
            raise ValueError(exc_desc)

        dataframe = self.dataframe
        fig = go.Figure()

        # Add bar traces for each component's percentage
        self._add_bar_trace_percentage(
            fig,
            'Liquidità',
            dataframe['Anno_Mese_str'],
            dataframe['Liquidità_Percentage'],
            'rgba(50, 171, 96, 0.7)',
        )
        self._add_bar_trace_percentage(
            fig,
            'Fondo Pensione Netto',
            dataframe['Anno_Mese_str'],
            dataframe['Fondo Pensione (Netto)_Percentage'],
            'rgba(26, 118, 255, 0.7)',
        )
        self._add_bar_trace_percentage(
            fig,
            'Fondo Pensione Tax',
            dataframe['Anno_Mese_str'],
            dataframe['Fondo Pensione Tax_Percentage'],
            'rgba(26, 118, 255, 0.3)',
        )
        self._add_bar_trace_percentage(
            fig,
            'Obbligazioni Netto',
            dataframe['Anno_Mese_str'],
            dataframe['Obbligazioni (Netto)_Percentage'],
            'rgba(247, 238, 127, 0.7)',
        )
        self._add_bar_trace_percentage(
            fig,
            'Obbligazioni Tax',
            dataframe['Anno_Mese_str'],
            dataframe['Obbligazioni Tax_Percentage'],
            'rgba(247, 238, 127, 0.3)',
        )
        self._add_bar_trace_percentage(
            fig,
            'Azioni ed ETF Netto',
            dataframe['Anno_Mese_str'],
            dataframe['Azioni ed ETF (Netto)_Percentage'],
            'rgba(219, 64, 82, 0.7)',
        )
        self._add_bar_trace_percentage(
            fig,
            'Azioni ed ETF Tax',
            dataframe['Anno_Mese_str'],
            dataframe['Azioni ed ETF Tax_Percentage'],
            'rgba(219, 64, 82, 0.3)',
        )

        # Update layout for 100% stacked bars
        fig.update_layout(
            barmode='stack',
            title='Portfolio Composition Over Time (Percentage)',
            xaxis_title='Month',
            yaxis_title='Percentage (%)',
            showlegend=False,
            template='plotly_white',
            xaxis={'tickangle': -45, 'type': 'category'},
            margin={'b': 150, 't': 150},
            width=1980,
            height=1080,
        )

        # Range from 0 to 100% on the y-axis
        fig.update_yaxes(range=[0, 100])
        fig.update_xaxes(categoryorder='category ascending')

        # Optionally add a "100%" label on top
        fig.add_trace(
            go.Scatter(
                x=dataframe['Anno_Mese_str'],
                y=[100] * len(dataframe),
                mode='text',
                text=['100%'] * len(dataframe),
                textposition='top center',
                textfont={'size': 12, 'color': 'black', 'family': 'Arial'},
                showlegend=False,
            )
        )

        return fig

    def generate_report(self, output_path: Union[str, Path]) -> None:
        """
        Generate a report with two stacked bar charts (absolute and percentage).

        This method generates the two bar charts.
        The charts are then saved to an HTML file at the specified path.

        Parameters
        ----------
        output_path : Union[str, Path]
            The path to the HTML file where the report will be saved.

        Raises
        ------
        ValueError
            If the data is not loaded or the output path is invalid.
        """
        if self.dataframe is None:
            exc_desc = "Dataframe is not loaded. Call 'read_and_clean_data' first."
            raise ValueError(exc_desc)

        output_path = Path(output_path)
        if not output_path.parent.is_dir():
            exc_desc = 'Output directory does not exist.'
            raise ValueError(exc_desc)

        # Compute metrics and create the figures
        fig_absolute = self.create_absolute_figure()
        fig_percentage = self.create_percentage_figure()

        # Save the figures to an HTML file
        self._save_list_of_plotly_figs(
            output_path, [fig_absolute, fig_percentage], title='Networth Report'
        )
