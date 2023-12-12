"""
Dataviz Utils
"""

# Custom Pylint rules for the file
# pylint: disable=W0718 C0301 E0402 C0302
# W0718:broad-exception-caught
# C0301:line-too-long
# E0402:relative-beyond-top-level
# C0302:too-many-lines


# Libraries
import os
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Custom functions
from .preprocessing import preprocess_csv

# Settings
pd.set_option('mode.chained_assignment', None)


# global variables

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

class ReportGenerator:
    """
    ReportGenerator Class
    """
    def __init__(
        self,
        csv_path,
        save_path,
        app="dummy",
        app_custom_dict=None,
        category_color_dict=None
        ):

        # instantiate class variables
        self.datasets_expenses = {}
        self.datasets_transfers = {}
        self.report_generators = {}
        self.reports = {}
        self.palette_colors = None
        self.save_path = save_path

        # handle datasets
        data_dict = preprocess_csv(csv_path, app=app, app_custom_dict=app_custom_dict)
        self.__handle_datasets(data_dict)

        # handle colors
        self.__handle_colors(category_color_dict)

        # create save path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print("Report Generator Ready")


    # MAIN FUNCTION
    def generate_reports(self, show_plots=False):
        """
        TODO: generate docstring
        """

        # for each year
        for key,value in self.datasets_expenses.items():

            # * YEARLY SPIDERPLOT *
            # Spyderplot of Yearly Total Income and Expenses
            plot = self.plot_yearly_spyderplot(dataset=value)
            self.reports[key].append(plot)
            if show_plots:
                plot.show()

            # * YEARLY TREEMAP *
            # Spyderplot of Yearly Total Income and Expenses
            plot = self.plot_yearly_treemap(dataset=value)
            self.reports[key].append(plot)
            if show_plots:
                plot.show()

            # * LINEPLOT OF INCOME AND EXPENSES *
            plot = self.plot_lineplot_income_and_expenses(dataset=value)
            self.reports[key].append(plot)
            if show_plots:
                plot.show()

            # * DELTA *
            plot = self.plot_monthly_delta(dataset=value)
            self.reports[key].append(plot)
            if show_plots:
                plot.show()

            # * LINEPLOT OF INCOME AND EXPENSES (CUMULATIVE) *
            plot = self.plot_lineplot_income_and_expenses_cum(dataset=value)
            self.reports[key].append(plot)
            if show_plots:
                plot.show()

            # * INCOME BY MONTH *
            plot = self.plot_income_by_month(dataset=value)
            self.reports[key].append(plot)
            if show_plots:
                plot.show()

            # * EXPENSES BY MONTH *
            plot = self.plot_expenses_by_month(dataset=value)
            self.reports[key].append(plot)
            if show_plots:
                plot.show()

            # * INCOME AND EXPENSES BY MONTH *
            plot = self.plot_income_and_expenses_by_month(dataset=value)
            self.reports[key].append(plot)
            if show_plots:
                plot.show()

        # for each year
        for key,value in self.datasets_transfers.items():
            plot = self.plot_barplot_transfers(dataset=value)
            self.reports[key].append(plot)
            if show_plots:
                plot.show()

        # * SAVE REPORTS *
        for key,value in self.reports.items():

            report_name = f'report_{key}.html'
            report_path = os.path.join(self.save_path, report_name)

            if key != 'all':
                title = f'Financial Report {key}'
            else:
                title = f'Financial Report<br><i>{self.min_year} - {self.max_year}</i>'

            self.__save_list_of_plotly_figs(
                path=report_path,
                fig_list=value,
                title=title
            )



    # DATA VISUALIZATIONS

    def plot_yearly_spyderplot(self, dataset):
        """
        Generate a Plotly figure representing yearly income, grouped by category, as a spyder plot.

        This function preprocesses the input DataFrame to extract income-relevant data.
        It then generates a spyder (radar) plot, showing the total amount of income for each category.

        Parameters
        ----------
        dataset : pd.DataFrame
            The input DataFrame containing financial transaction data.
            It should include the following columns: 'Transaction Type', 'Category', 'Amount'.

        Returns
        -------
        plotly.graph_objs._figure.Figure
            A Plotly figure representing the yearly income by category in a spyder plot.
        """

        dataset_income = dataset.copy()
        dataset_income = dataset_income.loc[dataset_income['Transaction Type'] == 'Reddito']
        dataset_income = dataset_income.groupby(['Category']).agg({
                'Amount': 'sum'
            }).reset_index()
        dataset_income['Amount'] = round(dataset_income['Amount']).astype(int)

        dataset_expenses = dataset.copy()
        dataset_expenses = dataset_expenses.loc[dataset_expenses['Transaction Type'] == 'Spesa']
        dataset_expenses = dataset_expenses.groupby(['Category']).agg({
                'Amount': 'sum'
            }).reset_index()
        dataset_expenses['Amount'] = round(dataset_expenses['Amount']).astype(int)

        #fig = go.Figure()
        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'polar'}] * 2] * 1)

        # additional info
        income_total = round(sum(dataset_income['Amount']))
        expenses_total = round(sum(dataset_expenses['Amount']))
        profit = income_total - expenses_total
        profit_perc = round((profit / income_total) * 100, 2)

        # determine the color and sign of the profit value
        if profit >= 0:
            profit_str = f'<span style="color: {self.income_color};">{profit}€</span>'
        else:
            profit_str = f'<span style="color: {self.expenses_color};">{profit}€</span>'

        subtitle = f"<br><sup><br>Total Income: <b>{income_total}€</b><br>Total Expenses: <b>{expenses_total}€</b><br>Profit: <b>{profit_str}</b> ({profit_perc}%)</sup>"


        theta_with_amount = [f'<b>{category}</b><br>{amount}€' for category, amount in zip(list(dataset_income['Category']), list(dataset_income['Amount']))]
        fig.add_trace(
            go.Scatterpolar(
                r=list(dataset_income['Amount']),
                theta=theta_with_amount, #list(dataset_income['Category']),
                mode='markers+text',
                name='Income',
                fill='toself',
                hoverinfo='r',
                hovertemplate='Income by %{theta}',
                line=dict(color=self.income_color)
            ),
            row=1,
            col=1,
        )

        theta_with_amount = [f'<b>{category}</b><br>{amount}€' for category, amount in zip(list(dataset_expenses['Category']), list(dataset_expenses['Amount']))]
        fig.add_trace(
            go.Scatterpolar(
                r=list(dataset_expenses['Amount']),
                theta=theta_with_amount,
                mode='markers+text',
                name='Expenses',
                fill='toself',
                hoverinfo='r',
                hovertemplate='Expenses by %{theta}',
                line=dict(color=self.expenses_color)
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            polar1=dict(
                radialaxis=dict(
                    visible=True
                ),
                angularaxis=dict(
                    tickfont=dict(
                        size=12
                    )
                )
            ),
            polar2=dict(
                radialaxis=dict(
                    visible=True
                ),
                angularaxis=dict(
                    tickfont=dict(
                        size=12
                    )
                )
            ),
            showlegend=False,
            width=1980,
            height=600,
            title=f"Yearly Income and Expenses by Category{subtitle}"
        )

        return fig


    def plot_yearly_treemap(self, dataset):
        """
        Generate Plotly figures representing yearly income and expenses, grouped by category, as treemaps.

        Parameters
        ----------
        dataset : pd.DataFrame
            The input DataFrame containing financial transaction data.
            It should include the following columns: 'Transaction Type', 'Category', 'Amount'.

        Returns
        -------
        plotly.graph_objs._figure.Figure
            A Plotly figure representing the yearly income and expenses by category in treemaps.
        """

        # Income
        dataset_income = dataset.loc[dataset['Transaction Type'] == 'Reddito']
        dataset_income['Notes'] = dataset_income['Notes'].fillna('Non specificato')
        dataset_income['Amount_label'] = dataset_income['Amount'].astype(str) + '€'
        dataset_income['Notes'] = ' • ' + dataset_income['Date'] + ': ' + dataset_income['Notes'] + ' → ' + dataset_income['Amount_label']
        dataset_income = dataset_income.groupby(['Category']).agg({
            'Amount': 'sum',
            'Notes': lambda x: '\n<br>'.join(x)
        }).reset_index()
        dataset_income['Amount'] = round(dataset_income['Amount']).astype(int)

        # Expenses
        dataset_expenses = dataset.loc[dataset['Transaction Type'] == 'Spesa']
        dataset_expenses['Notes'] = dataset_expenses['Notes'].fillna('Non specificato')
        dataset_expenses['Amount_label'] = dataset_expenses['Amount'].astype(str) + '€'
        dataset_expenses['Notes'] = ' • ' + dataset_expenses['Date'] + ': ' + dataset_expenses['Notes'] + ' → ' + dataset_expenses['Amount_label']
        dataset_expenses = dataset_expenses.groupby(['Category']).agg({
            'Amount': 'sum',
            'Notes': lambda x: '\n<br>'.join(x)
        }).reset_index()
        dataset_expenses['Amount'] = round(dataset_expenses['Amount']).astype(int)

        # Function to get colors for categories
        def get_colors(data):
            return [self.category_color_dict_expenses.get(category, '#FFFFFF') for category in data['Category']]

        # Create figures for income and expenses treemap
        fig_income = go.Figure(go.Treemap(
            labels=dataset_income['Category'],
            parents=[""] * len(dataset_income),
            values=dataset_income['Amount'],
            textinfo='label+text',
            texttemplate="<b>%{label}</b><br>%{value} €",
            textposition='middle center',
            marker_colors=get_colors(dataset_income),  # Set colors
            domain={'x': [0, 0.48], 'y': [0, 1]},  # Set domain for left side
            hoverinfo='none'  # Disable hover effect
        ))

        fig_expenses = go.Figure(go.Treemap(
            labels=dataset_expenses['Category'],
            parents=[""] * len(dataset_expenses),
            values=dataset_expenses['Amount'],
            textinfo='label+text',
            texttemplate="<b>%{label}</b><br>%{value} €",
            textposition='middle center',
            marker_colors=get_colors(dataset_expenses),  # Set colors
            domain={'x': [0.52, 1], 'y': [0, 1]},  # Set domain for right side
            hoverinfo='none'  # Disable hover effect
        ))

        # Create a single figure to display both treemaps side by side
        fig = go.Figure(data=[fig_income.data[0], fig_expenses.data[0]])

        # Update layout
        fig.update_layout(
            title='',
            grid= {'columns': 2, 'rows': 1},
            width=1980,
            height=600,
            margin=dict(t=0)
        )

        return fig


    def plot_lineplot_income_and_expenses(self, dataset):
        """
        Generate a plotly figure showing monthly income, income+support and expenses.

        The function groups data by month, and then generates a line chart showing the
        sum of the amount for income, income+support, and expenses per month.

        Parameters
        ----------
        dataset : DataFrame
            The dataset that contains the transactions. Expected columns are
            - 'Transaction Type'
            - 'Category'
            - 'Amount'
            - 'Month'
            'Transaction Type' should contain values like 'Reddito' for income and 'Spesa' for expenses.
            'Month' should be in the format YYYY-MM-DD.

        Returns
        -------
        fig : plotly.graph_objs._figure.Figure
            A plotly figure containing the generated plot.

        Notes
        -----
        The function uses the self.category_color_dict_expenses constant which should be a dictionary
        mapping each category to a specific color in hexadecimal form.
        """

        # prepare data for dataviz
        dataset = dataset.copy()
        income_df = dataset.loc[
            (dataset['Transaction Type'] == 'Reddito') & (dataset['Category'] != 'Supporto Famiglia')
            ]

        income_df = income_df.groupby(['Month']).agg({'Amount': 'sum'}).reset_index()
        income_df['Amount_label'] = round(income_df['Amount']).astype(int).astype(str) + '€'

        dataset = dataset.copy()
        income_and_support_df = dataset.loc[(dataset['Transaction Type'] == 'Reddito')]
        income_and_support_df = income_and_support_df.groupby(['Month']).agg({'Amount': 'sum'}).reset_index()
        income_and_support_df['Amount_label'] = round(income_and_support_df['Amount']).astype(int).astype(str) + '€'

        dataset = dataset.copy()
        expenses_df = dataset.loc[dataset['Transaction Type'] == 'Spesa']
        expenses_df = expenses_df.groupby(['Month']).agg({'Amount': 'sum'}).reset_index()
        expenses_df['Amount_label'] = round(expenses_df['Amount']).astype(int).astype(str) + '€'

        # calculate means
        income_mean = np.mean(income_df['Amount'])
        income_support_mean = np.mean(income_and_support_df['Amount'])
        expenses_mean = np.mean(expenses_df['Amount'])

        # create plot
        fig = go.Figure()

        # Income
        fig.add_trace(
            go.Scatter(
                x=income_df['Month'],
                y=income_df['Amount'],
                text=income_df['Amount_label'],
                mode='lines+markers+text',
                name='Income',
                textposition='top center',
                line=dict(color=self.income_color),
                hovertemplate='Income of %{x}: %{y}€',
                ))

        # Income + Support
        fig.add_trace(
            go.Scatter(
                x=income_and_support_df['Month'],
                y=income_and_support_df['Amount'],
                text=income_and_support_df['Amount_label'],
                mode='lines+markers+text',
                name='Income + Support',
                textposition='top center',
                line=dict(color=self.income_and_support_color),
                hovertemplate='Income+Support of %{x}: %{y}€',
                ))

        # Expenses
        fig.add_trace(
            go.Scatter(
                x=expenses_df['Month'],
                y=expenses_df['Amount'],
                text=expenses_df['Amount_label'],
                mode='lines+markers+text',
                name='Expenses',
                textposition='top center',
                line=dict(color=self.expenses_color),
                hovertemplate='Expenses of %{x}: %{y}€',
                ))

        # add mean line for Income
        fig.add_trace(
            go.Scatter(
                x=income_df['Month'],
                y=[income_mean]*len(income_df['Month']),
                mode='lines',
                line=dict(color=self.income_color, width=1.5, dash='dash'),
                name='Mean Income',
                hovertemplate='Mean Income: %{y}€',
                visible='legendonly'
            )
        )

        # add mean line for Income + Support
        fig.add_trace(
            go.Scatter(
                x=income_and_support_df['Month'],
                y=[income_support_mean]*len(income_and_support_df['Month']),
                mode='lines',
                line=dict(color=self.income_and_support_color, width=1.5, dash='dash'),
                name='Mean Income + Support',
                hovertemplate='Mean Income+Support: %{y}€',
                visible='legendonly'
            )
        )

        # add mean line for Expenses
        fig.add_trace(
            go.Scatter(
                x=expenses_df['Month'],
                y=[expenses_mean]*len(expenses_df['Month']),
                mode='lines',
                line=dict(color=self.expenses_color, width=1.5, dash='dash'),
                name='Mean Expenses',
                hovertemplate='Mean Expenses: %{y}€',
                visible='legendonly'
            )
        )

        fig.update_layout(
            title='Income, Income+Support and Expenses<br><sup>Monthly Report</sup>',
            width=1980,
            height=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        months_list = sorted(list(set(dataset['Month'])))
        fig.update_xaxes(
            tickvals=months_list,  # list of all months
            tickmode='array',      # use provided tick values as coordinates
            tickformat="%b %Y"     # custom date format
        )

        if len(months_list) > 12:
            fig.update_xaxes(
                tickangle=-30,         # rotate labels
            )

        return fig


    def plot_monthly_delta(self, dataset):
        """
        Generate a plotly figure showing the monthly delta between income (or income+support) and expenses.

        Parameters
        ----------
        dataset : DataFrame
            The dataset that contains the transactions. Expected columns are
            - 'Transaction Type'
            - 'Category'
            - 'Amount'
            - 'Month'

        Returns
        -------
        fig : plotly.graph_objs._figure.Figure
            A plotly figure containing the generated bar plot.
        """

        # Prepare data
        income_df = dataset.loc[
            (dataset['Transaction Type'] == 'Reddito') & (dataset['Category'] != 'Supporto Famiglia')
        ]
        income_df = income_df.groupby(['Month']).agg({'Amount': 'sum'}).reset_index()

        income_and_support_df = dataset.loc[(dataset['Transaction Type'] == 'Reddito')]
        income_and_support_df = income_and_support_df.groupby(['Month']).agg({'Amount': 'sum'}).reset_index()

        expenses_df = dataset.loc[dataset['Transaction Type'] == 'Spesa']
        expenses_df = expenses_df.groupby(['Month']).agg({'Amount': 'sum'}).reset_index()

        # Calculate deltas
        delta_income = income_df['Amount'] - expenses_df['Amount'].reindex_like(income_df, method='ffill')
        delta_income_support = income_and_support_df['Amount'] - expenses_df['Amount'].reindex_like(income_and_support_df, method='ffill')

        # Moving Average Calculation
        window_size = 3  # You can adjust this window size
        income_df['Moving_Avg'] = delta_income.rolling(window=window_size).mean()
        income_and_support_df['Moving_Avg'] = delta_income_support.rolling(window=window_size).mean()

        # Define function to determine text color based on value
        def get_text_color(values):
            return ['green' if v >= 0 else 'red' for v in values]

        # Create plot
        fig = go.Figure()

        # Delta relative to Income
        fig.add_trace(
            go.Bar(
                x=income_df['Month'],
                y=delta_income,
                text=delta_income.apply(lambda x: f"{x:,.0f}€"),
                textposition='outside',
                marker_color=self.income_color,
                name='Delta (Income)',
                insidetextfont=dict(color=get_text_color(delta_income)),
                outsidetextfont=dict(color=get_text_color(delta_income))
            )
        )

        # Delta relative to Income + Support
        fig.add_trace(
            go.Bar(
                x=income_and_support_df['Month'],
                y=delta_income_support,
                text=delta_income_support.apply(lambda x: f"{x:,.0f}€"),
                textposition='outside',
                marker_color=self.income_and_support_color,
                name='Delta (Income + Support)',
                insidetextfont=dict(color=get_text_color(delta_income_support)),
                outsidetextfont=dict(color=get_text_color(delta_income_support))
            )
        )

        # # Adding Moving Average Trend Line for Income
        # fig.add_trace(
        #     go.Scatter(
        #         x=income_df['Month'],
        #         y=income_df['Moving_Avg'],
        #         mode='lines',
        #         line=dict(color=self.income_color, dash='dot'),
        #         name='Income Trend'
        #     )
        # )

        # # Adding Moving Average Trend Line for Income + Support
        # fig.add_trace(
        #     go.Scatter(
        #         x=income_and_support_df['Month'],
        #         y=income_and_support_df['Moving_Avg'],
        #         mode='lines',
        #         line=dict(color=self.income_and_support_color, dash='dot'),
        #         name='Income + Support Trend'
        #     )
        # )

        # Update layout
        fig.update_layout(
            title='Monthly Delta between Income (Income+Support) and Expenses',
            barmode='group',
            width=1980,
            height=800,
            xaxis_title='Month',
            yaxis_title='Delta Amount (€)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Update x-axis format for dates
        months_list = sorted(list(set(dataset['Month'])))
        fig.update_xaxes(
            tickvals=months_list,
            tickmode='array',
            tickformat="%b %Y"
        )

        # Rotate labels if there are more than 12 months
        if len(months_list) > 12:
            fig.update_xaxes(tickangle=-30)

        return fig


    def plot_lineplot_income_and_expenses_cum(self, dataset):
        """
        Generate a plotly figure showing cumulative monthly income, income+support and expenses.

        The function groups data by month, then generates a line chart showing the cumulative sum
        of the amount for income, income+support, and expenses per month.

        Parameters
        ----------
        dataset : DataFrame
            The dataset that contains the transactions. Expected columns are:
            - 'Transaction Type'
            - 'Category'
            - 'Amount'
            - 'Month'
            'Transaction Type' should contain values like 'Reddito' for income and 'Spesa' for expenses.
            'Month' should be in the format YYYY-MM-DD.

        Returns
        -------
        fig : plotly.graph_objs._figure.Figure
            A plotly figure containing the generated plot.

        Notes
        -----
        The function uses the self.category_color_dict_expenses constant which should be a dictionary
        mapping each category to a specific color in hexadecimal form.
        """

        # prepare data for dataviz
        dataset = dataset.copy()
        income_df = dataset.loc[
            (dataset['Transaction Type'] == 'Reddito') & (dataset['Category'] != 'Supporto Famiglia')
            ]
        income_df = income_df.groupby(['Month']).agg({'Amount': 'sum'}).reset_index()
        income_df['Amount Cumulative'] = income_df['Amount'].cumsum()
        income_df['Amount_label'] = round(income_df['Amount Cumulative']).astype(int).astype(str) + '€'

        income_and_support_df = dataset.loc[(dataset['Transaction Type'] == 'Reddito')]
        income_and_support_df = income_and_support_df.groupby(['Month']).agg({'Amount': 'sum'}).reset_index()
        income_and_support_df['Amount Cumulative'] = income_and_support_df['Amount'].cumsum()
        income_and_support_df['Amount_label'] = round(income_and_support_df['Amount Cumulative']).astype(int).astype(str) + '€'

        expenses_df = dataset.loc[dataset['Transaction Type'] == 'Spesa']
        expenses_df = expenses_df.groupby(['Month']).agg({'Amount': 'sum'}).reset_index()
        expenses_df['Amount Cumulative'] = expenses_df['Amount'].cumsum()
        expenses_df['Amount_label'] = round(expenses_df['Amount Cumulative']).astype(int).astype(str) + '€'

        # create plot
        fig = go.Figure()

        # Income
        fig.add_trace(
            go.Scatter(
                x=income_df['Month'],
                y=income_df['Amount Cumulative'],
                text=income_df['Amount_label'],
                mode='lines+markers+text',
                name='Income',
                textposition='top center',
                line=dict(color=self.income_color),
                hovertemplate='Cumulative Income up to <b>%{x}</b>: %{y}€',
                ))

        # Income + Support
        fig.add_trace(
            go.Scatter(
                x=income_and_support_df['Month'],
                y=income_and_support_df['Amount Cumulative'],
                text=income_and_support_df['Amount_label'],
                mode='lines+markers+text',
                name='Income + Support',
                textposition='top center',
                line=dict(color=self.income_and_support_color),
                hovertemplate='Cumulative Income+Support up to <b>%{x}</b>: %{y}€',
                ))

        # Expenses
        fig.add_trace(
            go.Scatter(
                x=expenses_df['Month'],
                y=expenses_df['Amount Cumulative'],
                text=expenses_df['Amount_label'],
                mode='lines+markers+text',
                name='Expenses',
                textposition='top center',
                line=dict(color=self.expenses_color),
                hovertemplate='Cumulative Expenses up to <b>%{x}</b>: %{y}€',
                ))

        fig.update_layout(
            title='Income, Income+Support and Expenses<br><sup>Cumulative Report</sup>',
            width=1980,
            height=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        months_list = sorted(list(set(dataset['Month'])))
        fig.update_xaxes(
            tickvals=months_list,  # list of all months
            tickmode='array',      # use provided tick values as coordinates
            tickformat="%b %Y"     # custom date format
        )

        if len(months_list) > 12:
            fig.update_xaxes(
                tickangle=-30,         # rotate labels
            )

        return fig


    def plot_income_by_month(self, dataset):
        """
        Generate a Plotly figure representing income by month, grouped by category.

        This function preprocesses the input DataFrame to extract income-relevant data.
        It then generates a stacked line plot, with one line for each category of income.
        The plot also includes information on total, mean, minimum, and maximum income.

        Parameters
        ----------
        dataset : pd.DataFrame
            The input DataFrame containing financial transaction data.
            It should include the following columns:
            - 'Transaction Type'
            - 'Month'
            - 'Category'
            - 'Amount'
            - 'Notes'

        Returns
        -------
        plotly.graph_objs._figure.Figure
            A Plotly figure representing the income by month, with lines colored by category.

        Notes
        -----
        The input DataFrame is expected to have been preprocessed such that it only contains
        income transactions ('Transaction Type' == 'Reddito').
        Any missing 'Notes' are filled with 'Non specificato'.
        """

        dataset = dataset.copy()

        expenses_df = dataset.loc[dataset['Transaction Type'] == 'Spesa']
        expenses_df = expenses_df.groupby(['Month']).agg({'Amount': 'sum'}).reset_index()
        expenses_df['Amount_label'] = expenses_df['Amount'].astype(str) + '€'

        dataset = dataset.loc[dataset['Transaction Type'] == 'Reddito']
        dataset['Notes'] = dataset['Notes'].fillna('Non specificato')
        dataset['Amount_label'] = dataset['Amount'].astype(str) + '€'
        dataset['Notes'] = ' • ' + dataset['Date'] + ': ' + dataset['Notes'] + ' → ' + dataset['Amount_label']
        dataset = dataset.groupby(['Month', 'Category']).agg({
            'Amount': 'sum',
            'Notes': lambda x: '\n<br>'.join(x)
        }).reset_index()
        dataset['Month'] = pd.to_datetime(dataset['Month'])

        # setup padding of x-axis
        x_range_padding = pd.DateOffset(days=10)  # change the value for more or less padding
        min_date = dataset['Month'].min() - x_range_padding
        max_date = dataset['Month'].max() + x_range_padding

        income_total = round(sum(dataset['Amount']))
        income_mean = round(np.mean(dataset.groupby(['Month']).agg({'Amount': 'sum'}).reset_index()['Amount']))
        income_min = round(min(dataset.groupby(['Month']).agg({'Amount': 'sum'}).reset_index()['Amount']))
        income_max = round(max(dataset.groupby(['Month']).agg({'Amount': 'sum'}).reset_index()['Amount']))
        subtitle = f"<br><sup>Total Income: {income_total}€, Min: {income_min}€, Max: {income_max}€, Mean: {income_mean}€</sup>"

        fig = go.Figure()


        # Ensure every category has an entry for every month
        months = list(set(dataset['Month']))

        for category, color in self.category_color_dict_expenses.items():

            df_category = dataset[dataset['Category'] == category]

            # Ensure every category has an entry for every month
            temp_months = list(df_category['Month'])
            if not df_category.empty:
                for month in months:
                    if month not in temp_months:
                        new_row = pd.DataFrame({
                            'Month': [month],
                            'Category': [category],
                            'Amount': [0],
                            'Notes': ['Nessuna Entrata']
                        })
                        df_category = pd.concat([df_category,new_row])
                df_category = df_category.sort_values(['Month'], ascending=True)

            # Do not display the value of "Amount" if it is 0
            text = [f"{int(round(amount))}" if amount > 100 else "" for amount in df_category['Amount']]

            fig.add_trace(go.Scatter(
                x=df_category['Month'],
                y=df_category['Amount'],
                mode='lines+markers+text',
                line=dict(width=0.5, color=color),
                stackgroup='one',
                name=category,
                text=text,
                textposition='top center',
                hovertext=df_category['Notes'],
                hovertemplate=f'<b>{category}</b>: ' + '%{y}€<br><br>%{hovertext}',
            ))

        # Add Expenses Line
        fig.add_trace(
            go.Scatter(
                x=expenses_df['Month'],
                y=expenses_df['Amount'],
                hovertext=expenses_df['Amount_label'],
                mode='lines+markers+text',
                name='Spese',
                textposition='top center',
                line=dict(color=self.expenses_color_translucent, dash='dot'),
                hovertemplate='Total Expenses of <b>%{x}</b>: %{y}€',
                ))

        fig.update_layout(
            title=f'Income by Month{subtitle}',
            width=1980,
            height=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # update the x-axis range with the padding
        fig.update_xaxes(range=[min_date, max_date])

        months_list = sorted(list(set(dataset['Month'])))
        fig.update_xaxes(
            tickvals=months_list,  # list of all months
            tickmode='array',      # use provided tick values as coordinates
            tickformat="%b %Y"     # custom date format
        )

        if len(months_list) > 12:
            fig.update_xaxes(
                tickangle=-30,         # rotate labels
            )

        return fig


    def plot_expenses_by_month(self, dataset):
        """
        Generate a plotly figure that shows expenses by month.

        The function groups data by month and category, and then generates a stacked
        line chart showing the sum of the amount for each category per month.
        The chart also includes additional information like total, min, max, and mean expenses.

        Parameters
        ----------
        dataset : DataFrame
            The dataset that contains the transactions.
            Expected columns are
            - 'Transaction Type'
            - 'Notes'
            - 'Category'
            - 'Amount'
            - 'Month'
            'Transaction Type' should contain the value 'Spesa' for expenses.
            'Month' should be in the format YYYY-MM-DD.
            Missing 'Notes' are filled with 'Non specificato'.

        Returns
        -------
        fig : plotly.graph_objs._figure.Figure
            A plotly figure containing the generated plot.

        Notes
        -----
        The function uses the self.category_color_dict_expenses constant which should be a dictionary
        mapping each category to a specific color in hexadecimal form.
        """

        dataset = dataset.copy()
        dataset = dataset.loc[dataset['Transaction Type'] == 'Spesa']
        dataset['Notes'] = dataset['Notes'].fillna('Non specificato')
        dataset['Amount_str'] = dataset['Amount'].astype(str)
        dataset['Notes'] = ' • ' + dataset['Date'] + ': ' + dataset['Notes'] + ' → ' + dataset['Amount_str'] + '€'

        dataset = dataset.groupby(['Month', 'Category']).agg({
            'Amount': 'sum',
            'Notes': lambda x: '\n<br>'.join(x)
        }).reset_index()
        dataset['Month'] = pd.to_datetime(dataset['Month'])

        # setup padding of x-axis
        x_range_padding = pd.DateOffset(days=10)  # change the value for more or less padding
        min_date = dataset['Month'].min() - x_range_padding
        max_date = dataset['Month'].max() + x_range_padding

        expenses_total = round(sum(dataset['Amount']))
        expenses_mean = round(np.mean(dataset.groupby(['Month']).agg({'Amount': 'sum'}).reset_index()['Amount']))
        expenses_min = round(min(dataset.groupby(['Month']).agg({'Amount': 'sum'}).reset_index()['Amount']))
        expenses_max = round(max(dataset.groupby(['Month']).agg({'Amount': 'sum'}).reset_index()['Amount']))
        subtitle = f"<br><sup>Total Expenses: {expenses_total}€, Min: {expenses_min}€, Max: {expenses_max}€, Mean: {expenses_mean}€</sup>"

        fig = go.Figure()

        # Ensure every category has an entry for every month
        months = list(set(dataset['Month']))

        for category, color in self.category_color_dict_expenses.items():

            df_category = dataset[dataset['Category'] == category]

            # Ensure every category has an entry for every month
            temp_months = list(df_category['Month'])
            if not df_category.empty:
                for month in months:
                    if month not in temp_months:
                        new_row = pd.DataFrame({
                            'Month': [month],
                            'Category': [category],
                            'Amount': [0],
                            'Notes': ['Nessuna Spesa']
                        })
                        df_category = pd.concat([df_category,new_row])
                df_category = df_category.sort_values(['Month'], ascending=True)

            fig.add_trace(go.Scatter(
                x=df_category['Month'],
                y=df_category['Amount'],
                mode='lines',
                line=dict(width=0.5, color=color),
                stackgroup='one',
                name=category,
                hovertext=df_category['Notes'],
                hovertemplate=f'<b>{category}</b>: ' + '%{y}€<br><br>%{hovertext}',
            ))

        fig.update_layout(
            title=f'Expenses by Month{subtitle}',
            width=1980,
            height=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # update the x-axis range with the padding
        fig.update_xaxes(range=[min_date, max_date])

        months_list = sorted(list(set(dataset['Month'])))
        fig.update_xaxes(
            tickvals=months_list,  # list of all months
            tickmode='array',      # use provided tick values as coordinates
            tickformat="%b %Y"     # custom date format
        )

        if len(months_list) > 12:
            fig.update_xaxes(
                tickangle=-30,         # rotate labels
            )

        return fig


    def plot_income_and_expenses_by_month(self, dataset):
        """
        TODO: write
        """

        # preprocess dataset

        income_df = dataset.copy()
        income_df = income_df.loc[income_df['Transaction Type'] == 'Reddito']
        income_df['Notes'] = income_df['Notes'].fillna('Non specificato')
        income_df['Amount_str'] = income_df['Amount'].astype(str)
        income_df['Notes'] = ' • ' + income_df['Date'] + ': ' + income_df['Notes'] + ' → ' + income_df['Amount_str'] + '€'

        income_df = income_df.groupby(['Month', 'Category']).agg(
            {
                'Amount': 'sum',
                'Notes': lambda x: '\n<br>'.join(x)

            }
        ).reset_index()
        income_df['Amount'] = round(income_df['Amount']).astype(int)

        expenses_df = dataset.copy()
        expenses_df = expenses_df.loc[expenses_df['Transaction Type'] == 'Spesa']
        expenses_df['Notes'] = expenses_df['Notes'].fillna('Non specificato')
        expenses_df['Amount_str'] = expenses_df['Amount'].astype(str)
        expenses_df['Notes'] = ' • ' + expenses_df['Date'] + ': ' + expenses_df['Notes'] + ' → ' + expenses_df['Amount_str'] + '€'
        expenses_df = expenses_df.groupby(['Month', 'Category']).agg(
            {
                'Amount': 'sum',
                'Notes': lambda x: '\n<br>'.join(x)
            }
        ).reset_index()
        expenses_df['Amount'] = round(expenses_df['Amount']).astype(int)

        # create dataviz

        # get unique months for the slider steps
        income_months = income_df['Month'].unique()
        expenses_months = expenses_df['Month'].unique()

        # create empty figure
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=("", ""),
            specs=[[{'type':'pie'}, {'type':'pie'}]]
            )

        # loop over each month and create a pie chart
        for month in income_months:

            # income
            df_month_income = income_df[income_df['Month'] == month]
            fig.add_trace(
                go.Pie(
                    labels=df_month_income['Category'],
                    values=df_month_income['Amount'],
                    visible=False,
                    name=month,
                    hole=0.4,
                    marker=dict(
                        colors=[self.category_color_dict_expenses[cat] for cat in df_month_income['Category']]
                        ),
                    textinfo='label+value',
                    texttemplate='<b>%{label}</b><br>%{value}€',
                    hovertext=df_month_income['Notes'],
                    hovertemplate='<b>%{label}</b>: %{value}€ <br><br>%{hovertext}',
                    automargin=False,
                    opacity=1
                    #domain=dict(x=[0.45, 0.45], y=[0.45, 0.45]) # restricts the plot to the middle X% of the area
                ), 1, 1
            )

            # expenses
            df_month_expenses = expenses_df[expenses_df['Month'] == month]
            fig.add_trace(
                go.Pie(
                    labels=df_month_expenses['Category'],
                    values=df_month_expenses['Amount'],
                    visible=False,
                    name=month,
                    hole=0.4,
                    marker=dict(
                        colors=[self.category_color_dict_expenses[cat] for cat in df_month_expenses['Category']]
                        ),
                    textinfo='label+value',
                    texttemplate='<b>%{label}</b> %{value}€',
                    hovertext=df_month_expenses['Notes'],
                    hovertemplate='<b>%{label}</b>: %{value}€ <br><br>%{hovertext}',
                    automargin=False,
                    opacity=1
                    #domain=dict(x=[0.45, 0.45], y=[0.45, 0.45]) # restricts the plot to the middle X% of the area
                ), 1, 2
            )

        # make first trace visible
        fig.data[0].visible = True
        fig.data[1].visible = True

        # create and add slider
        steps = []
        for i in list(range(0, len(income_months))):

            # dynamic subtitle for info on total income and expenses of the month
            df_temp = income_df[income_df['Month'] == income_months[i]]
            tot_income = int(round(sum(df_temp['Amount'])))
            df_temp = expenses_df[expenses_df['Month'] == expenses_months[i]]
            tot_expenses = int(round(sum(df_temp['Amount'])))
            profit = tot_income - tot_expenses
            profit_perc = round((profit / tot_income) * 100, 2)

            # determine the color and sign of the profit value
            if profit >= 0:
                profit_str = f'<span style="color: {self.income_color};">{profit}€</span>'
            else:
                profit_str = f'<span style="color: {self.expenses_color};">{profit}€</span>'

            subtitle = f"<br><sup><br>Monthly Income: <b>{tot_income}€</b><br>Monthly Expenses: <b>{tot_expenses}€</b><br>Profit: <b>{profit_str}</b> ({profit_perc}%)</sup>"

            custom_label = pd.to_datetime(expenses_months[i]).strftime("%b %Y")
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)}, {"title": f"Income and Expenses by Month{subtitle}"}],  # layout attribute
                label=custom_label, # set the name of each month
            )
            idx_1 = 2*i
            idx_2 = 2*i + 1
            step["args"][0]["visible"][idx_1] = True
            step["args"][0]["visible"][idx_2] = True
            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Month: "},
            pad={"t": 90, "b":0},
            steps=steps
        )]

        # dynamic subtitle for info on total income and expenses of first month
        df_temp = income_df[income_df['Month'] == income_months[0]]
        tot_income = int(round(sum(df_temp['Amount'])))
        df_temp = expenses_df[expenses_df['Month'] == expenses_months[0]]
        tot_expenses = int(round(sum(df_temp['Amount'])))
        profit = tot_income - tot_expenses
        profit_perc = round((profit / tot_income) * 100, 2)

        # determine the color and sign of the profit value
        if profit >= 0:
            profit_str = f'<span style="color: {self.income_color};">{profit}€</span>'
        else:
            profit_str = f'<span style="color: {self.expenses_color};">{profit}€</span>'

        subtitle = f"<br><sup><br>Monthly Income: <b>{tot_income}€</b><br>Monthly Expenses: <b>{tot_expenses}€</b><br>Profit: <b>{profit_str}</b> ({profit_perc}%)</sup>"

        # add title, width, legend, ...
        fig.update_layout(
            sliders=sliders,
            title=f"Income and Expenses by Month{subtitle}",
            width=1980,
            height=800,
            showlegend=False
        )

        return fig


    def plot_barplot_transfers(self, dataset):
        """
        TODO: update
        """

        # Convert 'Month' to datetime
        dataset['Month'] = pd.to_datetime(dataset['Month'])

        # Fix notes
        dataset['Notes'] = dataset['Notes'].fillna('-')

        # Create a bar plot using Graph Objects
        fig = go.Figure()

        # Define a consistent text font size
        text_font_size = 12  # You can adjust this size as needed

        # Calculate total transfers for each 'To' category
        total_transfers = dataset.groupby('To')['Amount'].sum()

        # Subtitle text with total transfers
        subtitle_text = '<br><sup><br><b>Total Transfers</b><br>' + ', '.join([f'{to}: {amount:.2f} €' for to, amount in total_transfers.items()]) + '</sup>'

        # Add bars for each 'To' category
        categories = dataset['To'].unique()
        for category in categories:
            filtered_dataset = dataset[dataset['To'] == category]
            fig.add_trace(go.Bar(
                x=filtered_dataset['Month'],
                y=filtered_dataset['Amount'],
                name=f"<i>{category}</i><br>Total: {total_transfers[category]:.2f} €",  # Update legend name with total
                text=filtered_dataset['Amount'],
                textposition='inside',
                textfont=dict(size=text_font_size),
                hovertemplate="<b>Date:</b> %{x|%Y-%m}<br>" +
                            "<b>From:</b> %{customdata[0]}<br>" +
                            "<b>To:</b> %{customdata[1]}<br>" +
                            "<b>Amount:</b> %{y} €<br>" +
                            "<b>Notes:</b> %{customdata[2]}<extra></extra>",
                customdata=filtered_dataset[['From', 'To', 'Notes']].values,
                marker=dict(color=self.category_color_dict_transfers[category])  # Set custom color for each category
            ))

        # Customize the layout
        fig.update_layout(
            title=f'Money Transfers from Primary Account{subtitle_text}',
            xaxis_title='Date',
            yaxis_title='Amount (€)',
            barmode='stack',
            margin=dict(t=160),
            width=1980,
            height=800,
            legend=dict(
                orientation="v",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Setup ticks
        year = dataset['Month'].min().year
        months_list = [datetime(year, month, 1).strftime('%Y-%m') for month in range(1, 13)]
        fig.update_xaxes(
            tickvals=months_list,  # list of all months
            tickmode='array',      # use provided tick values as coordinates
            tickformat="%b %Y"     # custom date format
        )

        if len(months_list) > 12:
            fig.update_xaxes(
                tickangle=-30,         # rotate labels
            )

        return fig


    # UTILITY FUNCTIONS

    def __handle_datasets(self, data_dict):

        available_expenses_datasets = data_dict['expenses']
        self.datasets_expenses['all'] = pd.concat(available_expenses_datasets)

        available_transfers_datasets = data_dict['transfers']
        self.datasets_transfers['all'] = pd.concat(available_transfers_datasets)

        self.reports['all'] = []
        for dataset in available_expenses_datasets:
            if not dataset.empty:
                year = str(dataset['Year'].iloc[0])
                self.datasets_expenses[year] = dataset
                self.reports[year] = []

        for dataset in available_transfers_datasets:
            if not dataset.empty:
                year = str(dataset['Year'].iloc[0])
                self.datasets_transfers[year] = dataset

        years = sorted(self.datasets_expenses.keys())
        self.min_year = years[0]
        self.max_year = years[-2]


    def __handle_colors(self, category_color_dict):

        # * General Logic *

        # setup allowed colors and fixed colors
        self.palette_colors = [
            'rgba(241, 243, 206, 0.8)',
            'rgba(245, 228, 102, 0.8)',
            'rgba(242, 206, 203, 0.8)',
            'rgba(224, 168, 144, 0.8)',
            'rgba(140,  57,  57, 0.8)',
            'rgba(232, 156, 232, 0.8)',
            'rgba(194, 133, 255, 0.8)',
            'rgba(147, 129, 255, 0.8)',
            'rgba(208, 254, 245, 0.8)',
            'rgba(129, 210, 199, 0.8)',
            'rgba( 73, 202, 253, 0.8)',
            'rgba(100, 141, 229, 0.8)',
            'rgba(140, 160, 215, 0.8)',
            'rgba( 65, 103, 136, 0.8)',
            'rgba(237, 235, 242, 0.8)'
        ]

        self.expenses_color = 'rgba(204, 122, 171, 1)'
        self.expenses_color_translucent = 'rgba(204, 122, 171, 0.8)'
        self.income_color = 'rgba(138, 204, 122, 1)'
        self.income_color_translucent = 'rgba(138, 204, 122, 0.8)'
        self.income_and_support_color = 'rgba(100, 141, 229, 1)'
        self.income_and_support_color_translucent = 'rgba(100, 141, 229, 0.8)'

        # * Expenses Logic *

        # setup color palette
        self.category_color_dict_expenses = category_color_dict
        if self.category_color_dict_expenses is None:
            self.category_color_dict_expenses = {}

        # find categories with no color assigned
        unique_categories_expenses = self.__get_unique_categories_expenses()
        already_assigned_categories = set(self.category_color_dict_expenses.keys())
        unique_categories_expenses = sorted(list(unique_categories_expenses - already_assigned_categories))

        # obtain colors and assign them
        colors_to_assign_expenses = self.__get_distant_colors(n_colors=len(unique_categories_expenses))
        for idx,category in enumerate(unique_categories_expenses):
            self.category_color_dict_expenses[category] = colors_to_assign_expenses[idx]

        # finally, sort the dict by key (for better datavizs)
        self.category_color_dict_expenses = dict(sorted(self.category_color_dict_expenses.items()))

        # * Transfers Logic *

        # setup color palette
        self.category_color_dict_transfers = {}

        # find categories with no color assigned
        unique_categories_transfers = self.__get_unique_categories_transfers()
        already_assigned_categories = set(self.category_color_dict_transfers.keys())
        unique_categories_transfers = sorted(list(unique_categories_transfers - already_assigned_categories))

        # obtain colors and assign them
        colors_to_assign_transfers = self.__get_distant_colors(n_colors=len(unique_categories_transfers))
        for idx,category in enumerate(unique_categories_transfers):
            self.category_color_dict_transfers[category] = colors_to_assign_transfers[idx]

        # finally, sort the dict by key (for better datavizs)
        self.category_color_dict_transfers = dict(sorted(self.category_color_dict_transfers.items()))


    def __get_unique_categories_expenses(self):

        list_of_datasets = [dataset for dataset in self.datasets_expenses.values()]
        complete_dataset = pd.concat(list_of_datasets)
        unique_categories = set(complete_dataset['Category'].unique())
        return unique_categories


    def __get_unique_categories_transfers(self):

        list_of_datasets = [dataset for dataset in self.datasets_transfers.values()]
        complete_dataset = pd.concat(list_of_datasets)
        unique_categories = set(complete_dataset['To'].unique())
        return unique_categories


    def __get_distant_colors(self, n_colors):

        if n_colors <= 0 or n_colors > len(self.palette_colors):
            self.palette_colors = self.palette_colors + self.palette_colors
            return self.__get_distant_colors(n_colors)

        if n_colors == 1:
            return [self.palette_colors[5]]

        step = len(self.palette_colors) // (n_colors - 1)

        return [self.palette_colors[min(i * step, len(self.palette_colors)-1)] for i in range(n_colors)]


    def __save_list_of_plotly_figs(self, path, fig_list, title="My Report"):
        """
        Saves a list of Plotly figures to a single HTML file.

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

            icon_path = str(os.path.join(PACKAGE_DIR, '..', 'images', 'chart.svg'))
            gif_path = str(os.path.join(PACKAGE_DIR, '..', 'images', 'finance-cards.webp'))

            with open(path, "w", encoding="utf-8") as output_file:
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
                        <img src={gif_path} alt="Finance Cards" width="125">
                    """
                )

            with open(path, "a", encoding="utf-8") as output_file:
                output_file.write(
                    fig_list[0].to_html(full_html=False, include_plotlyjs="cdn")
                )

            with open(path, "a", encoding="utf-8") as output_file:
                for fig in fig_list[1:]:
                    output_file.write(fig.to_html(
                        full_html=False,
                        include_plotlyjs="cdn"
                    ))

            with open(path, "a", encoding="utf-8") as output_file:
                output_file.write("</div>") # Close the div after all figures have been written.

            print(f"The plots were correctly saved in '{path}'")
        except Exception as exc:
            print(f"There were errors in saving the plot. Details: {exc}")
