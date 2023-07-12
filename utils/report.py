"""
Dataviz Utils
"""

# Custom Pylint rules for the file
# pylint: disable=W0718 C0301
# W0718:broad-exception-caught
# C0301:line-too-long

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# global variables

PASTEL_COLORS = [
    '#cc7a7a',
    '#cc8a7a',
    '#cc9b7a',
    '#ccab7a',
    '#cccc7a',
    '#bbcc7a',
    '#abcc7a',
    '#9bcc7a',
    '#7acc7a',
    '#7acc8a',
    '#7acc9b',
    '#7accab',
    '#7accbb',
    '#7acccc',
    '#7abbcc',
    '#7aabcc',
    '#7a8acc',
    '#7a7acc',
    '#8a7acc',
    '#ab7acc',
    '#bb7acc',
    '#cc7acc',
    '#cc7abb',
    '#cc7a9b',
    '#cc7a8a'
    ]


class ReportGenerator:
    """
    ReportGenerator Class
    """
    def __init__(self, datasets_path, save_path, category_color_dict=None):

        self.datasets = {}
        self.report_generators = {}
        self.reports = {}
        self.save_path = save_path

        # iterate over available years
        available_dfs = os.listdir(datasets_path)
        for df_path in available_dfs:
            year = df_path.split('.')[0][-4:]
            self.datasets[year] = pd.read_csv(os.path.join(datasets_path,df_path))
            self.reports[year] = []

        # setup color palette
        self.category_color_dict = category_color_dict
        if self.category_color_dict is None:
            self.category_color_dict = {}

        # find categories with no color assigned
        unique_categories = self.__get_unique_categories()
        already_assigned_categories = set(self.category_color_dict.keys())
        unique_categories = unique_categories - already_assigned_categories

        # obtain colors and assign them
        colors_to_assign = self.__get_distant_colors(n_colors=len(unique_categories))
        for idx,category in enumerate(unique_categories):
            self.category_color_dict[category] = colors_to_assign[idx]

        # finally, sort the dict by key (for better datavizs)
        self.category_color_dict = dict(sorted(self.category_color_dict.items()))

        print("Report Generator Ready")


    # MAIN FUNCTION
    def generate_reports(self, show_plots=False):
        """
        TODO: generate docstring
        """

        # for each year
        for key,value in self.datasets.items():

            # * YEARLY SPIDERPLOT *
            # Spyderplot of Yearly Total Income and Expenses
            plot = self.plot_yearly_spyderplot(dataset=value)
            self.reports[key].append(plot)
            if show_plots:
                plot.show()

            # * LINEPLOT OF INCOME AND EXPENSES *
            plot = self.plot_lineplot_income_and_expenses(dataset=value)
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

        # * SAVE REPORTS *
        for key,value in self.reports.items():

            report_name = f'report_{key}.html'
            report_path = os.path.join(self.save_path, report_name)
            self.__save_list_of_plotly_figs(
                path=report_path,
                fig_list=value,
                title=f'Financial Report {key}'
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
        dataset_income['Amount_label'] = np.where(
            dataset_income['Amount'] < 1000,
            "",
            round(dataset_income['Amount']).astype(str) + '€'
            )

        dataset_expenses = dataset.copy()
        dataset_expenses = dataset_expenses.loc[dataset_expenses['Transaction Type'] == 'Spesa']
        dataset_expenses = dataset_expenses.groupby(['Category']).agg({
                'Amount': 'sum'
            }).reset_index()
        dataset_expenses['Amount'] = round(dataset_expenses['Amount']).astype(int)
        dataset_expenses['Amount_label'] = np.where(
            dataset_expenses['Amount'] < 1000,
            "",
            round(dataset_expenses['Amount']).astype(str) + '€'
            )

        #fig = go.Figure()
        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'polar'}] * 2] * 1)

        # additional info
        income_total = round(sum(dataset_income['Amount']))
        expenses_total = round(sum(dataset_expenses['Amount']))
        subtitle = f"<br><sup><br>Total Income: {income_total}€<br>Total Expenses: {expenses_total}€</sup>"

        fig.add_trace(
            go.Scatterpolar(
                r=list(dataset_income['Amount']),
                theta=list(dataset_income['Category']),
                mode='markers+text',
                name='markers',
                #text=list(dataset_income['Amount_label']),
                #textposition='middle right',
                fill='toself',
                hoverinfo='r',
                hovertemplate='Income by %{theta}: %{r} (€)',
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatterpolar(
                r=list(dataset_expenses['Amount']),
                theta=list(dataset_expenses['Category']),
                mode='markers+text',
                name='markers',
                #text=list(dataset_expenses['Amount_label']),
                #textposition='middle right',
                fill='toself',
                hoverinfo='r',
                hovertemplate='Expenses by %{theta}: %{r} (€)',
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
                        size=15
                    )
                )
            ),
            polar2=dict(
                radialaxis=dict(
                    visible=True
                ),
                angularaxis=dict(
                    tickfont=dict(
                        size=15
                    )
                )
            ),
            showlegend=False,
            width=1980,
            height=600,
            title=f"Yearly Income and Expenses by Category{subtitle}"
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
        The function uses the self.category_color_dict constant which should be a dictionary
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
                line=dict(color=self.category_color_dict['Stipendio']),
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
                line=dict(color=self.category_color_dict['Supporto Famiglia']),
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
                line=dict(color='#cc7aab'),
                hovertemplate='Expenses of %{x}: %{y}€',
                ))
        
        # add mean line for Income
        fig.add_trace(
            go.Scatter(
                x=income_df['Month'],
                y=[income_mean]*len(income_df['Month']),
                mode='lines',
                line=dict(color=self.category_color_dict['Stipendio'], width=1.5, dash='dash'),
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
                line=dict(color=self.category_color_dict['Supporto Famiglia'], width=1.5, dash='dash'),
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
                line=dict(color='#cc7aab', width=1.5, dash='dash'),
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
        The function uses the self.category_color_dict constant which should be a dictionary
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
                line=dict(color=self.category_color_dict['Stipendio']),
                hovertemplate='Cumulative Income up to %{x}: %{y}€',
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
                line=dict(color=self.category_color_dict['Supporto Famiglia']),
                hovertemplate='Cumulative Income+Support up to %{x}: %{y}€',
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
                line=dict(color='#cc7aab'),
                hovertemplate='Cumulative Expenses up to %{x}: %{y}€',
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
        expenses_df['Amount_label'] = round(expenses_df['Amount']).astype(int).astype(str) + '€'

        dataset = dataset.loc[dataset['Transaction Type'] == 'Reddito']
        dataset['Notes'] = dataset['Notes'].fillna('Non specificato')
        dataset['Notes'] = ' • ' + dataset['Date'] + ': ' + dataset['Notes']
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

        for category, color in self.category_color_dict.items():
            df_category = dataset[dataset['Category'] == category]
            fig.add_trace(go.Scatter(
                x=df_category['Month'],
                y=df_category['Amount'],
                mode='lines+markers+text',
                line=dict(width=0.5, color=color),
                stackgroup='one',
                name=category,
                text=df_category['Amount'],  # Text to display
                textposition='top center',  # Position of the text
                hovertext=df_category['Notes'],
                hovertemplate='Total Income: %{y}€<br><br>%{hovertext}',
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
                line=dict(color='rgba(255, 100, 70, 0.8)', dash='dot'),
                hovertemplate='Total Expenses: %{y}€',
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
        The function uses the self.category_color_dict constant which should be a dictionary
        mapping each category to a specific color in hexadecimal form.
        """

        dataset = dataset.copy()
        dataset = dataset.loc[dataset['Transaction Type'] == 'Spesa']
        dataset['Notes'] = dataset['Notes'].fillna('Non specificato')
        dataset['Notes'] = ' • ' + dataset['Date'] + ': ' + dataset['Notes']
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

        for category, color in self.category_color_dict.items():
            df_category = dataset[dataset['Category'] == category]
            fig.add_trace(go.Scatter(
                x=df_category['Month'],
                y=df_category['Amount'],
                mode='lines',
                line=dict(width=0.5, color=color),
                stackgroup='one',
                name=category,
                #text=df_category['Amount'],  # Text to display
                #textposition='top center',  # Position of the text
                hovertext=df_category['Notes'],
                hovertemplate='Total Expenses: %{y}€<br><br>%{hovertext}',
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

        return fig


    def plot_income_and_expenses_by_month(self, dataset):
        """
        TODO: write
        """

        # preprocess dataset

        income_df = dataset.copy()
        income_df = income_df.loc[income_df['Transaction Type'] == 'Reddito']
        income_df['Notes'] = income_df['Notes'].fillna('Non specificato')
        income_df['Notes'] = ' • ' + income_df['Date'] + ': ' + income_df['Notes']
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
        expenses_df['Notes'] = ' • ' + expenses_df['Date'] + ': ' + expenses_df['Notes']
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
                        colors=[self.category_color_dict[cat] for cat in df_month_income['Category']]
                        ),
                    textinfo='label+value',
                    hovertext=df_month_income['Notes'],
                    hovertemplate='%{label}: %{value}€ <br><br>%{hovertext}',
                    automargin=False,
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
                        colors=[self.category_color_dict[cat] for cat in df_month_expenses['Category']]
                        ),
                    textinfo='label+value',
                    hovertext=df_month_expenses['Notes'],
                    hovertemplate='%{label}: %{value}€ <br><br>%{hovertext}',
                    automargin=False,
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
            subtitle = f"<br><sup><br>Monthly Income: {tot_income}€<br>Monthly Expenses: {tot_expenses}€</sup>"

            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)}, {"title": f"Income and Expenses by Month{subtitle}"}],  # layout attribute
                label=expenses_months[i], # set the name of each month
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
        subtitle = f"<br><sup><br>Monthly Income: {tot_income}€<br>Monthly Expenses: {tot_expenses}€</sup>"

        # add title, width, legend, ...
        fig.update_layout(
            sliders=sliders,
            title=f"Income and Expenses by Month{subtitle}",
            width=1980,
            height=800,
            showlegend=False
        )

        return fig


    # UTILITY FUNCTIONS

    def __get_unique_categories(self):

        list_of_datasets = [dataset for dataset in self.datasets.values()]
        complete_dataset = pd.concat(list_of_datasets)
        unique_categories = set(complete_dataset['Category'].unique())
        return unique_categories


    def __get_distant_colors(self, n_colors):

        if n_colors <= 0 or n_colors > len(PASTEL_COLORS):
            raise ValueError(f"n must be between 1 and {len(PASTEL_COLORS)}")

        step = len(PASTEL_COLORS) // (n_colors - 1)

        return [PASTEL_COLORS[min(i * step, len(PASTEL_COLORS)-1)] for i in range(n_colors)]


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
            with open(path, "w", encoding="utf-8") as output_file:
                output_file.write(
                    f"""
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
                    <div class='centered'>
                        <br><h1>{title}</h1><br>
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
