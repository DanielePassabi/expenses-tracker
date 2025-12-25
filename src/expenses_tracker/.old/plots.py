"""ReportGenerator."""

# ⚙️ Ruff Settings

# Libraries

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Custom functions

# Settings
pd.set_option('mode.chained_assignment', None)


def plot_yearly_spyderplot(self, dataset):
    """
    Generate a Plotly figure representing yearly income, grouped by category, as a spyder plot.

    This function preprocesses the input DataFrame to extract income-relevant data.
    It then generates a spyder (radar) plot, showing the total amount of income for each
    category.

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
    dataset_income = dataset_income.loc[dataset_income['Transaction Type'] == 'Entrate']
    dataset_income = dataset_income.groupby(['Category']).agg({'Amount': 'sum'}).reset_index()
    dataset_income['Amount'] = round(dataset_income['Amount']).astype(int)

    dataset_expenses = dataset.copy()
    dataset_expenses = dataset_expenses.loc[dataset_expenses['Transaction Type'] == 'Spesa']
    dataset_expenses = dataset_expenses.groupby(['Category']).agg({'Amount': 'sum'}).reset_index()
    dataset_expenses['Amount'] = round(dataset_expenses['Amount']).astype(int)

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'polar'}] * 2] * 1)

    # additional info
    income_total = round(sum(dataset_income['Amount']))
    expenses_total = round(sum(dataset_expenses['Amount']))
    profit = income_total - expenses_total

    # handle case in which income_total is 0
    if income_total == 0:
        income_total = 1
    profit_perc = round((profit / income_total) * 100, 2)

    # determine the color and sign of the profit value
    if profit >= 0:
        profit_str = f'<span style="color: {self.income_color};">{profit}€</span>'
    else:
        profit_str = f'<span style="color: {self.expenses_color};">{profit}€</span>'

    subtitle = (
        f'<br><sub>Total Income: <b>{income_total}€</b><br>Total Expenses: '
        f'<b>{expenses_total}€</b><br>Profit: <b>{profit_str}</b> ({profit_perc}%)</sub>'
    )

    theta_with_amount = [
        f'<b>{category}</b><br>{amount}€'
        for category, amount in zip(
            list(dataset_income['Category']), list(dataset_income['Amount'])
        )
    ]
    fig.add_trace(
        go.Scatterpolar(
            r=list(dataset_income['Amount']),
            theta=theta_with_amount,  # list(dataset_income['Category']),
            mode='markers+text',
            name='Income',
            fill='toself',
            hoverinfo='r',
            hovertemplate='Income by %{theta}',
            line={'color': self.income_color},
        ),
        row=1,
        col=1,
    )

    theta_with_amount = [
        f'<b>{category}</b><br>{amount}€'
        for category, amount in zip(
            list(dataset_expenses['Category']), list(dataset_expenses['Amount'])
        )
    ]
    fig.add_trace(
        go.Scatterpolar(
            r=list(dataset_expenses['Amount']),
            theta=theta_with_amount,
            mode='markers+text',
            name='Expenses',
            fill='toself',
            hoverinfo='r',
            hovertemplate='Expenses by %{theta}',
            line={'color': self.expenses_color},
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        polar1={
            'radialaxis': {'visible': True},
            'angularaxis': {'tickfont': {'size': 12}},
        },
        polar2={
            'radialaxis': {'visible': True},
            'angularaxis': {'tickfont': {'size': 12}},
        },
        showlegend=False,
        width=1980,
        height=600,
        title=f'Yearly Income and Expenses by Category{subtitle}',
        margin={'t': 160},  # Increase top margin for padding
    )

    return fig


def plot_yearly_treemap(self, dataset):
    """
    Generate figures representing yearly income and expenses, grouped by category, as treemaps.

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
    dataset_income = dataset.loc[dataset['Transaction Type'] == 'Entrate']
    dataset_income['Notes'] = dataset_income['Notes'].fillna('Non specificato')
    dataset_income['Amount_label'] = dataset_income['Amount'].astype(str) + '€'
    dataset_income['Notes'] = (
        ' • '
        + dataset_income['Date']
        + ': '
        + dataset_income['Notes']
        + ' → '
        + dataset_income['Amount_label']
    )
    dataset_income = (
        dataset_income.groupby(['Category'])
        .agg({'Amount': 'sum', 'Notes': lambda x: '\n<br>'.join(x)})
        .reset_index()
    )
    dataset_income['Amount'] = round(dataset_income['Amount']).astype(int)

    # Expenses
    dataset_expenses = dataset.loc[dataset['Transaction Type'] == 'Spesa']
    dataset_expenses['Notes'] = dataset_expenses['Notes'].fillna('Non specificato')
    dataset_expenses['Amount_label'] = dataset_expenses['Amount'].astype(str) + '€'
    dataset_expenses['Notes'] = (
        ' • '
        + dataset_expenses['Date']
        + ': '
        + dataset_expenses['Notes']
        + ' → '
        + dataset_expenses['Amount_label']
    )
    dataset_expenses = (
        dataset_expenses.groupby(['Category'])
        .agg({'Amount': 'sum', 'Notes': lambda x: '\n<br>'.join(x)})
        .reset_index()
    )
    dataset_expenses['Amount'] = round(dataset_expenses['Amount']).astype(int)

    # Function to get colors for categories
    def get_colors(data):
        return [
            self.category_color_dict_expenses.get(category, '#FFFFFF')
            for category in data['Category']
        ]

    # Create figures for income and expenses treemap
    fig_income = go.Figure(
        go.Treemap(
            labels=dataset_income['Category'],
            parents=[''] * len(dataset_income),
            values=dataset_income['Amount'],
            textinfo='label+text',
            texttemplate='<b>%{label}</b><br>%{value} €',
            textposition='middle center',
            marker_colors=get_colors(dataset_income),  # Set colors
            domain={'x': [0, 0.48], 'y': [0, 1]},  # Set domain for left side
            hoverinfo='none',  # Disable hover effect
        )
    )

    fig_expenses = go.Figure(
        go.Treemap(
            labels=dataset_expenses['Category'],
            parents=[''] * len(dataset_expenses),
            values=dataset_expenses['Amount'],
            textinfo='label+text',
            texttemplate='<b>%{label}</b><br>%{value} €',
            textposition='middle center',
            marker_colors=get_colors(dataset_expenses),  # Set colors
            domain={'x': [0.52, 1], 'y': [0, 1]},  # Set domain for right side
            hoverinfo='none',  # Disable hover effect
        )
    )

    # Create a single figure to display both treemaps side by side
    fig = go.Figure(data=[fig_income.data[0], fig_expenses.data[0]])

    # Update layout
    fig.update_layout(
        title='',
        grid={'columns': 2, 'rows': 1},
        width=1980,
        height=600,
        margin={'t': 0},
    )

    return fig


def plot_income_and_expenses_by_month(self, dataset):
    """
    Generate a fig. with dual pie charts representing income and expenses categorized by month.

    This function preprocesses the given dataset to separate income and expenses, then creates a
    pie chart for each month. Each pie chart shows the distribution of income or expenses across
    different categories.

    The function also calculates and displays the total income, expenses, and profit for each
    month. A slider is added to the plot to navigate through different months.

    Parameters
    ----------
    dataset : DataFrame
        The dataset containing transaction data.
        Expected columns include 'Transaction Type', 'Category', 'Amount', 'Month', 'Date', and
        'Notes'.

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        A plotly figure containing the dual pie charts for income and expenses by month.

    Notes
    -----
    The pie charts are interactive, allowing the user to hover over sections to see detailed
    notes about each category. The slider at the bottom of the plot facilitates the navigation
    through different months, updating the pie charts and the displayed totals and profit
    percentage for the selected month.

    Examples
    --------
    >>> dataset = pd.read_csv('financial_data.csv')
    >>> fig = plot_income_and_expenses_by_month(dataset)
    >>> fig.show()
    """
    # preprocess dataset

    income_df = dataset.copy()
    income_df = income_df.loc[income_df['Transaction Type'] == 'Entrate']
    income_df['Notes'] = income_df['Notes'].fillna('Non specificato')
    income_df['Amount_str'] = income_df['Amount'].astype(str)
    income_df['Notes'] = (
        ' • '
        + income_df['Date']
        + ': '
        + income_df['Notes']
        + ' → '
        + income_df['Amount_str']
        + '€'
    )

    income_df = (
        income_df.groupby(['Month', 'Category'])
        .agg({'Amount': 'sum', 'Notes': lambda x: '\n<br>'.join(x)})
        .reset_index()
    )
    income_df['Amount'] = round(income_df['Amount']).astype(int)

    expenses_df = dataset.copy()
    expenses_df = expenses_df.loc[expenses_df['Transaction Type'] == 'Spesa']
    expenses_df['Notes'] = expenses_df['Notes'].fillna('Non specificato')
    expenses_df['Amount_str'] = expenses_df['Amount'].astype(str)
    expenses_df['Notes'] = (
        ' • '
        + expenses_df['Date']
        + ': '
        + expenses_df['Notes']
        + ' → '
        + expenses_df['Amount_str']
        + '€'
    )
    expenses_df = (
        expenses_df.groupby(['Month', 'Category'])
        .agg({'Amount': 'sum', 'Notes': lambda x: '\n<br>'.join(x)})
        .reset_index()
    )
    expenses_df['Amount'] = round(expenses_df['Amount']).astype(int)

    # create dataviz

    # get unique months for the slider steps
    income_months = income_df['Month'].unique()
    expenses_months = expenses_df['Month'].unique()

    # create empty figure
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=('', ''), specs=[[{'type': 'pie'}, {'type': 'pie'}]]
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
                    colors=[
                        self.category_color_dict_expenses[cat]
                        for cat in df_month_income['Category']
                    ]
                ),
                textinfo='label+value',
                texttemplate='<b>%{label}</b><br>%{value}€',
                hovertext=df_month_income['Notes'],
                hovertemplate='<b>%{label}</b>: %{value}€ <br><br>%{hovertext}',
                automargin=False,
                opacity=1,
            ),
            1,
            1,
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
                    colors=[
                        self.category_color_dict_expenses[cat]
                        for cat in df_month_expenses['Category']
                    ]
                ),
                textinfo='label+value',
                texttemplate='<b>%{label}</b> %{value}€',
                hovertext=df_month_expenses['Notes'],
                hovertemplate='<b>%{label}</b>: %{value}€ <br><br>%{hovertext}',
                automargin=False,
                opacity=1,
            ),
            1,
            2,
        )

    # make first trace visible
    fig.data[0].visible = True
    fig.data[1].visible = True

    # create and add slider
    steps = []
    for i in list(range(len(income_months))):
        # dynamic subtitle for info on total income and expenses of the month
        df_temp = income_df[income_df['Month'] == income_months[i]]
        tot_income = round(sum(df_temp['Amount']))
        df_temp = expenses_df[expenses_df['Month'] == expenses_months[i]]
        tot_expenses = round(sum(df_temp['Amount']))
        profit = tot_income - tot_expenses
        profit_perc = round((profit / tot_income) * 100, 2)

        # determine the color and sign of the profit value
        if profit >= 0:
            profit_str = f'<span style="color: {self.income_color};">{profit}€</span>'
        else:
            profit_str = f'<span style="color: {self.expenses_color};">{profit}€</span>'

        subtitle = (
            f'<br><br><sub>Monthly Income: <b>{tot_income}€</b><br>Monthly Expenses: '
            f'<b>{tot_expenses}€</b><br>Profit: <b>{profit_str}</b> ({profit_perc}%)</sub>'
        )

        custom_label = pd.to_datetime(expenses_months[i]).strftime('%b %Y')
        step = dict(
            method='update',
            args=[
                {'visible': [False] * len(fig.data)},
                {'title': f'Income and Expenses by Month{subtitle}'},
            ],  # layout attribute
            label=custom_label,  # set the name of each month
        )
        idx_1 = 2 * i
        idx_2 = 2 * i + 1
        step['args'][0]['visible'][idx_1] = True
        step['args'][0]['visible'][idx_2] = True
        steps.append(step)

    sliders = [
        dict(active=0, currentvalue={'prefix': 'Month: '}, pad={'t': 90, 'b': 0}, steps=steps)
    ]

    # dynamic subtitle for info on total income and expenses of first month
    df_temp = income_df[income_df['Month'] == income_months[0]]
    tot_income = round(sum(df_temp['Amount']))
    df_temp = expenses_df[expenses_df['Month'] == expenses_months[0]]
    tot_expenses = round(sum(df_temp['Amount']))
    profit = tot_income - tot_expenses
    profit_perc = round((profit / tot_income) * 100, 2)

    # determine the color and sign of the profit value
    if profit >= 0:
        profit_str = f'<span style="color: {self.income_color};">{profit}€</span>'
    else:
        profit_str = f'<span style="color: {self.expenses_color};">{profit}€</span>'

    subtitle = (
        f'<br><br><sub>Monthly Income: <b>{tot_income}€</b><br>Monthly Expenses: '
        f'<b>{tot_expenses}€</b><br>Profit: <b>{profit_str}</b> ({profit_perc}%)</sub>'
    )

    # add title, width, legend, ...
    fig.update_layout(
        sliders=sliders,
        title=f'Income and Expenses by Month{subtitle}',
        width=1980,
        height=800,
        showlegend=False,
    )

    return fig
