import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from itertools import product
from typing import List, Dict
import numpy as np


def boxplot_biomass_by_group(df: pd.DataFrame) -> None:
    # Create the boxplot
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    sns.boxplot(data=df, x='group_num', y='sum_biomass_ug_ml')
    plt.title('Boxplot of sum_biomass_ug_ml by group_num')
    plt.xlabel('Group Number')
    plt.ylabel('Sum Biomass (ug/ml)')
    plt.show()

def violin_biomass_by_group(df: pd.DataFrame) -> None:
    groups = df['group_num'].unique()
    fig, axes = plt.subplots(nrows=len(groups), ncols=1, figsize=(10, 7 * len(groups)), sharex=True)
    for i, group in enumerate(groups):
        ax = axes[i]
        group_df = df[df['group_num'] == group]
        sns.violinplot(data=group_df, y='sum_biomass_ug_ml', ax=ax)
        ax.set_title(f"Violin Plot of sum_biomass_ug_ml Group {group}")
        ax.set_ylabel("Sum Biomass (ug/ml)")
        ax.set_yticks(np.linspace(0, group_df['sum_biomass_ug_ml'].max(), 35))
    
    plt.tight_layout()
    plt.show()

def boxplot_by_depth(df: pd.DataFrame, signals: List=None, by_col: str='depth_discrete') -> None:
    # List of signals you want to plot
    if not signals:
        signals = ['red', 'green', 'yellow', 'orange', 'violet', 'brown', 'blue', 'pressure', 'temp_sample', 'yellow_sub']

    # Create subplots
    fig, axes = plt.subplots(nrows=len(signals), ncols=1, figsize=(10, 20), sharex=True)

    # Loop through each signal and create a boxplot
    for idx, signal in enumerate(signals):
        ax = axes[idx]
        sns.boxplot(x=by_col, y=signal, data=df, ax=ax)
        ax.set_ylabel(signal)
        ax.set_xlabel(f'{by_col}')
        ax.set_title(f'Boxplot of {signal} by {by_col}')

    plt.tight_layout()
    plt.show()

def remove_outliers_IQR(df: pd.DataFrame, q1: float=0.1, q3: float=0.9) -> pd.DataFrame:
    # Calculate the IQR for each group
    grouped = df.groupby('group_num')
    Q1 = grouped['sum_biomass_ug_ml'].quantile(q1)
    Q3 = grouped['sum_biomass_ug_ml'].quantile(q3)
    IQR = Q3 - Q1

    # Define a function to filter outliers based on the IQR
    def filter_outliers(group):
        group_num = group.name
        Q1_val = Q1[group_num]
        Q3_val = Q3[group_num]
        iqr_val = IQR[group_num]
        return group[(group['sum_biomass_ug_ml'] >= Q1_val - 1.5 * iqr_val) &
                    (group['sum_biomass_ug_ml'] <= Q3_val + 1.5 * iqr_val)]

    # Apply the filter_outliers function to each group
    filtered_df = grouped.apply(filter_outliers).reset_index(drop=True)

    return filtered_df

def correlation_per_group(df: pd.DataFrame) -> None:
    columns_to_drop = ['year', 'month', 'Depth']
    correlations_per_group = df.drop(columns=columns_to_drop).groupby('group_num').corr()

    for group_number in df.group_num.unique():
        correlation_for_group = correlations_per_group.loc[group_number]

        # Create a heatmap using seaborn
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_for_group, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(f"Correlation Heatmap for Group {group_number}")
        plt.show()

def plot_tsne(orig_df: pd.DataFrame) -> None:
    # Find the row index with the highest 'sum_biomass_ug_ml' value for each 'year_month' and signal group
    df = orig_df.copy()
    # df['year_month_week'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)+ '-' + df['week'].astype(str).str.zfill(2)

    df = df.sort_values(by=['year', 'month', 'week', 'group_num'])

    # Step 2: Calculate the mean over 'year', 'month', and 'group_num'
    signal_columns = ['red', 'green', 'yellow', 'orange', 'violet', 'brown', 'blue', 'pressure', 'temp_sample', 'yellow_sub']

    idx = df.groupby(['year', 'month', 'week'])[signal_columns + ['sum_biomass_ug_ml']].idxmax()

    df = df.loc[idx.sum_biomass_ug_ml.values].reset_index(drop=True)

    # Argument dictionary for t-SNE
    tsne_vars = {
        'perplexity': [5], #, 10, 30, 50],
        'learning_rate': [50], #, 200, 500, 1000],
        'init': ['random'], #, 'pca'],
        'angle': [0.5]
        # 'angle': [0.2, 0.5, 0.8]
    }

    # Loop through all combinations of t-SNE arguments and plot the graphs
    for perplexity, learning_rate, init, angle in product(tsne_vars['perplexity'], tsne_vars['learning_rate'],
                                                        tsne_vars['init'], tsne_vars['angle']):
        # t-SNE for dimensionality reduction to 2 dimensions
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, init=init, angle=angle,
                random_state=42)
        reduced_features = tsne.fit_transform(df[signal_columns])

        # Add the reduced features to the DataFrame
        df['t-SNE_1'] = reduced_features[:, 0]
        df['t-SNE_2'] = reduced_features[:, 1]

        # Plotting the scatter plot with size representing 'sum_biomass_ug_ml'
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='t-SNE_1', y='t-SNE_2', hue='group_num', size='sum_biomass_ug_ml',
                        palette='tab10', sizes=(20, 600), legend='brief')
        plt.title(f't-SNE Visualization - Perplexity: {perplexity}, Learning Rate: {learning_rate}, Init: {init}, Angle: {angle}')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(title='Group Number', loc='upper right', bbox_to_anchor=(1.3, 1))
        plt.show()


def plot_fluorprobe_prediction(df: pd.DataFrame, fluor_groups_map: Dict) -> None:
    # Generate unique colors for each unique value of 'Depth'
    unique_depths = df['Depth'].unique()
    depth_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_depths)))

    # Visualize predictions along with test points
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for i, group_num in enumerate(fluor_groups_map.keys()):
        row = i // 2
        col = i % 2
        
        group_y_test = df[df['group_num'] == group_num]['sum_biomass_ug_ml']
        group_y_fluor_pred = df[df['group_num'] == group_num][fluor_groups_map[group_num]]
        group_depth = df[df['group_num'] == group_num]['Depth']


        # Get the index of the corresponding unique depth color for each data point
        depth_color_idx = [np.where(unique_depths == d)[0][0] for d in group_depth]

        # Create a scatter plot with unique colors for each unique value of 'Depth'
        axes[row, col].scatter(group_y_test, group_y_fluor_pred, c=depth_colors[depth_color_idx], alpha=0.5)
        axes[row, col].plot([group_y_test.min(), group_y_test.max()], [group_y_test.min(), group_y_test.max()], 'r--', lw=2)  # Add a diagonal line for reference
        axes[row, col].set_xlabel('Actual Test Values')
        axes[row, col].set_ylabel('Fluor Predicted Values')
        axes[row, col].set_title(f'Group {group_num} - Actual vs. Fluor Predicted')

    # Create a separate scatter plot for the global legend
    legend_handles = []
    for depth, color in zip(unique_depths, depth_colors):
        legend_handles.append(plt.scatter([], [], c=color, label=depth))

    # Place the legend outside the subplots on the top-right corner
    fig.legend(handles=legend_handles, title='Depth', loc='upper right', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.tight_layout()
    plt.show()

def plot_corr_per_feature_per_group(df: pd.DataFrame, fluor_groups_map: Dict) -> None:
    # Visualize predictions along with test points
    signal_cols = ['red', 'green', 'yellow', 'orange', 'violet', 'brown', 'blue', 'pressure', 'temp_sample', 'yellow_sub']
    for group_num in fluor_groups_map.keys():
        fig, axes = plt.subplots(5, 2, figsize=(16, 12))

        for i, col_name in enumerate(signal_cols):
            row = i % 5
            col = i % 2

            group_y = df[df['group_num'] == group_num]['sum_biomass_ug_ml']
            feat = df[df['group_num'] == group_num][col_name]

            axes[row, col].scatter(group_y, feat)
            axes[row, col].set_xlabel('Biomass [ug/ml]')
            axes[row, col].set_ylabel(f'Signal {col_name}')

        fig.suptitle(f'Group {group_num}')
        plt.tight_layout()
        plt.show()

def groups_pie_chart(df: pd.DataFrame, by_biomass=False) -> None:
    # Count the occurrences of each unique group_num
    if by_biomass:
        group_counts = df.groupby('group_num')['sum_biomass_ug_ml'].sum()
    else:
        group_counts = df['group_num'].value_counts()

    # Create a pie chart
    plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
    plt.pie(group_counts, labels=group_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Proportion of Each Unique group_num')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Display the pie chart
    plt.show()