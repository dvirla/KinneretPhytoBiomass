import pandas as pd
import numpy as np
from typing import List, Dict

def get_biomass_data(phyt_cod_path: str, phyto_path: str) -> pd.DataFrame:
    phyt_cod_df = pd.read_csv(phyt_cod_path)
    phyto_df = pd.read_csv(phyto_path)

    # Extracting taxonomic group number
    def split_na(x):
        if x is not np.nan:
            return x.split('-')[0]
        return x

    phyt_cod_df['group_num'] = phyt_cod_df['Genus/Sp'].apply(split_na)

    # Filtering currently non relevant phytoplankton groups
    phyt_cod_df = phyt_cod_df[phyt_cod_df['group_num'].isin(['2', '3', '4', '5', '6'])]
    phyt_cod_df['group_num'] = phyt_cod_df['group_num'].apply(lambda x: int(x))

    # Merging lab data and group codes
    merged_phyto = phyto_df.merge(phyt_cod_df, on='Code', how='inner')
    merged_phyto = merged_phyto[['Date', 'Depth', 'Code', 'biomass_ug_ml', 'group_num']]
    merged_phyto.reset_index(inplace=True, drop=True)

    # Filtering measurements with depts below 3 meters since the fluorprobe has low-reliablity around 1.5 meters deep
    merged_phyto = merged_phyto[merged_phyto.Depth > 1]

    merged_phyto.Date = pd.to_datetime(merged_phyto.Date)

    # Extract week, year, and depth from the date column
    merged_phyto['week'] = merged_phyto['Date'].dt.isocalendar().week
    merged_phyto['year'] = merged_phyto['Date'].dt.year
    merged_phyto['month'] = merged_phyto['Date'].dt.month

    merged_phyto.drop('Code', axis=1, inplace=True)

    # Summing biomass for same week-year-month-group-depth
    biomass_by_week_year_group = merged_phyto.groupby(['week', 'year', 'month', 'group_num', 'Depth']).sum()
    biomass_by_week_year_group.rename(columns={'biomass_ug_ml': 'sum_biomass_ug_ml'}, inplace=True)
    biomass_by_week_year_group.reset_index(inplace=True)

    return biomass_by_week_year_group

def get_fluorprobe_data(path: str) -> pd.DataFrame:
    # Reading fluorprobe's data
    fp_df = pd.read_csv(path)
    fp_df = fp_df[fp_df['station'] == 'A'].reset_index(drop=True)
    fp_df = fp_df[['Date_time', 'Date','depth', 'Trans 700 nm', 'LED 3 525 nm', 'LED 4  570 nm', 'LED 5  610 nm',
                'LED 6  370 nm', 'LED 7  590 nm', 'LED 8  470 nm', 'Pressure', 'Temp Sample', 'Yellow substances',
                    'Green Algae', 'Bluegreen', 'Diatoms', 'Cryptophyta', 'Total conc']]

    color_names = {
        'Trans 700 nm': 'red',
        'LED 3 525 nm': 'green',
        'LED 4  570 nm': 'yellow',
        'LED 5  610 nm': 'orange',
        'LED 6  370 nm': 'violet',
        'LED 7  590 nm': 'brown', # yellow_2?
        'LED 8  470 nm': 'blue',
        'Pressure': 'pressure',
        # 'Temp Sensor': 'temp_sensor',
        'Temp Sample': 'temp_sample',
        'Yellow substances': 'yellow_sub'
    }
    fp_df.rename(columns=color_names, inplace=True)
    fp_df.dropna(inplace=True)
    fp_df.reset_index(drop=True, inplace=True)
    fp_df.Date = pd.to_datetime(fp_df.Date)
    fp_df.Date_time = pd.to_datetime(fp_df.Date_time)

    # Extract week, year, and depth from the date column
    fp_df['week'] = fp_df['Date'].dt.week
    fp_df['year'] = fp_df['Date'].dt.year
    fp_df['month'] = fp_df['Date'].dt.month

    fp_df.drop(['Date_time', 'Date'], inplace=True, axis=1)

    fp_df = fp_df[(fp_df >= 0).all(axis=1)] # Filtering rows with negative values
    
    # Filtering measurements with depts below 3 meters since the fluorprobe has low-reliablity around 1.5 meters deep
    fp_df = fp_df[fp_df.depth >= 1.5]

    fp_df.drop_duplicates(['week', 'year', 'month', 'depth'], inplace=True)


    return fp_df

def merge_fp_biomass_df(fp_df: pd.DataFrame, biomass_df: pd.DataFrame, is_train=True) -> pd.DataFrame:
    if is_train:
        if 'depth_discrete' not in fp_df.columns:
            fp_df['depth_discrete'] = fp_df['depth'].apply(lambda x: min(biomass_df['Depth'], key=lambda y: abs(y - x)))

        fp_df.rename(columns={'depth_discrete': 'Depth'}, inplace=True)
        # fp_df.drop('depth', axis=1, inplace=True)
        fp_df.drop_duplicates(inplace=True)

        result_df = fp_df.merge(biomass_df, on=['week', 'year', 'month', 'Depth'])
    else:
        # Merge df1 and df2 on the common columns ['week', 'year', 'month']
        merged_df = pd.merge(fp_df, biomass_df, on=['week', 'year', 'month'], suffixes=('_df1', '_df2'))

        # Calculate the absolute difference between 'depth_df1' and 'Depth_df2'
        merged_df['depth_diff'] = np.abs(merged_df['depth'] - merged_df['Depth'])

        # Find the index of the minimum depth difference for each group
        idx = merged_df.groupby(['week', 'year', 'month', 'group_num', 'Depth'])['depth_diff'].idxmin()

        # Select the rows from merged_df using the calculated index
        result_df = merged_df.loc[idx].drop(['depth', 'depth_diff'], axis=1)

    return result_df.reset_index(drop=True)

def proportionalize(df: pd.DataFrame, row_wise=True, new_col_prefix='') -> None:
    # Transforming df to add new column or replace 'sum_biomass_ug_ml
    if row_wise:
        # Row-wise proportionalization
        row_proportional_cols = ['Green Algae', 'Bluegreen', 'Diatoms', 'Cryptophyta']
        row_sums = df[row_proportional_cols].sum(axis=1)
        df[row_proportional_cols].div(row_sums, axis=0) * 100
        df[row_proportional_cols] = df[row_proportional_cols].div(row_sums, axis=0) * 100

    else:
        # # Column-wise proportionalization
        col_proportional_cols = ['week', 'month', 'year', 'Depth']
        col_sum = df.groupby(col_proportional_cols)['sum_biomass_ug_ml'].transform('sum')
        df[f'{new_col_prefix}sum_biomass_ug_ml'] = df['sum_biomass_ug_ml'].div(col_sum) * 100

def biomass_estimation(df: pd.DataFrame) -> None:
    # Calculate estimated sum_biomass_ug_ml
    estimated_biomass = []
    current_step = 0  # Initialize the step for the current week-year-month frame
    depth_diffs = []

    prev_depths = {
        3: 0,
        5: 3,
        10: 5,
        15: 10,
        20: 15,
        21: 20,
        25: 21,
        30: 25
    }

    for _, row in df.iterrows():
        # Calculate step for the current week-year-month frame
        next_df = df[(df.week == row.week) & 
                            (df.year == row.year) & 
                            (df.month == row.month) & 
                            (df.group_num == row.group_num) &
                            (df.Depth == prev_depths[row.Depth])]
        if next_df.shape[0] > 0:
            step_numerator = row['sum_biomass_ug_ml'] - next_df.iloc[0]['sum_biomass_ug_ml']
            step_denominator = row['Depth'] - next_df.iloc[0]['Depth']
        else:
            step_numerator = row['sum_biomass_ug_ml']
            step_denominator = row['Depth']

        current_step = step_numerator / step_denominator
            
        depth_diff = row['Depth'] - row['depth']
        depth_diffs.append(depth_diff)
        
        estimated_biomass_value = row['sum_biomass_ug_ml']
        estimated_biomass_value += current_step * depth_diff
        
        estimated_biomass.append(estimated_biomass_value)

    df['estimated_sum_biomass_ug_ml'] = estimated_biomass
    df['depth_diffs'] = depth_diffs

    df.drop(['sum_biomass_ug_ml', 'depth_diffs'], axis=1, inplace=True)
    df.rename(columns={'estimated_sum_biomass_ug_ml': 'sum_biomass_ug_ml'}, inplace=True)

def filter_signals_by_boundaries(df: pd.DataFrame, signals: List, boundaries: Dict) -> None:
    # Dictionary to store records to be removed for each signal
    records_to_remove = {signal: [] for signal in signals}

    # Loop through each signal and create a boxplot
    for _, signal in enumerate(signals):
        
        lower_bound = boundaries[signal]['lower_bound']
        upper_bound = boundaries[signal]['upper_bound']
        
        # Identify records to be removed
        outliers = df[(df[signal] < lower_bound) | (df[signal] > upper_bound)]
        
        # Accumulate records to be removed
        records_to_remove[signal].extend(outliers.index.tolist())
    
    # Flatten the list of indices to remove
    indices_to_remove = set(idx for lst in records_to_remove.values() for idx in lst)

    # Remove accumulated records from 'fp_df'
    df.drop(index=indices_to_remove, inplace=True)

def filter_biomass_by_group_boundaries(df: pd.DataFrame, boundaries: List) -> None:
    groups = df['group_num'].unique()
    indices_to_remove = []
    for group in groups:
        lb, ub = boundaries[group]
        indices_to_remove.append(df[(df['sum_biomass_ug_ml'] < lb) | (df['sum_biomass_ug_ml'] > ub)].index)
    
    indices_to_remove = set(indices_to_remove)
    df.drop(indices_to_remove, inplace=True)