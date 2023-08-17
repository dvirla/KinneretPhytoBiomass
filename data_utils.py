import pandas as pd
import numpy as np

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
    phyt_cod_df = phyt_cod_df[phyt_cod_df['group_num'].isin(['2', '3', '4', '5', '6', '7', '9'])]
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

def merge_fp_biomass_df(fp_df: pd.DataFrame, biomass_df: pd.DataFrame) -> pd.DataFrame:
    fp_df['depth_discrete'] = fp_df['depth'].apply(lambda x: min(biomass_df['Depth'], key=lambda y: abs(y - x)))

    fp_df.rename(columns={'depth_discrete': 'Depth'}, inplace=True)
    fp_df.drop('depth', axis=1, inplace=True)
    fp_df.drop_duplicates(inplace=True)

    result_df = fp_df.merge(biomass_df, on=['week', 'year', 'month', 'Depth'])

    # Merge df1 and df2 on the common columns ['week', 'year', 'month']
    # merged_df = pd.merge(fp_df, biomass_df, on=['week', 'year', 'month'], suffixes=('_df1', '_df2'))

    # # Calculate the absolute difference between 'depth_df1' and 'Depth_df2'
    # merged_df['depth_diff'] = np.abs(merged_df['depth'] - merged_df['Depth'])

    # # Find the index of the minimum depth difference for each group
    # idx = merged_df.groupby(['week', 'year', 'month', 'group_num', 'Depth'])['depth_diff'].idxmin()

    # # Select the rows from merged_df using the calculated index
    # result_df = merged_df.loc[idx].drop(['depth', 'depth_diff'], axis=1)

    return result_df.reset_index(drop=True)