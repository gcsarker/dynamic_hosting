import pandas as pd
import numpy as np


def prepare_ev_profile(file_path='EV/dataset_3a_aggregated.csv'):
    """
    Processing EV dataset and return resampled dataframe and normalized profile (lhat).
    """

    # Load data
    ev_df = pd.read_csv(file_path, sep=';')
    ev_df = ev_df[['date_from', 'Synthetic_7_2kW']]

    # Fix decimal format
    ev_df['Synthetic_7_2kW'] = (
        ev_df['Synthetic_7_2kW']
        .astype(str)
        .str.replace(',', '.', regex=False)
        .astype(float)
    )

    # Parse datetime
    ev_df['date_from'] = pd.to_datetime(ev_df['date_from'], dayfirst=True)
    ev_df = ev_df.set_index('date_from').sort_index()

    ev_df.drop_duplicates(inplace=True)

    # Resample to 15-minute intervals
    ev_df = ev_df.resample('15T').interpolate(method='linear')

    # Select time window
    ev_df = ev_df.loc['2019-01-01 00:15:00':'2020-01-01 00:00:00']

    # Shift year
    ev_df.index = ev_df.index - pd.DateOffset(years=12)

    ev_profile = ev_df["Synthetic_7_2kW"].copy()

    # Normalize
    lhat = ev_profile.to_numpy(dtype=float)
    lhat = lhat / lhat.max()

    print("Length of lhat:", len(lhat))
    print("Min/Max of lhat:", lhat.min(), lhat.max())

    return ev_df, lhat