import json
from sys import version

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Extracts features from patient data encoded in a specific format

    The input DataFrame must be formatted in the same way as 
    diabetic_data.csv. A few variations are acceptable:
    1. The columns 'encounter_id', 'patient_nbr', and 'readmitted' may be
    omitted. If present, they will be ignored.
    2. Any columns with unexpected headings will be ignored.
    3. The 'citoglipton' column may be called 'sitagliptin' instead.
    4. Missing values may be represented with '?' or in standard Pandas ways

    The features are extracted by the same rules used in Readmission.ipynb

    Args:
        df: a Pandas DataFrame containing patient data formatted like 
            diabetic_data.csv

    Returns:
        a Pandas DataFrame with 475 feature columns, and the same number of 
        rows as df
    '''
    if 'citoglipton' in df.columns:
        df.rename(columns={'citoglipton': 'sitagliptin'}, inplace=True)
    df.replace({'?': np.nan}, inplace=True)

    # Read the codes and their meanings from IDS_mapping.csv
    admission_type = pd.read_csv(
        'IDS_mapping.csv',
        index_col=0,
        nrows=8
    )
    discharge_disposition = pd.read_csv(
        'IDS_mapping.csv',
        index_col=0,
        skiprows=10,
        nrows=30
    )
    admission_source = pd.read_csv(
        'IDS_mapping.csv',
        index_col=0,
        skiprows=42
    )

    # Create dicts to make codes to meanings
    admission_type_dict = admission_type.description.to_dict()
    discharge_disposition_dict = discharge_disposition.description.to_dict()
    admission_source_dict = admission_source.description.to_dict()

    # Add columns to df that have the meanings of the codes
    df['admission_type'] = df.admission_type_id.replace(
        admission_type_dict,
        inplace=False)
    df['discharge_disposition'] = df.discharge_disposition_id.replace(
        discharge_disposition_dict,
        inplace=False)
    df['admission_source'] = df.admission_source_id.replace(
        admission_source_dict,
        inplace=False)
    
    
