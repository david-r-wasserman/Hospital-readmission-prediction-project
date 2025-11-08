'''
The DiabetesReadmissionPredictor class predicts readmission from patient
data encoded in a specific format.

For more detail about the data format, see Readmission.ipynb.
For an explanation of the feature selections, see Readmission.ipynb.
'''

import json

import joblib
import numpy as np
import pandas as pd
# import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


class DiabetesReadmissionPredictor:
    '''
    Predicts readmission from patient data encoded in a specific format
    '''

    def __init__(self):
        self.model = joblib.load('xgb_readmission_model.pkl')

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        '''
        Predicts readmission from patient data encoded in a specific format

        The input DataFrame must be formatted in the same way as
        diabetic_data.csv. A few variations are acceptable:
        1. The columns 'encounter_id', 'patient_nbr', and 'readmitted' may be
        omitted. If present, they will be ignored.
        2. Any columns with unexpected headings will be ignored.
        3. The 'citoglipton' column may be called 'sitagliptin' instead.
        4. Missing values may be represented with '?' or in standard pandas
        ways

        The features are extracted by the same rules used in Readmission.ipynb

        Args:
            df: a pandas DataFrame, shape (n_patients, n_cols) containing
                patient data formatted like diabetic_data.csv

        Returns:
            numpy array, float, shape (n_patients,), with values in the range
            [0, 1], given the estimated probability of each patient to be
            readmitted within 30 days
        '''
        features = _extract_features(df)
        return self.model.predict_proba(features)[:, 1]


def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Extracts features from patient data encoded in a specific format

    The input DataFrame must be formatted in the same way as
    diabetic_data.csv. A few variations are acceptable:
    1. The columns 'encounter_id', 'patient_nbr', and 'readmitted' may be
    omitted. If present, they will be ignored.
    2. Any columns with unexpected headings will be ignored.
    3. The 'citoglipton' column may be called 'sitagliptin' instead.
    4. Missing values may be represented with '?' or in standard pandas ways

    The features are extracted by the same rules used in Readmission.ipynb

    Args:
        df: a pandas DataFrame containing patient data formatted like
            diabetic_data.csv

    Returns:
        a pandas DataFrame with 475 feature columns, and the same number of
        rows as df
    '''
    with open('feature_data.json', 'r') as file:
        feature_data_dict = json.load(file)

    if 'citoglipton' in df.columns:
        df.rename(columns={'citoglipton': 'sitagliptin'}, inplace=True)
    df.replace({'?': np.nan}, inplace=True)

    df = _decode_nominal_columns(df)

    feature_df = pd.DataFrame(
        index=df.index,
        columns=feature_data_dict['all_columns'],
        dtype=float,
    )

    _extract_code_features(df, feature_df, feature_data_dict)
    _extract_code_range_features(df, feature_df, feature_data_dict)
    _extract_categorical_features(df, feature_df, feature_data_dict)
    _edit_medical_specialty_features(df, feature_df)
    _extract_specific_medication_features(df, feature_df, feature_data_dict)
    _extract_discharge_group_features(df, feature_df, feature_data_dict)
    _extract_numeric_features(df, feature_df, feature_data_dict)

    return feature_df


def _extract_numeric_features(df, feature_df, feature_data_dict):
    '''
    Edit numeric features

    Args:
        df: a pandas DataFrame containing patient data formatted like
            diabetic_data.csv
        feature_df: a pandas DataFrame with the same index as df
        feature_data_dict: a dict with a key 'specific_med_columns'
    Returns:
        None

    Effect: columns in feature_df representing groups of discharge dispositions
        will be populated
    '''
    feature_df.loc[:, 'gender'] = np.nan
    feature_df.loc[df.gender == 'Female', 'gender'] = 0
    feature_df.loc[df.gender == 'Male', 'gender'] = 1

    for age in range(0, 100, 10):
        age_string = f'[{age}-{age + 10})'
        feature_df.loc[df.age == age_string, 'age'] = age + 5

    for weight in range(0, 200, 25):
        weight_string = f'[{weight}-{weight + 25})'
        feature_df.loc[
            df.weight == weight_string, 'weight'] = weight + 12.5
    feature_df.loc[df.weight == '>200', 'weight'] = 212.5

    for col in [
        'time_in_hospital',
        'num_lab_procedures',
        'num_procedures',
        'num_medications',
        'number_outpatient',
        'number_emergency',
        'number_inpatient',
        'number_diagnoses',
    ]:
        feature_df.loc[:, col] = df[col]

    specific_med_columns = feature_data_dict['specific_med_columns']
    feature_df.loc[:, 'num_specific_meds'] = (
        df[specific_med_columns] != 'No'
    ).astype(float).sum(axis=1)

    feature_df.loc[:, 'change'] = (df.change == 'Ch').astype(float)


def _extract_discharge_group_features(df, feature_df, feature_data_dict):
    '''
    Edit features for groups of discharge dispositions

    Args:
        df: a pandas DataFrame containing patient data formatted like
            diabetic_data.csv
        feature_df: a pandas DataFrame with the same index as df
        feature_data_dict: a dict with a key 'discharge_keywords'
    Returns:
        None

    Effect: columns in feature_df representing groups of discharge dispositions
        will be populated
    '''
    for group, keywords in feature_data_dict['discharge_keywords'].items():
        # Initially create feature as all False
        feature = pd.Series(False, index=df.index)
        # Then find patients with discharge_disposition containing a keyword
        for keyword in keywords:
            feature = (
                feature | df.discharge_disposition.str.contains(keyword)
            )
        feature_df.loc[:, 'discharge_' + group] = feature.astype(float)


def _extract_specific_medication_features(df, feature_df, feature_data_dict):
    '''
    Edit features for specific medication columns in diabetic_data.csv

    Args:
        df: a pandas DataFrame containing patient data formatted like
            diabetic_data.csv
        feature_df: a pandas DataFrame with the same index as df
        feature_data_dict: a dict with a key 'med_features'
    Returns:
        None

    Effect: columns in feature_df representing specific medications will be
        populated
    '''
    for col in feature_data_dict['med_features']:
        parts = col.split('_')
        if len(parts) == 2:
            feature_df[col] = (
                df[parts[0]].str.lower() == parts[1]).astype(float)
        else:
            feature_df[col] = (df[col] != 'No').astype(float)


def _edit_medical_specialty_features(df, feature_df):
    '''
    Edit features for medical specialty column in diabetic_data.csv

    Args:
        df: a pandas DataFrame containing patient data formatted like
            diabetic_data.csv
        feature_df: a pandas DataFrame with the same index as df

    Returns:
        None

    Effect: columns in feature_df representing medical specialties will be
        populated and edited, and a column 'specialty_Pediatrics-Endocrinology'
        will be removed
    '''
    # 1 A "Hematology" feature that includes "Hematology/Oncology"
    feature_df.loc[:, 'specialty_Hematology'] = (
        (df.medical_specialty == 'Hematology')
        | (df.medical_specialty == 'Hematology/Oncology')
    ).astype(float)
    # 2 An "Oncology" feature that includes "Hematology/Oncology"
    feature_df.loc[:, 'specialty_Oncology'] = (
        (df.medical_specialty == 'Oncology')
        | (df.medical_specialty == 'Hematology/Oncology')
    ).astype(float)
    # 3 A "Hematology/Oncology" feature that has value 1 for
    # "Hematology/Oncology", 0.5 for "Hematology" and "Oncology",
    # and 0 for everything else.
    feature_df.loc[
        df.medical_specialty == 'Hematology',
        'specialty_Hematology/Oncology'
    ] = 0.5
    feature_df.loc[
        df.medical_specialty == 'Hematology',
        'specialty_Hematology/Oncology'
    ] = 0.5
    # 4 include "Orthopedics-Reconstructive" patients under "Orthopedics"
    feature_df.loc[:, 'specialty_Orthopedics'] = (
        (df.medical_specialty == 'Orthopedics')
        | (df.medical_specialty == 'Orthopedics-Reconstructive')
    ).astype(float)
    # 5 The "Pediatrics" feature will include all values that contain "Pediat"
    # case insensitive
    feature_df.loc[:, 'specialty_Pediatrics'] = (
        ~df.medical_specialty.isna()
        & df.medical_specialty.str.lower().str.contains('pediat')
    ).astype(float)
    # 6 "Pediatrics-Endocrinology" patients will also be included in an
    # "Endocrinology" feature; there will be no "Pediatrics-Endocrinology"
    # feature
    feature_df.drop(
        columns='specialty_Pediatrics-Endocrinology',
        inplace=True
    )
    feature_df.loc[:, 'specialty_Endocrinology'] = (
        ~df.medical_specialty.isna()
        & df.medical_specialty.str.lower().str.contains('endocrin')
    ).astype(float)
    # 7 The "Psychiatry" feature will include "Psychology" patients
    feature_df.loc[:, 'specialty_Psychiatry'] = (
        (df.medical_specialty == 'Psychiatry')
        | (df.medical_specialty == 'Psychology')
    ).astype(float)
    # 8 The "Surgery-General" feature will include "Surgeon" and
    # "SurgicalSpecialty"
    feature_df.loc[:, 'specialty_Surgery-General'] = (
        (df.medical_specialty == 'Surgery-General')
        | (df.medical_specialty == 'Surgeon')
        | (df.medical_specialty == 'SurgicalSpecialty')
    ).astype(float)
    # 9. add a "Surgery" feature, including all values that contain "Surg"
    feature_df.loc[:, 'specialty_Surgery'] = (
        ~df.medical_specialty.isna()
        & df.medical_specialty.str.lower().str.contains('surg')
    ).astype(float)
    # 10. The "ObstetricsandGynecology" feature will include "Gynecology"
    feature_df.loc[:, 'specialty_ObstetricsandGynecology'] = (
        ~df.medical_specialty.isna()
        & df.medical_specialty.str.lower().str.contains('gynec')
    ).astype(float)
    # 11. a "Surgery-Thoracic" feature that includes
    # "Surgery-Cardiovascular/Thoracic"
    feature_df.loc[:, 'specialty_Surgery-Thoracic'] = (
        (df.medical_specialty == 'Surgery-Thoracic')
        | (df.medical_specialty == 'Surgery-Cardiovascular/Thoracic')
    ).astype(float)
    # 12. There will be a "Surgery-Cardiovascular" feature that includes
    # "Surgery-Cardiovascular/Thoracic"
    feature_df.loc[:, 'specialty_Surgery-Cardiovascular'] = (
        (df.medical_specialty == 'Surgery-Cardiovascular')
        | (df.medical_specialty == 'Surgery-Cardiovascular/Thoracic')
    ).astype(float)
    # 13. The "Surgery-Cardiovascular/Thoracic" feature will have value 0.5
    # for "Surgery-Thoracic" and "Surgery-Cardiovascular"
    feature_df.loc[
        df.medical_specialty == 'Surgery-Thoracic',
        'specialty_Surgery-Cardiovascular/Thoracic'
    ] = 0.5
    feature_df.loc[
        df.medical_specialty == 'Surgery-Cardiovascular',
        'specialty_Surgery-Cardiovascular/Thoracic'
    ] = 0.5
    # 14. The "Radiologist" feature will include "Radiology".
    feature_df.loc[:, 'specialty_Radiologist'] = (
        ~df.medical_specialty.isna()
        & df.medical_specialty.str.lower().str.contains('radiol')
    ).astype(float)


def _extract_categorical_features(df, feature_df, feature_data_dict):
    '''
    Create features for categorical columns in diabetic_data.csv

    Args:
        df: a pandas DataFrame containing patient data formatted like
            diabetic_data.csv
        feature_df: a pandas DataFrame with the same index as df
        feature_data_dict: a dict with keys 'races', 'common_payers',
            'common_specialties', 'glucose_results', 'a1c_results',
            'common_admit_types', 'common_admit_sources', and
            'common_discharges'

    Returns:
        None

    Effect: columns in feature_df representing categorical features
        will be populated, and a column 'specialty_Pediatrics-Endocrinology'
        will be added
    '''
    category_fields = [
        ('races', 'race', 'race_'),
        ('common_payers', 'payer_code', 'payer_'),
        ('common_specialties', 'medical_specialty', 'specialty_'),
        ('glucose_results', 'max_glu_serum', 'glucose_'),
        ('a1c_results', 'A1Cresult', 'a1c_'),
        ('common_admit_types', 'admission_type', 'admit_type_'),
        ('common_admit_sources', 'admission_source', 'admit_source_'),
        ('common_discharges', 'discharge_disposition', 'discharge_'),
    ]
    for data_dict_key, df_col, feature_stem in category_fields:
        for feature in feature_data_dict[data_dict_key]:
            feature_df.loc[:, feature_stem + feature] = (
                df[df_col] == feature
            ).astype(float)


def _decode_nominal_columns(df, drop=False):
    '''
    Add columns for admission_type, admission_source, and
    discharge_disposition

    diabetic_data.csv has columns 'admission_type_id', 'admission_source_id',
    and 'discharge_disposition_id'. This function creates columns
    'admission_type', 'admission_source', and 'discharge_disposition'
    containing the corresponding string values, which are found in
    IDS_mapping.csv.

    Args:
        df: a pandas DataFrame with columns 'admission_type_id',
            'admission_source_id', and 'discharge_disposition_id'
        drop: (Optional) boolean. If True, the columns 'admission_type_id',
            'admission_source_id', and 'discharge_disposition_id' will
            be removed from the output

    Returns:
        the input pandas DataFrame, with the added columns 'admission_type',
        'admission_source', and 'discharge_disposition'
    '''
    # Read the codes and their meanings from IDS_mapping.csv
    admission_type = pd.read_csv(
        'IDS_mapping.csv',
        index_col=0,
        nrows=8,
    )
    discharge_disposition = pd.read_csv(
        'IDS_mapping.csv',
        index_col=0,
        skiprows=10,
        nrows=30,
    )
    admission_source = pd.read_csv(
        'IDS_mapping.csv',
        index_col=0,
        skiprows=42,
    )

    # Create dicts to make codes to meanings
    admission_type_dict = admission_type.description.to_dict()
    discharge_disposition_dict = discharge_disposition.description.to_dict()
    admission_source_dict = admission_source.description.to_dict()

    # Add columns to df that have the meanings of the codes
    df['admission_type'] = df.admission_type_id.replace(
        admission_type_dict,
        inplace=False,
    )
    df['discharge_disposition'] = df.discharge_disposition_id.replace(
        discharge_disposition_dict,
        inplace=False,
    )
    df['admission_source'] = df.admission_source_id.replace(
        admission_source_dict,
        inplace=False,
    )

    if drop:
        df = df.drop(columns=[
            'admission_type_id',
            'admission_source_id',
            'discharge_disposition_id'
        ])

    return df


def _extract_code_features(df, feature_df, feature_data_dict):
    '''
    Create features for individual ICD-9 codes

    Args:
        df: a pandas DataFrame containing patient data formatted like
            diabetic_data.csv
        feature_df: a pandas DataFrame with the same index as df
        feature_data_dict: a dict with keys 'common_codes' and
            'common_primary_codes'

    Returns:
        None

    Effect: columns in feature_df representing individual ICD-9 codes
        will be populated
    '''
    for code in feature_data_dict['common_codes']:
        _make_code_feature(
            df,
            code,
            ['diag_1', 'diag_2', 'diag_3'],
            feature_df,
            'ICD9_' + code,
        )

    for code in feature_data_dict['common_primary_codes']:
        _make_code_feature(
            df,
            code,
            ['diag_1'],
            feature_df,
            'primary_ICD9_' + code,
        )


def _extract_code_range_features(df, feature_df, feature_data_dict):
    '''
    Create features for ICD-9 codes ranges

    Args:
        df: a pandas DataFrame containing patient data formatted like
            diabetic_data.csv
        feature_df: a pandas DataFrame with the same index as df
        feature_data_dict: a dict with keys 'code_ranges' and
            'primary_code_ranges'

    Returns:
        None

    Effect: columns in feature_df representing ICD-9 code ranges
        will be populated
    '''
    code_nums = df[['diag_1', 'diag_2', 'diag_3']].apply(
        pd.to_numeric,
        errors='coerce',
    )

    for low, high in feature_data_dict['code_ranges']:
        mask = code_nums.apply(
            lambda col: col.between(low, high),
            axis=0
        )
        feature_df.loc[:, f'ICD9_{low}-{high}'] = (
            mask.any(axis=1).astype(float)
        )

    for low, high in feature_data_dict['primary_code_ranges']:
        feature_df.loc[:, f'primary_ICD9_{low}-{high}'] = (
            code_nums['diag_1'].between(low, high)
        ).astype(float)


def _make_code_feature(
    X: pd.DataFrame,
    code: str,
    code_columns: list,
    feature_df: pd.DataFrame,
    feature_name: str,
) -> None:
    '''
    Make a feature representing an ICD-9 code.

    Args:
        X: pandas DataFrame; its columns must include everything in
            code_columns
        code: string, ICD-9 code
        code_columns: list of strings
        feature_df: pandas DataFrame with same index as X
        feature_name: string, name of one of the columns in feature_df

    Returns:
        None

    Effect:
        feature_df[feature_name] values will be overwritten, indicating which
            patients have this code in one of the code_columns
    '''
    if code.startswith('250'):
        matches = X[code_columns].map(
            lambda x: (not pd.isna(x)) and x.startswith(code)
            )
    else:
        # we can't use startswith for all the codes, because some of them
        # are less than 3 characters (the leading zeroes are missing), so
        # startswith would pick up other 3-digit codes
        matches = (X[code_columns] == code)

    feature_df.loc[:, feature_name] = matches.any(axis=1).astype(float)


if __name__ == '__main__':
    # Test that we can reproduce the results in Readmission.ipynb
    df = pd.read_csv(
        'diabetic_data.csv',
        na_values='?',
        keep_default_na=False,
        dtype={
            'diag_1': 'object',
            'diag_2': 'object',
            'diag_3': 'object',
            'payer_code': 'object'}
        )
    discharge_disposition = pd.read_csv(
        'IDS_mapping.csv',
        index_col=0,
        skiprows=10,
        nrows=30)
    discharge_disposition_dict = discharge_disposition.description.to_dict()
    df['discharge_disposition'] = df.discharge_disposition_id.replace(
        discharge_disposition_dict,
        inplace=False)
    df = df[~df['discharge_disposition'].str.contains(
        'expired',
        case=False,
        na=False,
    )]
    df.drop_duplicates(subset=['patient_nbr'], inplace=True)

    y = (df['readmitted'] == '<30')
    df.drop(columns=['readmitted'], inplace=True)
    _, X_test, _, y_test = train_test_split(
        df,
        y,
        test_size=5000,
        random_state=0
    )
    y_pred = DiabetesReadmissionPredictor().predict(X_test)

    score = roc_auc_score(y_test, y_pred)
    print('ROC AUC score:', score)
    if score > 0.672:
        print('Test passed')
    else:
        print('Test failed')
