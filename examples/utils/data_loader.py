"""
Data loading utilities for Aurora-GLM examples.

Automatically downloads datasets from public sources and caches locally.
All datasets use permissive licenses (CC0, GPL, Public Domain).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests


def _download_file(url: str, output_path: str) -> None:
    """Download file from URL with progress indicator."""
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\rProgress: {percent:.1f}%", end='', flush=True)

    print(f"\n✓ Downloaded to {output_path}")


def load_insurance_data(
    cache_dir: str = '../data', force_download: bool = False
) -> pd.DataFrame:
    """
    Load insurance claims dataset.

    Source: Medical Cost Personal Dataset
    License: CC0 (Public Domain)
    URL: https://github.com/stedy/Machine-Learning-with-R-datasets

    Dataset contains medical insurance charges for individuals with various
    characteristics. Useful for Gamma GLM modeling (positive continuous outcome).

    Parameters
    ----------
    cache_dir : str, default='../data'
        Directory to cache downloaded data
    force_download : bool, default=False
        Force re-download even if cached

    Returns
    -------
    df : DataFrame
        Insurance data with columns:
        - age: Age of primary beneficiary
        - sex: Gender (male/female)
        - bmi: Body mass index
        - children: Number of children/dependents
        - smoker: Smoking status (yes/no)
        - region: Residential area (northeast/southeast/southwest/northwest)
        - charges: Medical costs billed by insurance (target variable)

    Examples
    --------
    >>> df = load_insurance_data()
    >>> print(df.shape)
    (1338, 7)
    >>> print(df['charges'].describe())
    """
    cache_path = Path(cache_dir) / 'insurance.csv'

    if not cache_path.exists() or force_download:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
        _download_file(url, cache_path)
    else:
        print(f"✓ Using cached data: {cache_path}")

    df = pd.read_csv(cache_path)
    return df


def load_sleepstudy_data(
    cache_dir: str = '../data', force_download: bool = False
) -> pd.DataFrame:
    """
    Load sleep deprivation study dataset.

    Source: lme4 R package
    License: GPL-2
    URL: https://github.com/vincentarelbundock/Rdatasets

    Classic longitudinal dataset from a sleep deprivation study. Reaction time
    was measured for subjects with restricted sleep (3 hours per night).
    Useful for demonstrating longitudinal GAMM with random slopes.

    Parameters
    ----------
    cache_dir : str, default='../data'
        Directory to cache downloaded data
    force_download : bool, default=False
        Force re-download even if cached

    Returns
    -------
    df : DataFrame
        Sleep study data with columns:
        - reaction: Average reaction time (ms)
        - days: Days of sleep deprivation (0-9)
        - subject: Subject identifier (18 subjects)

    Notes
    -----
    Original study: Belenky et al. (2003). "Patterns of performance degradation
    and restoration during sleep restriction and subsequent recovery"

    Examples
    --------
    >>> df = load_sleepstudy_data()
    >>> print(df.shape)
    (180, 3)
    >>> print(df.groupby('subject').size())
    """
    cache_path = Path(cache_dir) / 'sleepstudy.csv'

    if not cache_path.exists() or force_download:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/lme4/sleepstudy.csv"
        _download_file(url, cache_path)
    else:
        print(f"✓ Using cached data: {cache_path}")

    df = pd.read_csv(cache_path)

    # Clean column names and remove index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    df = df.rename(
        columns={'Reaction': 'reaction', 'Days': 'days', 'Subject': 'subject'}
    )

    return df


def load_airquality_data(
    cache_dir: str = '../data', force_download: bool = False
) -> pd.DataFrame:
    """
    Load air quality dataset.

    Source: R datasets package
    License: GPL-3
    URL: https://github.com/vincentarelbundock/Rdatasets

    Daily air quality measurements in New York, May to September 1973.
    Useful for GAM modeling with smooth terms (temperature, wind, etc.).

    Parameters
    ----------
    cache_dir : str, default='../data'
        Directory to cache downloaded data
    force_download : bool, default=False
        Force re-download even if cached

    Returns
    -------
    df : DataFrame
        Air quality data with columns:
        - ozone: Mean ozone (ppb)
        - solar_r: Solar radiation (lang)
        - wind: Average wind speed (mph)
        - temp: Maximum daily temperature (°F)
        - month: Month (5-9)
        - day: Day of month (1-31)

    Notes
    -----
    Original source: New York State Department of Conservation and the
    National Weather Service.

    Examples
    --------
    >>> df = load_airquality_data()
    >>> print(df.shape)
    (111, 6)  # After removing missing values
    >>> print(df['temp'].describe())
    """
    cache_path = Path(cache_dir) / 'airquality.csv'

    if not cache_path.exists() or force_download:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/airquality.csv"
        _download_file(url, cache_path)
    else:
        print(f"✓ Using cached data: {cache_path}")

    df = pd.read_csv(cache_path)

    # Clean column names
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    df = df.rename(
        columns={
            'Ozone': 'ozone',
            'Solar.R': 'solar_r',
            'Wind': 'wind',
            'Temp': 'temp',
            'Month': 'month',
            'Day': 'day',
        }
    )

    # Remove missing values
    df = df.dropna()

    return df


def load_species_data(
    cache_dir: str = '../data', force_download: bool = False
) -> pd.DataFrame:
    """
    Generate synthetic species abundance dataset.

    Since real species data from GBIF requires API keys and can be very large,
    this function generates synthetic but realistic species count data for
    ecological modeling demonstrations.

    Parameters
    ----------
    cache_dir : str, default='../data'
        Not used (data is generated, not cached)
    force_download : bool, default=False
        Not used (data is generated)

    Returns
    -------
    df : DataFrame
        Species data with columns:
        - species: Species identifier (A, B, C)
        - count: Observed count at site
        - temperature: Mean temperature (°C)
        - precipitation: Annual precipitation (mm)
        - elevation: Elevation (m)
        - latitude: Latitude (degrees)
        - longitude: Longitude (degrees)

    Notes
    -----
    Data is generated to mimic realistic ecological patterns:
    - Poisson distribution for counts
    - Environmental predictors influence abundance
    - Species-specific environmental preferences

    Examples
    --------
    >>> df = load_species_data()
    >>> print(df.shape)
    (500, 7)
    >>> print(df['count'].describe())
    """
    np.random.seed(42)
    n = 500

    # Environmental predictors
    temperature = np.random.uniform(10, 35, n)
    precipitation = np.random.uniform(500, 2500, n)
    elevation = np.random.uniform(0, 3000, n)
    latitude = np.random.uniform(30, 50, n)
    longitude = np.random.uniform(-120, -80, n)

    # Species assignment
    species = np.random.choice(['species_A', 'species_B', 'species_C'], n)

    # Species-specific responses to environment
    species_effects = {'species_A': 0.3, 'species_B': 0.0, 'species_C': -0.3}

    # Generate counts (Poisson process)
    log_lambda = np.zeros(n)
    for i in range(n):
        sp_effect = species_effects[species[i]]
        log_lambda[i] = (
            1.5
            + sp_effect
            - 0.05 * temperature[i]
            + 0.0005 * precipitation[i]
            - 0.0003 * elevation[i]
            + np.random.randn() * 0.3
        )

    count = np.random.poisson(np.exp(log_lambda))

    df = pd.DataFrame(
        {
            'species': species,
            'count': count,
            'temperature': temperature,
            'precipitation': precipitation,
            'elevation': elevation,
            'latitude': latitude,
            'longitude': longitude,
        }
    )

    print(f"✓ Generated synthetic species data ({len(df)} observations)")
    return df


def load_clinical_trial_data(
    cache_dir: str = '../data', force_download: bool = False
) -> pd.DataFrame:
    """
    Generate synthetic clinical trial dataset.

    Creates realistic clinical trial data with nested random effects
    (patients within clinics) for demonstrating binary GAMM models.

    Parameters
    ----------
    cache_dir : str, default='../data'
        Not used (data is generated, not cached)
    force_download : bool, default=False
        Not used (data is generated)

    Returns
    -------
    df : DataFrame
        Clinical trial data with columns:
        - outcome: Binary treatment outcome (0=failure, 1=success)
        - treatment: Treatment group (0=control, 1=treated)
        - time: Time point (0-3, weeks)
        - age: Patient age (years)
        - severity: Baseline severity score (0-10)
        - patient: Patient identifier (0-99)
        - clinic: Clinic identifier (0-9)

    Notes
    -----
    Data structure:
    - 100 patients across 10 clinics
    - 4 time points per patient (400 total observations)
    - Treatment effect increases over time
    - Random effects for both clinic and patient

    Examples
    --------
    >>> df = load_clinical_trial_data()
    >>> print(df.shape)
    (400, 7)
    >>> print(df['outcome'].mean())  # Overall success rate
    """
    np.random.seed(123)

    n_patients = 100
    n_clinics = 10
    n_timepoints = 4
    n = n_patients * n_timepoints

    # Random effects
    clinic_effects = np.random.randn(n_clinics) * 0.5
    patient_effects = np.random.randn(n_patients) * 0.3

    # Patient-level covariates
    patient_age = np.random.uniform(30, 70, n_patients)
    patient_severity = np.random.uniform(2, 8, n_patients)

    # Generate observations
    data = []
    for patient in range(n_patients):
        clinic = patient % n_clinics
        treatment = 1 if patient >= n_patients // 2 else 0

        for time in range(n_timepoints):
            # Log-odds of success
            log_odds = (
                -0.5  # baseline
                + 0.8 * treatment  # treatment effect
                + 0.1 * time  # time trend
                + 0.3 * treatment * time  # treatment × time interaction
                - 0.05 * patient_age[patient]  # age effect
                - 0.2 * patient_severity[patient]  # severity effect
                + clinic_effects[clinic]
                + patient_effects[patient]
            )

            prob = 1 / (1 + np.exp(-log_odds))
            outcome = np.random.binomial(1, prob)

            data.append(
                {
                    'outcome': outcome,
                    'treatment': treatment,
                    'time': time,
                    'age': patient_age[patient],
                    'severity': patient_severity[patient],
                    'patient': patient,
                    'clinic': clinic,
                }
            )

    df = pd.DataFrame(data)
    print(f"✓ Generated synthetic clinical trial data ({len(df)} observations)")
    return df
