from typing import List, Tuple

import numpy as np
import pandas as pd


def get_correct_date(dt_in: str, timestep_freq: str) -> np.datetime64:
    """
    Determine the correct date based on the input date and compositing window.
    """
    # Extract year, month, and day
    year = np.datetime64(dt_in, "D").astype("object").year
    month = np.datetime64(dt_in, "D").astype("object").month
    day = np.datetime64(dt_in, "D").astype("object").day

    if timestep_freq == "dekad":
        if day <= 10:
            correct_date = np.datetime64(f"{year}-{month:02d}-01")
        elif 11 <= day <= 20:
            correct_date = np.datetime64(f"{year}-{month:02d}-11")
        else:
            correct_date = np.datetime64(f"{year}-{month:02d}-21")
    elif timestep_freq == "month":
        correct_date = np.datetime64(f"{year}-{month:02d}-01")
    else:
        raise ValueError(f"Unknown compositing window: {timestep_freq}")

    return correct_date

def get_dekadal_dates(start_date: np.datetime64, num_timesteps: int) -> Tuple[list, list, list]:
    # Extract year, month, and day
    year = start_date.astype("object").year
    month = start_date.astype("object").month
    day = start_date.astype("object").day

    days, months, years = [day], [month], [year]
    while len(days) < num_timesteps:
        if day < 21:
            day += 10
        else:
            month = month + 1 if month < 12 else 1
            year = year + 1 if month == 1 else year
            day = 1
        days.append(day)
        months.append(month)
        years.append(year)
    return days, months, years

def get_monthly_dates(start_date: str, num_timesteps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # truncate to month precision
    start_month = np.datetime64(start_date, "M")
    # generate date vector based on the number of timesteps
    date_vector = start_month + np.arange(
        num_timesteps, dtype="timedelta64[M]"
    )

    # generate day, month and year vectors with numpy operations
    days = np.ones(num_timesteps, dtype=int)
    months = (date_vector.astype("datetime64[M]").astype(int) % 12) + 1
    years = (date_vector.astype("datetime64[Y]").astype(int)) + 1970
    return days, months, years

def get_timestamps(start_date: str, timestep_freq: str, num_timesteps: int) -> np.ndarray:
    """
    Generate an array of dates based on the specified compositing window.
    """
    # adjust start date depending on the compositing window
    start_date = get_correct_date(start_date, timestep_freq)

    # Generate date vector depending on the compositing window
    if timestep_freq == "dekad":
        days, months, years = get_dekadal_dates(start_date, num_timesteps)
    elif timestep_freq == "month":
        days, months, years = get_monthly_dates(start_date, num_timesteps)
    else:
        raise ValueError(f"Unknown compositing window: {timestep_freq}")

    return  [
        np.array(days),
        np.array(months),
        np.array(years),
        ]
