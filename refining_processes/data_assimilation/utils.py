"""
Utility functions for data assimilation.
"""

import numpy as np
import pandas as pd


def svd_inverse(matrix: np.ndarray, pseudo: bool = False,
                threshold: float = 1e-10) -> np.ndarray:
    """
    Calculate the inverse of a matrix using SVD method.

    Inputs:
        matrix: Matrix to invert
        pseudo: If True, calculate pseudo-inverse (works for any matrix)
                If False, calculate regular inverse (needs square matrix)
        threshold: Small number to ignore tiny singular values (avoid division by zero)

    Outputs:
        inverse_matrix: The calculated inverse matrix

    Description:
        Breaks matrix into three parts using SVD, then inverts the important parts.
        Good for solving equations when matrix is difficult to invert normally.
        Used in data assimilation to solve linear systems.
    """
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    if pseudo:
        S_inv = np.zeros_like(matrix.T, dtype=float)
        diag = np.where(S > threshold, 1 / S, 0.0)
        S_inv[:len(S), :len(S)] = np.diag(diag)
    else:
        if matrix.shape[0] != matrix.shape[1] or not np.all(S > threshold):
            raise ValueError("Matrix is not invertible")
        S_inv = np.diag(1.0 / S)

    return Vt.T @ S_inv @ U.T



def generate_date_list(start_date, end_date, date_format='%Y%m%d'):
    """
    Make a list of all dates between start and end dates.

    Inputs:
        start_date: First date (string or date object)
        end_date: Last date (string or date object)
        date_format: How to format dates (default: YYYYMMDD)

    Outputs:
        result_dates: List of date strings from start to end

    Description:
        Creates daily date list between two dates.
        Example: start='20231201', end='20231203' gives 3 dates.
        Useful for looping through many days of data or making file lists.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    result_dates = dates.strftime('%Y%m%d').tolist()

    return result_dates
    
