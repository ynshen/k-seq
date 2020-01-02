"""
This module contains the methods for input and output
"""

# from .pre_processing import SequenceSet



def read_table_files(file_path, col_name):
    from pathlib import Path
    import pandas as pd

    file_path = Path(file_path)
    if file_path.suffix in ['.xls', 'xlsx']:
        df = pd.read_excel(io=file_path, sheet_name=0, header=1)
    elif file_path.suffix in ['csv']:
        df = pd.read_csv(file_path, header=1)
    return df[col_name]
