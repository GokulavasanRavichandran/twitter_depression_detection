# -*- coding: utf-8 -*-
import pandas as pd
import sys

def clean_data(file_name):
    df = pd.read_csv(file_name)
    df.dropna(inplace=True)
    cleaned_file_name = file_name.replace('.csv', '_cleaned.csv')
    df.to_csv(cleaned_file_name, index=False)
    print(f'Cleaned data saved to {cleaned_file_name}')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python clean.py <file_name>')
    else:
        file_name = sys.argv[1]
        clean_data(file_name)
        