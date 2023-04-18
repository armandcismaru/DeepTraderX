# pylint: skip-file

import csv
import seaborn as sns
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy import stats

df = pd.DataFrame(columns=["AA", "DTR"])

def moving_average(arr, window_size):
    i = 0
    moving_averages = []
    
    while i < len(arr) - window_size + 1:
        window = arr[i : i + window_size]
        window_average = round(sum(window) / window_size, 2)
        moving_averages.append(window_average)

        i += 1

    return moving_averages

with open("00-00-00-19-00-00-01r.csv", newline="") as csvfile:
    reader = csv.reader(csvfile)
    AA_sum = 0
    DTR_sum = 0
    aac, dtrc = 0, 0
    for row in reader:
        if float(row[11]) < 40:
            continue
        AA_sum += float(row[4])
        DTR_sum += float(row[11])

        if float(row[11]) > float(row[4]):
            dtrc += 1
        else:
            aac += 1

        new_row = {"AA": float(row[4]), "DTR": float(row[11])}
        df = df.append(new_row, ignore_index=True)

    # Print the result
    print(f"AA_sum: {AA_sum}")
    print(f"DTR_sum: {DTR_sum}")

    fig, ax = plt.subplots()

    dtr_smooth = moving_average(df['DTR'], 4)
    aa_smooth = moving_average(df['AA'], 4)

    plt.ylabel('Percentage difference of profits')
    plt.xlabel('Number of market session (1 session = 1 hour)') 
    x_ticks = [i for i in range(1, len(df["DTR"]) + 1)]

    # line, = ax.plot(x_ticks, aa_smooth, 'r-')  
    # line.set_label('AA profis')

    # line, = ax.plot(x_ticks, dtr_smooth, 'b-')
    # line.set_label('DTR profits')

    diff_profits = [(df['DTR'][i] / df['AA'][i]) - 1 for i in range(len(df['DTR']))]
    colors = ['#BF2F38' if p > 0 else '#002855' for p in diff_profits]
    print(diff_profits)
    plt.axhline(y=0, color='#CCCCCC', linestyle='--')
    
    plt.bar(x_ticks, diff_profits, color=colors)
    plt.plot(moving_average(diff_profits, 6), color='#002855')

    # sns.boxplot(data=df)
    # sns.violinplot(data=df)
    plt.plot()
    plt.show()

    sys.exit()
    for col in df.columns:
        shapiro_test = stats.shapiro(df[col])
        print("test for col", col, ":", shapiro_test)

    print(stats.ttest_ind(df["DTR"], df["AA"]))
    print(stats.f_oneway(df["DTR"], df["AA"]))
    print("DTR wins:", dtrc, "AA wins:", aac)
