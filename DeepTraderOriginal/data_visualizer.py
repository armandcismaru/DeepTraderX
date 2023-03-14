# pylint: skip-file
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from data_handler import get_end_results

# import seaborn as sns
from scipy.stats import skew
import matplotlib as mpl
import seaborn as sns
import pandas as pd


def time_series_plot(filename):
    d_types = ["TIME", "MID", "MIC", "IMB", "SPR", "BID", "ASK", "TAR"]
    data = {}
    for d in d_types:
        data[d] = np.array([])
        data[d] = data_handler.read_data3("./Data/" + filename + ".csv", d)

    for i in range(1, len(d_types)):
        plt.plot(data["TIME"], data[d_types[i]], label=d_types[i])
        plt.legend()
        plt.savefig(f"./Diagrams/{filename}-{d_types[i]}")
        plt.close()


def hists(filename):
    d_types = ["TIME", "MID", "MIC", "IMB", "SPR", "BID", "ASK", "TAR"]
    data = {}
    print(filename)
    for d in d_types:
        data[d] = np.array([])
        data[d] = data_handler.read_data3("./Data/" + filename + ".csv", d)
        print(d, "skew :", skew(data[d]))
        print(d, "log skew: ", skew(np.sqrt(data[d])))

    for i in range(1, len(d_types)):
        plt.hist(data[d_types[i]], bins=20)
        plt.savefig(f"./Diagrams/Histogram-{filename}-{d_types[i]}")
        plt.close()


def accuracy_plot(actual, preds, baseline=[]):
    time = np.arange(len(actual))
    mpl.style.use("seaborn")
    plt.plot(time, actual, label="actual", color="red")
    plt.plot(time, preds, label="preds", color="green")

    plt.title("Predicting the Micro Price in Bristol Stock Exchage")

    # plt.plot(time, baseline,  label="mean", color='green')
    plt.xlabel("Prediction Number")
    plt.ylabel("Micro Price")
    plt.legend()
    plt.show()


def box_plots():
    tests = []

    for i in range(7):
        test = []
        for j in range(1, 101):
            file = i * 100 + j
            # print(file)
            test.append(get_end_results(file))
        tests.append(test)

    dtr_results = []
    trader_results = []
    for t in tests:
        temp = []
        temp1 = []
        for res in t:
            temp.append(res[list(res.keys())[0]])
            temp1.append((res[list(res.keys())[1]]))
        dtr_results.append(temp)
        trader_results.append(temp1)

    labels = [
        "DeepTrader",
        "DeepTrader",
        "DeepTrader",
        "DeepTrader",
        "DeepTrader",
        "DeepTrader",
        "DeepTrader",
        "SNIPER",
        "GDX",
        "AA",
        "GIVEAWAY",
        "ZIC",
        "ZIP",
        "SHAVER",
    ]

    temp = dtr_results[2][:]
    dtr_results[2][:] = trader_results[2][:]
    trader_results[2][:] = temp

    results = {}

    results["Test Number"] = [
        x % 7 for x in range(len(labels)) for y in range(len(dtr_results[0]))
    ]
    results["Trader"] = [
        labels[x] for x in range(len(labels)) for y in range(len(dtr_results[0]))
    ]
    # results["Blank"] = [labels[x]
    #  for x in range(len(labels)) for y in range(len(dtr_results[0]))]
    # results["trader_name"] = [labels[x] for x in range(len(labels)) for y in range(len(dtr_results[0]))]
    # results["trader_name"] = [labels[i] for x in range(
    #     len(dtr_results[1])) for y in range(len(dtr_results[0]))]
    l = dtr_results + trader_results
    flat_list = [item for sublist in l for item in sublist]
    results["Profit Per Trader"] = flat_list

    # print([len(results[k]) for  k in results])
    mdf = pd.DataFrame(results)
    # df.to_csv("one.csv")

    for opt in range(7):
        df = mdf.loc[mdf["Test Number"] == opt]
        df = df.reset_index()
        cols = ["SNIPER", "GDX", "AA", "GIVEAWAY", "ZIC", "ZIP", "SHAVER", "DeepTrader"]
        sns.set_style("darkgrid")
        f = plt.figure(figsize=(3.5, 4.5))
        grid = sns.boxplot(
            data=df,
            x="Trader",
            y="Profit Per Trader",
            color=["blue", "red"],
            orient="v",
            palette="bright",
            hue="Trader",
            width=0.5,
        )
        name = "Balanced Group Test - " + cols[opt] + " Trader"
        plt.title(name)
        grid.legend_.remove()
        plt.tight_layout()
        f.savefig(name + ".png")

    print("Done.")


def relationships():
    n = 3
    file = f"./Data/Training/trial{(n):04}.csv"
    data = data_handler.read_all_data(file)

    corr = np.corrcoef([data[:][d] for d in range(data.shape[1])])
    sns.heatmap(corr)
    plt.show()


def profit_time(number):
    market_data, trader_data = data_handler.collect_time_series_results(number)
    # mpl.style.use('seaborn')

    for t in trader_data.keys():
        plt.plot(market_data["TIME"], trader_data[t]["PPT"], label=t)

    plt.title("Profit per Trader over Time in BSE")
    plt.grid()
    plt.xlabel("Time (s)")
    plt.ylabel("Profit per Trader")
    plt.legend()
    plt.show()


def placement():
    results = [0, 0, 0, 0]
    for i in range(1, 36):
        data = data_handler.get_end_results(i)
        ppt = []

        for d in data:
            ppt.append(data[d]["PPT"])

        ppt = sorted(ppt, reverse=True)

        for i in range(len(ppt)):
            if data["DTR"]["PPT"] == ppt[i]:
                results[i] += 1

    pos = ["1st", "2nd", "3rd", "4th"]
    for i in range(len(pos)):
        print(f"{pos[i]}: {results[i]}")


def loss_train():
    epochs = np.arange(1, 21)
    loss = [
        0.2514,
        0.1702,
        0.0970,
        0.0428,
        0.0143,
        0.0057,
        0.0043,
        0.0038,
        0.0032,
        0.0026,
        0.0022,
        0.0019,
        0.0017,
        0.0016,
        0.0015,
        0.0014,
        0.0014,
        0.0013,
        0.0012,
        0.0012,
    ]
    plt.plot(epochs, loss, label="train", color="orange")

    plt.title("Loss")
    plt.xticks(np.arange(min(epochs), max(epochs) + 1, 1.0))
    # plt.yticks(np.linspace(max(0.005))
    # plt.plot(time, baseline,  label="mean", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    # plt.grid()
    # plt.legend()
    plt.savefig(f"./loss_train.png")
    plt.close()


def bar_chart():
    tests = []

    for i in range(7):
        test = []
        for j in range(1, 101):
            file = i * 100 + j
            # print(file)
            test.append(get_end_results(file))
        tests.append(test)

    results = []
    for t in tests:
        result = [0, 0]
        for res in t:
            result[0] += res[list(res.keys())[0]]
            result[1] += res[list(res.keys())[1]]

        result[0] /= 100
        result[1] /= 100

        results.append(result)

    dtr = []
    trader = []
    for i in range(len(results)):
        if i == 2:
            dtr.append(results[i][1])
            trader.append(results[i][0])
            continue
        dtr.append(results[i][0])
        trader.append(results[i][1])

    for d in range(len(dtr)):
        dtr[d] = round(dtr[d], 1)
        trader[d] = round(trader[d], 1)

    labels = ["SNPR", "GDX", "AA", "GVWY", "ZIC", "ZIP", "SHVR"]

    # , "GVWY", "ZIC", "ZIP", "SHVR"]

    x = np.arange(len(labels))  # the label locations
    colors = ["orange", "yellow", "green", "red", "grey", "black"]

    width = 0.35  # the width of the bars
    # colors = ['orange', 'green','red', 'purple','brown', 'pink','gray','black']

    fig, ax = plt.subplots(figsize=(20, 10))
    rects1 = ax.bar(x - width / 2, dtr, width, label="DeepTrader")
    rects2 = ax.bar(x + width / 2, trader, width, label="BSE Trading Strategy")

    # fig.figsize = (20, 10)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Profit Per Trader")
    ax.set_title("Balanced Group Test Results for DeepTrader")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    def autolabel(rects):
        # """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                "{}".format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)

    # childrenLS = ax.get_children()
    # barlist = list(filter(lambda x: isinstance(x, patches.Rectangle), childrenLS))
    # count = 0
    # for i in range(len(barlist)):
    #     if i ==(len(barlist)-1): continue
    #     elif (i < 15) == 0:
    #         barlist[i].set_color('blue')
    #         barlist[i+1].set_color('blue')

    #     elif(i> 15):
    #         barlist[i].set_color(colors[count])
    #         barlist[i+1].set_color(colors[count])
    #         count+=1

    ax.legend(loc="upper left")
    # fig.tight_layout()

    plt.show()

    print(trader)
    print(dtr)


def main():
    # loss_train()
    # bar_chart()
    # relationships()
    # for i in range(1,11):
    #     profit_time(i)
    # placement()
    box_plots()


if __name__ == "__main__":
    main()
