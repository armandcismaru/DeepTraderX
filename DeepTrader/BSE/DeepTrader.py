from DeepBSE import Trader, Order
import NeuralNetwork as nn
import numpy as np


class DeepTrader(Trader):

    def __init__(self, ttype, tid, balance, time, filename):
        self.ttype = ttype      # what type / strategy this trader is
        self.tid = tid          # trader unique ID code
        self.balance = balance  # money in the bank
        self.blotter = []       # record of trades executed
        # customer orders currently being worked (fixed at 1)
        self.orders = []
        self.n_quotes = 0       # number of quotes live on LOB
        self.willing = 1        # used in ZIP etc
        self.able = 1           # used in ZIP etc
        self.birthtime = time   # used when calculating age of a trader/strategy
        self.profitpertime = 0  # profit per unit time
        self.n_trades = 0       # how many trades has this trader done?
        self.lastquote = None   # record of what its last quote was
        self.filename = filename
        self.model = nn.load_network(self.filename)
        self.n_features = 12
        self.max_vals, self.min_vals = nn.normalization_values(self.filename)
        self.count = [0, 0]

    def create_input(self, lob):
        time = lob['time']
        bids = lob['bids']
        asks = lob['asks']
        qid = lob['QID']
        tape = lob['tape']
        limit = self.orders[0].price
        otype = self.orders[0].otype
        val = 0

        if otype == "Ask":
            val = 1

        mid_price = 0
        micro_price = 0
        imbalances = 0
        spread = 0
        delta_t = 0
        weighted_moving_average = 0
        smiths_alpha = 0

        if len(tape) != 0:

            tape = reversed(tape)
            trades = list(filter(lambda d: d['type'] == "Trade", tape))
            trade_prices = [t['price'] for t in trades]
            weights = [pow(0.9, t) for t in range(len(trades))]
            p_estimate = np.average(trade_prices, weights=weights)
            smiths_alpha = np.sqrt(
                np.sum(np.square(trade_prices-p_estimate)/len(trade_prices)))

            if (time == trades[0]['time']):

                trade_prices = trade_prices[1:]
                if len(trades) == 1:
                    delta_t = trades[0]['time'] - 0
                else:
                    delta_t = trades[0]['time'] - trades[1]['time']

        else:
            delta_t = time

        if (bids['best'] == None):
            x = 0
        else:
            x = bids['best']

        if (asks['best'] == None):
            y = 0
        else:
            y = asks['best']

        n_x = bids['n']
        n_y = asks['n']
        total = n_x + n_y

        spread = abs(y - x)
        mid_price = (x + y) / 2
        if (n_x + n_y != 0):
            micro_price = ((n_x * y) + (n_y * x)) / (n_x + n_y)
            imbalances = (n_x - n_y) / (n_x + n_y)

        market_conditions = np.array(
            [time, val, limit, mid_price, micro_price, imbalances, spread, x, y, delta_t, total, smiths_alpha, p_estimate])

        return market_conditions

    def getorder(self, time, countdown, lob):

        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:

            qid = lob['QID']
            tape = lob['tape']
            otype = self.orders[0].otype
            limit = self.orders[0].price

            # creating the input for the network
            x = self.create_input(lob)
            # print self.min_vals
            # print x
            normalized_input = (x-self.min_vals[:self.n_features]) / (
                self.max_vals[:self.n_features]-self.min_vals[:self.n_features])
            normalized_input = np.reshape(
                normalized_input, (1, -1, 1))
            # print normalized_input

            # dealing witht the networks output
            normalized_output = self.model.predict(normalized_input)[0][0]
            denormalized_output = ((normalized_output) * (
                self.max_vals[self.n_features] - self.min_vals[self.n_features])) + self.min_vals[self.n_features]
            model_price = int(round(denormalized_output, 0))

            if otype == "Ask":
                if model_price < limit:
                    self.count[1] += 1
                    model_price = limit
            else:
                if model_price > limit:
                    self.count[0] += 1
                    model_price = limit

            # print(seld.tid, self.count)

            order = Order(self.tid, otype, model_price,
                          self.orders[0].qty, time, qid)
            self.lastquote = order
        return order