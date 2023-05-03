# pylint: disable=C0103,C0413,E0401,C0302
"""-*- coding: utf-8 -*-

TBSE: The Threaded Bristol Stock Exchange

Version 1.0; Augusts 1st, 2020.

------------------------
Copyright (c) 2020, Michael Rollins

MIT Open-Source License:
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

------------------------



TBSE is a very simple simulation of automated execution traders
operating on a very simple model of a limit order book (LOB) exchange
extended from Dave Cliff's Bristol Stock Exchange (BSE). TBSE uses
Python multi-threading to allow multiple traders to operate simultaneously
which means that the execution t of trading algorithms can affect
their performance.

major simplifications in this version:
      (a) only one financial instrument being traded
      (b) traders can only trade contracts of size 1 (will add variable quantities later)
      (c) each trader can have max of one order per single orderbook.
      (d) traders can replace/overwrite earlier orders, and/or can cancel

NB this code has been written to be readable/intelligible, not efficient!"""

import csv
import math
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import queue
import random
import sys
import threading
import time
from pathlib import Path
from datetime import datetime
from random import randint
import boto3
import src.config

myDir = os.getcwd()
sys.path.append(myDir)

path = Path(myDir)

a = str(path.parent.absolute())
sys.path.append(a)

from src.tbse.tbse_exchange import Exchange
from src.tbse.tbse_customer_orders import customer_orders
from src.tbse.tbse_trader_agents import (
    TraderGiveaway,
    TraderShaver,
    TraderSniper,
    TraderZic,
    TraderZip,
    TraderAA,
    TraderGdx,
    DeepTrader,
)


# Adapted from original BSE code
def trade_stats(expid, traders, dumpfile):
    """dump CSV statistics on exchange data and trader population to file for later analysis
    this makes no assumptions about the number of types of traders, or
    the number of traders of any one type -- allows either/both to change
    between successive calls, but that does make it inefficient as it has to
    re-analyse the entire set of traders on each call"""
    trader_types = {}
    for t in traders:
        trader_type = traders[t].ttype
        t_time1 = 0
        t_time2 = 0
        if trader_type in trader_types:
            t_balance = trader_types[trader_type]["balance_sum"] + traders[t].balance
            t_trades = trader_types[trader_type]["trades_sum"] + traders[t].n_trades
            if traders[t].last_quote is not None:
                t_time1 = (
                    trader_types[trader_type]["time1"]
                    + traders[t].times[0] / traders[t].times[2]
                )
                t_time2 = (
                    trader_types[trader_type]["time2"]
                    + traders[t].times[1] / traders[t].times[3]
                )
            n = trader_types[trader_type]["n"] + 1
        else:
            t_balance = traders[t].balance
            if traders[t].last_quote is not None:
                t_time1 = traders[t].times[0] / traders[t].times[2]
                t_time2 = traders[t].times[1] / traders[t].times[3]
            n = 1
            t_trades = traders[t].n_trades
        trader_types[trader_type] = {
            "n": n,
            "balance_sum": t_balance,
            "trades_sum": t_trades,
            "time1": t_time1,
            "time2": t_time2,
        }

    dumpfile.write(f"{expid}")
    for trader_type in sorted(list(trader_types.keys())):
        n = trader_types[trader_type]["n"]
        s = trader_types[trader_type]["balance_sum"]
        t = trader_types[trader_type]["trades_sum"]
        time1 = trader_types[trader_type]["time1"]
        time2 = trader_types[trader_type]["time2"]
        dumpfile.write(
            f", {trader_type}, {s}, {n}, {(s / float(n)):.2f}, "
            f"{(t / float(n)):.2f}, {(time1 / float(n)):.8f}, {(time2 / float(n)):.8f}"
        )

    dumpfile.write("\n")


# From original BSE code
def populate_market(trader_spec, traders, shuffle, verbose):
    """create a bunch of trader_list from trader_spec
    returns tuple (n_buyers, n_sellers)
    optionally shuffles the pack of buyers and the pack of sellers"""

    # pylint: disable=too-many-return-statements
    def create_trader(robot_type, name):
        """
        Function that creates instances of the different Trader Types
        :param robot_type: String representing type of trader to be created
        :param name: String, name given to trader
        :return: Instantiated Trader object
        """
        if robot_type == "GVWY":
            return TraderGiveaway("GVWY", name, 0.00, 0)
        if robot_type == "ZIC":
            return TraderZic("ZIC", name, 0.00, 0)
        if robot_type == "SHVR":
            return TraderShaver("SHVR", name, 0.00, 0)
        if robot_type == "SNPR":
            return TraderSniper("SNPR", name, 0.00, 0)
        if robot_type == "ZIP":
            return TraderZip("ZIP", name, 0.00, 0)
        if robot_type == "AA":
            return TraderAA("AA", name, 0.00, 0)
        if robot_type == "GDX":
            return TraderGdx("GDX", name, 0.00, 0)
        if robot_type == "DTR":
            return DeepTrader("DTR", name, 0.00, 0, "DeepTrader2_2")
        sys.exit(f"FATAL: don't know robot type {robot_type}\n")

    def shuffle_traders(ttype_char, n, trader_list):
        """
        Shuffles traders to avoid any biases caused by trader position.
        :param ttype_char: 'B' if buyers, 'S' if sellers
        :param n: int - number of traders being shuffles
        :param trader_list: list of traders to shuffle
        """
        for swap in range(n):
            t1 = (n - 1) - swap
            t2 = random.randint(0, t1)
            t1name = f"{ttype_char}{str(t1).zfill(2)}"
            t2name = f"{ttype_char}{str(t2).zfill(2)}"
            trader_list[t1name].tid = t2name
            trader_list[t2name].tid = t1name
            temp = trader_list[t1name]
            trader_list[t1name] = trader_list[t2name]
            trader_list[t2name] = temp

    n_buyers = 0
    for bs in trader_spec["buyers"]:
        trader_type = bs[0]
        for _ in range(bs[1]):
            trader_name = f"B{str(n_buyers).zfill(2)}"  # buyer i.d. string
            traders[trader_name] = create_trader(trader_type, trader_name)
            n_buyers = n_buyers + 1

    if n_buyers < 1:
        sys.exit("FATAL: no buyers specified\n")

    if shuffle:
        shuffle_traders("B", n_buyers, traders)

    n_sellers = 0
    for ss in trader_spec["sellers"]:
        trader_type = ss[0]
        for _ in range(ss[1]):
            trader_name = f"S{str(n_sellers).zfill(2)}"  # buyer i.d. string
            traders[trader_name] = create_trader(trader_type, trader_name)
            n_sellers = n_sellers + 1

    if n_sellers < 1:
        sys.exit("FATAL: no sellers specified\n")

    if shuffle:
        shuffle_traders("S", n_sellers, traders)

    if verbose:
        for t in range(n_buyers):
            bname = f"B{str(t).zfill(2)}"
            print(traders[bname])
        for t in range(n_sellers):
            bname = f"S{str(t).zfill(2)}"
            print(traders[bname])

    return {"n_buyers": n_buyers, "n_sellers": n_sellers}


# pylint: disable=too-many-arguments,too-many-locals,too-many-branches
def run_exchange(
    sess_id,
    traders,
    exchange,
    order_q,
    trader_qs,
    kill_q,
    start_event,
    start_time,
    sess_length,
    virtual_end,
    process_verbose,
    tdump_file,
    dump_each_trade,
    lobframes,
    lob_out,
    data_file,
):
    """
    Function for running of the exchange.
    :param exchange: Exchange object
    :param order_q: Queue on which new orders are sent to the queue
    :param trader_qs: Queues by which traders receive updates from the exchange
    :param kill_q: Queue where orders to be removed from the exchange are placed
    :param start_event: Event indicating if the exchange is active
    :param start_time: float, represents the start t (seconds since 1970)
    :param sess_length: int, number of seconds the
    :param virtual_end: The number of virtual seconds the trading day lasts for
    :param process_verbose: Flag indicating whether additional information about order processing should be printed
                            to console
    :return: Returns 0 on completion of trading day
    """
    completed_coid = {}
    start_event.wait()
    while start_event.is_set():
        virtual_time = (time.time() - start_time) * (virtual_end / sess_length)

        while kill_q.empty() is False:
            exchange.del_order(virtual_time, kill_q.get())

        order = order_q.get()
        if order.coid in completed_coid:
            if completed_coid[order.coid]:
                continue
        else:
            completed_coid[order.coid] = False

        limits = [0, 0]
        if order is not None:
            if order.otype == "Bid":
                limits[0] = order.price
            if order.otype == "Ask":
                limits[1] = order.price

        (trade, lob) = exchange.process_order2(
            virtual_time, order, process_verbose, None
        )

        if trade is not None:
            completed_coid[order.coid] = True
            completed_coid[trade["counter"]] = True
            for q in trader_qs:
                q.put([trade, order, lob])
            if lobframes is not None:
                _ = exchange.publish_lob(virtual_time, lobframes, False)

            if dump_each_trade:
                exchange.trade_stats(
                    sess_id,
                    traders,
                    tdump_file,
                    virtual_time,
                    exchange.publish_lob(virtual_time, lobframes, False),
                )

            if lob_out:
                exchange.lob_data_out(virtual_time, data_file, limits)
    return 0


# pylint: disable=too-many-arguments,too-many-locals
def run_trader(
    trader,
    exchange,
    order_q,
    trader_q,
    start_event,
    start_time,
    sess_length,
    virtual_end,
    respond_verbose,
    bookkeep_verbose,
):
    """
    Function for running a single trader. Multiple of these are run on a number of threads created in market_session()
    :param trader: The trader this function is controlling
    :param exchange: The exchange object
    :param order_q: Queue where the trader places new orders to send to the exchange
    :param trader_q: Queue where the exchange updates this trader on activities in the market
    :param start_event: Event flagging whether the market session is in progress
    :param start_time: Time at which market session begins
    :param sess_length: Length of market session in real world seconds
    :param virtual_end: Virtual number of seconds the market session ends at
    :param respond_verbose: Should the trader display additional information on its response
    :param bookkeep_verbose: Should there be additional bookkeeping information displayed on the console
    :return: Returns 0 at the end of the market session
    """
    start_event.wait()
    # dtr_quoted_prices = []
    # aa_quoted_prices = []
    while start_event.is_set():
        time.sleep(0.01)
        virtual_time = (time.time() - start_time) * (virtual_end / sess_length)
        time_left = (virtual_end - virtual_time) / virtual_end
        trade = None
        while trader_q.empty() is False:
            [trade, order, lob] = trader_q.get(block=False)
            if trade["party1"] == trader.tid:
                trader.bookkeep(trade, order, bookkeep_verbose, virtual_time)
            if trade["party2"] == trader.tid:
                trader.bookkeep(trade, order, bookkeep_verbose, virtual_time)
            time1 = time.time()
            trader.respond(virtual_time, lob, trade, respond_verbose)
            time2 = time.time()
            trader.times[1] += time2 - time1
            trader.times[3] += 1

        lob = exchange.publish_lob(virtual_time, None, False)
        time1 = time.time()
        trader.respond(virtual_time, lob, trade, respond_verbose)
        time2 = time.time()
        order = trader.get_order(virtual_time, time_left, lob)
        # if order is not None and trader.ttype == "DTR":
        #     dtr_quoted_prices.append(order.price)
        # if order is not None and trader.ttype == "AA":
        #     aa_quoted_prices.append(order.price)

        time3 = time.time()
        trader.times[1] += time2 - time1
        trader.times[3] += 1
        if order is not None:
            if order.otype == "Ask" and order.price < trader.orders[order.coid].price:
                sys.exit("Bad ask")
            if order.otype == "Bid" and order.price > trader.orders[order.coid].price:
                sys.exit("Bad bid")
            trader.n_quotes = 1
            order_q.put(order)
            trader.times[0] += time3 - time2
            trader.times[2] += 1

    # if dtr_quoted_prices:
    #     print(dtr_quoted_prices)
    # if aa_quoted_prices:
    #     print()

    return 0


# one session in the market
# pylint: disable=too-many-arguments,too-many-locals,consider-using-with,too-many-statements
def market_session(
    sess_id,
    sess_length,
    virtual_end,
    trader_spec,
    order_schedule,
    start_event,
    verbose,
    dumpfile,
    dump_data,
    schedule_n=0,
    lob_out=True,
):
    """
    Function representing a market session
    :param sess_id: ID of the session
    :param sess_length: Length of session in real world seconds
    :param virtual_end: Number of virtual seconds before the session ends
    :param trader_spec: JSON data representing the number and types of traders on the market
    :param order_schedule: JSON data representing the supply/demand curve of the market
    :param start_event: Event showing whether the market session is in progress
    :param verbose: Should additional information be printed to the console
    :return: Returns the number of threads operating at the end of the session. Used to check threads didn't crash.
    """

    # initialise the exchange
    exchange = Exchange()
    order_q = queue.Queue()
    kill_q = queue.Queue()

    start_time = time.time()

    orders_verbose = False
    process_verbose = False
    respond_verbose = False
    bookkeep_verbose = False

    # if dump_data:
    #     lobframes = open(sess_id + "_LOB_frames.csv", "w", encoding="utf-8")
    # else:
    lobframes = None

    # create a bunch of traders
    traders = {}
    trader_threads = []
    trader_qs = []
    trader_stats = populate_market(trader_spec, traders, True, verbose)

    lob_file_name = str(schedule_n) + "-" + str(time.time())
    data_file = None
    if lob_out:
        data_file = open(f"{lob_file_name}.csv", "w", encoding="utf-8")

    # create threads and queues for traders
    for i in range(0, len(traders)):
        trader_qs.append(queue.Queue())
        tid = list(traders.keys())[i]
        trader_threads.append(
            threading.Thread(
                target=run_trader,
                args=(
                    traders[tid],
                    exchange,
                    order_q,
                    trader_qs[i],
                    start_event,
                    start_time,
                    sess_length,
                    virtual_end,
                    respond_verbose,
                    bookkeep_verbose,
                ),
            )
        )

    ex_thread = threading.Thread(
        target=run_exchange,
        args=(
            sess_id,
            traders,
            exchange,
            order_q,
            trader_qs,
            kill_q,
            start_event,
            start_time,
            sess_length,
            virtual_end,
            process_verbose,
            dumpfile,
            dump_data,
            lobframes,
            lob_out,
            data_file,
        ),
    )

    # start exchange thread
    ex_thread.start()

    # start trader threads
    for thread in trader_threads:
        thread.start()

    start_event.set()

    pending_cust_orders = []

    if verbose:
        print(f"\n{sess_id};  ")

    cuid = 0  # Customer order id

    while time.time() < (start_time + sess_length):
        # print(
        #     f"Time: {time.time() - start_time} / {sess_length} ({(time.time() - start_time) / sess_length * 100}%)"
        # )
        virtual_time = (time.time() - start_time) * (virtual_end / sess_length)
        # distribute customer orders
        [pending_cust_orders, kills, cuid] = customer_orders(
            virtual_time,
            cuid,
            traders,
            trader_stats,
            order_schedule,
            pending_cust_orders,
            orders_verbose,
        )
        # if any newly-issued customer orders mean quotes on the LOB need to be cancelled, kill them
        if len(kills) > 0:
            if verbose:
                print(f"Kills: {kills}")
            for kill in kills:
                if verbose:
                    print(f"last_quote={traders[kill].last_quote}")
                if traders[kill].last_quote is not None:
                    kill_q.put(traders[kill].last_quote)
                    if verbose:
                        print(f"Killing order {str(traders[kill].last_quote)}")
        time.sleep(0.01)
    print("Session complete")

    start_event.clear()
    len_threads = len(threading.enumerate())

    # close exchange thread
    ex_thread.join()

    # close trader threads
    for thread in trader_threads:
        thread.join()

    # close lobframes
    if lobframes is not None:
        lobframes.close()

    # end of an experiment -- dump the tape
    if dump_data:
        exchange.tape_dump("transactions.csv", "a", "keep")

    # write trade_stats for this experiment NB end-of-session summary only
    if len_threads == len(traders) + 2:
        trade_stats(sess_id, traders, tdump)

    # print(f"Session {sess_id} complete")
    if lob_out:
        data_file.close()
        s3.upload_file(
            f"{lob_file_name}.csv", "output-data-fz19792", f"{lob_file_name}.csv"
        )
        print(f"uploading to s3 for {sess_id}...")
        os.remove(lob_file_name + ".csv")

    return len_threads


#############################


def get_order_schedule():
    """
    Produces order schedule as defined in src.config file.
    :return: Order schedule representing the supply/demand curve of the market
    """
    range_max = random.randint(
        src.config.supply["rangeMax"]["rangeLow"],
        src.config.supply["rangeMax"]["rangeHigh"],
    )
    range_min = random.randint(
        src.config.supply["rangeMin"]["rangeLow"],
        src.config.supply["rangeMin"]["rangeHigh"],
    )

    if src.config.useInputFile:
        offset_function_event_list = get_offset_event_list()
        range_s = (
            range_min,
            range_max,
            [real_world_schedule_offset_function, [offset_function_event_list]],
        )
    elif src.config.useOffset:
        range_s = (range_min, range_max, schedule_offset_function)
    else:
        range_s = (range_min, range_max)

    supply_schedule = [
        {
            "from": 0,
            "to": src.config.virtualSessionLength,
            "ranges": [range_s],
            "stepmode": src.config.stepmode,
        }
    ]

    if not src.config.symmetric:
        range_max = random.randint(
            src.config.demand["rangeMax"]["rangeLow"],
            src.config.demand["rangeMax"]["rangeHigh"],
        )
        range_min = random.randint(
            src.config.demand["rangeMin"]["rangeLow"],
            src.config.demand["rangeMin"]["rangeHigh"],
        )

    if src.config.useInputFile:
        offset_function_event_list = get_offset_event_list()
        range_d = (
            range_min,
            range_max,
            [real_world_schedule_offset_function, [offset_function_event_list]],
        )
    elif src.config.useOffset:
        range_d = (range_min, range_max, schedule_offset_function)
    else:
        range_d = (range_min, range_max)

    demand_schedule = [
        {
            "from": 0,
            "to": src.config.virtualSessionLength,
            "ranges": [range_d],
            "stepmode": src.config.stepmode,
        }
    ]

    return {
        "sup": supply_schedule,
        "dem": demand_schedule,
        "interval": src.config.interval,
        "timemode": src.config.timemode,
    }


def schedule_offset_function(t):
    """
    schedule_offset_function returns t-dependent offset on schedule prices
    :param t: Time at which we are retrieving the offset
    :return: The offset
    """
    print(t)
    pi2 = math.pi * 2
    c = math.pi * 3000
    wavelength = t / c
    gradient = 100 * t / (c / pi2)
    amplitude = 100 * t / (c / pi2)
    offset = gradient + amplitude * math.sin(wavelength * t)
    return int(round(offset, 0))


def real_world_schedule_offset_function(t, params):
    """
    Returns offset based on real world data read in via CSV
    :param t: Time at which the offset is being calculated
    :param params: Parameters used to find offset
    :return: The offset
    """
    end_time = float(params[0])
    offset_events = params[1]
    # this is quite inefficient: on every call it walks the event-list
    # come back and make it better
    percent_elapsed = t / end_time
    offset = 0
    for event in offset_events:
        offset = event[1]
        if percent_elapsed < event[0]:
            break
    return offset


# pylint: disable:too-many-locals


def get_offset_event_list():
    """
    read in a real-world-data data-file for the SDS offset function
    having this here means it's only read in once
    this is all quite skanky, just to get it up and running
    assumes data file is all for one date, sorted in t order, in correct format, etc. etc.
    :return: list of offset events
    """
    with open(src.config.input_file, "r", encoding="utf-8") as input_file:
        rwd_csv = csv.reader(input_file)
        scale_factor = 80
        # first pass: get t & price events, find out how long session is, get min & max price
        min_price = None
        max_price = None
        first_time_obj = None
        price_events = []
        time_since_start = 0
        for line in rwd_csv:
            t = line[1]
            if first_time_obj is None:
                first_time_obj = datetime.strptime(t, "%H:%M:%S")
            time_obj = datetime.strptime(t, "%H:%M:%S")
            price = float(line[2])
            if min_price is None or price < min_price:
                min_price = price
            if max_price is None or price > max_price:
                max_price = price
            time_since_start = (time_obj - first_time_obj).total_seconds()
            price_events.append([time_since_start, price])
        # second pass: normalise times to fractions of entire t-series duration
        #              & normalise price range
        price_range = max_price - min_price
        end_time = float(time_since_start)
        offset_function_event_list = []
        for event in price_events:
            # normalise price
            normld_price = (event[1] - min_price) / price_range
            # clip
            normld_price = min(normld_price, 1.0)
            normld_price = max(0.0, normld_price)
            # scale & convert to integer cents
            price = int(round(normld_price * scale_factor))
            normld_event = [event[0] / end_time, price]
            offset_function_event_list.append(normld_event)
        return offset_function_event_list


# # Below here is where we set up and run a series of experiments

if __name__ == "__main__":
    if not src.config.parse_config():
        sys.exit()

    # Input configuration
    USE_CONFIG = False
    USE_CSV = False
    USE_COMMAND_LINE = False

    NUM_ZIC = src.config.numZIC
    NUM_ZIP = src.config.numZIP
    NUM_GDX = src.config.numGDX
    NUM_AA = src.config.numAA
    NUM_GVWY = src.config.numGVWY
    NUM_SHVR = src.config.numSHVR
    NUM_DTR = src.config.numDTR

    NUM_OF_ARGS = len(sys.argv)
    if NUM_OF_ARGS == 1:
        USE_CONFIG = True
    elif NUM_OF_ARGS == 2:
        USE_CSV = True
    elif NUM_OF_ARGS == 8:
        USE_COMMAND_LINE = True
        try:
            NUM_ZIC = int(sys.argv[1])
            NUM_ZIP = int(sys.argv[2])
            NUM_GDX = int(sys.argv[3])
            NUM_AA = int(sys.argv[4])
            NUM_GVWY = int(sys.argv[5])
            NUM_SHVR = int(sys.argv[6])
            NUM_DTR = int(sys.argv[7])
        except ValueError:
            print("ERROR: Invalid trader schedule. Please enter seven integer values.")
            sys.exit()
    else:
        print("Invalid input arguements.")
        print("Options for running TBSE:")
        print("	$ python3 tbse.py  ---  Run using trader schedule from src.config.")
        print(
            " $ python3 tbse.py <string>.csv  ---  Enter name of csv file describing a series of trader schedules."
        )
        print(
            " $ python3 tbse.py <int> <int> <int> <int> <int> <int> <int>  ---  Enter 7 integer values representing \
        trader schedule."
        )
        sys.exit()
    # pylint: disable=too-many-boolean-expressions
    if (
        NUM_ZIC < 0
        or NUM_ZIP < 0
        or NUM_GDX < 0
        or NUM_AA < 0
        or NUM_GVWY < 0
        or NUM_SHVR < 0
        or NUM_DTR < 0
    ):
        print("ERROR: Invalid trader schedule. All input integers should be positive.")
        sys.exit()

    # This section of code allows for the same order and trader schedules
    # to be tested src.config.numTrials times.

    if USE_CONFIG or USE_COMMAND_LINE:
        order_sched = get_order_schedule()

        buyers_spec = [
            ("ZIC", NUM_ZIC),
            ("ZIP", NUM_ZIP),
            ("GDX", NUM_GDX),
            ("AA", NUM_AA),
            ("GVWY", NUM_GVWY),
            ("SHVR", NUM_SHVR),
            ("DTR", NUM_DTR),
        ]

        sellers_spec = buyers_spec
        traders_spec = {"sellers": sellers_spec, "buyers": buyers_spec}

        file_name = (
            f"{str(NUM_ZIC).zfill(2)}-{str(NUM_ZIP).zfill(2)}-{str(NUM_GDX).zfill(2)}-"
            f"{str(NUM_AA).zfill(2)}-{str(NUM_GVWY).zfill(2)}-{str(NUM_SHVR).zfill(2)}-{str(NUM_DTR).zfill(2)}.csv"
        )
        with open(file_name, "w", encoding="utf-8") as tdump:
            trader_count = 0
            for ttype in buyers_spec:
                trader_count += ttype[1]
            for ttype in sellers_spec:
                trader_count += ttype[1]

            if trader_count > 40:
                print("WARNING: Too many traders can cause unstable behaviour.")

            trial = 1
            dump_all = True

            while trial < (src.config.numTrials + 1):
                trial_id = f"trial{str(trial).zfill(7)}"
                start_session_event = threading.Event()
                try:
                    NUM_THREADS = market_session(
                        trial_id,
                        src.config.sessionLength,
                        src.config.virtualSessionLength,
                        traders_spec,
                        order_sched,
                        start_session_event,
                        False,
                        tdump,
                        dump_all,
                        0,
                        False,
                    )

                    print(f"Trial {trial} complete, {NUM_THREADS} threads running.")
                    if NUM_THREADS != trader_count + 2:
                        trial = trial - 1
                        start_session_event.clear()
                        time.sleep(0.5)
                except Exception as e:  # pylint: disable=broad-except
                    print("Error: Market session failed, trying again.")
                    print(e)
                    trial = trial - 1
                    start_session_event.clear()
                    time.sleep(0.5)
                tdump.flush()
                trial = trial + 1

    # To use this section of code run TBSE with 'python3 tbse.py <csv>'
    # and have a CSV file with name <string>.csv with a list of values
    # representing the number of each trader type present in the
    # market you wish to run. The order is:
    # 				ZIC,ZIP,GDX,AA,GVWY,SHVR,DTR
    # So an example entry would be: 5,5,0,0,5,5,0
    # which would be 5 ZIC traders, 5 ZIP traders, 5 Giveaway traders and
    # 5 Shaver traders. To have different buyer and seller specs modifications
    # would be needed.

    elif USE_CSV:
        server = sys.argv[1]
        ratios = []
        try:
            with open(server, newline="", encoding="utf-8") as csv_file:
                reader = csv.reader(csv_file, delimiter=",")
                for row in reader:
                    ratios.append(row)
        except FileNotFoundError:
            print("ERROR: File " + server + " not found.")
            sys.exit()
        except IOError as e:
            print("ERROR: " + e)
            sys.exit()

        trial_number = 1
        for no_of_schedule, ratio in enumerate(ratios):
            try:
                NUM_ZIC = int(ratio[0])
                NUM_ZIP = int(ratio[1])
                NUM_GDX = int(ratio[2])
                NUM_AA = int(ratio[3])
                NUM_GVWY = int(ratio[4])
                NUM_SHVR = int(ratio[5])
                NUM_DTR = int(ratio[6])
            except ValueError:
                print(
                    "ERROR: Invalid trader schedule. Please enter seven, comma-separated, integer values. Skipping "
                    "this trader schedule."
                )
                continue
            except Exception as e:  # pylint: disable=broad-except
                print(
                    "ERROR: Unknown input error. Skipping this trader schedule."
                    + str(e)
                )
                continue
            # pylint: disable=too-many-boolean-expressions
            if (
                NUM_ZIC < 0
                or NUM_ZIP < 0
                or NUM_GDX < 0
                or NUM_AA < 0
                or NUM_GVWY < 0
                or NUM_SHVR < 0
                or NUM_DTR < 0
            ):
                print(
                    "ERROR: Invalid trader schedule. All input integers should be positive. Skipping this trader"
                    " schedule."
                )
                continue

            file_name = (
                f"{randint(10, 99999)}-"
                f"{str(NUM_ZIC).zfill(2)}-{str(NUM_ZIP).zfill(2)}-{str(NUM_GDX).zfill(2)}-"
                f"{str(NUM_AA).zfill(2)}-{str(NUM_GVWY).zfill(2)}-{str(NUM_SHVR).zfill(2)}-{str(NUM_DTR).zfill(2)}.csv"
            )

            # pylint: disable=line-too-long
            try:
                s3 = boto3.client(
                    "s3",
                    aws_access_key_id="AWS_ACCESS_KEY_ID",
                    aws_secret_access_key="AWS_SECRET_ACCESS_KEY",
                    aws_session_token="AWS_SESSION_TOKEN",
                )
            except Exception as e:  # pylint: disable=broad-except
                print(e)

            with open(file_name, "w", encoding="utf-8") as tdump:
                for _ in range(0, src.config.numSchedulesPerRatio):
                    order_sched = get_order_schedule()

                    buyers_spec = [
                        ("ZIC", NUM_ZIC),
                        ("ZIP", NUM_ZIP),
                        ("GDX", NUM_GDX),
                        ("AA", NUM_AA),
                        ("GVWY", NUM_GVWY),
                        ("SHVR", NUM_SHVR),
                        ("DTR", NUM_DTR),
                    ]

                    sellers_spec = buyers_spec
                    traders_spec = {"sellers": sellers_spec, "buyers": buyers_spec}

                    trader_count = 0
                    for ttype in buyers_spec:
                        trader_count += ttype[1]
                    for ttype in sellers_spec:
                        trader_count += ttype[1]

                    if trader_count > 40:
                        print("WARNING: Too many traders can cause unstable behaviour.")

                    trial = 1
                    dump_all = False

                    while trial <= src.config.numTrialsPerSchedule:
                        trial_id = f"trial{str(trial_number).zfill(7)}"
                        start_session_event = threading.Event()
                        try:
                            NUM_THREADS = market_session(
                                trial_id,
                                src.config.sessionLength,
                                src.config.virtualSessionLength,
                                traders_spec,
                                order_sched,
                                start_session_event,
                                False,
                                tdump,
                                dump_all,
                                no_of_schedule,
                                lob_out=False,
                            )

                            if NUM_THREADS != trader_count + 2:
                                trial = trial - 1
                                trial_number = trial_number - 1
                                start_session_event.clear()
                                time.sleep(0.5)
                        except Exception as e:  # pylint: disable=broad-except
                            print("Market session failed. Trying again. " + str(e))
                            trial = trial - 1
                            trial_number = trial_number - 1
                            start_session_event.clear()
                            time.sleep(0.5)

                        tdump.flush()
                        trial = trial + 1
                        trial_number = trial_number + 1

                    s3.upload_file(
                        f"{file_name}", "experiment-data-fz19792", f"{file_name}"
                    )
                    print(f"Uploading {file_name} to s3...")
            # os.remove(file_name)

        sys.exit("Done Now")

    else:
        print("ERROR: An unknown error has occurred. Something is very wrong.")
        sys.exit()
