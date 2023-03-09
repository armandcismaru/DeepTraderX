"""
Module containing code for production of customer orders
"""
import random
import sys

import config
from tbse_msg_classes import Order
from tbse_sys_consts import TBSE_SYS_MAX_PRICE, TBSE_SYS_MIN_PRICE


# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
def customer_orders(time, coid, traders, trader_stats, order_sched, pending, verbose):
    """
    Produce and distribute customer orders to traders
    Mostly unaltered from original BSE code by Dave Cliff
    :param time: current curr_time
    :param coid: last used customer order ID
    :param traders: List of traders
    :param trader_stats: number of buyers and number of sellers
    :param order_sched: order schedule
    :param pending: pending orders to be distributed
    :param verbose: should verbose logging be printed to console
    :return: List containing left over pending orders, cancellations to be made and the final customer ID used
    """
    def sys_min_check(price):
        """
        Check if order price is below system minimum price and sets price to be minimum if it is
        :param price: Order price
        :return: new order price
        """
        if price < TBSE_SYS_MIN_PRICE:
            print('WARNING: price < bse_sys_min -- clipped')
            price = TBSE_SYS_MIN_PRICE
        return price

    def sys_max_check(price):
        """
        Check if order price is above system maximum price and sets price to be maximum if it is
        :param price: Order price
        :return: new order price
        """
        if price > TBSE_SYS_MAX_PRICE:
            print('WARNING: price > bse_sys_max -- clipped')
            price = TBSE_SYS_MAX_PRICE
        return price

    def get_order_price(i, schedule, schedule_end, n, stepmode, time_of_issue):
        """
        Calculates order price for a new customer order
        :param i: Index of either buyer or seller trader
        :param schedule: Order schedule
        :param schedule_end: End curr_time of order schedule
        :param n: number of buyers or sellers
        :param stepmode: Stepmode of order schedule
        :param time_of_issue: Time order should be issued after
        :return: Order price
        """
        # pylint: disable=too-many-branches,too-many-statements
        if config.useInputFile:
            if len(schedule[0]) > 2:
                offset_function = schedule[0][2][0]
                offset_function_params = [schedule_end] + list(schedule[0][2][1])
                if callable(offset_function):
                    # same offset for min and max
                    offset_min = offset_function(time_of_issue, offset_function_params)
                    offset_max = offset_min
                else:
                    sys.exit('FAIL: 3rd argument of schedule in get_order_price() should be [callable_fn [params]]')
                if len(schedule[0]) > 3:
                    # if second offset function is specfied, that applies only to the max value
                    offset_function = schedule[0][3][0]
                    offset_function_params = [schedule_end] + list(schedule[0][3][1])
                    if callable(offset_function):
                        # this function applies to max
                        offset_max = offset_function(time_of_issue, offset_function_params)
                    else:
                        sys.exit('FAIL: 4th argument of schedule in get_order_price() should be [callable_fn [params]]')
            else:
                offset_min = 0.0
                offset_max = 0.0
        else:
            # does the first schedule range include optional dynamic offset function(s)?
            if len(schedule[0]) > 2:
                offset_function = schedule[0][2]
                if callable(offset_function):
                    # same offset for min and max
                    offset_min = offset_function(time_of_issue)
                    offset_max = offset_min
                else:
                    sys.exit('FAIL: 3rd argument of schedule in get_order_price() not callable')
                if len(schedule[0]) > 3:
                    # if second offset function is specfied, that applies only to the max value
                    offset_function = schedule[0][3]
                    if callable(offset_function):
                        # this function applies to max
                        offset_max = offset_function(time_of_issue)
                    else:
                        sys.exit('FAIL: 4th argument of schedule in get_order_price() not callable')
            else:
                offset_min = 0.0
                offset_max = 0.0

        p_min = sys_min_check(offset_min + min(schedule[0][0], schedule[0][1]))
        p_max = sys_max_check(offset_max + max(schedule[0][0], schedule[0][1]))
        p_range = p_max - p_min
        step_size = p_range / (n - 1)
        half_step = round(step_size / 2.0)

        if stepmode == 'fixed':
            new_order_price = p_min + int(i * step_size)
        elif stepmode == 'jittered':
            new_order_price = p_min + int(i * step_size) + random.randint(-half_step, half_step)
        elif stepmode == 'random':
            if len(schedule) > 1:
                # more than one schedule: choose one equiprobably
                s = random.randint(0, len(schedule) - 1)
                p_min = sys_min_check(min(schedule[s][0], schedule[s][1]))
                p_max = sys_max_check(max(schedule[s][0], schedule[s][1]))
            new_order_price = random.randint(p_min, p_max)
        else:
            sys.exit('ERROR: Unknown stepmode in schedule')
        new_order_price = sys_min_check(sys_max_check(new_order_price))
        return new_order_price

    # pylint: disable=too-many-branches
    def get_issue_times(n_traders, stepmode, interval, shuffle, fit_to_interval):
        """
        Produces issue times for orders
        :param n_traders: The number of traders
        :param stepmode:
        :param interval: Gap between each set of orders
        :param shuffle: Boolean of whether it should be shuffled or not
        :param fit_to_interval:
        :return: Issue times
        """
        interval = float(interval)
        if n_traders < 1:
            sys.exit('FAIL: n_traders < 1 in get_issue_times()')
        elif n_traders == 1:
            t_step = interval
        else:
            t_step = interval / (n_traders - 1)
        arr_time = 0
        order_issue_times = []
        for i in range(n_traders):
            if stepmode == 'periodic':
                arr_time = interval
            elif stepmode == 'drip-fixed':
                arr_time = i * t_step
            elif stepmode == 'drip-jitter':
                arr_time = i * t_step + t_step * random.random()
            elif stepmode == 'drip-poisson':
                # poisson requires a bit of extra work
                inter_arrival_time = random.expovariate(n_traders / interval)
                arr_time += inter_arrival_time
            else:
                sys.exit('FAIL: unknown t-stepmode in get_issue_times()')
            order_issue_times.append(arr_time)

        # at this point, arr_time is the last arrival i
        if fit_to_interval and ((arr_time > interval) or (arr_time < interval)):
            # generated sum of inter-arrival times longer than the interval
            # squish them back so that last arrival falls at t=interval
            for i in range(n_traders):
                order_issue_times[i] = interval * (order_issue_times[i] / arr_time)
        # optionally randomly shuffle the times
        if shuffle:
            for i in range(n_traders):
                i = (n_traders - 1) - i
                j = random.randint(0, i)
                tmp = order_issue_times[i]
                order_issue_times[i] = order_issue_times[j]
                order_issue_times[j] = tmp
        return order_issue_times

    def get_sched_mode(curr_time, order_schedule):
        """
        Extract details of the order_schedule
        :param curr_time: Current time
        :param order_schedule: Order Schedule
        :return: The range of values the orders can take, the stepmode and the schedule end time.
        """
        schedrange = None
        stepmode = None
        sched_end_time = None
        got_one = False
        for schedule in order_schedule:
            if schedule['from'] <= curr_time < schedule['to']:
                # within the timezone for this schedule
                schedrange = schedule['ranges']
                stepmode = schedule['stepmode']
                sched_end_time = schedule['to']
                got_one = True
                break  # jump out the loop -- so the first matching timezone has priority over any others
        if not got_one:
            sys.exit(f'Fail: t={curr_time:5.2f} not within any timezone in order_schedule={order_schedule}')
        return schedrange, stepmode, sched_end_time

    n_buyers = trader_stats['n_buyers']
    n_sellers = trader_stats['n_sellers']

    shuffle_times = True

    cancellations = []

    if len(pending) < 1:
        # list of pending (to-be-issued) customer orders is empty, so generate a new one
        new_pending = []

        # demand side (buyers)
        issue_times = get_issue_times(n_buyers, order_sched['timemode'], order_sched['interval'], shuffle_times, True)
        order_type = 'Bid'
        (sched, mode, sched_end) = get_sched_mode(time, order_sched['dem'])
        for t in range(n_buyers):
            issue_time = time + issue_times[t]
            t_name = f'B{str(t).zfill(2)}'
            order_price = get_order_price(t, sched, sched_end, n_buyers, mode, issue_time)
            order = Order(t_name, order_type, order_price, 1, issue_time, coid, -3.14)
            new_pending.append(order)
            coid += 1

        # supply side (sellers)
        issue_times = get_issue_times(n_sellers, order_sched['timemode'], order_sched['interval'], shuffle_times, True)
        order_type = 'Ask'
        (sched, mode, sched_end) = get_sched_mode(time, order_sched['sup'])
        for t in range(n_sellers):
            issue_time = time + issue_times[t]
            t_name = f'S{str(t).zfill(2)}'
            order_price = get_order_price(t, sched, sched_end, n_sellers, mode, issue_time)
            order = Order(t_name, order_type, order_price, 1, issue_time, coid, -3.14)
            new_pending.append(order)
            coid += 1
    else:
        # there are pending future orders: issue any whose timestamp is in the past
        new_pending = []
        for order in pending:
            if order.time < time:
                # this order should have been issued by now
                # issue it to the trader
                t_name = order.tid
                response = traders[t_name].add_order(order, verbose)
                if verbose:
                    print(f'Customer order: {response} {order}')
                if response == 'LOB_Cancel':
                    cancellations.append(t_name)
                    if verbose:
                        print(f'Cancellations: {cancellations}')
                # and then don't add it to new_pending (i.e., delete it)
            else:
                # this order stays on the pending list
                new_pending.append(order)
    return [new_pending, cancellations, coid]
