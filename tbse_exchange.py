""""
Module containing classes for describing a simulated exchange

Minor adaptions from the original BSE code by Dave Cliff
"""
import sys

from tbse_sys_consts import TBSE_SYS_MIN_PRICE, TBSE_SYS_MAX_PRICE


# pylint: disable=too-many-instance-attributes
class OrderbookHalf:
    """
    OrderbookHalf is one side of the book: a list of bids or a list of asks, each sorted best-first
    """
    def __init__(self, book_type, worst_price):
        # book_type: bids or asks?
        self.book_type = book_type
        # dictionary of orders received, indexed by Trader ID
        self.orders = {}
        # limit order book, dictionary indexed by price, with order info
        self.lob = {}
        # anonymized LOB, lists, with only price/qty info
        self.lob_anon = []
        # summary stats
        self.best_price = None
        self.best_tid = None
        self.worst_price = worst_price
        self.n_orders = 0  # how many orders?
        self.lob_depth = 0  # how many different prices on lob?

    def anonymize_lob(self):
        """
        anonymize a lob, strip out order details, format as a sorted list
        NB for asks, the sorting should be reversed
        """
        self.lob_anon = []
        for price in list(sorted(self.lob)):
            qty = self.lob[price][0]
            self.lob_anon.append([price, qty])

    def build_lob(self):
        """"
        take a list of orders and build a limit-order-book (lob) from it
        NB the exchange needs to know arrival times and trader-id associated with each order
        returns lob as a dictionary (i.e., unsorted)
        also builds anonymized version (just price/quantity, sorted, as a list) for publishing to traders
        """
        lob_verbose = False
        self.lob = {}

        for tid in list(self.orders):
            # 	order = orders_cp.get(tid)
            order = self.orders.get(tid)
            price = order.price
            if price in self.lob:
                # update existing entry
                qty = self.lob[price][0]
                order_list = self.lob[price][1]
                order_list.append([order.time, order.qty, order.tid, order.toid])
                self.lob[price] = [qty + order.qty, order_list]
            else:
                # create a new dictionary entry
                self.lob[price] = [order.qty, [[order.time, order.qty, order.tid, order.toid]]]
        # create anonymized version
        self.anonymize_lob()
        # record best price and associated trader-id
        if len(self.lob) > 0:
            if self.book_type == 'Bid':
                self.best_price = self.lob_anon[-1][0]
            else:
                self.best_price = self.lob_anon[0][0]
            self.best_tid = self.lob[self.best_price][1][0][2]
        else:
            self.best_price = None
            self.best_tid = None

        if lob_verbose:
            print(self.lob)

    def book_add(self, order):
        """
        add order to the dictionary holding the list of orders
        either overwrites old order from this trader
        or dynamically creates new entry in the dictionary
        so, max of one order per trader per list
        checks whether length or order list has changed, to distinguish addition/overwrite
        """

        n_orders = self.n_orders
        self.orders[order.tid] = order
        self.n_orders = len(self.orders)
        self.build_lob()

        if n_orders != self.n_orders:
            return 'Addition'
        return 'Overwrite'

    def book_del(self, order):
        """
        delete order from the dictionary holding the orders
        assumes max of one order per trader per list
        checks that the Trader ID does actually exist in the dict before deletion
        :param order: Order to be deleted
        """

        if self.orders.get(order.tid) is not None:
            del self.orders[order.tid]
            self.n_orders = len(self.orders)
            self.build_lob()

    def delete_best(self):
        """
        delete order: when the best bid/ask has been hit, delete it from the book
        the TraderID of the deleted order is return-value, as counterparty to the trade
        :return: Trader ID of the counterparty to the trade
        """
        best_price_orders = self.lob[self.best_price]
        best_price_qty = best_price_orders[0]
        best_price_counterparty = best_price_orders[1][0][2]
        if best_price_qty == 1:
            # here the order deletes the best price
            del self.lob[self.best_price]
            del self.orders[best_price_counterparty]
            self.n_orders = self.n_orders - 1
            if self.n_orders > 0:
                if self.book_type == 'Bid':
                    self.best_price = max(self.lob.keys())
                else:
                    self.best_price = min(self.lob.keys())
                self.lob_depth = len(self.lob.keys())
            else:
                self.best_price = self.worst_price
                self.lob_depth = 0
        else:
            # best_bid_qty>1 so the order decrements the quantity of the best bid
            # update the lob with the decremented order data
            self.lob[self.best_price] = [best_price_qty - 1, best_price_orders[1][1:]]

            # update the bid list: counterparty's bid has been deleted
            del self.orders[best_price_counterparty]
            self.n_orders = self.n_orders - 1
        self.build_lob()
        return best_price_counterparty


class Orderbook:
    """
    Orderbook for a single instrument: list of bids and list of asks
    """

    def __init__(self):
        self.bids = OrderbookHalf('Bid', TBSE_SYS_MIN_PRICE)
        self.asks = OrderbookHalf('Ask', TBSE_SYS_MAX_PRICE)
        self.tape = []
        self.quote_id = 0  # unique ID code for each quote accepted onto the book
        self.lob_string = '' # string representation of the limit order book

    def get_quote_id(self):
        """
        :return: Returns current quote id
        """
        return self.quote_id

    def increment_quote_id(self):
        """
        Increments quote_id by 1
        """
        self.quote_id += 1


class Exchange(Orderbook):
    """
    Exchange's internal orderbook
    """
    def add_order(self, order, verbose):
        """
        add a quote/order to the exchange and update all internal records; return unique i.d.
        :param order: order to be added to the exchange
        :param verbose: should verbose logging be printed to console
        :return: List containing order trader ID and the response from the OrderbookHalf (Either addition or overwrite)
        """
        order.toid = self.get_quote_id()
        self.increment_quote_id()

        if verbose:
            print(f'QUID: order.quid={order.qid} self.quote.id={self.quote_id}')

        if order.otype == 'Bid':
            response = self.bids.book_add(order)
            best_price = self.bids.lob_anon[-1][0]
            self.bids.best_price = best_price
            self.bids.best_tid = self.bids.lob[best_price][1][0][2]
        else:
            response = self.asks.book_add(order)
            best_price = self.asks.lob_anon[0][0]
            self.asks.best_price = best_price
            self.asks.best_tid = self.asks.lob[best_price][1][0][2]
        return [order.toid, response]

    def del_order(self, time, order):
        """
        delete a trader's quote/order from the exchange, update all internal records
        :param time: Time when the order is being deleted
        :param order: The order to delete
        """

        if order.otype == 'Bid':
            self.bids.book_del(order)
            if self.bids.n_orders > 0:
                best_price = self.bids.lob_anon[-1][0]
                self.bids.best_price = best_price
                self.bids.best_tid = self.bids.lob[best_price][1][0][2]
            else:  # this side of book is empty
                self.bids.best_price = None
                self.bids.best_tid = None
            cancel_record = {'type': 'Cancel', 't': time, 'order': order}
            self.tape.append(cancel_record)

        elif order.otype == 'Ask':
            self.asks.book_del(order)
            if self.asks.n_orders > 0:
                best_price = self.asks.lob_anon[0][0]
                self.asks.best_price = best_price
                self.asks.best_tid = self.asks.lob[best_price][1][0][2]
            else:  # this side of book is empty
                self.asks.best_price = None
                self.asks.best_tid = None
            cancel_record = {'type': 'Cancel', 't': time, 'order': order}
            self.tape.append(cancel_record)
        else:
            # neither bid nor ask?
            sys.exit('bad order type in del_quote()')

    # pylint: disable=consider-using-f-string
    def publish_lob(self, time, lob_file, verbose):
        """
        this returns the LOB data "published" by the exchange, i.e., what is accessible to the traders
        :param time: Current t
        :param verbose: Flag indicate whether additional information should be printed to console
        :return: JSON object representing the current state of the LOB
        """
        public_data = {
            't': time,
            'bids':
                {
                    'best': self.bids.best_price,
                    'worst': self.bids.worst_price,
                    'n': self.bids.n_orders,
                    'lob': self.bids.lob_anon
                },
            'asks':
                {
                    'best': self.asks.best_price,
                    'worst': self.asks.worst_price,
                    'n': self.asks.n_orders,
                    'lob': self.asks.lob_anon
                },
            'QID': self.quote_id,
            'tape': self.tape
        }

        if lob_file is not None:
            # build a linear character-string summary of only those prices on LOB with nonzero quantities
            lobstring ='Bid:,'
            n_bids = len(self.bids.lob_anon)
            if n_bids > 0:
                lobstring += '%d,' % n_bids
                for lobitem in self.bids.lob_anon:
                    price_str = '%d,' % lobitem[0]
                    qty_str = '%d,' % lobitem[1]
                    lobstring = lobstring + price_str + qty_str
            else:
                lobstring += '0,'
            lobstring += 'Ask:,'
            n_asks = len(self.asks.lob_anon)
            if n_asks > 0:
                lobstring += '%d,' % n_asks
                for lobitem in self.asks.lob_anon:
                    price_str = '%d,' % lobitem[0]
                    qty_str = '%d,' % lobitem[1]
                    lobstring = lobstring + price_str + qty_str
            else:
                lobstring += '0,'
            # is this different to the last lob_string?
            if lobstring != self.lob_string:
                # write it
                lob_file.write('%.3f, %s\n' % (time, lobstring))
                lob_file.write('%s\n' % "------------------")
                # remember it
                self.lob_string = lobstring

        if verbose:
            print(f'publish_lob: t={time}')
            print(f'BID_lob={public_data["bids"]["lob"]}')
            print(f'ASK_lob={public_data["asks"]["lob"]}')

        return public_data

    # publish lob here
    # pylint: disable=too-many-locals,too-many-branches
    def process_order2(self, time, order, verbose, lobframes):
        """
        receive an order and either add it to the relevant LOB (ie treat as limit order)
        or if it crosses the best counterparty offer, execute it (treat as a market order)

        :param time: Current time
        :param order: Order being processed
        :param verbose: Should verbose logging be printed to the console
        :return: transaction record and updated LOB
        """
        o_price = order.price
        counterparty = None
        counter_coid = None
        [toid, response] = self.add_order(order, verbose)  # add it to the order lists -- overwriting any previous order
        order.toid = toid
        if verbose:
            print(f'TOID: order.toid={order.toid}')
            print(f'RESPONSE: {response}')
        best_ask = self.asks.best_price
        best_ask_tid = self.asks.best_tid
        best_bid = self.bids.best_price
        best_bid_tid = self.bids.best_tid
        price = 0
        if order.otype == 'Bid':
            if self.asks.n_orders > 0 and best_bid >= best_ask:
                # bid lifts the best ask
                if verbose:
                    print(f"Bid ${o_price} lifts best ask")
                counterparty = best_ask_tid
                counter_coid = self.asks.orders[counterparty].coid
                price = best_ask  # bid crossed ask, so use ask price
                if verbose:
                    print('counterparty, price', counterparty, price)
                # delete the ask just crossed
                self.asks.delete_best()
                # delete the bid that was the latest order
                self.bids.delete_best()
        elif order.otype == 'Ask':
            if self.bids.n_orders > 0 and best_ask <= best_bid:
                # ask hits the best bid
                if verbose:
                    print("Ask ${o_price} hits best bid")
                # remove the best bid
                counterparty = best_bid_tid
                counter_coid = self.bids.orders[counterparty].coid
                price = best_bid  # ask crossed bid, so use bid price
                if verbose:
                    print('counterparty, price', counterparty, price)
                # delete the bid just crossed, from the exchange's records
                self.bids.delete_best()
                # delete the ask that was the latest order, from the exchange's records
                self.asks.delete_best()
        else:
            # we should never get here
            sys.exit('process_order() given neither Bid nor Ask')
        # NB at this point we have deleted the order from the exchange's records
        # but the two traders concerned still have to be notified
        if verbose:
            print(f'counterparty {counterparty}')

        lob = self.publish_lob(time, lobframes, False)
        if counterparty is not None:
            # process the trade
            if verbose:
                print(f'>>>>>>>>>>>>>>>>>TRADE t={time:5.2f} ${price} {counterparty} {order.tid}')
            transaction_record = {
                'type': 'Trade',
                't': time,
                'price': price,
                'party1': counterparty,
                'party2': order.tid,
                'qty': order.qty,
                'coid': order.coid,
                'counter': counter_coid
            }
            self.tape.append(transaction_record)
            return transaction_record, lob
        return None, lob

    def tape_dump(self, file_name, file_mode, tape_mode):
        """
        Dumps current tape to file
        :param file_name: Name of file to dump tape to
        :param file_mode: mode by which to access file (R / R/W / W)
        :param tape_mode: Should tape be wiped after dump
        """
        with open(file_name, file_mode, encoding="utf-8") as dumpfile:
            for tape_item in self.tape:
                if tape_item['type'] == 'Trade':
                    dumpfile.write(f'{tape_item["t"]}, {tape_item["price"]}\n')
            dumpfile.close()
            if tape_mode == 'wipe':
                self.tape = []
