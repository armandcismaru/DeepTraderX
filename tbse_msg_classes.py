"""
From original BSE code by Dave Cliff
Module holding the Order class
"""

# pylint: disable=too-many-arguments,too-few-public-methods
class Order:
    """
    an Order/quote has a trader id, a type (buy/sell) price, quantity, timestamp, and unique i.d.
    """
    def __init__(self, tid, otype, price, qty, time, coid, toid):
        self.tid = tid  # trader i.d.
        self.otype = otype  # order type
        self.price = price  # price
        self.qty = qty  # quantity
        self.time = time  # timestamp
        self.coid = coid  # customer order i.d. (unique to each quote customer order)
        self.toid = toid  # trader order i.d. (unique to each order posted by the trader)

    def __str__(self):
        return f'[{self.tid} {self.otype} P={str(self.price).zfill(3)} Q={self.qty} ' \
               f'T={self.time:5.2f} COID:{self.coid} TOID:{self.toid}]'
