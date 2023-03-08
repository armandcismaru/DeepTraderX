import csv
import sys
import numpy as np
import pickle
# from progress.bar import IncrementalBar as Bar
from keras.utils import Sequence


def normalize_data2(x, max=0, min=0, train=True):
    if train:
        max = np.max(x)
        min = np.min(x)
    
    normalized = (2*(x-min/(max-min)) - 1)
    return normalized
def normalize_data(x, max=0, min=0, train=True):
    if train:
        max = np.max(x)
        min = np.min(x)

    normalized = (x-min)/(max-min)
    return normalized


class DataGenerator(Sequence):
    def __init__(self, dataset_path, batch_size, n_features):
        self.no_items = 0
        self.files = 0
        self.dataset_path = dataset_path
        with open(self.dataset_path, 'rb') as f:
            while 1:
                try:
                    # print(np.array(pickle.load(f)).astype(float))
                    # sys.exit(1)
                    # print(len(pickle.load(f)))
                    self.no_items += len(pickle.load(f))
                except EOFError:
                    break  # no more data in the file
        print(self.no_items)

        # self.dataset = [float(s) for sublist in self.dataset for s in sublist]
        self.batch_size = batch_size
        self.n_features = n_features
        # self.dataset = np.reshape(np.array(self.dataset), (-1,self.n_features+1))
       
        # print(np.array(self.dataset).shape)
        self.train_max = np.empty((self.n_features+1))
        self.train_min = np.empty((self.n_features+1))
        
        # # normalizing data
        # # note: treating the test set the same way as the training sets
        # for c in range(self.n_features + 1):
        #     # normalizing each feature using the only the training data to scale
        #     self.train_max[c]= np.max(self.dataset[:,c])
        #     self.train_min[c]= np.min(self.dataset[:,c])
        #     self.dataset[:,c] = normalize_data(self.dataset[:,c], max=self.train_max[c], min=self.train_min[c], train=False)
        
       
    def __getitem__(self, index):  
        # Generate indexes of the batch
        indexes = [x for x in range(index*self.batch_size, (index+1)*self.batch_size)]
       
        x = np.empty((self.batch_size, 1, self.n_features))
        y = np.empty((self.batch_size, 1))
        
        # for i in range(len(indexes)):
        #     item = self.dataset[indexes[i]()
        #     x[i, ] = np.reshape(item[:self.n_features], (1,-1))
        #     y[i, ] = np.reshape(item[self.n_features], (1, 1))
        
        with open(self.dataset_path, 'rb') as f:
            count = 0
            number = 0
            i = 0
            while 1:
                try:
                    number = len(pickle.load(f)) + count
                    if((number < indexes[0])):
                        count = number
                        break
                    elif(len(pickle.load(f)) == 0 ):
                        break

                    file = np.array(pickle.load(f))
                    for item in file:
                        item = item.astype(np.float)
                        if count in indexes:
                            x[i, ] = np.reshape(item[:self.n_features], (1,-1))
                            y[i, ] = np.reshape(item[self.n_features], (1, 1))
                            
                            
                        count += 1
                        i += 1
                        if(i > self.batch_size - 1): 
                            i = 0
                            # print(x)
                            return(x,y)

                except EOFError:
                    break  # no more data in the file
        # print(x.shape)
            


    def __len__(self):
        # print((self.no_items // self.batch_size))
        return (self.no_items // self.batch_size)



# class SlowBar(Bar):
#     suffix = '%(index)d/%(max)d, %(percent).1f%%, %(remaining_hours)d hours remaining'
#     @property
#     def remaining_hours(self):
#         return self.eta // 3600



# def pickle_files(pkl_path, no_files):
    
#     file_list = []
#     bar = SlowBar('Pickling Data', max=no_files)
#     # retrieving data from multiple files
#     for i in range(no_files):
#         filename = f"trial{(i+1):06d}.csv"
#         file_list.append(read_all_data(filename))
#         bar.next()
#     bar.finish()
    
#     with open(pkl_path, "wb") as fileobj:
#         pickle.dump(file_list, fileobj)


def standardize_data(x):
    standardized = (x - np.mean(x)) / np.std(x)
    return standardized

# obselete functions
def read_data(filename, d_type):
    data = np.array([])

    with open(filename, "r") as f:
        f_data = csv.reader(f)

        for row in f_data:
            # print(row)
            if d_type == "MID":
                data = np.append(data, float(row[1]))
            elif d_type == "MIC":
                data = np.append(data, float(row[2]))
            elif d_type == "IMB":
                data = np.append(data, float(row[3]))
            elif d_type == "SPR":
                    data = np.append(data, float(row[4]))
            
            else:
                data = np.append(data, float(row[0]))

    return data
def read_data2(filename, d_type):
    data = np.array([])

    with open(filename, "r") as f:
        f_data = csv.reader(f)

        for row in f_data:
            # print(row)
            if d_type == "MID":
                data = np.append(data, float(row[1]))
            elif d_type == "MIC":
                data = np.append(data, float(row[2]))
            elif d_type == "IMB":
                data = np.append(data, float(row[3]))
            elif d_type == "SPR":
                data = np.append(data, float(row[4]))
            elif d_type == "BID":
                data = np.append(data, float(row[5]))
            elif d_type == "ASK":
                data = np.append(data, float(row[6]))
            elif d_type == "TAR":
                data = np.append(data, float(row[10]))
            elif d_type == "OCC":
                data = np.append(data, float(row[8]))
            elif d_type == "DT":
                data = np.append(data, float(row[9]))
            else:
                data = np.append(data, float(row[0]))

    data = normalize_data(data)

    return data
def read_data3(filename, d_type):
    data = np.array([])

    with open(filename, "r") as f:
        f_data = csv.reader(f)

        for row in f_data:
            # print(row)
            if d_type == "MID":
                data = np.append(data, float(row[1]))
            elif d_type == "MIC":
                data = np.append(data, float(row[2]))
            elif d_type == "IMB":
                data = np.append(data, float(row[3]))
            elif d_type == "SPR":
                data = np.append(data, float(row[4]))
            elif d_type == "BID":
                data = np.append(data, float(row[5]))
            elif d_type == "ASK":
                data = np.append(data, float(row[6]))
            elif d_type == "TAR":
                data = np.append(data, float(row[10]))
            else:
                data = np.append(data, float(row[0]))

    return data

def read_all_data(filename):
    data = {}
    
    with open(filename, "r") as f:
        f_data = csv.reader(f)
        data["TIME"] = np.array([])
        data["TYP"] = np.array([])
        data["LIM"] = np.array([])
        data["MID"] = np.array([])
        data["MIC"] = np.array([])
        data["IMB"] = np.array([])
        data["SPR"] = np.array([])
        data["BID"] = np.array([])
        data["ASK"] = np.array([])
        data["DT"] = np.array([])
        data["TOT"] = np.array([])
        data["ALP"] = np.array([])
        data["TAR"] = np.array([])
       
        for row in f_data:
            data["TIME"] = np.append(data["TIME"],float(row[0]))
            data["TYP"] = np.append(data["TYP"], float(row[1]))
            data["LIM"] = np.append(data["LIM"], float(row[2]))
            data["MID"] = np.append(data["MID"], float(row[3]))
            data["MIC"] = np.append(data["MIC"], float(row[4]))
            data["IMB"] = np.append(data["IMB"], float(row[5]))
            data["SPR"] = np.append(data["SPR"], float(row[6]))
            data["BID"] = np.append(data["BID"], float(row[7]))
            data["ASK"] = np.append(data["ASK"], float(row[8]))
            data["DT"]  = np.append(data["DT"],  float(row[9]))
            data["TOT"] = np.append(data["TOT"], float(row[10]))
            data["ALP"] = np.append(data["ALP"], float(row[11]))
            data["TAR"] = np.append(data["TAR"], float(row[12]))
     
    temp = np.array([])
    temp = np.column_stack([data[d] for d in data])
  
    return temp

def read_data_from_multiple_files(no_files, no_features):
    
    X = np.array([[]])
    Y = np.array([])
    
    # retrieving data from multiple files
    for i in range(no_files):
        filename = f"./Data/Training/trial{(i+1):07}.csv"
        data = read_all_data(filename)
        transaction_prices = read_data3(filename, "TAR")
        X = np.append(X, data)
        Y = np.append(Y, transaction_prices)

    # reshaping input data
    X = np.reshape(X, (-1, no_features))


    return X, Y

def normalization_values(X, Y, no_features):

    train_max = np.array([float(0)]*(no_features + 1))
    train_min = np.array([float(0)]*(no_features + 1))

    for c in range(no_features):
        # storing values used to scale
        train_max[c] = np.max(X[:, c])
        train_min[c] = np.min(X[:, c])

    # normalizing target data in the same way
    train_max[no_features] = np.max(Y)
    train_min[no_features] = np.min(Y)

    return train_max, train_min

# obselete function
def get_data(no_files, no_features):

    # obtaining data
    X, Y = read_data_from_multiple_files(no_files, no_features)
    # print(X.shape, Y.shape)
    # ratio of split as an array
    ratio = [9,1]

    # splitting train and test data for targets and input
    train_X, test_X = split_train_test_data(X, ratio)
    train_Y, test_Y = split_train_test_data(Y, ratio)

    # reshaping input to be correct
    train_X = np.reshape(train_X,(-1, no_features))
    test_X = np.reshape(test_X, (-1, no_features))

    train_max, train_min = normalization_values(train_X, train_Y, no_features)

    # normalizing data
    # note: treating the test set the same way as the training set
    for c in range(no_features):
        # normalizing each feature using the only the training data to scale
        train_X[:, c] = normalize_data(train_X[:,c])
        test_X[:, c] = normalize_data(test_X[:, c], max=train_max[c], min=train_min[c], train=False)
        # print(np.max(train_X[:, c]), np.min(train_X[:, c]))

    train_Y = normalize_data(train_Y)
    test_Y = normalize_data(test_Y, max=train_max[no_features], min=train_min[no_features], train=False)

    print(train_max)
    print(train_min)
    # print(train_X)
    # print(test_X)
    # print(train_Y)
    # print(test_Y)
    
    # reshaping input and target data for nn
    train_X = np.reshape(train_X, (-1, 1, no_features))
    train_Y = np.reshape(train_Y, (-1, 1))
    test_X = np.reshape(test_X, (-1, 1, no_features))
    test_Y = np.reshape(test_Y, (-1, 1))

    return train_X, train_Y, test_X, test_Y, train_max, train_min

# splitting data into input and output signal
# n_steps is the number of steps taken until a split occurs will have to formalize this with time steps
# for now is just for every n_steps, we have a y
def split_data(data, n_steps):

    A = np.array([])
    B = np.array([])

    step = 0
    for d in np.nditer(data):

        if step == n_steps + 1:
            B = np.append(B, d)
            step = 0
        else:
            A = np.append(A, d)
        step += 1
    
    
    A = A[:-1]
    A = np.reshape(A, (-1, n_steps,1))

    A = (A - np.mean(A)) / np.max(A)
    B = (B - np.mean(B)) / np.max(B)

    return A, B
def multi_split_data(data, x_steps, y_steps):
    
    A = np.array([])
    B = np.array([])

    step = 0
    add_A = True
    for d in np.nditer(data):

        if add_A:
            
            A = np.append(A, d)
            step += 1
            
            if step == x_steps: 
                add_A = False
                step = 0
        
        else:
           
            B = np.append(B, d)
            step += 1

            if step == y_steps:
                add_A = True
                step = 0

    
   
    A = A[:-1]
    A = np.reshape(A, (-1, x_steps,1))

    B = np.reshape(B, (-1, y_steps, 1))
    return A, B

# ratio is train first and then test  
def split_train_test_data(data, ratio):
    
    A = np.array([])
    B = np.array([])

    split_index = int( ratio[0] / (ratio[0] + ratio[1]) * len(data) )

    A = np.append(A, data[:split_index])
    B = np.append(B, data[split_index:])

    return A, B
def collect_time_series_results(file_no):
    market_data = {}
    market_data["TIME"] = np.array([])
    market_data["ASK"] = np.array([])
    market_data["BID"] = np.array([])

    trader_data = {}
    session_id = ""
    filename = f"./Balanced/avg_balance{(file_no):04d}.csv"
    
    with open(filename, "r") as f:
        f_data = list(csv.reader(f))
        first_row= f_data[0]    
        session_id = first_row[0]
        no_traders = int((len(first_row) - 5) / 4)
        
        for i in range(no_traders):
            trader = str(first_row[(i*no_traders) + 2]).strip()
            trader_data[trader] = {}
            trader_data[trader]["Balance"] = np.array([])
            trader_data[trader]["n"] = int(str(first_row[(i*no_traders) + 4]).strip())
            trader_data[trader]["PPT"] = np.array([])

        
        for row in f_data:
            # print(row)
        
            market_data["TIME"] = np.append( market_data["TIME"], float( str(row[1]).strip()))
            market_data["ASK"] = np.append( market_data["ASK"],   float( str(row[no_traders*4 + 2]).strip() ) )
            market_data["BID"] = np.append( market_data["BID"],   float( str(row[no_traders*4 + 3]).strip() ) )
            
            for i in range(no_traders):
                trader = str(row[(i*no_traders) + 2]).strip()
                trader_data[trader]["Balance"] = np.append(trader_data[trader]["Balance"], int(str(row[(i*no_traders + 3)]).strip()))
                trader_data[trader]["PPT"] = np.append(trader_data[trader]["PPT"], float(str(row[(i*no_traders)+5]).strip()))

    return market_data, trader_data


def get_end_results(file_no):

    with open(f"./Balanced/avg_balance{(file_no):04d}.csv", 'r') as f:
        lines = list(csv.reader(f))
        
        no_trades = len(lines)
        # print (len(lines))
        try:
            last_line = lines[-1]
        except:
            print(file_no)
        # print(last_line)
        trader_data = {}
    
    no_traders = int((len(last_line) - 5) / 4)
    # print(no_traders)
    for i in range(no_traders):
        trader = str(last_line[(i*4) + 2]).strip()
        # print(trader)
        trader_data[trader] = {}
        # trader_data[trader]["Balance"] =  float(str(last_line[(i*no_traders + 3)]).strip())
        # trader_data[trader]["n"] = int(str(last_line[(i*no_traders) + 4]).strip())
        trader_data[trader] = float(str(last_line[(i*4 + 5)]).strip())

    return trader_data

if __name__ == "__main__":
   
    pkl_path = "./Train_Dataset2.pkl"
    # pickle_files(pkl_path, )
    train_data = DataGenerator(pkl_path)
    
