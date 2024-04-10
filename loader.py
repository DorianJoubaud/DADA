import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from scipy.interpolate import interp1d

class Loader:
    def __init__(self, name, path = 'datasets/CMAPSS/', seq_len = 30, shift = 1, early_rul = 130):
        
        self.name = name
        self.path = path
        self.seq_len = seq_len
        self.shift = shift
        self.early_rul = early_rul
        self.interpolated_test_data = {}
        self.original_test_data = {}
        
        
    
    
    
    
    def createEarlyRul(self, data_length):
        early_rul_duration = data_length - self.early_rul
        if early_rul_duration <= 0:
            return np.arange(data_length-1, -1, -1)
        else:
            return np.append(self.early_rul*np.ones(shape = (early_rul_duration,)), np.arange(self.early_rul-1, -1, -1))
        
    
    def process_input_data_with_targets(self, input_data, target_data = None):
        """Depending on values of seq_len and shift, this function generates batchs of data and targets 
        from input_data and target_data.
        
        Number of batches = np.floor((len(input_data) - seq_len)/shift) + 1
        
        **We don't check input dimensions uisng exception handling. So readers should be careful while using these
        functions. If input data are not of desired dimension, either error occurs or something undesirable is 
        produced as output.**
        
        Arguments:
            input_data: input data to function (Must be 2 dimensional)
            target_data: input rul values (Must be 1D array)s
            seq_len: window length of data
            shift: Distance by which the window moves for next batch. This is closely related to overlap
                between data. For example, if window length is 30 and shift is 1, there is an overlap of 
                29 data points between two consecutive batches.
            
        """
        num_batches = int(np.floor((len(input_data) - self.seq_len)/self.shift)) + 1
        num_features = input_data.shape[1]
        output_data = np.repeat(np.nan, repeats = num_batches * self.seq_len * num_features).reshape(num_batches, self.seq_len,
                                                                                                    num_features)
        if target_data is None:
            for batch in range(num_batches):
                output_data[batch,:,:] = input_data[(0+self.shift*batch):(0+self.shift*batch+self.seq_len),:]
            return output_data
        else:
            output_targets = np.repeat(np.nan, repeats = num_batches)
            for batch in range(num_batches):
                output_data[batch,:,:] = input_data[(0+self.shift*batch):(0+self.shift*batch+self.seq_len),:]
                output_targets[batch] = target_data[(self.shift*batch + (self.seq_len-1))]
        
            return output_data, output_targets
        
    
    def process_test_data(self,test_data_for_an_engine, num_test_windows = 1):
        """ This function takes test data for an engine as first input. The next two inputs
        seq_len and shift are same as other functins. 
        
        Finally it takes num_test_windows as the last input. num_test_windows sets how many examplles we
        want from test data (from last). By default it extracts only the last example.
        
        The function return last examples and number of last examples (a scaler) as output. 
        We need the second output later. If we are extracting more than 1 last examples, we have to 
        average their prediction results. The second scaler halps us do just that.
        """
        max_num_test_batches = int(np.floor((len(test_data_for_an_engine) - self.seq_len)/self.shift)) + 1
        if max_num_test_batches < num_test_windows:
            required_len = (max_num_test_batches -1)* self.shift + self.seq_len
            batched_test_data_for_an_engine = self.process_input_data_with_targets(test_data_for_an_engine[-required_len:, :],
                                                                            target_data = None)
            return batched_test_data_for_an_engine, max_num_test_batches
        else:
            required_len = (num_test_windows - 1) * self.shift + self.seq_len
            batched_test_data_for_an_engine = self.process_input_data_with_targets(test_data_for_an_engine[-required_len:, :],
                                                                            target_data = None)
            return batched_test_data_for_an_engine, num_test_windows
        
    def load_data(self):
        train_data = pd.read_csv(f'{self.path}train_{self.name}.txt', sep = " ", header = None)
        # if more than 26 columns, drop the last two columns (problem due to formart)
        if train_data.shape[1] > 26:
            train_data.drop(train_data.columns[[26, 27]], axis=1, inplace=True)
        
        test_data = pd.read_csv(f'{self.path}test_{self.name}.txt', sep = " ", header = None)
        if test_data.shape[1] > 26:
            test_data.drop(test_data.columns[[26, 27]], axis=1, inplace=True)
            
        true_rul = pd.read_csv(f'{self.path}RUL_{self.name}.txt', sep = " ", header = None)
        if true_rul.shape[1] > 1:
            true_rul.drop(true_rul.columns[[1]], axis=1, inplace=True)

                 
        processed_train_data = []
        processed_train_targets = []

        # How many test windows to take for each engine. If set to 1 (this is the default), only last window of test data for 
        # each engine is taken. If set to a different number, that many windows from last are taken. 
        # Final output is the average output of all windows.
        num_test_windows = 1
        processed_test_data = []
        num_test_windows_list = []

        columns_to_be_dropped = [0,1,2]

        train_data_first_column = train_data[0]
        test_data_first_column = test_data[0]

        # Scale data for all engines
        scaler = MinMaxScaler(feature_range = (-1,1))
        train_data = scaler.fit_transform(train_data.drop(columns = columns_to_be_dropped))
        test_data = test_data.drop(columns = columns_to_be_dropped)

        train_data = pd.DataFrame(data = np.c_[train_data_first_column, train_data])
        test_data = pd.DataFrame(data = np.c_[test_data_first_column, test_data])

        num_train_machines = len(train_data[0].unique())
        num_test_machines = len(test_data[0].unique())

        # Process training and test data sepeartely as number of engines in training and test set may be different.
        # As we are doing scaling for full dataset, we are not bothered by different number of engines in training and test set.

        # Process trianing data
        for i in np.arange(1, num_train_machines + 1):
            temp_train_data = train_data[train_data[0] == i].drop(columns = [0]).values
            
            # Verify if data of given window length can be extracted from training data
            if (len(temp_train_data) < self.seq_len):
                print("Train engine {} doesn't have enough data for window_length of {}".format(i, self.seq_len))
                raise AssertionError("Window length is larger than number of data points for some engines. "
                                    "Try decreasing window length.")
                
            temp_train_targets = self.createEarlyRul(data_length = temp_train_data.shape[0])
            # normalize the RUL values
            temp_train_targets = temp_train_targets / self.early_rul
            data_for_a_machine, targets_for_a_machine = self.process_input_data_with_targets(temp_train_data, temp_train_targets)
            
            processed_train_data.append(data_for_a_machine)
            processed_train_targets.append(targets_for_a_machine)

        processed_train_data = np.concatenate(processed_train_data)
        processed_train_targets = np.concatenate(processed_train_targets)
        
        
        

        # Process test data
        for i in np.arange(1, num_test_machines + 1):
            temp_test_data = test_data[test_data[0] == i].drop(columns = [0]).values
            
            # Verify if data of given window length can be extracted from test data
            if (len(temp_test_data) < self.seq_len):
                # if not, interpolate data to required length
                self.original_test_data[i] = temp_test_data
                original_indices = np.arange(len(temp_test_data))
                required_indices = np.linspace(0, len(temp_test_data) - 1, self.seq_len)
                interpolated_data = np.zeros((self.seq_len, temp_test_data.shape[1]))
                
                for col in range(temp_test_data.shape[1]):
                    f = interp1d(original_indices, temp_test_data[:, col], kind='nearest')  # You can change 'linear' to other types
                    interpolated_data[:, col] = f(required_indices)
                
                temp_test_data = interpolated_data
                
                self.interpolated_test_data[i] = interpolated_data
                
                    
                
            temp_test_data = scaler.transform(temp_test_data)
            # Prepare test data
            test_data_for_an_engine, num_windows = self.process_test_data(temp_test_data, num_test_windows = num_test_windows)
            
            processed_test_data.append(test_data_for_an_engine)
            num_test_windows_list.append(num_windows)

        processed_test_data = np.concatenate(processed_test_data)
        true_rul = true_rul[0].values/ self.early_rul

        return processed_train_data, processed_train_targets, processed_test_data, true_rul