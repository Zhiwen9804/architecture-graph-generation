import os
import pickle

def load_dataset(data_dir='data', file_name=None):
    file_path = os.path.join(data_dir, file_name)
    with open(file_path + '.pkl', 'rb') as f:
        graph_list = pickle.load(f)
    return graph_list 

def save_dataset(data_dir, graphs, save_name):
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    with open(data_dir + save_name + '.pkl', 'wb') as f:
        pickle.dump(obj=graphs, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Save Success!')

def make_dir(dirs):
    try:
        if not os.path.exists(dirs):
            os.makedirs(dirs)
    except Exception as err:
        print("create_dirs error!")
        print(dirs)
        exit()