#!../../bin/python3
import sys
from preprocessing.baselines_dataloader import divide_data
from fed_baselines.server_base import FedServer
from fed_baselines.server_scaffold import ScaffoldServer
from fed_baselines.server_fednova import FedNovaServer
import yaml
import json
import os
from tqdm import tqdm
from postprocessing.recorder import Recorder
from json import JSONEncoder
import pickle
import time
import random
import shutil
import csv
import time
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Util.Padding import unpad
import io
import hashlib      
os.chdir(sys.path[0])
import torch

class TreeNode:
    def __init__(self):
        self.blocks = [None] * 4  
class WPathORAM:
    def __init__(self, depth, storage_dir):
        self.depth = depth  
        self.tree_size = 2 ** (depth + 1) - 1 
        self.tree = [TreeNode() for _ in range(self.tree_size)]
        self.position_map = {} 
        self.storage_dir = storage_dir  
    def reset_storage(self):
        shutil.rmtree(self.storage_dir, ignore_errors=True) 
        os.makedirs(self.storage_dir, exist_ok=True) 
        for i in range(self.tree_size):
            node_path = os.path.join(self.storage_dir, str(i))
            os.makedirs(node_path, exist_ok=True)
            for j in range(1, 5): 
                os.makedirs(os.path.join(node_path, str(j)), exist_ok=True)
        print("Storage reset completed.")
    def _get_path(self, leaf):
        path = []  
        node_idx = leaf
        while node_idx > 0:
            path.append(node_idx)
            node_idx = (node_idx - 1) // 2 
        path.append(0)
        print(f"Path for leaf {leaf}: {path}")
        return path
    def random_leaf(self):
        leaf_start = 2 ** self.depth - 1  
        leaf_end = self.tree_size - 1  
        leaf = random.randint(leaf_start, leaf_end)  
        print(f"Random leaf chosen: {leaf}")
        return leaf
    def accesswrite(self,new_data,filename):
        self.reset_storage() 
        self.position_map.clear()
        chosen_block_idx = random.randint(1, 4)
        if filename not in self.position_map:
            self.position_map[filename] = (self.random_leaf(), chosen_block_idx)
        leaf, block_idx = self.position_map[filename]
        print(f"Position map: {self.position_map}")
        path = self._get_path(leaf) 
        chosen_node_idx = random.choice(path)
        print(f"Chosen node for real data: {chosen_node_idx}, block: {chosen_block_idx}")
        self.position_map[filename] = (chosen_node_idx, chosen_block_idx)
        for node_idx in path:
            node_path = os.path.join(self.storage_dir, str(node_idx))
            for idx in range(1, 5):
                if node_idx == chosen_node_idx and idx == chosen_block_idx:
                    block_path = os.path.join(node_path, str(idx), filename)
                    data_to_write = new_data
                    print(f"Writing real data to {block_path}")
                else:
                    block_path = os.path.join(node_path, str(idx), f'fake_data_block_{filename}')
                    data_to_write = os.urandom(len(new_data))
                    print(f"Writing fake data to {block_path}")
                with open(block_path, 'wb') as f:
                    f.write(data_to_write)
        return self.position_map
    #read data
    def accessread(self, position_map, filename):
        if filename.startswith('C'):
            storage_dir = '/host/ctosfile'
        elif filename.startswith('F'):
            storage_dir = '/host/stocfile'
        else:
            raise ValueError("Invalid filename provided. Must be 'Ci' or 'Fi'.")
        file_path = position_map['file_path']
        path_parts = file_path.split('/')
        leaf = int(path_parts[-2])  
        block_idx = int(path_parts[-1]) 
        path = self._get_path(leaf)
        data_block = None
        real_block_volume_serial_number = None
        all_volume_serial_numbers = []  
        for node_idx in path:
            node_path = os.path.join(storage_dir, str(node_idx))
            for idx in range(1, 5):
                block_path = os.path.join(node_path, str(idx), filename if node_idx == leaf and idx == block_idx else f'fake_data_block_{filename}')
                volume_serial_number = get_volume_serial_number(block_path)
                all_volume_serial_numbers.append(volume_serial_number)
                if os.path.exists(block_path):
                    with open(block_path, 'rb') as f:
                        data = f.read()
                        print(f"Read {'real' if node_idx == leaf and idx == block_idx else 'fake'} data from {block_path}")
                        if node_idx == leaf and idx == block_idx:
                            data_block = data 
                            real_block_volume_serial_number = volume_serial_number 
        if data_block is None:
            raise Exception("Data block not found")
        return data_block, real_block_volume_serial_number

def get_model_hash(model_weights):
    if not isinstance(model_weights, dict):
        raise ValueError("Input must be a dict")
    buffer = io.BytesIO()
    torch.save(model_weights, buffer)
    model_weights_bytes = buffer.getvalue()
    model_hash = hashlib.sha256(model_weights_bytes).hexdigest()
    return model_hash

def encrypt_file(data, key):#AES encrypt
    buffer = io.BytesIO()
    torch.save(data, buffer)
    serialized_data = buffer.getvalue()
    cipher = AES.new(key, AES.MODE_CBC)
    encrypted_data = cipher.encrypt(pad(serialized_data, AES.block_size))
    return cipher.iv + encrypted_data

def decrypt_file(input_data, key):#AES decrypt
    if isinstance(input_data, str) and os.path.exists(input_data):
        with open(input_data, 'rb') as file:
            encrypted_data = file.read()
    elif isinstance(input_data, bytes):
        encrypted_data = input_data
    else:
        raise ValueError("Input must be a valid file path or bytes object containing encrypted data.")
    iv = encrypted_data[:16]
    encrypted_content = encrypted_data[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_data = unpad(cipher.decrypt(encrypted_content), AES.block_size)
    buffer = io.BytesIO(decrypted_data)
    return torch.load(buffer, map_location=torch.device('cpu'))

key = b'v \xf35$\x90{\xbd-\xa2v\xc3\xbf\xb0\xf3\xa3' #key
received_key = b'v \xf35$\x90{\xbd-\xa2v\xc3\xbf\xb0\xf3\xa3'  
def get_volume_serial_number(path):
    volume_info = os.stat(path)
    return volume_info.st_dev+volume_info.st_ino  
def load_checkpoint():  #SGX reload
    files = [f for f in os.listdir('/host/TTP/') if f.startswith('F') and f.endswith('_PM')] 
    if not files:
        return 0, None, None  
    latest_file = sorted(files, key=lambda x: int(x[1:].split('_')[0]), reverse=True)[0]
    global_round = int(latest_file[1:].split('_')[0]) -1 
    PM_server = torch.load(os.path.join('/host/TTP/', latest_file))     
    PM_server = decrypt_file(PM_server, key)
    file_path = PM_server['file_path']  
    fi_files = [f for f in os.listdir(file_path) if f.startswith('F')]
    if not fi_files:
        return 0, None, None  
    fi_file_path = os.path.join(file_path, fi_files[0])
    with open(fi_file_path, 'rb') as f:
        Fi = f.read()
    data_to_save = decrypt_file(Fi, key)
    return global_round, data_to_save, file_path
json_types = (list, dict, str, int, float, bool, type(None))

class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}

def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(dct['_python_object'].encode('latin-1'))
    return dct

recorder = Recorder()
with open("config/test_config.yaml", "r") as yaml_file:
    try:
        config = yaml.safe_load(yaml_file)
    except yaml.YAMLError as exc:
        print(exc)

trainset_config, testset = divide_data(num_client=config["system"]["num_client"], num_local_class=config["system"]["num_local_class"], dataset_name=config["system"]["dataset"],
                                        i_seed=config["system"]["i_seed"])   
pbar = tqdm(range(config["system"]["num_round"]))
current_round, data_to_save, file_path = load_checkpoint()
if current_round > 0:
    if config["client"]["fed_algo"] == 'FedAvg' or config["client"]["fed_algo"] == 'FedProx' or config["client"]["fed_algo"] == 'FedNova':
        fed_server = FedServer(trainset_config['users'], dataset_id=config["system"]["dataset"], model_name=config["system"]["model"])
        fed_server.load_testset(testset)
        fed_server.state_dict().update(data_to_save['global_state_dict'])
    elif config["client"]["fed_algo"] == 'SCAFFOLD':
        fed_server = ScaffoldServer(trainset_config['users'], dataset_id=config["system"]["dataset"], model_name=config["system"]["model"])
        fed_server.load_testset(testset)
        fed_server.state_dict().update(data_to_save['global_state_dict'])
        fed_server.scv.load_state_dict(data_to_save['scv_state'])
    elif config["client"]["fed_algo"] == 'FedProx':
        fed_server = FedServer(trainset_config['users'], dataset_id=config["system"]["dataset"], model_name=config["system"]["model"])
        fed_server.load_testset(testset)       
        fed_server.state_dict().update(data_to_save['global_state_dict'])
    elif config["client"]["fed_algo"] == 'FedNova':
        fed_server = FedNovaServer(trainset_config['users'], dataset_id=config["system"]["dataset"], model_name=config["system"]["model"])
        fed_server.load_testset(testset)    
        fed_server.state_dict().update(data_to_save['global_state_dict'])
    model_save_directory = '/host/stocfile'  #result
    if not os.path.exists(model_save_directory):  #  res_root: "results"  
        os.makedirs(model_save_directory)
    oram = WPathORAM(depth=3, storage_dir=model_save_directory) #init ORAM
else:    
    if config["client"]["fed_algo"] == 'FedAvg':
        fed_server = FedServer(trainset_config['users'], dataset_id=config["system"]["dataset"], model_name=config["system"]["model"])
        fed_server.load_testset(testset)        
        global_state_dict = fed_server.state_dict()
        data_to_save = {
            'global_state_dict': global_state_dict,
            }
    elif config["client"]["fed_algo"] == 'SCAFFOLD':
        fed_server = ScaffoldServer(trainset_config['users'], dataset_id=config["system"]["dataset"], model_name=config["system"]["model"])
        scv_state = fed_server.scv.state_dict()
        fed_server.load_testset(testset)       
        global_state_dict = fed_server.state_dict()
        data_to_save = {
            'global_state_dict': global_state_dict,
            'scv_state':scv_state
            }
    elif config["client"]["fed_algo"] == 'FedProx':
        fed_server = FedServer(trainset_config['users'], dataset_id=config["system"]["dataset"], model_name=config["system"]["model"])
        fed_server.load_testset(testset)        
        global_state_dict = fed_server.state_dict()
        data_to_save = {
            'global_state_dict': global_state_dict,
            }
    elif config["client"]["fed_algo"] == 'FedNova':
        fed_server = FedNovaServer(trainset_config['users'], dataset_id=config["system"]["dataset"], model_name=config["system"]["model"])
        fed_server.load_testset(testset)        
        global_state_dict = fed_server.state_dict()
        data_to_save = {
            'global_state_dict': global_state_dict,
            }

    model_save_directory = '/host/stocfile'  #result
    if not os.path.exists(model_save_directory):  
        os.makedirs(model_save_directory)
    oram = WPathORAM(depth=3, storage_dir=model_save_directory)
    H1=get_model_hash(data_to_save)   
    Fi=encrypt_file(data_to_save, key)         
    time.sleep(0.1)
    file_path_Fi = f'F{current_round+1}'
    position_map = oram.accesswrite(Fi,file_path_Fi) 
    file_path = os.path.join(model_save_directory, str(position_map[file_path_Fi][0]), str(position_map[file_path_Fi][1]))
    serial_number = get_volume_serial_number(os.path.join(file_path,str(os.listdir(file_path)[0])))
    positon_to_save = {
        'file_path': file_path,
        'serial_number':serial_number,
        'H1':H1
        }
    PM_server=encrypt_file(positon_to_save, key)
    if not os.path.exists("/host/TTP"): 
        os.makedirs("/host/TTP")
    FiPMsavename = "TTP/"+f'F{current_round+1}_PM'
    torch.save(PM_server, "/host/"+FiPMsavename)
    print("Model stored at (after second write):", file_path)
    print(f"Volume Serial Number for {file_path}: {serial_number}")
    print("Data writed successfully.")
max_acc = 0
with open('/host/'+config["system"]["csv_file"], mode='w', newline='') as file: #-csv
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['Global Round', 'Average Loss', 'Accuracy', 'Max Accuracy', 'time'])
    for global_round in range(current_round , config["system"]["num_round"]):
        i=0
        start_time = time.time()
        while True:
            CiPMsavename = "TTP/"+f'C{global_round + 1}_PM'
            if os.path.exists('/host/'+CiPMsavename):
                time.sleep(0.5)
                ctos_position = torch.load('/host/'+CiPMsavename)#get PM  ../ctosfile/14/ctosflmodel.pt
                ctos_position = decrypt_file(ctos_position, received_key) 
                A1=ctos_position['serial_number']
                H2=ctos_position['H2']
                file_path_Ci=f'C{global_round + 1}'
                encrypt_Ci,A2=oram.accessread(ctos_position,file_path_Ci)
                if A1==A2:
                    data=decrypt_file(encrypt_Ci, received_key)  
                    H2_Ci=get_model_hash(data)
                else:
                    raise ValueError("Error! The Ci address does not match.")

                if H2==H2_Ci:
                    if config["client"]["fed_algo"] == 'FedAvg':   #FedAvg
                        all_state_dicts = data['all_state_dicts']
                        all_n_data = data['all_n_data']
                        all_losses = data['all_losses']
                    elif config["client"]["fed_algo"] == 'SCAFFOLD':
                        all_state_dicts = data['all_state_dicts']
                        all_n_data = data['all_n_data']
                        all_losses = data['all_losses']
                        all_delta_ccv_state = data['all_delta_ccv_state']
                    elif config["client"]["fed_algo"] == 'FedProx':   #
                        all_state_dicts = data['all_state_dicts']
                        all_n_data = data['all_n_data']
                        all_losses = data['all_losses']
                    elif config["client"]["fed_algo"] == 'FedNova':   #FedNova
                        all_state_dicts = data['all_state_dicts']
                        all_n_data = data['all_n_data']
                        all_losses = data['all_losses']
                        all_coeff=data['all_coeff']
                        all_norm_grad=data['all_norm_grad']
                    print("Data loaded successfully.")
                    break
                else:
                    raise ValueError("Error! The Ci hashvalue does not match.")
            else:
                time.sleep(0.1)  # wait for 1 second before checking again
        for client_id in trainset_config['users']:
            if config["client"]["fed_algo"] == 'FedAvg':   #FedAvg
                fed_server.rec(client_id, all_state_dicts[i], all_n_data[i], all_losses[i])
            elif config["client"]["fed_algo"] == 'SCAFFOLD':
                fed_server.rec(client_id, all_state_dicts[i], all_n_data[i], all_losses[i],all_delta_ccv_state[i])
            elif config["client"]["fed_algo"] == 'FedProx':
                fed_server.rec(client_id, all_state_dicts[i], all_n_data[i], all_losses[i])
            elif config["client"]["fed_algo"] == 'FedNova':
                fed_server.rec(client_id, all_state_dicts[i], all_n_data[i], all_losses[i],all_coeff[i], all_norm_grad[i])
            i=i+1    

        fed_server.select_clients()
        if config["client"]["fed_algo"] == 'FedAvg':
            global_state_dict, avg_loss, _ = fed_server.agg() 
        elif config["client"]["fed_algo"] == 'SCAFFOLD':
            global_state_dict, avg_loss, _, scv_state = fed_server.agg()  # scarffold
        elif config["client"]["fed_algo"] == 'FedProx':
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == 'FedNova':
            global_state_dict, avg_loss, _ = fed_server.agg()
        accuracy = fed_server.test()
        fed_server.flush()

        # Record the results
        recorder.res['server']['iid_accuracy'].append(accuracy)
        recorder.res['server']['train_loss'].append(avg_loss)

        if max_acc < accuracy:
            max_acc = accuracy
        pbar.set_description(
            'Global Round: %d' % global_round +
            '| Train loss: %.4f ' % avg_loss +
            '| Accuracy: %.4f' % accuracy +
            '| Max Acc: %.4f' % max_acc)
        if config["client"]["fed_algo"] == 'FedAvg':
            data_to_save = {
                'global_state_dict': global_state_dict,
                }
        elif config["client"]["fed_algo"] == 'SCAFFOLD':
            data_to_save = {
                'global_state_dict': global_state_dict,
                'scv_state':scv_state
                }
        elif config["client"]["fed_algo"] == 'FedProx':
            data_to_save = {
                'global_state_dict': global_state_dict,
                }
        elif config["client"]["fed_algo"] == 'FedNova': 
            data_to_save = {
                'global_state_dict': global_state_dict,
                }
        model_save_directory = '/host/stocfile'  #result
        if not os.path.exists(model_save_directory):  
            os.makedirs(model_save_directory)
        oram = WPathORAM(depth=3, storage_dir=model_save_directory) 
        H1=get_model_hash(data_to_save) 
        Fi=encrypt_file(data_to_save, key) 
        time.sleep(0.2)
        file_path_Fi = f'F{global_round + 2}'
        position_map = oram.accesswrite(Fi,file_path_Fi) 
        file_path = os.path.join(model_save_directory, str(position_map[file_path_Fi][0]), str(position_map[file_path_Fi][1]))
        serial_number = get_volume_serial_number(os.path.join(file_path,str(os.listdir(file_path)[0])))
        encrypt_file(file_path, key)
        positon_to_save = {
            'file_path': file_path,
            'serial_number':serial_number,
            'H1':H1
            }
        positon_to_save=encrypt_file(positon_to_save, key)
        if global_round + 1 > 0:
            os.remove("/host/TTP/"+f'F{global_round + 1}_PM')    
        FiPMsavename = "TTP/"+f'F{global_round + 2}_PM'  
        torch.save(positon_to_save, "/host/"+FiPMsavename) 
        print("Model stored at (after second write):", file_path)
        print(f"Volume Serial Number for {file_path}: {serial_number}")
        print("Data writed successfully.")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Program running time: {elapsed_time:.2f} seconds")
        formatted_elapsed_time = "{:.2f}".format(elapsed_time)
        writer.writerow([global_round, avg_loss, accuracy, max_acc,formatted_elapsed_time])
