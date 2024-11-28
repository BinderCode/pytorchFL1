#!../../bin/python3
import os
import random
import shutil
import yaml
import pickle
import io
from tqdm import tqdm
import hashlib 
from fed_baselines.client_base import FedClient
from fed_baselines.client_fedprox import FedProxClient
from fed_baselines.client_scaffold import ScaffoldClient
from fed_baselines.client_fednova import FedNovaClient
from fed_baselines.server_base import FedServer
from fed_baselines.server_scaffold import ScaffoldServer
from fed_baselines.server_fednova import FedNovaServer
from preprocessing.baselines_dataloader import divide_data
from utils.models import *
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Util.Padding import unpad

class TreeNode:
    def __init__(self):
        self.blocks = [None] * 4
class WPathORAM:
    def __init__(self, depth, storage_dir,):
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

    def accessread(self, position_map, filename):

        if filename.startswith('C'):
            storage_dir = '../ctosfile'
        elif filename.startswith('F'):
            storage_dir = '../stocfile'
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

def encrypt_file(data, key):
    buffer = io.BytesIO()
    torch.save(data, buffer)
    serialized_data = buffer.getvalue()
    cipher = AES.new(key, AES.MODE_CBC)
    encrypted_data = cipher.encrypt(pad(serialized_data, AES.block_size))   
    return cipher.iv + encrypted_data

def decrypt_file(input_data, key):
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
    return torch.load(buffer,map_location=torch.device('cpu')) 
key = b'v \xf35$\x90{\xbd-\xa2v\xc3\xbf\xb0\xf3\xa3' #key
received_key = b'v \xf35$\x90{\xbd-\xa2v\xc3\xbf\xb0\xf3\xa3'  

def get_volume_serial_number(path):
    volume_info = os.stat(path)
    return volume_info.st_dev+volume_info.st_ino 
def get_model_hash(model_weights):
    if not isinstance(model_weights, dict):
        raise ValueError("Input must be a dict")
    buffer = io.BytesIO()
    torch.save(model_weights, buffer)
    model_weights_bytes = buffer.getvalue()

    model_hash = hashlib.sha256(model_weights_bytes).hexdigest()
    return model_hash

def fed_run():
    """
    Main function for the baselines of federated learning
    """
    import sys
    os.chdir(sys.path[0])
    with open("config/test_config.yaml", "r") as yaml_file:
        try:
            config = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)

    algo_list = ["FedAvg", "SCAFFOLD", "FedProx", "FedNova"]
    assert config["client"]["fed_algo"] in algo_list, "The federated learning algorithm is not supported"
    dataset_list = ['MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN', 'CIFAR100']
    assert config["system"]["dataset"] in dataset_list, "The dataset is not supported"
    model_list = ["LeNet", 'AlexCifarNet', "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "CNN"]
    assert config["system"]["model"] in model_list, "The model is not supported"

    np.random.seed(config["system"]["i_seed"])
    torch.manual_seed(config["system"]["i_seed"])
    random.seed(config["system"]["i_seed"])
    client_dict = {}
    trainset_config, testset = divide_data(num_client=config["system"]["num_client"], num_local_class=config["system"]["num_local_class"], dataset_name=config["system"]["dataset"],
                                           i_seed=config["system"]["i_seed"])   

    for client_id in trainset_config['users']:
        if config["client"]["fed_algo"] == 'FedAvg':
            client_dict[client_id] = FedClient(client_id, dataset_id=config["system"]["dataset"], epoch=config["client"]["num_local_epoch"], model_name=config["system"]["model"])
        elif config["client"]["fed_algo"] == 'SCAFFOLD':
            client_dict[client_id] = ScaffoldClient(client_id, dataset_id=config["system"]["dataset"], epoch=config["client"]["num_local_epoch"], model_name=config["system"]["model"])
        elif config["client"]["fed_algo"] == 'FedProx':
            client_dict[client_id] = FedProxClient(client_id, dataset_id=config["system"]["dataset"], epoch=config["client"]["num_local_epoch"], model_name=config["system"]["model"])
        elif config["client"]["fed_algo"] == 'FedNova':
            client_dict[client_id] = FedNovaClient(client_id, dataset_id=config["system"]["dataset"], epoch=config["client"]["num_local_epoch"], model_name=config["system"]["model"])
        client_dict[client_id].load_trainset(trainset_config['user_data'][client_id])
    pbar = tqdm(range(config["system"]["num_round"]))
    model_save_directory = '../ctosfile'
    if not os.path.exists(model_save_directory):  #  res_root: "results"
        os.makedirs(model_save_directory)
    oram = WPathORAM(depth=3, storage_dir=model_save_directory)

    for global_round in pbar:
        all_state_dicts = []
        all_n_data = []
        all_losses = []
        all_delta_ccv_state=[]
        all_client_dict=[]
        all_coeff=[]
        all_norm_grad=[]
        while True:
            FiPMsavename = "TTP/"+f'F{global_round + 1}_PM'
            if os.path.exists('../'+FiPMsavename):
                time.sleep(0.1)
                stoc_position1= torch.load('../'+FiPMsavename)
                stoc_position=decrypt_file(stoc_position1, received_key)
                A1=stoc_position['serial_number']
                H1=stoc_position['H1']
                file_path_Fi = f'F{global_round + 1}'
                encrypt_Fi,A2=oram.accessread(stoc_position, file_path_Fi)
                if A1==A2:
                    data=decrypt_file(encrypt_Fi, received_key)
                    H1_Fi=get_model_hash(data)
                else:
                    raise ValueError("Error! The Fi address does not match.")
                if H1==H1_Fi:
                    if config["client"]["fed_algo"] == 'FedAvg':
                        global_state_dict = data['global_state_dict']
                    elif config["client"]["fed_algo"] == 'SCAFFOLD':
                        global_state_dict = data['global_state_dict']
                        scv_state=data['scv_state']
                    elif config["client"]["fed_algo"] == 'FedProx':
                        global_state_dict = data['global_state_dict']
                    elif config["client"]["fed_algo"] == 'FedNova':
                        global_state_dict = data['global_state_dict']
                    print("Data loaded successfully.")
                    break
                else:
                    raise ValueError("Error! The Fi hashvalue does not match.")
            else:
                time.sleep(0.1)  # wait for 1 second before checking again
            
        for client_id in trainset_config['users']:
            if config["client"]["fed_algo"] == 'FedAvg':   #FedAvg
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss = client_dict[client_id].train() 
                all_state_dicts.append(state_dict)
                all_n_data.append(n_data)
                all_losses.append(loss)           
            elif config["client"]["fed_algo"] == 'SCAFFOLD':
                client_dict[client_id].update(global_state_dict, scv_state)
                state_dict, n_data, loss, delta_ccv_state = client_dict[client_id].train()
                all_state_dicts.append(state_dict)
                all_n_data.append(n_data)
                all_losses.append(loss)   
                all_delta_ccv_state.append(delta_ccv_state)
            elif config["client"]["fed_algo"] == 'FedProx':
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss = client_dict[client_id].train()
                all_state_dicts.append(state_dict)
                all_n_data.append(n_data)
                all_losses.append(loss)    
            elif config["client"]["fed_algo"] == 'FedNova':
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss, coeff, norm_grad = client_dict[client_id].train()
                all_state_dicts.append(state_dict)
                all_n_data.append(n_data)
                all_losses.append(loss)
                all_coeff.append(coeff)
                all_norm_grad.append(norm_grad)

        if config["client"]["fed_algo"] == 'FedAvg':
            data_to_save = {
                'all_state_dicts': all_state_dicts, 
                'all_n_data': all_n_data,
                'all_losses': all_losses
                }
        elif config["client"]["fed_algo"] == 'SCAFFOLD':
            data_to_save = {
                'all_state_dicts': all_state_dicts, 
                'all_n_data': all_n_data,
                'all_losses': all_losses,
                'all_delta_ccv_state': all_delta_ccv_state
                }
        elif config["client"]["fed_algo"] == 'FedProx':
            data_to_save = {
                'all_state_dicts': all_state_dicts, 
                'all_n_data': all_n_data,
                'all_losses': all_losses,
                'all_client_dict':all_client_dict
                }
        elif config["client"]["fed_algo"] == 'FedNova':
            data_to_save = {
                'all_state_dicts': all_state_dicts,
                'all_n_data': all_n_data,
                'all_losses': all_losses,
                'all_coeff':all_coeff,
                'all_norm_grad':all_norm_grad
                }
        Ci=encrypt_file(data_to_save, key)
        data_to_save=decrypt_file(Ci,received_key)
        H2=get_model_hash(data_to_save)
        file_path_Ci = f'C{global_round + 1}'
        position_map = oram.accesswrite(Ci,file_path_Ci) 
        file_path = os.path.join(model_save_directory, str(position_map[file_path_Ci][0]), str(position_map[file_path_Ci][1])) #输出存储路径改
        serial_number = get_volume_serial_number(os.path.join(file_path,str(os.listdir(file_path)[0])))
        positon_to_save = {
            'file_path': file_path,
            'serial_number':serial_number,
            'H2':H2
            }
        PM_client=encrypt_file(positon_to_save, key) 
        time.sleep(1)
        if global_round > 0:
            os.remove("../TTP/"+f'C{global_round}_PM') 
        CiPMsavename = 'TTP/'+f'C{global_round + 1}_PM'
        torch.save(PM_client, "../"+CiPMsavename) 
        print("Model stored at (after second write):", file_path)
        print(f"Volume Serial Number for {file_path}: {serial_number}")
        print("Data writed successfully.")

if __name__ == "__main__":
    import time
    start_time = time.time()
    fed_run()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Program running time: {elapsed_time:.2f} seconds")







