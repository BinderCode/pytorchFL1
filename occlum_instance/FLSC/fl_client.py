#!/usr/bin/env python
#ORAM+SGX+FLclient
#注意参数从GPU-》CPUhash值会变化  map_location=torch.device('cpu')  会使hash值不对应，解决办法 参数传输前先调用这个map_location=torch.device('cpu')。
import os
import random
import shutil#oram用到了
import yaml
import pickle
import io
from tqdm import tqdm
import hashlib      #-----20241116---新增hash验证

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
# 树节点类
class TreeNode:
    def __init__(self):
        self.blocks = [None] * 4  # 初始化4个子块

# PathORAM 类
class WPathORAM:
    def __init__(self, depth, storage_dir,):
        self.depth = depth  # 树的深度
        self.tree_size = 2 ** (depth + 1) - 1  # 计算树的总节点数
        self.tree = [TreeNode() for _ in range(self.tree_size)]  # 初始化树节点
        self.position_map = {}  # 位置映射表
        self.storage_dir = storage_dir  # 存储目录

    # 重置存储
    def reset_storage(self):
        shutil.rmtree(self.storage_dir, ignore_errors=True)  # 删除现有存储
        os.makedirs(self.storage_dir, exist_ok=True)  # 创建存储目录
        for i in range(self.tree_size):
            node_path = os.path.join(self.storage_dir, str(i))
            os.makedirs(node_path, exist_ok=True)
            for j in range(1, 5):  # 为每个子块创建文件夹
                os.makedirs(os.path.join(node_path, str(j)), exist_ok=True)
        print("Storage reset completed.")

    # 获取从叶子节点到根节点的路径
    def _get_path(self, leaf):
        path = []  # 路径列表
        node_idx = leaf
        while node_idx > 0:
            path.append(node_idx)
            node_idx = (node_idx - 1) // 2  # 计算父节点索引
        path.append(0)  # 包含根节点
        print(f"Path for leaf {leaf}: {path}")
        return path

    # 获取随机叶子节点
    def random_leaf(self):
        leaf_start = 2 ** self.depth - 1  # 叶子节点起始索引
        leaf_end = self.tree_size - 1  # 叶子节点结束索引
        leaf = random.randint(leaf_start, leaf_end)  # 返回随机叶子节点
        print(f"Random leaf chosen: {leaf}")
        return leaf

    # 写入数据块
    def accesswrite(self,new_data,filename):
        self.reset_storage()  # 重置存储
        # 清空位置映射表（如果需要每次写入前清空）
        self.position_map.clear()
        # 随机选择该节点中的一个数据块
        chosen_block_idx = random.randint(1, 4)
        # 如果文件名不在位置映射表中，为其分配一个随机叶子节点
        if filename not in self.position_map:
            self.position_map[filename] = (self.random_leaf(), chosen_block_idx)

        leaf, block_idx = self.position_map[filename]
        print(f"Position map: {self.position_map}")
        path = self._get_path(leaf)  # 获取从叶子节点到根节点的路径
        # 随机选择路径中的一个节点
        chosen_node_idx = random.choice(path)
        print(f"Chosen node for real data: {chosen_node_idx}, block: {chosen_block_idx}")
        # 更新position_map为真实写入位置
        self.position_map[filename] = (chosen_node_idx, chosen_block_idx)
        # 写入数据到路径上的每个节点
        for node_idx in path:
            node_path = os.path.join(self.storage_dir, str(node_idx))

            for idx in range(1, 5):
                if node_idx == chosen_node_idx and idx == chosen_block_idx:
                    # 写入真实数据
                    block_path = os.path.join(node_path, str(idx), filename)
                    data_to_write = new_data
                    print(f"Writing real data to {block_path}")
                else:
                    # 写入伪数据
                    block_path = os.path.join(node_path, str(idx), f'fake_data_block_{filename}')
                    data_to_write = os.urandom(len(new_data))
                    print(f"Writing fake data to {block_path}")

                with open(block_path, 'wb') as f:
                    f.write(data_to_write)

        # 返回位置映射表，包含文件名和对应的叶子节点位置及子块编号
        return self.position_map

    #read data----------------
    def accessread(self, position_map, filename):
        # 根据filename设置存储路径
        if filename.startswith('C'):
            storage_dir = '../ctosfile'
        elif filename.startswith('F'):
            storage_dir = '../stocfile'
        else:
            raise ValueError("Invalid filename provided. Must be 'Ci' or 'Fi'.")

        # 获取文件路径
        file_path = position_map['file_path']
        
        # 提取叶子节点和块索引
        path_parts = file_path.split('/')
        leaf = int(path_parts[-2])  # 获取倒数第二部分作为叶子节点
        block_idx = int(path_parts[-1])  # 获取最后一部分作为块索引

        # 获取从叶子节点到根节点的路径
        path = self._get_path(leaf)

        data_block = None
        real_block_volume_serial_number = None  # 存储真实数据块的卷号
        all_volume_serial_numbers = []  # 储存所有访问过的块的卷号，包括虚拟和真实

        # 读取路径上的每个节点数据块
        for node_idx in path:
            node_path = os.path.join(storage_dir, str(node_idx))  # 使用动态设置的存储路径
            for idx in range(1, 5):  # 读取所有子块
                block_path = os.path.join(node_path, str(idx), filename if node_idx == leaf and idx == block_idx else f'fake_data_block_{filename}')
                # 获取每个块的卷号，包括虚拟块
                volume_serial_number = get_volume_serial_number(block_path)
                all_volume_serial_numbers.append(volume_serial_number)
                if os.path.exists(block_path):
                    with open(block_path, 'rb') as f:
                        data = f.read()
                        print(f"Read {'real' if node_idx == leaf and idx == block_idx else 'fake'} data from {block_path}")
                        if node_idx == leaf and idx == block_idx:
                            data_block = data  # 真实数据块
                            real_block_volume_serial_number = volume_serial_number  # 保存真实数据块的卷号

        if data_block is None:
            raise Exception("Data block not found")

        # 返回真实数据块和它的卷号
        return data_block, real_block_volume_serial_number

def encrypt_file(data, key):
    # 将数据保存到一个字节流中
    buffer = io.BytesIO()
    torch.save(data, buffer)
    serialized_data = buffer.getvalue()
    
    # 创建cipher对象并加密
    cipher = AES.new(key, AES.MODE_CBC)
    encrypted_data = cipher.encrypt(pad(serialized_data, AES.block_size))
    
    # 返回加密数据和初始化向量
    return cipher.iv + encrypted_data

def decrypt_file(input_data, key):
    # 检查输入数据类型并读取加密内容
    if isinstance(input_data, str) and os.path.exists(input_data):
        # 当输入是文件路径并且文件存在时，从文件读取加密数据
        with open(input_data, 'rb') as file:
            encrypted_data = file.read()
    elif isinstance(input_data, bytes):
        # 当输入直接是字节串时，使用字节串作为加密数据
        encrypted_data = input_data
    else:
        raise ValueError("Input must be a valid file path or bytes object containing encrypted data.")

    # 提取IV和加密内容
    iv = encrypted_data[:16]
    encrypted_content = encrypted_data[16:]

    # 创建cipher对象并解密
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_data = unpad(cipher.decrypt(encrypted_content), AES.block_size)

    # 使用torch.load反序列化数据到字典，并映射到CPU
    buffer = io.BytesIO(decrypted_data)
    return torch.load(buffer,map_location=torch.device('cpu')) #在SGX中   从GPU-CPUhash值会改变，所以需要多解密一次，同时设置解密函数在cpu上。
 
key = b'v \xf35$\x90{\xbd-\xa2v\xc3\xbf\xb0\xf3\xa3' #密钥
# 接收到的密钥
received_key = b'v \xf35$\x90{\xbd-\xa2v\xc3\xbf\xb0\xf3\xa3'  # 此处填入接收到的密钥

def get_volume_serial_number(path):#获取卷号
    volume_info = os.stat(path)
    return volume_info.st_dev+volume_info.st_ino   #二者相加防止克隆，防止移动文件夹  
def get_model_hash(model_weights):  #--20241116---新加
    """
    计算给定字典数据（模型权重）的哈希值。
    参数:    model_weights (dict): 模型的权重字典。
    返回:    str: 模型哈希值的十六进制字符串表示。
    """
    # 检查输入是否为字典
    if not isinstance(model_weights, dict):
        raise ValueError("输入数据必须是一个字典类型")

    # 将模型的权重序列化为字节流
    buffer = io.BytesIO()
    torch.save(model_weights, buffer)
    model_weights_bytes = buffer.getvalue()
    # 计算模型的SHA-256哈希值
    model_hash = hashlib.sha256(model_weights_bytes).hexdigest()
    return model_hash

def fed_run():
    """
    Main function for the baselines of federated learning
    """
    # 读取配置文件：通过yaml.safe_load函数，加载一个YAML格式的配置文件。
    import sys
    os.chdir(sys.path[0])#使用当前目录作为根目录
    print("当前目录是~",sys.path[0])
    with open("config/test_config.yaml", "r") as yaml_file:
        try:
            config = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)
    # 数据验证和准备：确保指定的算法、数据集和模型是支持的，并分割数据集。
    algo_list = ["FedAvg", "SCAFFOLD", "FedProx", "FedNova"]
    assert config["client"]["fed_algo"] in algo_list, "The federated learning algorithm is not supported"

    dataset_list = ['MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN', 'CIFAR100']
    assert config["system"]["dataset"] in dataset_list, "The dataset is not supported"

    model_list = ["LeNet", 'AlexCifarNet', "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "CNN"]
    assert config["system"]["model"] in model_list, "The model is not supported"
    # 随机数生成器种子设置：为了可重复的实验结果，设置了NumPy和PyTorch的随机数生成器种子。
    np.random.seed(config["system"]["i_seed"])
    torch.manual_seed(config["system"]["i_seed"])
    random.seed(config["system"]["i_seed"])
    # 初始化客户端和服务端：根据配置文件和算法类型，初始化客户端和服务端实例。
    client_dict = {}
    #根据客户端数量划分数据集   
    trainset_config, testset = divide_data(num_client=config["system"]["num_client"], num_local_class=config["system"]["num_local_class"], dataset_name=config["system"]["dataset"],
                                           i_seed=config["system"]["i_seed"])   
    #客户端数量  type(trainset_config)== <class 'dict'>    type(testset)== <class 'torchvision.datasets.mnist.MNIST'>
    # Initialize the clients w.r.t. the federated learning algorithms and the specific federated settings  每个客户端对象加载其训练数据集。
    # 在联邦学习算法和特定的联邦设置之外初始化客户机
    # Initialize the clients w.r.t. the federated learning algorithms and the specific federated settings  为每个客户端创建一个相应的对象
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

    # Main process of federated learning in multiple communication rounds
    # 主循环：在多个通讯轮次（Communication Rounds）中，各客户端进行本地训练，然后全局服务器进行模型更新。
    #state_dict: 这是该客户端训练完毕后的模型的状态字典，包含了模型的所有参数。
    #n_data是客户端数据集的大小或其他形式的权重，用于在全局模型更新时进行加权。  client_dict[client_id].name=f_00098, state_dict, n_data, loss
    #loss: 这是客户端在其本地数据集上训练模型所得到的损失值。
    pbar = tqdm(range(config["system"]["num_round"]))    #--------迭代次数
    model_save_directory = '../ctosfile'  #模型存储路径
    if not os.path.exists(model_save_directory):  #  res_root: "results"   ORAM模型保存在results文件夹下
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
            if os.path.exists('../'+FiPMsavename):  #不能使用host
                time.sleep(0.1)
                stoc_position1= torch.load('../'+FiPMsavename)# 读取加密数据
                stoc_position=decrypt_file(stoc_position1, received_key)  #解密server模型位置
                print("stoc_position=",stoc_position)
                print(stoc_position['file_path'])
                A1=stoc_position['serial_number']
                H1=stoc_position['H1']
                file_path_Fi = f'F{global_round + 1}'#更换名字
                encrypt_Fi,A2=oram.accessread(stoc_position, file_path_Fi)
                if A1==A2:
                    #验证地址，地址匹配再解密
                    data=decrypt_file(encrypt_Fi, received_key)  #解密客户端模型 返回数据
                    H1_Fi=get_model_hash(data)
                    print("H1在client中的hash===",H1)
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
                    #os.remove('../stoc_position.pt')
                    break
                else:
                    raise ValueError("Error! The Fi hashvalue does not match.")
            else:
                time.sleep(0.1)  # wait for 1 second before checking again
            
        for client_id in trainset_config['users']:
            # Local training
            if config["client"]["fed_algo"] == 'FedAvg':   #FedAvg
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss = client_dict[client_id].train() 
                all_state_dicts.append(state_dict)
                all_n_data.append(n_data)
                all_losses.append(loss)   #客户端loss值一样             
            elif config["client"]["fed_algo"] == 'SCAFFOLD':
                client_dict[client_id].update(global_state_dict, scv_state)
                state_dict, n_data, loss, delta_ccv_state = client_dict[client_id].train()#客户端训练查看
                #print(client_dict[client_id].name)
                all_state_dicts.append(state_dict)
                all_n_data.append(n_data)
                all_losses.append(loss)   #客户端loss值一样
                all_delta_ccv_state.append(delta_ccv_state)
            elif config["client"]["fed_algo"] == 'FedProx':
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss = client_dict[client_id].train()
                all_state_dicts.append(state_dict)
                all_n_data.append(n_data)
                all_losses.append(loss)   #客户端loss值一样  
            elif config["client"]["fed_algo"] == 'FedNova':
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss, coeff, norm_grad = client_dict[client_id].train()
                all_state_dicts.append(state_dict)
                all_n_data.append(n_data)
                all_losses.append(loss)   #客户端loss值一样
                all_coeff.append(coeff)
                all_norm_grad.append(norm_grad)

        if config["client"]["fed_algo"] == 'FedAvg':
            data_to_save = {
                'all_state_dicts': all_state_dicts,   #存放GPU训练的模型
                'all_n_data': all_n_data,
                'all_losses': all_losses
                }
        elif config["client"]["fed_algo"] == 'SCAFFOLD':
            data_to_save = {
                'all_state_dicts': all_state_dicts,   #存放GPU训练的模型
                'all_n_data': all_n_data,
                'all_losses': all_losses,
                'all_delta_ccv_state': all_delta_ccv_state
                }
        elif config["client"]["fed_algo"] == 'FedProx':
            data_to_save = {
                'all_state_dicts': all_state_dicts,   #存放GPU训练的模型
                'all_n_data': all_n_data,
                'all_losses': all_losses,
                'all_client_dict':all_client_dict
                }
        elif config["client"]["fed_algo"] == 'FedNova':
            data_to_save = {
                'all_state_dicts': all_state_dicts,   #存放GPU训练的模型
                'all_n_data': all_n_data,
                'all_losses': all_losses,
                'all_coeff':all_coeff,
                'all_norm_grad':all_norm_grad
                }
        Ci=encrypt_file(data_to_save, key)#加密模型
        data_to_save=decrypt_file(Ci,received_key)  #在SGX中   从GPU-CPUhash值会改变，所以需要多解密一次
        H2=get_model_hash(data_to_save)  #获取hash值
        file_path_Ci = f'C{global_round + 1}'
        position_map = oram.accesswrite(Ci,file_path_Ci) #写入数据返回ci位置表
        print("position_map=",position_map)
        file_path = os.path.join(model_save_directory, str(position_map[file_path_Ci][0]), str(position_map[file_path_Ci][1])) #输出存储路径改
        print("file_path=",file_path)
        serial_number = get_volume_serial_number(os.path.join(file_path,str(os.listdir(file_path)[0])))
        positon_to_save = {
            'file_path': file_path,
            'serial_number':serial_number,
            'H2':H2
            }
        PM_client=encrypt_file(positon_to_save, key) 
        time.sleep(1)
        if global_round > 0:
            os.remove("../TTP/"+f'C{global_round}_PM')       #删除旧版本 
        CiPMsavename = 'TTP/'+f'C{global_round + 1}_PM'
        torch.save(PM_client, "../"+CiPMsavename)#保存位置 
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
# 这是整个脚本的核心部分，完成以下几个主要任务：

# 读取配置文件：通过yaml.safe_load函数，加载一个YAML格式的配置文件。
# 数据验证和准备：确保指定的算法、数据集和模型是支持的，并分割数据集。
# 随机数生成器种子设置：为了可重复的实验结果，设置了NumPy和PyTorch的随机数生成器种子。
# 初始化客户端和服务端：根据配置文件和算法类型，初始化客户端和服务端实例。
# 主循环：在多个通讯轮次（Communication Rounds）中，各客户端进行本地训练，然后全局服务器进行模型更新。
# 测试和记录：每个全局轮次后，测试模型的准确度，并记录相关信息。