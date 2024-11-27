#!../../bin/python3
#存在问题，准确率不上升，loss不下降，怀疑是参数传递和读取出现问题
#FL+SGX+ORAM+保存csv
#与原始fl_main比时间大概多了10s检查问题。
#应该是跑fl_sever  fl_client.py  再继续跑。
#注意参数从GPU-》CPUhash值会变化  map_location=torch.device('cpu')  会使hash值不对应，解决办法 参数传输前先调用这个map_location=torch.device('cpu')。
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
import shutil#oram用到了
import csv
import time

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Util.Padding import unpad
import io
import hashlib      #-----20241116---新增hash验证

os.chdir(sys.path[0])#使用当前目录作为根目录
print("当前目录是~",sys.path[0])
import torch
# # 将PyTorch映射到CPU
# device = torch.device('cpu')
# # 在加载模型之前设置默认设备
# torch.set_default_tensor_type(torch.FloatTensor)
# 树节点类
class TreeNode:
    def __init__(self):
        self.blocks = [None] * 4  # 初始化4个子块
# PathORAM 类
class WPathORAM:
    def __init__(self, depth, storage_dir):
        self.depth = depth  # 树的深度
        self.tree_size = 2 ** (depth + 1) - 1  # 计算树的总节点数
        self.tree = [TreeNode() for _ in range(self.tree_size)]  # 初始化树节点
        self.position_map = {}  # 位置映射表
        self.storage_dir = storage_dir  # 存储目录
        #self.reset_storage()  # 重置存储

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

    #read data
    def accessread(self, position_map, filename):
        # 根据filename设置存储路径
        if filename.startswith('C'):
            storage_dir = '/host/ctosfile'
        elif filename.startswith('F'):
            storage_dir = '/host/stocfile'
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
    return torch.load(buffer, map_location=torch.device('cpu'))

key = b'v \xf35$\x90{\xbd-\xa2v\xc3\xbf\xb0\xf3\xa3' #密钥
received_key = b'v \xf35$\x90{\xbd-\xa2v\xc3\xbf\xb0\xf3\xa3'  # 此处填入接收到的密钥

def get_volume_serial_number(path):#获取卷号--改
    volume_info = os.stat(path)
    return volume_info.st_dev+volume_info.st_ino   #二者相加防止克隆，防止移动文件夹  

def load_checkpoint():  #检查是否断电
    files = [f for f in os.listdir('/host/') if f.startswith('F') and f.endswith('_PM')]  #SGX内要改路径
    print("Fi_files==",files)
    if not files:
        return 0, None, None  # 没有找到 Fi_PM 文件，返回轮次为 0
    latest_file = sorted(files, key=lambda x: int(x[1:].split('_')[0]), reverse=True)[0]
    global_round = int(latest_file[1:].split('_')[0]) -1 #获取轮次
    PM_server = torch.load(os.path.join('/host/', latest_file))     #SGX内要改路径
    PM_server = decrypt_file(PM_server, key)
    print('PM_server=',PM_server)

    file_path = PM_server['file_path']  #能读出来
    print('file_path=',file_path)
    fi_files = [f for f in os.listdir(file_path) if f.startswith('F')]
    if not fi_files:
        return 0, None, None  # 没有找到 Fi 文件，返回轮次为 0
    fi_file_path = os.path.join(file_path, fi_files[0])
    with open(fi_file_path, 'rb') as f:
        Fi = f.read()
    data_to_save = decrypt_file(Fi, key)
    print("重加载的global_round=",global_round)
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
#根据客户端数量划分数据集
trainset_config, testset = divide_data(num_client=config["system"]["num_client"], num_local_class=config["system"]["num_local_class"], dataset_name=config["system"]["dataset"],
                                        i_seed=config["system"]["i_seed"])   

pbar = tqdm(range(config["system"]["num_round"]))  #迭代次数
# 加载断点续练状态   如过不需要重加载  那么就直接把current_round=0就好
current_round, data_to_save, file_path = load_checkpoint()
#current_round=0
if current_round > 0:
    # 如果存在断点续练状态，使用加载的状态进行初始化
    if config["client"]["fed_algo"] == 'FedAvg' or config["client"]["fed_algo"] == 'FedProx' or config["client"]["fed_algo"] == 'FedNova':
        fed_server = FedServer(trainset_config['users'], dataset_id=config["system"]["dataset"], model_name=config["system"]["model"])
        fed_server.load_testset(testset)
        fed_server.state_dict().update(data_to_save['global_state_dict'])
    elif config["client"]["fed_algo"] == 'SCAFFOLD':
        fed_server = ScaffoldServer(trainset_config['users'], dataset_id=config["system"]["dataset"], model_name=config["system"]["model"])
        fed_server.load_testset(testset)
        fed_server.state_dict().update(data_to_save['global_state_dict'])
        fed_server.scv.load_state_dict(data_to_save['scv_state'])
    #------------------------------重新初始化oram
    model_save_directory = '/host/stocfile'  #result
    if not os.path.exists(model_save_directory):  #  res_root: "results"   ORAM模型保存在results文件夹下
        os.makedirs(model_save_directory)
    oram = WPathORAM(depth=3, storage_dir=model_save_directory) #初始化ORAM
else:    
    if config["client"]["fed_algo"] == 'FedAvg':
        fed_server = FedServer(trainset_config['users'], dataset_id=config["system"]["dataset"], model_name=config["system"]["model"])
        fed_server.load_testset(testset)        #加载测试集到服务器: 服务器加载测试数据集。  优化代码  if到上面
        global_state_dict = fed_server.state_dict()
        data_to_save = {
            'global_state_dict': global_state_dict,
            }
    elif config["client"]["fed_algo"] == 'SCAFFOLD':
        fed_server = ScaffoldServer(trainset_config['users'], dataset_id=config["system"]["dataset"], model_name=config["system"]["model"])
        scv_state = fed_server.scv.state_dict()
        fed_server.load_testset(testset)        #加载测试集到服务器: 服务器加载测试数据集。  优化代码  if到上面
        global_state_dict = fed_server.state_dict()
        data_to_save = {
            'global_state_dict': global_state_dict,
            'scv_state':scv_state
            }
    elif config["client"]["fed_algo"] == 'FedProx':
        fed_server = FedServer(trainset_config['users'], dataset_id=config["system"]["dataset"], model_name=config["system"]["model"])
        fed_server.load_testset(testset)        #加载测试集到服务器: 服务器加载测试数据集。  优化代码  if到上面
        global_state_dict = fed_server.state_dict()
        data_to_save = {
            'global_state_dict': global_state_dict,
            }
    elif config["client"]["fed_algo"] == 'FedNova':
        fed_server = FedNovaServer(trainset_config['users'], dataset_id=config["system"]["dataset"], model_name=config["system"]["model"])
        fed_server.load_testset(testset)        #加载测试集到服务器: 服务器加载测试数据集。  优化代码  if到上面
        global_state_dict = fed_server.state_dict()
        data_to_save = {
            'global_state_dict': global_state_dict,
            }
    #----------------初始化server ORAM 写入   model-->position+物理位置；
    model_save_directory = '/host/stocfile'  #result
    if not os.path.exists(model_save_directory):  #  res_root: "results"   ORAM模型保存在results文件夹下
        os.makedirs(model_save_directory)
    oram = WPathORAM(depth=3, storage_dir=model_save_directory) #初始化ORAM
    H1=get_model_hash(data_to_save)                 #获取模型hash值
    print("H1在server中的hash===",H1)
    Fi=encrypt_file(data_to_save, key)              #加密
    time.sleep(0.1)
    file_path_Fi = f'F{current_round+1}'
    position_map = oram.accesswrite(Fi,file_path_Fi)  # 写入数据
    print("position_map=",position_map)
    file_path = os.path.join(model_save_directory, str(position_map[file_path_Fi][0]), str(position_map[file_path_Fi][1]))
    print("file_path=",file_path)
    serial_number = get_volume_serial_number(os.path.join(file_path,str(os.listdir(file_path)[0])))
    positon_to_save = {
        'file_path': file_path,
        'serial_number':serial_number,
        'H1':H1
        }
    PM_server=encrypt_file(positon_to_save, key)
    FiPMsavename = "TTP/"+f'F{current_round+1}_PM'

    torch.save(PM_server, "/host/"+FiPMsavename)#保存位置--------------修改PM存储位置  修改为TTP  进行模拟

    print("Model stored at (after second write):", file_path)
    print(f"Volume Serial Number for {file_path}: {serial_number}")
    print("Data writed successfully.")
max_acc = 0
with open('/host/'+config["system"]["csv_file"], mode='w', newline='') as file: #----------------写入csv文件
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
                ctos_position = torch.load('/host/'+CiPMsavename)#获取位置表  ../ctosfile/14/ctosflmodel.pt
                ctos_position = decrypt_file(ctos_position, received_key)  #解密server模型位置
                print("ctos_position=",ctos_position)
                A1=ctos_position['serial_number']
                H2=ctos_position['H2']
                file_path_Ci=f'C{global_round + 1}'
                encrypt_Ci,A2=oram.accessread(ctos_position,file_path_Ci)
                if A1==A2:
                    #验证地址，地址匹配再解密
                    data=decrypt_file(encrypt_Ci, received_key)  #解密客户端模型 返回数据
                    H2_Ci=get_model_hash(data)
                    print("H2_Ci===",H2_Ci)
                else:
                    raise ValueError("Error! The Ci address does not match.")

                if H2==H2_Ci: # or H2_Ci==H3_Ci
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
                # Local training
            if config["client"]["fed_algo"] == 'FedAvg':   #FedAvg
                fed_server.rec(client_id, all_state_dicts[i], all_n_data[i], all_losses[i])
            elif config["client"]["fed_algo"] == 'SCAFFOLD':
                fed_server.rec(client_id, all_state_dicts[i], all_n_data[i], all_losses[i],all_delta_ccv_state[i])
            elif config["client"]["fed_algo"] == 'FedProx':
                fed_server.rec(client_id, all_state_dicts[i], all_n_data[i], all_losses[i])
            elif config["client"]["fed_algo"] == 'FedNova':
                fed_server.rec(client_id, all_state_dicts[i], all_n_data[i], all_losses[i],all_coeff[i], all_norm_grad[i])
            i=i+1    
        # Global aggregation  全局聚合
        fed_server.select_clients()
        if config["client"]["fed_algo"] == 'FedAvg':
            global_state_dict, avg_loss, _ = fed_server.agg()   ###需要每次都传输回去  存放的server的信息
        elif config["client"]["fed_algo"] == 'SCAFFOLD':
            global_state_dict, avg_loss, _, scv_state = fed_server.agg()  # scarffold
        elif config["client"]["fed_algo"] == 'FedProx':
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == 'FedNova':
            global_state_dict, avg_loss, _ = fed_server.agg()
        #print("avg_loss====",avg_loss)
        
        # Testing and flushing
        accuracy = fed_server.test()
        #print("accuracy====",accuracy)   #bug准确率一直没变，模型没更新？loss不降反而增长
        fed_server.flush()

        # Record the results
        # # 测试和记录：每个全局轮次后，测试模型的准确度，并记录相关信息。
        recorder.res['server']['iid_accuracy'].append(accuracy)
        recorder.res['server']['train_loss'].append(avg_loss)

        if max_acc < accuracy:
            max_acc = accuracy
        pbar.set_description(
            'Global Round: %d' % global_round +
            '| Train loss: %.4f ' % avg_loss +
            '| Accuracy: %.4f' % accuracy +
            '| Max Acc: %.4f' % max_acc)
  
        #------保存发送到client的模型--------
        if config["client"]["fed_algo"] == 'FedAvg':    #不同算法  模型参数不同
            data_to_save = {
                'global_state_dict': global_state_dict,
                }
        elif config["client"]["fed_algo"] == 'SCAFFOLD':
            data_to_save = {
                'global_state_dict': global_state_dict,
                'scv_state':scv_state
                }
        elif config["client"]["fed_algo"] == 'FedProx':    #不同算法  模型参数不同
            data_to_save = {
                'global_state_dict': global_state_dict,
                }
        elif config["client"]["fed_algo"] == 'FedNova':    #不同算法  模型参数不同
            data_to_save = {
                'global_state_dict': global_state_dict,
                }
        #--------oram保存模型----------
        model_save_directory = '/host/stocfile'  #result
        if not os.path.exists(model_save_directory):  #  res_root: "results"   ORAM模型保存在results文件夹下
            os.makedirs(model_save_directory)
        oram = WPathORAM(depth=3, storage_dir=model_save_directory) #初始化ORAM
        H1=get_model_hash(data_to_save) 
        print("server到client的Fi哈希值=-----",H1)
        Fi=encrypt_file(data_to_save, key)              #加密
        time.sleep(0.2)
        file_path_Fi = f'F{global_round + 2}'
        position_map = oram.accesswrite(Fi,file_path_Fi)  # 写入数据
        print("position_map=",position_map)
        file_path = os.path.join(model_save_directory, str(position_map[file_path_Fi][0]), str(position_map[file_path_Fi][1]))
        print("file_path=",file_path)
        serial_number = get_volume_serial_number(os.path.join(file_path,str(os.listdir(file_path)[0])))
        encrypt_file(file_path, key)
        positon_to_save = {
            'file_path': file_path,
            'serial_number':serial_number,
            'H1':H1
            }
        positon_to_save=encrypt_file(positon_to_save, key)
        if global_round + 1 > 0:
            os.remove("/host/TTP/"+f'F{global_round + 1}_PM')       #删除旧版本
        FiPMsavename = "TTP/"+f'F{global_round + 2}_PM'   #名字必须是+2,因为初始化是1,不然重复了。

        torch.save(positon_to_save, "/host/"+FiPMsavename)  #保存位置
        print("Model stored at (after second write):", file_path)
        print(f"Volume Serial Number for {file_path}: {serial_number}")
        print("Data writed successfully.")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Program running time: {elapsed_time:.2f} seconds")
        formatted_elapsed_time = "{:.2f}".format(elapsed_time)#保留两位小数
        # Save the results 在整个训练过程中，记录服务器端的准确度和训练损失，并保存为pt文件。
        
        writer.writerow([global_round, avg_loss, accuracy, max_acc,formatted_elapsed_time])
