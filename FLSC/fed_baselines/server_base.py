from utils.models import *
import torch
from torch.utils.data import DataLoader
from utils.fed_utils import assign_dataset, init_model


class FedServer(object):
    def __init__(self, client_list, dataset_id, model_name):
        """
        Initialize the server for federated learning.  初始化FL服务器
        :param client_list: List of the connected clients in networks   网络中已连接的客户端列表
        :param dataset_id: Dataset name for the application scenario    应用场景的数据集名称
        :param model_name: Machine learning model name for the application scenario  应用场景的机器学习模型名称
        """
        # Initialize the dict and list for system settings
        self.client_state = {}
        self.client_loss = {}
        self.client_n_data = {}
        self.selected_clients = []
        # batch size for testing
        self._batch_size = 200   #初始化用于测试的批处理大小
        self.client_list = client_list

        # Initialize the test dataset
        self.testset = None

        # Initialize the hyperparameter for federated learning in the server
        self.round = 0
        self.n_data = 0
        self._dataset_id = dataset_id
  
        # Testing on GPU   服务器检测是否有GPU可用，并据此设置计算设备（self._device）。
        gpu = -1      #=0使用gpu   -1cpu
        self._device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")   #gpu利用率低
        # Initialize the global machine learning model  服务器也初始化一个全局机器学习模型。
        self._num_class, self._image_dim, self._image_channel = assign_dataset(dataset_id)
        self.model_name = model_name
        self.model = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)

    def load_testset(self, testset):
        """
        Server loads the test dataset.
        :param data: Dataset for testing.服务器可以加载一个用于测试的数据集。
        """
        self.testset = testset

    def state_dict(self):
        """
        Server returns global model dict.
        :return: Global model dict  返回当前全局模型的状态字典。
        """
        return self.model.state_dict()

    def test(self):
        """
        Server tests the model on test dataset.  在测试数据集上测试当前全局模型，并返回准确率。
        """
        test_loader = DataLoader(self.testset, batch_size=self._batch_size, shuffle=True)
        self.model.to(self._device)
        accuracy_collector = 0
        for step, (x, y) in enumerate(test_loader):
            with torch.no_grad():
                b_x = x.to(self._device)  # Tensor on GPU
                b_y = y.to(self._device)  # Tensor on GPU

                test_output = self.model(b_x)
                pred_y = torch.max(test_output, 1)[1].to(self._device).data.squeeze()
                accuracy_collector = accuracy_collector + sum(pred_y == b_y)
        accuracy = accuracy_collector / len(self.testset)

        return accuracy.cpu().numpy()

    def select_clients(self, connection_ratio=1):
        """
        Server selects a fraction of clients.
        :param connection_ratio: connection ratio in the clients
        """
        # select a fraction of clients
        self.selected_clients = []
        self.n_data = 0
        for client_id in self.client_list:
            b = np.random.binomial(np.ones(1).astype(int), connection_ratio)
            if b:
                self.selected_clients.append(client_id)
                self.n_data += self.client_n_data[client_id]

    def agg(self):
        """
        这是服务器端模型聚合的主要逻辑。
        它收集来自各个客户端的模型权重和损失，然后根据客户端数据点的数量进行加权平均。
        然后将这个加权平均模型设置为新的全局模型。
        Server aggregates models from connected clients. 
        :return: model_state: Updated global model after aggregation
        :return: avg_loss: Averaged loss value
        :return: n_data: Number of the local data points
        """
        client_num = len(self.selected_clients)
        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0, 0

        # Initialize a model for aggregation   初始化聚合模型
        model = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        model_state = model.state_dict()
        avg_loss = 0

        # Aggregate the local updated models from selected clients   聚合来自选定客户端的本地更新模型
        for i, name in enumerate(self.selected_clients):
            if name not in self.client_state:
                continue
            for key in self.client_state[name]:
                if i == 0:
                    model_state[key] = self.client_state[name][key] * self.client_n_data[name] / self.n_data
                else:
                    model_state[key] = model_state[key] + self.client_state[name][key] * self.client_n_data[
                        name] / self.n_data

            avg_loss = avg_loss + self.client_loss[name] * self.client_n_data[name] / self.n_data  #n_data: Number of the local data points

        # Server load the aggregated model as the global model
        self.model.load_state_dict(model_state)
        self.round = self.round + 1
        n_data = self.n_data

        return model_state, avg_loss, n_data

    def rec(self, name, state_dict, n_data, loss):
        """
        服务器接收从特定客户端发送来的模型状态、本地数据点数量和本地损失。
        这些信息存储在服务器的字典中，用于后续的模型聚合。
        Server receives the local updates from the connected client k.
        :param name: Name of client k
        :param state_dict: Model dict from the client k
        :param n_data: Number of local data points in the client k
        :param loss: Loss of local training in the client k
        """
        self.n_data = self.n_data + n_data
        self.client_state[name] = {}
        self.client_n_data[name] = {}

        self.client_state[name].update(state_dict)
        self.client_n_data[name] = n_data
        self.client_loss[name] = {}
        self.client_loss[name] = loss

    def flush(self):
        """
        清空服务器中存储的客户端信息，以准备进行下一轮的联邦学习。
        Flushing the client information in the server
        """
        self.n_data = 0
        self.client_state = {}
        self.client_n_data = {}
        self.client_loss = {}
