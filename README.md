# Binder

This project demonstrates a prototype of the Binder framework.

## Install Binder

Install Occlum, Refer to [occlum/occlum: Occlum is a memory-safe, multi-process library OS for Intel SGX (github.com)](https://github.com/occlum/occlum/tree/master)

Step 1 (on the host): Start an Occlum container
```
docker pull occlum/occlum:0.29.3-ubuntu20.04
docker run -it --name=Binder --gpus all --device /dev/sgx/enclave occlum/occlum:0.29.3-ubuntu20.04 bash
```

Step 2  Copy pytorchFL1 to directory  `cd /root/demos/`

```
docker cp ./pytorchFL1.zip yourdocker:/root/demos/
docker attach yourdocker
cd /root/demos/
unzip pytorchFL1.zip
```

Step 3 (in the Occlum container): Download miniconda and install python to prefix position.

```
cd pytorchFL1
bash ./install_python_with_conda.sh
```

Step 3 Install FL package in SGX

```
./python-occlum/bin/pip install numpy~=1.21.5 scipy~=1.7.0 Pillow~=9.4.0 matplotlib~=3.4.2 tqdm~=4.61.1 opencv-python~=4.5.3.56 scikit-learn~=0.24.2 colorama~=0.4.4 pykeops~=2.1 pyyaml~=6.0
```

Step 4 Install anaconda3 in docker

Refer to [Anaconda | The Operating System for AI](https://www.anaconda.com/)

Step 5 Install FL environments in anaconda3 



## Run Binder

Step 1 (in the Occlum container): Run the sample code on Occlum

```
docker attach yourdocker
cd ~/demos/pytorchFL1
bash ./run_pytorch_on_occlum.sh
```



```
docker exec -it yourdocker /bin/bash
conda activate yourcondaname
cd ~/demos/pytorchFL1/occlum_instance/FLSC
python fl_client.py
```

