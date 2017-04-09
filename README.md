# dl-on-azure
A bunch of scrips and demos for deep learning on Azure.

At the time of writing there was no easy way to stand up the Azure Data Science VM.


# Creating an Ubuntu VM in Azure
I assume you are working from an Ubuntu client. Install the Azure CLI client
```
sudo pip install azure-cli
```

Start by logging the CLI client into your Azure account. Exactly how you go about this will depend on what authentication approach you have configured for your account.
```
az login
```

Once login is complete your subscriptions will be listed in the terminal window. There is a good chance that you'll have multiple subscriptions for your account. If you do and you may like to [change your default subscription](https://docs.microsoft.com/en-us/cli/azure/manage-azure-subscriptions-azure-cli) something like this.
```
az account set --subscription <your-sub-here>
```
Now we'll create a resource group to hold our VM and associated resources. We'll be using an NC series (GPU compute enabled) VM and so you'll need to check that you create things in the correct region. You can check availablility of the NC series VMs in various data centers at the [Azure Linux VM Pricing](https://azure.microsoft.com/pricing/details/virtual-machines/linux/) page.
```
az group create --name dlazuredemo --location SouthCentralUS
```

We need to generate an SSH keypair that we'll use to connect to the VM. 
```
ssh-keygen -f ~/.ssh/dlazuredemo_rsa -t rsa -b 2048 -C '' -N ''
```

Now we can create our VM. We'll choose an Ubuntu 16.04 workstation image because we'll take advantage of being able to connect to the machine GUI later. We are launching the smallest NC Series VM, an NC6; Note that this is still a $1/hr machine so you'll want to de-allocate it when done and we'll see that below. If you prefer to start with an alternative marketplace VM images you can [list these](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/cli-ps-findimage). This process will take a few minutes.
```
az vm create --resource-group dlazuredemo --name dlazuredemo --image Canonical:UbuntuServer:16.04-LTS:latest --size Standard_NC6 --storage-sku Standard_LRS --admin-username dldemoadmin --ssh-key-value ~/.ssh/dlazuredemo_rsa.pub
```

For convenience, and to save money, you'll want to have a start and stop shell script to start and stop your machine. These are the two lines repectively.
```
az vm deallocate --resource-group dlazuredemo --name dlazuredemo
az vm start --resource-group dlazuredemo --name dlazuredemo
```

## Connecting and Configuring a Desktop
We can now connect to our machine using SSH. You'll need to use the IP address that was output when you created the machine. You can featch the IP address in the future using `az vm list-ip-addresses`. You'll need to do this every time you `deallocate` and re-`start` the VM.
```
ssh dldemoadmin@127.0.0.1 -i ~/.ssh/dlazuredemo_rsa
```
**All of the commands below are now being executed on the remote machine over the SSH connection.**
We will start by configuring a full desktop environment on the machine. More options on [server GUI environemnts](https://help.ubuntu.com/community/ServerGUI) can be found in the Ubuntu documentation.
```
sudo apt-get install ubuntu-desktop
sudo apt-get install vnc4server
```

## TODO: MOre on getting GNOME Setup


## Installing nvidia bits
First we install the CUDA Toolkit. Current version at time of writing is 8.0.61-1
```
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
sudo apt-get update
sudo apt-get install -y cuda
rm cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
```

Next we instal CuDNN. We currently install CuDNN v5.1 rather than v6 as the later version is not yet well supported by the various Deep Learning Frameworks.
```
wget http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz
sudo tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local
rm cudnn-8.0-linux-x64-v5.1.tgz
sudo ldconfig
```

I'm a fan of the Anaconda Python distribution so we'll install that for our Python needs.
```
wget https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh -O ~/anaconda.sh
bash ~/anaconda.sh -b -p $HOME/anaconda
export PATH="$HOME/anaconda/bin:$PATH"
rm Anaconda3-4.3.1-Linux-x86_64.sh
```

Export environment variables. Ensure that you `exit` the SSH session and reconnect after modifying `.bashrc`.
```
sudo sed -i '$ a export CUDA_HOME=/usr/local/cuda-8.0' ~/.bashrc 
sudo sed -i '$ a export PATH=~/anaconda/bin${PATH:+:${PATH}}' ~/.bashrc
sudo sed -i '$ a export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}' ~/.bashrc
sudo sed -i '$ a export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' ~/.bashrc  
sudo sed -i '$ a export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' ~/.bashrc 
sudo sed -i '$ a export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' ~/.bashrc 
```

# Deep Learning Frameworks
You may not want to install all of these, but, the scripts below install each of the various Deep Learning frameworks into a Virtual Environment using `conda`. Pretty well everyone has an MNIST demo and so we run that after install to check that all is well.

## Dependencies
I 've found these dependencies need to be installed. I've pulled them all together even though they are not required by all of the below. I am using an OpenCV install script from [here](http://milq.github.io/install-opencv-ubuntu-debian/)
```
sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y dist-upgrade
sudo apt-get -y autoremove
sudo apt-get install libopencv-dev python-opencv
sudo apt-get install libjpeg-dev zlib1g-dev
sudo apt-get install libcurl4-openssl-dev
sudo apt-get install build-essential
sudo apt-get install llvm
sudo apt-get install clang
udo apt-get install libsox-dev sox
sudo apt-get install openmpi-bin
```

## Nirvana Systems (Intel) Neon
```
conda create --name neon pip numpy scipy jupyter
source activate neon
git clone https://github.com/NervanaSystems/neon.git
cd neon && make sysinstall
examples/mnist_mlp.py
source deactivate   
``` 
## Google TensorFlow
```
conda create --name tensorflow python=3.6 numpy scipy h5py jupyter
source activate tensorflow
mkdir tensorflow
cd tensorflow
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.1-cp36-cp36m-linux_x86_64.whl
wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py
python mnist_softmax.py
source deactivate
```

## Microsoft CNTK
Please note that we are installing the 1-Bit SGD variant of CNTK. This licensed under a specific license which is [more restrictive](https://github.com/Microsoft/CNTK/wiki/Enabling-1bit-SGD#license-difference-between-cntk-and-1bit-sgd) than the rest of CNTK. Details can be found [here](https://github.com/Microsoft/CNTK/wiki/Enabling-1bit-SGD). If you would prefer to install the standard GPU or CPU builds then the wheel files are [here](https://github.com/Microsoft/CNTK/wiki/Setup-Linux-Python).
```
conda create --name cntk python=3.5 numpy scipy h5py jupyter
activate cntk
mkdir cntk
cd cntk
pip install https://cntk.ai/PythonWheel/GPU-1bit-SGD/cntk-2.0rc1-cp35-cp35m-linux_x86_64.whl
TODO: Get path correctly exported
python -c "import cntk; print(cntk.__version__)"
python -m cntk.sample_installer
source deactivate
```

## XGBoost and LightGBM
These aren't Deep Learning frameworks, but, I have a couple of examples that use as well. We install GPU optimized support for XGBoost. 