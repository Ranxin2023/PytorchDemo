from tensor_demo import tensor_demo
from nn_training import redirect_nn_training
# import torch, torchvision
def main():
    tensor_demo()
    redirect_nn_training()
    # print(torch.__version__)
    # print(torchvision.__version__)
    
if __name__=="__main__":
    main()