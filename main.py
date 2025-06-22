from tensor_demo import tensor_demo
from nn_training import redirect_nn_training
from transfer_learning import transfer_learning_redirect_output
# import torch, torchvision
def main():
    # tensor_demo()
    # redirect_nn_training()
    # print(torch.__version__)
    # print(torchvision.__version__)
    transfer_learning_redirect_output()
    
if __name__=="__main__":
    main()