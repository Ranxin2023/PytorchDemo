'''
Why do we learn deep learning?
Answer:
1,  Problems with long list of rules
- when the traditional approach fails, machine learning 
2, continually changing environments
-deep learning can adapt ('learn') to new scenarios
3, discovering insights within large collections of data
4, when you need explainability
- the pattern learned by a deep learning model are typically uninterpretable by a human
5, When the traditional approach is a better option
- if you can accomplish what you need with a simple rule-based system
6, When errors are unacceptable
- since the outputs of deep learning model aren't always predictable.
'''

import torch 
import pandas as pandas
import numpy as numpy
import matplotlib.pyplot as pyplot
def tensordemo():
    scalar=torch.tensor(7)
    print(scalar)
    print("the dimension of scalar is ", scalar.ndim)
    vector=torch.tensor([7, 7])
    print("the dimension of vector is", vector.ndim)
    print("the shape of vector is", vector.shape)
    matrix=torch.tensor([[7, 8], [9, 10]])
def main():
    print(torch.__version__)
    tensordemo()
if __name__=='__main__':
    main()