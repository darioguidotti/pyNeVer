# pyNeVer

Neural networks Verifier (__NeVer__) is a tool for the training, pruning and verification of neural networks.
At present it supports sequential fully connected neural networks with ReLU and Sigmoid activation functions.
__pyNeVer__ is the corresponding python package providing all the main capabilities of the __NeVer__ tool
and can be easily installed using pip.

__REQUIREMENTS__
Given the characteristcs of PyTorch and ONNX we were not able to setup an auto-installation for these packages.
Therefore the user is required to install the torch, torchvision and onnx packages indipendently.
Guides on how to install such packages can be found at:
* <https://pytorch.org/get-started/locally/>
* <https://github.com/onnx/onnx>
