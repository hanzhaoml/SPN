# Parameter Learning of Sum-Product Networks

@Author: [Han Zhao](http://www.cs.cmu.edu/~hzhao1/)

@Note: Please cite the following paper if you use the tool developed in this package.

**A Unified Approach for Learning the Parameters of Sum-Product Networks**
by H. Zhao, P. Poupart and G. Gordon, NIPS 2016.

**Collapsed Variational Inference for Sum-Product Networks**
by H. Zhao, T. Adel, G. Gordon and B. Amos, ICML 2016.

@Required lib:
 - Boost (>= 1.55)
 - CMake (>= 2.80)

-------------------------------------------------------------------------------
The software is written in C++11 for the purpose of parameter learning 
in Sum-Product Networks. It supports batch learning, online learning as 
well as streaming learning. This software implements the following learning
algorithms for SPNs:

1.  Projected Gradient Descent.
2.  Exponentiated Gradient Method.
3.  Sequential Monomial Approximation (NIPS 2016).
4.  Concave-Convex Procedure/Expectation Maximization (NIPS 2016).
5.  Collapsed Variational Inference (ICML 2016).

-------------------------------------------------------------------------------


Usage: 

1.  batch\_learning.cpp, online\_learning.cpp and stream_learning.cpp should be the starting source files in order to understand the package.
2.  All the code about network structures is implemented in SPNNode.[h|cpp] and SPNetwork.[h|cpp].
3.  Learning algorithms under different learning scenarios, e.g., batch, online and streaming, are implemented separately in the corresponding *.cpp files.
