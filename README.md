# NN
Neural Net closely built off of Dave Miller's implementation url: https://www.millermattson.com/dave/ <br>
Also uses code built off of Lazy Foo' Productions' tutorial for rendering using SDL url: https://lazyfoo.net/tutorials/SDL/ <br>
rendering code is a bit mangled, but it gets the job done.

To run, requires: 
- ISO C++20 Standard
- SDL2.0 package

To configure neural net and drawing: <br>
Preprocessor definitions declared at the top of the main file (i.e., the NN.cpp file), change these values to obtain differnt networks. <br>
Network training constants:
- DATA_SET_SIZE: Corresponds to the size of the data set the network trains on. Data type = size_t.
- DISPLAY_FACTOR: Tells the program to display a test run on the network every (DATA_SET_SIZE / DISPLAY_FACTOR) many iterations since the last display. Data type = size_t.
- USE_SAVED_DATA: Tells the network to run on prebuilt data, a causing the network to be trained the exact same every time it is run with this set tot true (note: this option will override all other values set for training). Data type = bool.
- LEAKY: Tells the network to use a leaky ReLU as it's activation function if set to true, otherwise the network will use a hyperbolic tangent activation function. Data type = bool
- OPERATOR: Choice between AND, NAND, OR, NOR, XOR, and NXOR. Tells the network what operator to train to be by changing the built data set. Data type = enum boolOperator.
- TOPOLOGY: Tells the neural network what layers are desired from it. Ex: "1 9 9 3" corresponds to a network with an input layer with 1 input neuron, a hidden layer fully connected to the input layer with 9 neaurons, another hidden layer fully connected to the previous hidden layer with 9 neurons, and an output layer fully connected to the previous layer with 3 output neurons. Data type = string, format = "i s1 s2 s3... sn o", data type for values i, s1, s2, s3,... sn, o = size_t. 

Network drawing constants:
- NEURON_WIDTH: Indicates the width to draw neurons with. Data type = size_t.
- NEURON_HEIGHT: Indicates the height to draw neurons with. Data type = size_t.
