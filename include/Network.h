#include "Value.h"
#include <iostream>

class Neuron{

public:
    Neuron(int nin);
    void printer();
    Value activator(std::vector<double>);
    std::vector<Value> parameters();
    std::vector<Value> w;
    std::vector<Value> b;
};


class Layer{

public:
    Layer(int nin, int noun);
    std::vector<Value> activator(std::vector<double>);
    std::vector<Value> parameters();
    std::vector<Neuron> neurons;
};

class MLP{

public:
    MLP(std::vector<int> nin, std::vector<int> nout);
    Value activator(std::vector<double>);
    std::vector<Value> parameters();
    std::vector<Layer> layers;
};