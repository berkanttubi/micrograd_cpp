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