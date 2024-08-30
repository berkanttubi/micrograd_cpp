#pragma once
#include "Value.h"
#include <iostream>
#include <vector>
#include <memory>


class Neuron {
public:
    Neuron(int nin);
    void printer();
    std::shared_ptr<Value> activator(std::vector<std::shared_ptr<Value>>& x);
    std::vector<std::shared_ptr<Value>> parameters();
    
    std::vector<std::shared_ptr<Value>> w;
    std::shared_ptr<Value> b;
};

class Layer {
public:
    Layer(int nin, int nout);
    std::vector<std::shared_ptr<Value>> activator(std::vector<std::shared_ptr<Value>>& x);
    std::vector<std::shared_ptr<Value>> parameters();
    
    std::vector<Neuron> neurons;
};

class MLP {
public:
    MLP(const std::vector<int>& nin, const std::vector<int>& nout);
    std::vector<std::shared_ptr<Value>> activator(std::vector<std::shared_ptr<Value>>& x);
    std::vector<std::shared_ptr<Value>> parameters();
    
    std::vector<Layer> layers;
};