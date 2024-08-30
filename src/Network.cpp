#include "Network.h"
#include <random>
#include <sstream>
#include <memory>
#include <iostream>

// Constructor for Neuron
Neuron::Neuron(int nin) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for(int i = 0; i < nin; i++) {
        auto random_number = dis(gen);
        this->w.push_back(std::make_shared<Value>(random_number, "w"));
    }
    auto random_number = dis(gen);
    this->b = std::make_shared<Value>(random_number, "b");
}

// Print weights and biases
void Neuron::printer() {
    for(auto& i : this->w) {
        std::cout << i->data << std::endl;
    }
    std::cout << "b: " << this->b->data << std::endl;
}

// Activation function for Neuron
std::shared_ptr<Value> Neuron::activator(std::vector<std::shared_ptr<Value>>& x) {
    auto result = std::make_shared<Value>(0.0, "");
    for(size_t i = 0; i < x.size(); i++) {
        auto temp = *this->w[i] * x[i];
        result = *result + temp;
    }
    result = *result + this->b;
    return result->tanh();
}

// Get parameters for Neuron
std::vector<std::shared_ptr<Value>> Neuron::parameters() {
    std::vector<std::shared_ptr<Value>> _parameters;
    _parameters.insert(_parameters.end(), this->w.begin(), this->w.end());
    _parameters.push_back(this->b);
    return _parameters;
}

// Constructor for Layer
Layer::Layer(int nin, int nout) {
    for (int i = 0; i < nout; i++) {
        this->neurons.push_back(Neuron(nin));
    }
}

// Activation function for Layer
std::vector<std::shared_ptr<Value>> Layer::activator(std::vector<std::shared_ptr<Value>>& x) {
    std::vector<std::shared_ptr<Value>> out;
    for(auto& neuron : this->neurons) {
        out.push_back(neuron.activator(x));
    }
    return out;
}

// Get parameters for Layer
std::vector<std::shared_ptr<Value>> Layer::parameters() {
    std::vector<std::shared_ptr<Value>> _parameters;
    for(auto& neuron : this->neurons) {
        auto temp = neuron.parameters();
        _parameters.insert(_parameters.end(), temp.begin(), temp.end());
    }
    return _parameters;
}

// Constructor for MLP
MLP::MLP(const std::vector<int>& nin, const std::vector<int>& nout) {
    std::vector<int> sizes = nin;
    sizes.insert(sizes.end(), nout.begin(), nout.end());
    for (size_t i = 0; i < sizes.size() - 1; i++) {
        this->layers.push_back(Layer(sizes[i], sizes[i + 1]));
    }
}

// Activation function for MLP
std::vector<std::shared_ptr<Value>> MLP::activator(std::vector<std::shared_ptr<Value>>& x) {
    for(auto& layer : this->layers) {
        x = layer.activator(x);
    }
    return x;
}

// Get parameters for MLP
std::vector<std::shared_ptr<Value>> MLP::parameters() {
    std::vector<std::shared_ptr<Value>> _parameters;
    for(auto& layer : this->layers) {
        auto temp = layer.parameters();
        _parameters.insert(_parameters.end(), temp.begin(), temp.end());
    }
    return _parameters;
}
