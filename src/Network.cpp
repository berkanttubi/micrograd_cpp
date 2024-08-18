#include "Network.h"
#include <random>
#include <sstream>


Neuron::Neuron(int nin){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for(int i = 0; i<nin; i++){
        auto random_number = dis(gen);
        this->w.push_back(Value(random_number, "w", ""));
    }
    auto random_number = dis(gen);
    this->b.push_back(Value(random_number, "b", ""));
}

void Neuron::printer(){

    for(auto& i:this->w){
        std::cout<<i.data<<std::endl;
    }
    std::cout<<"b: "<<this->b[0].data;
}

Value Neuron::activator(std::vector<double> x){

    double result = 0.0;
    for(int i = 0; i< x.size(); i++){
        result += (this->w[i].data * x[i]);
    }

    return *Value(result + this->b[0].data,"","").tanh();
}

std::vector<Value> Neuron::parameters(){
    std::vector<Value> _parameters {};
    
    for(auto&i : this->w){
        _parameters.push_back(i);
    }

    for(auto&i : this->b){
        _parameters.push_back(i);
    }

    return _parameters;
}   


Layer::Layer(int nin, int nout){
    for (int i = 0; i<nout; i++){
        this->neurons.push_back(Neuron(nin));
    }
}

std::vector<Value> Layer::activator(std::vector<double> x){

    std::vector<Value> out {};

    for(auto& neuron:this->neurons){
        out.push_back(neuron.activator(x));
    }

    return out;

}
std::vector<Value> Layer::parameters(){
    std::vector<Value> _parameters {};
    
    for(auto& neuron:this->neurons){
        std::vector<Value> temp;
        temp = neuron.parameters();
        _parameters.insert(_parameters.end(),temp.begin(),temp.end());
    }

    return _parameters;
}

MLP::MLP(std::vector<int> nin, std::vector<int> nout){
    nin.insert(nin.end(),nout.begin(), nout.end());

    for (int i = 0; i< nout.size(); i++){
        this->layers.push_back(Layer(nin[i], nin[i+1]));
    }

}

Value MLP::activator(std::vector<double> x){

    std::vector<Value> out;

    for(auto& layer:this->layers){
        out = layer.activator(x);
    }

    return out[0];

}
std::vector<Value> MLP::parameters(){
    std::vector<Value> _parameters {};
    
    for(auto& layer:this->layers){
        std::vector<Value> temp;
        temp = layer.parameters();
        _parameters.insert(_parameters.end(),temp.begin(),temp.end());
    }

    return _parameters;
}