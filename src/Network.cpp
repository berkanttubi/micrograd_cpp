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

    return Value(result + this->b[0].data,"","");
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
