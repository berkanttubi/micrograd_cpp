#include "Value.h"


Value::Value(double data,std::string label, std::string op){

    this->data = data;
    this->grad = 0.0;
    this-> label = label;
    this->_op = op;
    //std::cout<< "Value "<<this->data<<" is created! \n";

}



Value * Value::operator+(Value & other){

    auto data = this->data + other.data;
    Value * temp = new Value(data,"","+");

    temp->_prev = {this, &other};
    temp->_backward = [this, &other, temp]{
        this->grad += temp->grad;
        other.grad += temp->grad;
    };

    return temp;
}

Value * Value::operator*(Value & other){

    auto data = this->data * other.data;
    Value * temp = new Value(data,"","*");
    temp->_prev = {this, &other};
    temp->_backward = [this, &other, temp]{
        this->grad += other.data * temp->grad;
        other.grad += this->data * temp->grad;
    };
    return temp;
}

Value * Value::tanh(){
    double n = this->data;
    double t = ((std::exp(2*n) -1) / (std::exp(2*n) + 1));
    Value * temp = new Value(data,"tanh","tan()");
    temp->_prev = {this};
    temp->_backward = [this, temp, t]{
        this->grad += (1 - std::pow(t,2)) * temp->grad;
    };

    return temp;
}



void Value::backward() {
    std::vector<Value*> topo;
    std::set<Value*> visited;


    std::function<void(Value*)> build_topo = [&](Value* v) {
            if (visited.find(v) == visited.end()) {
                visited.insert(v);
                for (Value* child : v->_prev) {
                    build_topo(child);
                }
                topo.push_back(v);
            }
        };


    build_topo(this);
    this->grad = 1.0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        try {
            (*it)->_backward();
        }catch (const std::exception& e) {
            // Handle other exceptions if necessary, or continue
            continue;
        }

    }
}

  
