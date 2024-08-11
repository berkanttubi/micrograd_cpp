#include "Value.h"


Value::Value(double data,std::string label){

    this->data = data;
    this->grad = 0.0;

    this-> label = label;

    std::cout<< "Value "<<this->data<<" is created! \n";

}



Value * Value::operator+(Value & other){

    auto data = this->data + other.data;
    Value * temp = new Value(data,"");

    temp->_prev = {this, &other};
    temp->_backward = [this, &other, temp]{
        this->grad += temp->grad;
        other.grad += temp->grad;
    };

    return temp;
}

Value * Value::operator*(Value & other){

    auto data = this->data * other.data;
    Value * temp = new Value(data,"");
    temp->_backward = [this, &other, temp]{
        this->grad *= temp->grad;
        other.grad *= temp->grad;
    };
    return temp;
}

