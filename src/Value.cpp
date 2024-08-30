#include "Value.h"

Value::Value() {
    this->label = "uninitialized";
}

Value::Value(double data, std::string label) {
    this->data = data;
    this->label = label;
    this->grad = 0.0;
}

Value::Value(double data, std::vector<std::shared_ptr<Value>> _children, std::string label) {
    this->data = data;
    this->label = label;
    this->_children = std::move(_children);

}



std::shared_ptr<Value> Value::operator+(std::shared_ptr<Value> &obj) {
    double new_data = (*obj).data + this->data;
    auto new_value = std::make_shared<Value>(new_data, std::vector<std::shared_ptr<Value>>{shared_from_this(), obj}, (*obj).label + "+" + this->label);
    new_value->_backward = [self = shared_from_this(), obj, new_value]() {
        (*obj).grad += 1.0 * new_value->grad;
        self->grad += 1.0 * new_value->grad;
    };
    return new_value;
}

std::shared_ptr<Value> Value::operator*(std::shared_ptr<Value> &obj){

    double new_data = (*obj).data * this->data;
    auto new_value = std::make_shared<Value>(new_data, std::vector<std::shared_ptr<Value>>{shared_from_this(), obj}, obj->label + "*" + this->label);

    new_value->_backward = [self = shared_from_this(), obj, new_value]() {
        (*obj).grad += self->data * new_value->grad;
        self->grad += (*obj).data * new_value->grad;
    };


    return new_value;
}

std::shared_ptr<Value> Value::operator-(std::shared_ptr<Value> &obj) {
    auto negative = std::make_shared<Value>(-1, "");
    auto negated_obj = *negative * obj;
    return shared_from_this()->operator+(negated_obj);
}


std::shared_ptr<Value> Value::operator^(std::shared_ptr<Value> &obj){

    auto data = this->data * (*obj).data ;
    auto new_value = std::make_shared<Value>(data,std::vector<std::shared_ptr<Value>>{shared_from_this(), obj},obj->label + "^" + this->label);
    new_value->_backward = [this, &obj, new_value]{
        this->grad += ((*obj).data * std::pow(this->data,((*obj).data)-1)) * (*obj).grad;

    };
    return new_value;
}

std::shared_ptr<Value> Value::tanh() {
    double n = this->data;
    double t = ((std::exp(2*n) - 1) / (std::exp(2*n) + 1));
    auto new_value = std::make_shared<Value>(t, std::vector<std::shared_ptr<Value>>{shared_from_this()}, "tanh()");
    new_value->_backward = [self = shared_from_this(), new_value, t]() {
        self->grad += (1 - std::pow(t, 2)) * new_value->grad;
    };
    return new_value;
}

void Value::backward() {
    std::vector<std::shared_ptr<Value>> topo;
    std::set<std::shared_ptr<Value>> visited;

    std::function<void(std::shared_ptr<Value>)> build_topo = [&](std::shared_ptr<Value> v) {
        if (visited.find(v) == visited.end()) {
            visited.insert(v);
            for (const auto& child : v->_children) {
                if (child != nullptr) {
                    build_topo(child);
                }
            }
            topo.push_back(v);
        }
    };


    build_topo(shared_from_this());
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

void Value::print_value() {
    std::cout << "Value: " << this->data << std::endl;
}
