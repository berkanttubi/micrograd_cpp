#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <memory>
#include <unordered_set>
#include <string>
#include <cmath>
#include <set>

class Value : public std::enable_shared_from_this<Value>
{
public:
    double data;
    std::string label;
    double grad;
    std::vector<std::shared_ptr<Value>> _children;
    std::function<void()> _backward;

    Value();
    Value(double data, std::string label);
    Value(double data,  std::vector<std::shared_ptr<Value>> _children, std::string label);

    // Operator overloads as non-member functions
    std::shared_ptr<Value> operator +(std::shared_ptr<Value> &obj);
    std::shared_ptr<Value> operator *(std::shared_ptr<Value> &obj);
    std::shared_ptr<Value> operator -(std::shared_ptr<Value> &obj);
    std::shared_ptr<Value>operator ^(std::shared_ptr<Value> &obj);
    std::shared_ptr<Value> tanh();
    void backward();


    void print_value();
};
