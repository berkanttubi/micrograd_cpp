#pragma once

#include <functional>
#include <unordered_set>
#include <string>
#include <memory>
#include <iostream>
#include <cmath>
#include <set>
class Value
{
public:
    double data;
    double grad;
    std::function<void()> _backward;
    std::vector<Value*> _prev;
    std::string _op;
    std::string label = "";


    Value(double data,std::string label,std::string op);
    Value * operator+(Value & other);
    Value * operator*(Value & other);
    Value * operator-(Value & other);
    Value* operator^(Value &other);

    Value * tanh();

    void backward();
};
