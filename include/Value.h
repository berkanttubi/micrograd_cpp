#pragma once

#include <functional>
#include <unordered_set>
#include <string>
#include <memory>
#include <iostream>

class Value
{
public:
    double data;
    double grad;
    std::function<void()> _backward;
    std::vector<Value*> _prev;
    std::string op;
    std::string label = "";


    Value(double data,std::string label);
    Value * operator+(Value & other);
    Value * operator*(Value & other);
};
