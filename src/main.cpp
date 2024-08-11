#include "Value.h"
#include <sstream>



int main() {
    // Değerlerin oluşturulması
    auto x1 = std::make_shared<Value>(2.0, "a", "");
    auto x2 = std::make_shared<Value>(0.0, "b", "");
    auto w1 = std::make_shared<Value>(-3.0, "w1", "");
    auto w2 = std::make_shared<Value>(1.0, "w2", "");
    auto b = std::make_shared<Value>(6.8813735870195432, "b", "");

    // Hesaplamaların yapılması
    auto x1w1 = *x1 * *w1;
    x1w1->label = "x1*w1";
    x1w1->_op = "";
    auto x2w2 = *x2 * *w2;
    x2w2->label = "x2*w2";
    x2w2->_op = "";
    auto x1w1x2w2 = *x1w1 + *x2w2;
    x1w1x2w2->label = "x1*w1 + x2*w2";
    x1w1x2w2->_op = "";
    auto n = *x1w1x2w2 + *b; 
    n->label = "n";
    n->_op = "";
    auto o = n->tanh(); 
    o->label = "o";
    o->_op = "";
    o->grad = 1;
    o->backward();

    std::cout << "Gradients:" << std::endl;
    std::cout << "o grad: " << o->grad << std::endl;
    std::cout << "n grad: " << n->grad << std::endl;
    std::cout << "b grad: " << b->grad << std::endl;
    std::cout << "x1w1x2w2 grad: " << x1w1x2w2->grad << std::endl;
    std::cout << "x2w2 grad: " << x2w2->grad << std::endl;
    std::cout << "x1w1 grad: " << x1w1->grad << std::endl;
    std::cout << "w2 grad: " << w2->grad << std::endl;
    std::cout << "w1 grad: " << w1->grad << std::endl;
    std::cout << "x2 grad: " << x2->grad << std::endl;
    std::cout << "x1 grad: " << x1->grad << std::endl;

    return 0;
}
