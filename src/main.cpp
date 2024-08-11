#include "Value.h"


int main(){

    Value *a = new Value(2.0, "a");
    Value *b = new Value(-3.0, "b");
    Value *c = new Value(10.0, "c");

    Value *d = *(*a * *b) + *c;

    std::cout<<d->_prev[0]->data<<" | "<< d->_prev[1]->data;

}