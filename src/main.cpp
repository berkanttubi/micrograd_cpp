#include "Value.h"
#include "Network.h"
#include <sstream>
#include <cmath>

Value calculate_loss(std::vector<double> ygt, std::vector<Value> ypreds);
int main() {
    Value square = Value(2,"","");
    std::vector<int> nin = {3};
    std::vector<int> nout = {1,1,1};
    auto network = MLP(nin, nout);

    std::vector<std::vector<double>> data = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}
    };

    std::vector<double> label = {1.0, -1.0, -1.0, 1.0};
    

    for (int epoch = 0; epoch<100; epoch++){
        std::vector<Value> ypreds {};

        for(auto& values: data){
        auto pred = network.activator(values);
        ypreds.push_back(pred);
        }

        Value loss = Value(0.0, "", "");
        for (size_t i = 0; i < label.size(); i++) {
            Value ygt = Value(label[i], "", "");
            Value diff = *(ypreds[i] - ygt);
            Value diff_square = *(diff ^ square);
            loss = *(loss+diff_square);
            std::cout<<loss.data<<std::endl;
        }
        for(auto& p:network.parameters()){
            p.grad = 0.0;
        }
        loss.backward();
        
        for(auto& p:network.parameters()){
            p.data += -0.1 * p.grad;
        }

        std::cout<<"Epoch: "<<epoch<<" |  Loss: "<<loss.data<<std::endl;
    }

    



    return 0;
}


Value calculate_loss(std::vector<double> ygt, std::vector<Value> ypreds){
    Value loss = Value(0.0,"","");
    Value square = Value(2,"","");
    for(int i = 0; i< ypreds.size(); i++){

        Value temp = Value(ygt[i],"","");
        Value new_ = *(ypreds[i] - temp);
        Value new_2 = *(new_ ^ square);
        loss = *(loss + new_2);
    }

    return loss;

}