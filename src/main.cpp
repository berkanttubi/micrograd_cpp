#include "Value.h"
#include "Network.h"
#include <memory>
#include <iostream>



std::vector<std::vector<std::shared_ptr<Value>>> convert_data(const std::vector<std::vector<double>>& data);

int main() {
    auto square = std::make_shared<Value>(2, "");
    std::vector<int> nin = {3};
    std::vector<int> nout = {4,4,1};
    auto network = MLP(nin, nout);
    std::vector<std::vector<double>> data = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}
    };
    std::vector<std::vector<std::shared_ptr<Value>>> converted_data = convert_data(data);
    std::vector<double> label = {1.0, -1.0, -1.0, 1.0};
    std::vector<std::shared_ptr<Value>> ypreds;
    std::shared_ptr<Value> final_loss;
    for (int epoch = 0; epoch < 300; epoch++) {
        ypreds.clear();
        for(auto& values : converted_data) {
            auto pred = network.activator(values);
            ypreds.push_back(pred[0]);
        }
        std::shared_ptr<Value> total_loss = std::make_shared<Value>(0.0, "");
        for (size_t i = 0; i < label.size(); i++) {
            auto ygt = std::make_shared<Value>(label[i], "");

            auto loss = *ypreds[i]-ygt;

            loss = loss->operator^(square);
            total_loss = *total_loss + loss;
        }
        //final_loss = total_loss / std::make_shared<Value>(label.size());

        for(auto& p : network.parameters()) {
            p->grad = 0.0;
        }

        total_loss->backward();

        for(auto& p : network.parameters()) {
            std::cout<<p->grad<<"\n";
            p->data += -0.1 * p->grad;
        }

        std::cout << "Epoch: " << epoch << " | Loss: " << total_loss->data << std::endl;
    }

    

    return 0;
}

std::vector<std::vector<std::shared_ptr<Value>>> convert_data(const std::vector<std::vector<double>>& data) {
    std::vector<std::vector<std::shared_ptr<Value>>> result;
    for (const auto& row : data) {
        std::vector<std::shared_ptr<Value>> row_ptrs;
        for (double value : row) {
            row_ptrs.push_back(std::make_shared<Value>(value, ""));
        }
        result.push_back(row_ptrs);
    }
    return result;
}
