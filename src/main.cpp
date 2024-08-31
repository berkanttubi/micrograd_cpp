#include "Value.h"
#include "Network.h"
#include <memory>
#include <iostream>



std::vector<std::vector<std::shared_ptr<Value>>> convert_data(const std::vector<std::vector<double>>& data);

int main() {
    std::vector<int> nin = {3};
    std::vector<int> nout = {4,1}; 
    auto network = MLP(nin, nout);

    std::vector<std::vector<double>> data = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}
    };
    std::vector<std::vector<std::shared_ptr<Value>>> converted_data = convert_data(data);
    std::vector<std::vector<std::shared_ptr<Value>>> test = convert_data(data);
     for(size_t i = 0; i < converted_data.size(); i++) {
        std::cout << "Input: ";
        for(auto& val : converted_data[i]) {
            std::cout << val->data << " ";
        }
     }
    std::vector<double> label = {1.0, 0.0, 0.0, 1.0};  // Changed to 0 and 1 for binary classification

    double learning_rate = 0.01;
    double clip_value = 1.0;

    for (int epoch = 0; epoch < 200; epoch++) {
        std::shared_ptr<Value> total_loss = std::make_shared<Value>(0.0, "total_loss");

        for(size_t i = 0; i < converted_data.size(); i++) {
            auto pred = network.activator(converted_data[i]);
            auto ygt = std::make_shared<Value>(label[i], "ygt");
            auto diff = *pred[0] - ygt;
            auto loss = *diff * diff;  // MSE loss
            total_loss = *total_loss + loss;
        }

        for(auto& p : network.parameters()) {
            p->grad = 0.0;
        }

        total_loss->backward();

        // Gradient clipping and update
        for(auto& p : network.parameters()) {
            if (std::isnan(p->grad)) {
                std::cout << "NaN gradient detected!" << std::endl;
                return 1;
            }
            p->grad = std::max(std::min(p->grad, clip_value), -clip_value);
            p->data += -learning_rate * p->grad;
        }

        if (epoch % 10 == 0) {
            std::cout << "Epoch: " << epoch << " | Loss: " << total_loss->data << std::endl;
        }
    }


    

    // Test the trained network
    for(size_t i = 0; i < test.size(); i++) {
        auto pred = network.activator(test[i]);
        
        std::cout << "| Prediction: " << pred[0]->data << " | Target: " << label[i] << std::endl;
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
