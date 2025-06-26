//trainer.h
#pragma once

#include <iostream>
#include <memory>
#include "Module.h"
#include "MSE.h"
#include "Adam.h"


template <typename MatrixType>
class Trainer{

    public:
        using Scalar = typename MatrixType::Scalar;

        //Initializer List
        Trainer(
            Module<MatrixType>& model,
            MSE<MatrixType>& loss_fn,
            Adam<MatrixType>& optimizer,
            Scalar learning_rate = 0.001
        ):
            model(model),
            loss_fn(loss_fn),
            optimizer(optimizer),
            learning_rate(learning_rate)
        {}

        void train(int num_epochs, int batch_size, int input_dim){
            for(int epoch =0; epoch < num_epochs; ++epoch){
                //Dummy data generator
                MatrixType input = MatrixType::Random(batch_size, input_dim); //TODO this is for PoC only, model cannot train on random data

                //forward
                MatrixType output = model.forward(input);
                Scalar loss = loss_fn.forward(output, input);


                //Backward
                MatrixType grad = loss_fn.backward(output, input);
                model.backward(grad);
                optimizer.step(model.get_layers());


                if (epoch % 2 == 0){
                    std::cout << "Epoch: " << epoch << ", Loss: " << loss << "\n";
                }
            }
        }


    private:
            Module<MatrixType>& model;
            MSE<MatrixType>& loss_fn;
            Adam<MatrixType>& optimizer;
            Scalar learning_rate;

};
