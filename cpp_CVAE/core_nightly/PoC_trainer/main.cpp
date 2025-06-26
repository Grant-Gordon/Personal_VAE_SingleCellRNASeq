#include "Module.h"
#include "MSE.h"
#include "Adam.h"
#include "trainer.h"
#include "RELU.h"
#include "param_init_utils.h"
//TODO: need to include init_param_helper stuff?


int main(){
    using MatrixType = Eigen::MatrixXd;
    using Scalar = typename MatrixType::Scalar;

    const int input_dim = 100;
    const int latent_dim = 16;
    const int batch_size = 32;
    const int SEED = 1000;
    const int NUM_EPOCHS = 4;

    //build model
    Module<MatrixType> model;
    model.add_layer(std::make_shared<LinearLayer<MatrixType>>(input_dim, latent_dim, SEED, he_init<Scalar>)); //TOOD he_ has /*output_dim*/ and std:mt19937 gen, not int SEED
    model.add_layer(std::make_shared<RELULayer<MatrixType>>());
    model.add_layer(std::make_shared<LinearLayer<MatrixType>>(latent_dim, input_dim, SEED, he_init<Scalar>));


    //Build Optimizer and Loss
    MSE<MatrixType> loss_fn;
    Adam<MatrixType> optimizer;
    
    //train
    Trainer<MatrixType> trainer(model, loss_fn, optimizer);
    trainer.train(NUM_EPOCHS, batch_size, input_dim);
    return 0;
}