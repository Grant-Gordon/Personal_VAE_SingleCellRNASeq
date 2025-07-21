//LinearLayer.tpp
#pragma once
#include <random>
#include <omp.h>
#include "config_values.h"
#include "custom_types.h"
//Constructor 
template <typename Scalar>
LinearLayer<Scalar>::LinearLayer(
    unsigned int input_dim, 
    unsigned int output_dim, 
    InitFn<Scalar> init_fn
){

  
    static_assert(std::is_floating_point<Scalar>::value, "LinearLayer: Scalar must be floating-point.");
    assert(input_dim > 0 && output_dim > 0 
        && "LinearLayer: input_dim and output_dim must be > 0.");


    std::mt19937 gen(configV::Global__seed); //TODO: confirm this is RNG I want

    this->weights = MatrixD<Scalar>(output_dim, input_dim);
    this->bias = VectorD<Scalar>::Zero(output_dim); //TODO confirm biases should Always be init to zeros

    for(unsigned int i =0; i < output_dim; ++i){
        this->bias(0, i) = init_fn(input_dim, output_dim, gen); 
        for(unsigned int j =0; j < input_dim; j++){
            this->weights(i, j) = init_fn(input_dim, output_dim, gen);
        }
    }

 
}


//Forward Dense - all layers except input layer will use this dense X dense
template <typename Scalar>
MatrixD<Scalar> LinearLayer<Scalar>::forward(const MatrixD<Scalar>& input){
    this->input_cache = input;
    //y = xW^T + b (broadcasted): where input = (batch_size X features), W = (inputFeature X outputFeature), bias = (1 X output_size), input*W = (batch_size X output_features)
    return (input * this->weights.transpose()).rowwise() + this->bias.transpose();
}

//Backward  Dense - all layers except input layer will use this dense X dense
template <typename Scalar>
MatrixD<Scalar> LinearLayer<Scalar>::backward(const MatrixD<Scalar>& upstream_grad){
    //B = batch_size, in_d & out_d = this layers input & ouput dimensionality 
    
    //upstream_grad [B * out_d]
    //weights = [out_d * in_d]
    //bias = [out_d * 1]
    // x = [B * in_d]
    //y = [B * out_d]
    
    // forward: y = Wx +b
    // find 
    //     dL/dW(gradient of weights)  - how much weightw contributes to to the loss function - deritivie of loss function wrt weights - if I adjust weights slightly how much does Loss change
    //     dL/db(gradients of bais)    - how much bias contributes to to the loss function - deritivie of loss function wrt bias - if I adjust bias slightly how much does Loss change
    //     dL/dx(gradients of input)   - how much input contribute to to the loss function - deritivie of loss function wrt input - if I adjust inputs slightly how much does Loss change

    //grad_weights [out_d * in_d]
    //  = dL/dW 
    //  = dL/dy *dy/dW
    //      dL/dy = upstream_grad
    //      dy/dW = x | specifically if y_i = [sum over j (w_ij * x_i + b_i)]  then dy_i/dW_ij = x_j
    //  = upstream_grad^T * input
    this->grad_weights = upstream_grad.transpose() * this->input_cache;

        //grad_bias [out_d *1]
        //  = dL/db 
        //  = dL/dy * dy/db
        //      dL/dy = upstream_grad
        //      dy/db = I | specifically if y_i = W_i * x + b_i, then dy/db_i = 1, for all i!=j, dy/db_i =0 i.e. I
        //  = upstream_grad * Identity
        //  = sum across rows
    this->grad_bias = upstream_grad.colwise().sum().transpose();
    
    //grad_input [B * in_d]
    //  = dL/dx 
    //  = dL/dy * dy/dx 
    //      dL/dy = upstream_grad
    //      dy/dx = W 
    //  = upstream_grad * Weights
    
    //this->grad_inputs  = upstream_grad * this->weights //TODO: only store grad_inputs if neccesarry?
    return upstream_grad * this->weights;
}


//Forward Sparse Row - Custom forward for handling the sparse inputs of the first Linear Layer
template <typename Scalar>
MatrixD<Scalar> LinearLayer<Scalar>::forward(const Batch<Scalar>& input){
    
    const unsigned int batch_size = static_cast<int>(input.size());
    MatrixD out(batch_size, this->output_dim); //Pre-allocate MatrixD to populate with SSR forward outputs
  
    #pragma omp parallel for 
    for (size_t i = 0; i < batch_size; ++i){
        VectorD ssr_output = this->bias;
        for(int j = 0; j < input.nnz; ++j){
            int idx = input.indices[j];
            Scalar val = input.data[j];
            ssr_output += val * this->weights.row(idx).transpose();
        }
        out.row(i) = ssr_output;
    }
    return out;
}


//Backward Sparse Row - Custom backward for handling the SingleSparseRow inputs of the first Linear Layer
template <typename Scalar>
VectorD<Scalar> LinearLayer<Scalar>::backward(const MatrixD<Scalar>& upstream_grad, const Batch<Scalar>& input){

    const unsigned int batch_size = static_cast<int>(input.size());
    MatrixD<Scalar> grad_out(batch_size, this->input_dim);
    
    //gradient of loss wrt bias
    // = dL/db 
    // = dL/dy * dy/db 
    //     dL/dy = upstream_grad    
    //     dy/db = I 
    // = upstream_grad * I
    this->grad_bias = upstream_grad.colwise().sum().transpose(); //Colwise sum because bias VectorD, input MatrixD


    //grad_weights [out_d * in_d]
    //  = dL/dW 
    //  = dL/dy *dy/dW
    //      dL/dy = upstream_grad
    //      dy/dW = x | specifically if y_i = [sum over j (w_ij * x_i + b_i)]  then dy_i/dW_ij = x_j
    //  = upstream_grad^T * input
    #pragma parallel for
    for(size_t i = 0; i < batch_size; ++i){

        for (size_t j =0; j< input[j].nzz; ++j){
            int col = input[i].indices[j];
            Scalar val = input[i].data[j];

            this->grad_weights.col(col) += upstream_grad * val;

            //gradient wrt inputs[in_d * 1] = 
            //  = dL/dx
            //  = upstream_grad * W 
            grad_out.row(i) = this->grad_weights.col(col).transpose * upstream_grad;



        }
    }
    return grad_out;
}



//update_weights
template <typename Scalar>
void LinearLayer<Scalar>::update_weights(Scalar learning_rate){

    this->weights -= learning_rate * this->grad_weights; 
    this->bias -= learning_rate * this->grad_bias;
}




//Getters - weights
template <typename Scalar>
MatrixD<Scalar>& LinearLayer<Scalar>::get_weights(){
    return  this->weights;
}

template <typename Scalar>
const MatrixD<Scalar>& LinearLayer<Scalar>::get_weights() const{
    return  this->weights;
}

template <typename Scalar>
const MatrixD<Scalar>& LinearLayer<Scalar>::get_grad_weights() const{
    return  this->grad_weights;
}

//Getters - biases

template <typename Scalar>
VectorD<Scalar>& LinearLayer<Scalar>::get_bias(){
    return  this->bias;
}

template <typename Scalar>
const VectorD<Scalar>& LinearLayer<Scalar>::get_bias() const{
    return  this->bias;
}

template <typename Scalar>
const VectorD<Scalar>& LinearLayer<Scalar>::get_grad_bias() const{
    return  this->grad_bias;
}



