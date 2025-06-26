//LinearLayer.tpp
#include <random>

//Constructor 
template <typename MatrixType>
LinearLayer<MatrixType>::LinearLayer(
    unsigned int input_dim, 
    unsigned int output_dim, 
    unsigned int seed, 
    InitFn init_fn
){

    using Scalar = typename MatrixType::Scalar;
    
    std::mt19937 gen(seed); //TODO: confirm this is RNG I want

    this->weights = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(output_dim, input_dim);
    this->bias = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>(1, output_dim);

    for(unsigned int i =0; i < output_dim; ++i){
        this->bias(0, i) = init_fn(input_dim, output_dim, gen); 
        for(unsigned int j =0; j < input_dim; j++){
            this->weights(i, j) = init_fn(input_dim, output_dim, gen);
        }
    }


}


//forward
template <typename MatrixType>
MatrixType LinearLayer<MatrixType>::forward(const MatrixType& input){
    this->input_cache = input;
    //y = xW^T + b (broadcasted)
    return (input * this->weights.transpose()).rowwise() + this->bias; //TODO: eigen doesn't support rowwise addition or some shit
}

//backward
template <typename MatrixType>
MatrixType LinearLayer<MatrixType>::backward(const MatrixType& grad_output){
    //grad_weights = dL/dW - grad_putput^T * input
    this->grad_weights = grad_output.transpose() * this->input_cache;

    //grad_bias = dL/db = sum across rows
    this->grad_bias = grad_output.colwise().sum();
    
    //grad_input = grad_output * W
    return grad_output * this->weights;
}

//update_weights
template <typename MatrixType>
void LinearLayer<MatrixType>::update_weights(typename MatrixType::Scalar learning_rate){
    this->weights -= learning_rate * this->grad_weights; //where are weights and bias. I dont see any namespace? Can we be more explicit? this.weights?
    this->bias -= learning_rate * this->grad_bias;
}

//Getters - weights
template <typename MatrixType>
MatrixType& LinearLayer<MatrixType>::get_weights(){
    return  this->weights;
}

template <typename MatrixType>
const MatrixType& LinearLayer<MatrixType>::get_weights() const{
    return  this->weights;
}

template <typename MatrixType>
const MatrixType& LinearLayer<MatrixType>::get_grad_weights() const{
    return  this->grad_weights;
}

//Getters - biases

template <typename MatrixType>
MatrixType& LinearLayer<MatrixType>::get_bias(){
    return  this->bias;
}

template <typename MatrixType>
const MatrixType& LinearLayer<MatrixType>::get_bias() const{
    return  this->bias;
}

template <typename MatrixType>
const MatrixType& LinearLayer<MatrixType>::get_grad_bias() const{
    return  this->grad_bias;
}
