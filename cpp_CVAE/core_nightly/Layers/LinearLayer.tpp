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
  
    static_assert(std::is_floating_point<typename MatrixType::Scalar>::value, "LinearLayer: Scalar must be floating-point.");
    assert(input_dim > 0 && output_dim > 0 
        && "LinearLayer: input_dim and output_dim must be > 0.");


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
    //y = xW^T + b (broadcasted): where input = (batch_size X features), W = (inputFeature X outputFeature), bias = (1 X output_size), input*W = (batch_size X output_features)
    assert(input.cols() == this->weights.cols() 
        && "LinearLayer::forward: input cols must match weights cols (input_dim).");
    assert(this->bias.cols() == this->weights.rows() 
        && "LinearLayer::forward: bias width must match number of output features."); //TODO: is this even true?

    const auto logits = input * this->weights.transpose(); //TODO: point of optimization - Biases replicated and y=xW^T + b is not inplace
    return logits + this->bias.colwise().replicate(input.rows());
}

//backward
template <typename MatrixType>
MatrixType LinearLayer<MatrixType>::backward(const MatrixType& grad_output){
    
    assert(grad_output.rows() == this->input_cache.rows() 
        && grad_output.cols() == this->weights.rows() 
        &&"LinearLayer::backward: grad_output must match forward output shape.");

    assert(this->weights.cols() == this->input_cache.cols() 
        && "LinearLayer::backward: input_cache must match weights shape.");

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
   assert(this->weights.rows() == this->grad_weights.rows() 
        && this->weights.cols() == this->grad_weights.cols() 
        && "LinearLayer::update_weights: grad_weights must match weights shape.");

    assert(this->bias.rows() == this->grad_bias.rows() 
        && this->bias.cols() == this->grad_bias.cols() 
        && "LinearLayer::update_weights: grad_bias must match bias shape.");

   
    this->weights -= learning_rate * this->grad_weights; 
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
