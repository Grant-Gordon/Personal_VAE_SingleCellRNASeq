//LinearLayer.tpp
#include <random>

template<typename Scalar>
using MatrixD = typename LinearLayer<Scalar>::MatrixD;
template<typename Scalar>
using VectorD = typename LinearLayer<Scalar>::VectorD;

//Constructor 
template <typename Scalar>
LinearLayer<Scalar>::LinearLayer(
    unsigned int input_dim, 
    unsigned int output_dim, 
    unsigned int seed, 
    InitFn init_fn
){

  
    static_assert(std::is_floating_point<Scalar>::value, "LinearLayer: Scalar must be floating-point.");
    assert(input_dim > 0 && output_dim > 0 
        && "LinearLayer: input_dim and output_dim must be > 0.");


    std::mt19937 gen(seed); //TODO: confirm this is RNG I want

    this->weights = MatrixD(output_dim, input_dim);
    this->bias = VectorD::Zero(output_dim);

    for(unsigned int i =0; i < output_dim; ++i){
        this->bias(0, i) = init_fn(input_dim, output_dim, gen); 
        for(unsigned int j =0; j < input_dim; j++){
            this->weights(i, j) = init_fn(input_dim, output_dim, gen);
        }
    }

 
}


//Forward Dense - all layers except input layer will use this dense X dense
template <typename Scalar>
MatrixD LinearLayer<Scalar>::forward(const MatrixD& input){
    this->input_cache = input;
    //y = xW^T + b (broadcasted): where input = (batch_size X features), W = (inputFeature X outputFeature), bias = (1 X output_size), input*W = (batch_size X output_features)
    return (input * this->weights.transpose()).rowwise() + this->bias.transpose();
}

//Backward  Dense - all layers except input layer will use this dense X dense
template <typename Scalar>
MatrixD LinearLayer<Scalar>::backward(const MatrixD& grad_output){
    

    //grad_weights = dL/dW - grad_putput^T * input
    this->grad_weights = grad_output.transpose() * this->input_cache;

    //grad_bias = dL/db = sum across rows
    this->grad_bias = grad_output.colwise().sum().transpose();
    
    //grad_input = grad_output * W
    return grad_output * this->weights;
}


//Forward Sparse Row - Custom forward for handling the sparse inputs of the first Linear Layer
template <typename Scalar>
VectorD LinearLayer<Scalar>::forward(const SingleSparseRow& input){
    this->input_cache_sparse = input;
    VectorD output = this->bias;
    
   
    for(int j = 0; j < input.nnz; ++j){
        int idx = input.indices[j];
        Scalar val = input.data[j];
        output += val * this->weights.row(idx).transpose();
    }
    return output;
}


//TODO: Review this function, not 100% on the logic 
//Backward Sparse Row - Custom backward for handling the sparse inputs of the first Linear Layer
template <typename Scalar>
VectorD LinearLayer<Scalar>::backward(const VectorD& grad_output){

    assert(this->input_cache_sparse.has_value() && "LinearLayer::backward (Sparse Row): no cached sparse input found");
    
    //gradient wrt bias
    this->grad_bias += grad_output;

    VectorD grad_input = VectorD::Zero(this->weights.cols()); //input_dim zeros
    
    for (int i = 0; i < input.nnz; ++i){
        //grad wrt weights
        const int idx = this->input_cache_sparse->indices[i];
        const Scalar val = this->input_cache_sparse->data[i];

        this->grad_weights.row(idx) += val * grad_output.transpose();

        //grad wrt inputs //TODO: I dont understand back prop well enough idk wth im using this for
        for (int j = 0; j < output_dim; ++ j){
            grad_input[idx] += grad_output[j] * this->weights(i, idx);
        }
    }
    return grad_input;
}



//update_weights
template <typename Scalar>
void LinearLayer<Scalar>::update_weights(Scalar learning_rate){

    this->weights -= learning_rate * this->grad_weights; 
    this->bias -= learning_rate * this->grad_bias;
}




//Getters - weights
template <typename Scalar>
MatrixD& LinearLayer<Scalar>::get_weights(){
    return  this->weights;
}

template <typename Scalar>
const MatrixD& LinearLayer<Scalar>::get_weights() const{
    return  this->weights;
}

template <typename Scalar>
const MatrixD& LinearLayer<Scalar>::get_grad_weights() const{
    return  this->grad_weights;
}

//Getters - biases

template <typename Scalar>
VectorD& LinearLayer<Scalar>::get_bias(){
    return  this->bias;
}

template <typename Scalar>
const VectorD& LinearLayer<Scalar>::get_bias() const{
    return  this->bias;
}

template <typename Scalar>
const VectorD& LinearLayer<Scalar>::get_grad_bias() const{
    return  this->grad_bias;
}



