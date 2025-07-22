//TrainableLayer.h
template <typename Scalar>
class TrainableLayer : public Layer<Scalar> {
    public:
        bool has_trainable_params() const  override{return true;}

        const MatrixD<Scalar>& get_weights() const{ return this->weights;}
        MatrixD<Scalar>& get_weights(){ return this->weights;}

        const MatrixD<Scalar>& get_grad_weights() const{ return this->grad_weights;}
        MatrixD<Scalar>& get_grad_weights(){ return this->grad_weights;}
        
        const VectorD<Scalar>& get_bias() const{ return this->bias;}
        VectorD<Scalar>& get_bias(){ return this->bias;}

        const VectorD<Scalar>& get_grad_bias() const{ return this->grad_bias;}
        VectorD<Scalar>& get_grad_bias(){ return this->grad_bias;}
        
        const MatrixD<Scalar>& get_input_cache() const{ return this->input_cache;}
        MatrixD<Scalar>& get_input_cache(){ return this->input_cache;}
    
        const unsigned int get_input_dim() const {return this->input_dim;}
        const unsigned int get_output_dim() const {return this->output_dim;}



    protected: 
        MatrixD<Scalar> weights;
        MatrixD<Scalar> grad_weights;

        VectorD<Scalar> bias;
        VectorD<Scalar> grad_bias;

        MatrixD<Scalar> input_cache;

        unsigned int input_dim;
        unsigned int output_dim;

};
