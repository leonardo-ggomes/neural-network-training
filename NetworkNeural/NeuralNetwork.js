import Matrix from "./Matrix.js";

class NeuralNetwork{

    learning_rating = 0.1

    constructor(i_nodes, h_nodes, o_nodes){
        this.i_nodes = i_nodes
        this.h_nodes = h_nodes
        this.o_nodes = o_nodes

        this.i_weights = new Matrix(h_nodes, i_nodes)
        this.i_weights.randomize()

        this.I_BIAS = new Matrix(h_nodes, 1)
        this.I_BIAS.randomize()
    }

    static sigmoid(x){
        return +(1 / (1 + Math.exp(-x))).toFixed(4)
    }

    static dsigmoid(x){
        return +(x * (1 - x)).toFixed(4)
    }

    train(inputs, target){
        /** FeedFoward */
        //Input to Hidden
        const I = Matrix.arrayToMatrix(inputs)
        const W_I = Matrix.multiply(this.i_weights, I)
        const W_H = Matrix.add(W_I, this.I_BIAS)

        //Activation function
        W_H.map(NeuralNetwork.sigmoid)    
        
        //Sigmoid Output
        const W_Ho = this.i_weights.sum()  
        const O = Matrix.multiply(W_Ho, W_H)
        const H = Matrix.add(O, this.I_BIAS)
      
        //Output
        const predict = H.map(NeuralNetwork.sigmoid)
      

        /** Backpropagation */
        // Step - 1 
        const matrix_error = Matrix.arrayToMatrix(target)       
        const E_o = Matrix.subtract(matrix_error, predict)
      
        // Step - 2 : Output to Hidden
        const dsigmoid_error = Matrix.map(E_o, NeuralNetwork.dsigmoid)

        //How many steps do I need get?
        const hadmard_product = Matrix.hadamard(dsigmoid_error, E_o)  
        const Go = Matrix.multiply_scalar(hadmard_product, this.learning_rating)       

        const W_Ht = Matrix.transpose(W_H)   
        const W_Hto = Matrix.outerProduct(Go, W_Ht)
        this.W_Hfinal = Matrix.add(W_Ho,W_Hto)

        // Step - 3 : Hidden to Input        
        const E_h = Matrix.outerProduct(E_o, W_Ho)
        const D_W_Ho = W_Ho.map(NeuralNetwork.dsigmoid)
        const D_Ho_hadamard = Matrix.hadamard(D_W_Ho,E_h)
        const G_h = Matrix.multiply_scalar(D_Ho_hadamard, this.learning_rating)
  
        const W_ih = Matrix.outerProduct(I, G_h)  
        this.i_weights = Matrix.transpose(W_ih)
    }


    predict(inputs){
        const I = Matrix.arrayToMatrix(inputs)
        const W_I = Matrix.multiply(this.i_weights, I)
        const W_H = Matrix.add(W_I, this.I_BIAS)

        //Activation function
        W_H.map(NeuralNetwork.sigmoid)            
      
        const O = Matrix.multiply(this.W_Hfinal, W_H)
        const H = Matrix.add(O, this.I_BIAS)
      
        //Output
        const predict = H.map(NeuralNetwork.sigmoid)        
        return predict;
    }
}

export default NeuralNetwork