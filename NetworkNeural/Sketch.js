import NeuralNetwork from "./NeuralNetwork.js";

const nn = new NeuralNetwork(2,2,1)

// XOR Problem
const dataset = {
    inputs:
        [[1, 1],
        [1, 0],
        [0, 1],
        [0, 0]],
    outputs:
        [[0],
        [1],
        [1],
        [0]]
}

let isTrained = true

console.log("✔️ Iniciando treinamento")
function update(timestamp){   
    if(isTrained){
        
        for (let index = 0; index < 1000; index++) {           
             const random = Math.floor(Math.random() * (dataset.inputs.length - 0) + 0)
             nn.train(dataset.inputs[random], dataset.outputs[random])
        }

        if(nn.predict([0,0])[0]  < 0.04 && nn.predict([1,0])[0] > 0.98){
            isTrained = false
            console.log("---------- Status ----------")
            console.log("✅ Treinamento concluído ")
            console.log("----------------------------")
        }
        
    }    

    requestAnimationFrame(update)    
}

requestAnimationFrame(update)  

// const predict = nn.predict([1,1])
// predict.print()



