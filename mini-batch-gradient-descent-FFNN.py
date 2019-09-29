import numpy as np
import pylab
import random

def sigmoid(x) :
    return 1/(1 + np.exp(-x))

def  derivate_sigmoid(x) :
    return sigmoid(x) * (1-sigmoid(x))

def single_node_process(inputs, weights) :
    v = np.dot(inputs , weigths)
    return v

def error(target,output) :
    return 0.5 * (target -output) ** 2

def activate(val) :
    return sigmoid(val)

def gradient_output(input,target,output) :
    return derivate_sigmoid(input) * (target- output)

def gradient_hidden(input,gradients,weights) :
    sum = 0
    for i in range(len(gradients)) :
        sum +=gradients[i] * weights[i]
    return derivate_sigmoid(input) * sum

def update_weight(weight,momentum,learning_rate,gradient,output) :
    return weight + momentum * weight + learning_rate * gradient * output

def initiate_weight(input_size, hidden_layer, nb_nodes):
    weights = []
    weights_input_hidden = []
    weights_hidden_hidden = []
    weights_hidden_output = []
    
    # Random weight from input layer to hidden layer
    for i in range(0, nb_nodes):
        nb_nodes_i = []
        for j in range(0, input_size):
            weight = random.uniform(0.0, 1.0)
            nb_nodes_i.append(weight)
        weights_input_hidden.append(nb_node_i)
    weights.append(weights_input_hidden)
    
    # Random weight from hidden layer to hidden layer
    for i in range(1, hidden_layer):
        hidden_layer_i = []
        for j in range(0, nb_nodes):
            nb_nodes_i = []
            for k in range (0, nb_nodes):
                weight = random.uniform(0.0, 1.0)
                nb_nodes_i.append(weight)
            hidden_layer_i.append(nb_nodes)
        weights_hidden_hidden.append(hidden_layer_i)
    weights.append(weights_hidden_hidden)

    # Random weight from hidden layer to output layer
    for i in range(0, nb_nodes):
        weight = random.uniform(0.0, 1.0)
        weights_hidden_output.append(weight)
    weights.append(weights_hidden_output)
    
    return weights

def mini_batch(data,batch_size) :
    mini = []
    np.random.shuffle(data)
    
    for i in range((len(data) // batch_size) +1) :
        batch = data[i * batch_size : (i+1) * batch_size]
        mini.append(batch)
    
    return mini

def feedforward(weights, inputs, nb_nodes, hidden_layer ) :
    output_feed = []
    output_weights = []
    hidden_feed = []
    hidden_weights =[]
    
    for i in range(nb_nodes) :
        output = single_node_process(inputs, weights[0][i])
        hidden_weight.append(output)
        activ = activate(output)
        hidden_feed.append(activ)
        
    output_feed.append(hidden_feed)
    output_weights.apped(hidden_weight)
    
    for j in range(hidden_layer -1) :
            inputs.clear()
            inputs.append(hidden_feed)
            
            hidden_feed.clear()
            hidden_weights.clear()
            
            for j in range(nb_nodes) :
                output = single_node_process(inputs, weights[i+1][j])
                hidden_weight.append(output)
                active = activate(output)
                hidden_feed.append(active)
            
            output_feed.append(hidden_feed)
            output_weights.apped(hidden_weight)  
        
    inputs.clear()
    inputs.append(hidden_feed))
    hidden_feed.clear()
    hidden_weight.clear()
    output = single_node_process(inputs, weights[1][0])
    hidden_weight.append(output)
    active = activate(output)
    hidden_feed.append(active)
    output_feed.append(hidden_feed)
    output_weights.apped(hidden_weight)  
    
    return active, output_feed, output_weights

        def initilialize_delta(weights):
        delta=[]
        for i in weights:
        temp=[]
        for j in i:
        temp_delta=[]
        for k in j:
            temp_delta.append(0.0)
        temp.append(temp_delta)
        delta.append(temp)

    return delta


