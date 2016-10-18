'''
Recurrent network example.  Trains a 2 layered LSTM network to learn
XOR function. The network can then be used to generate sequence using a short binary sequence as seed.
'''

from __future__ import print_function


import numpy as np
import theano
import theano.tensor as T
import lasagne


class ZeroLayer(lasagne.layers.Layer):
    def get_output_for(self, input = None):
        return 0*input
    '''
    def get_output_shape_for(self, input_shape):
        return [1,1]
    '''




generation_phrase = [0,1,0,0,1,1,0,1] #This phrase will be used as seed to generate text.

# as we generate the date on each take data_size gives a value for an epoch size
data_size = 10000


#Lasagne Seed for Reproducibility
lasagne.random.set_rng(np.random.RandomState(1))

# Sequence Length
SEQ_LENGTH = 8

# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 512

# Optimization learning rate
LEARNING_RATE = .005

# All gradients above this will be clipped
GRAD_CLIP = 100

# How often should we check the output?
PRINT_FREQ = 20 #1000

# Number of epochs to train the net
NUM_EPOCHS = 50

# Batch Size
BATCH_SIZE = 128

#Number of steps 2-50
NUM_STEPS = 2


def gen_xor_data(word_length=SEQ_LENGTH, batch_size = BATCH_SIZE):
    x_list = np.random.randint(2, size=(batch_size,word_length))
    x = np.array(x_list,dtype='int32')
    y = np.zeros(batch_size)  

    y = x_list.sum(axis=1) % 2

    return x, np.array(y,dtype='int32')
 

def gen_data(seq_length = SEQ_LENGTH, batch_size = BATCH_SIZE):
    return gen_xor_data(seq_length, batch_size)

def encode_phrase(input_phrase):
    phrase = np.zeros((len(input_phrase), 2),dtype='int32')
    for xi in range(len(input_phrase)):
        phrase[xi] [input_phrase[xi]] = 1
    return phrase




def main(num_epochs=NUM_EPOCHS):
    print("Building network ...")
   
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, SEQ_LENGTH, num_features)

    # l_in = lasagne.layers.InputLayer(shape=(1,len(generation_phrase),2))
    l_in = lasagne.layers.InputLayer(shape=(None,SEQ_LENGTH))
    l_in_zero = lasagne.layers.InputLayer(shape=(None, NUM_STEPS, 1))

    l_lin = lasagne.layers.DenseLayer(l_in, num_units = N_HIDDEN, nonlinearity = None)

    # We now build the LSTM layer which takes l_in as the input layer
    # We clip the gradients at GRAD_CLIP to prevent the problem of exploding gradients. 

    """

    l_forward_1 = lasagne.layers.LSTMLayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_forward_2 = lasagne.layers.LSTMLayer(
        l_forward_1, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh,
        only_return_final=True)
    """



    l_forward = lasagne.layers.LSTMLayer(
       l_in_zero, N_HIDDEN, 
        nonlinearity=lasagne.nonlinearities.tanh, hid_init = l_lin, only_return_final=True)

    

    # The output of l_forward_2 of shape (batch_size, N_HIDDEN) is then passed through the softmax nonlinearity to 
    # create probability distribution of the prediction
    # The output of this stage is (batch_size, vocab_size)
    l_lin_out = lasagne.layers.DenseLayer(l_forward, num_units = 2, nonlinearity = None)
    l_out = lasagne.layers.DenseLayer(l_lin_out, num_units=2, nonlinearity=lasagne.nonlinearities.softmax)

    # Theano tensor for the targets
    target_values = T.ivector('target_output')
    
    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(l_out)

    # The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.
    cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out,trainable=True)

    # Compute AdaGrad updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, l_in_zero.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
    compute_cost = theano.function([l_in.input_var, l_in_zero.input_var, target_values], cost, allow_input_downcast=True)

    # In order to generate text from the network, we need the probability distribution of the next character given
    # the state of the network and the input (a seed).
    # In order to produce the probability distribution of the prediction, we compile a function called probs. 
    
    probs = theano.function([l_in.input_var, l_in_zero.input_var],network_output,allow_input_downcast=True)

    # The next function generates text given a phrase of length at least SEQ_LENGTH.
    # The phrase is set using the variable generation_phrase.
    # The optional input "N" is used to set the number of characters of text to predict. 

    def try_it_out(N=50):
        '''
        This function uses the user-provided string "generation_phrase" and current state of the RNN generate text.
        The function works in three steps:
        1. It converts the string set in "generation_phrase" (which must be over SEQ_LENGTH characters long) 
           to encoded format. We use the gen_data function for this. By providing the string and asking for a single batch,
           we are converting the first SEQ_LENGTH characters into encoded form. 
        2. We then use the LSTM to predict the next character and store it in a (dynamic) list sample_ix. This is done by using the 'probs'
           function which was compiled above. Simply put, given the output, we compute the probabilities of the target and pick the one 
           with the highest predicted probability. 
        3. Once this character has been predicted, we construct a new sequence using all but first characters of the 
           provided string and the predicted character. This sequence is then used to generate yet another character.
           This process continues for "N" characters. 
        To make this clear, let us again look at a concrete example. 
        Assume that SEQ_LENGTH = 5 and generation_phrase = "The quick brown fox jumps". 
        We initially encode the first 5 characters ('T','h','e',' ','q'). The next character is then predicted (as explained in step 2). 
        Assume that this character was 'J'. We then construct a new sequence using the last 4 (=SEQ_LENGTH-1) characters of the previous
        sequence ('h','e',' ','q') , and the predicted letter 'J'. This new sequence is then used to compute the next character and 
        the process continues.
        '''

        assert(len(generation_phrase)>=SEQ_LENGTH)
        sample_ix = []
        x = np.zeros((1,len(generation_phrase)),dtype='int32')
        x [0,:] = generation_phrase
        x_zero = np.zeros((len(x),NUM_STEPS,1),dtype='int32')

        for i in range(N):
            # Pick the character that got assigned the highest probability
            ix = np.argmax(probs(x,x_zero).ravel())
            # Alternatively, to sample from the distribution instead:
            # ix = np.random.choice(np.arange(vocab_size), p=probs(x).ravel())
            sample_ix.append(ix)
            x[0,:SEQ_LENGTH - 1] = x[0,1:]
            x[0,SEQ_LENGTH - 1] = ix

        random_snippet = generation_phrase + sample_ix
        print("----\n %s \n----" % random_snippet)


    
    print("Training ...")
    print("Seed used for text generation is: " + ''.join(str(char) for char in generation_phrase))
    p = 0
    try:
        for it in xrange(data_size * num_epochs / BATCH_SIZE):
            try_it_out() 
            
            avg_cost = 0;
            for _ in range(PRINT_FREQ):
                x,y = gen_data()
                x_zero = np.zeros((BATCH_SIZE,NUM_STEPS,1),dtype='int32')

                avg_cost += train(x, x_zero, y)
            print("Epoch {} average loss = {}".format(it*1.0*PRINT_FREQ/data_size*BATCH_SIZE, avg_cost / PRINT_FREQ))
                    
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()