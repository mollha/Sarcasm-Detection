from keras.layers.core import*
from keras.models import Sequential
from keras.layers import Dense, Lambda, dot, Activation, concatenate


def attention_3d_block(hidden_states):
    """
    Many-to-one attention mechanism for Keras.
    @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
    @return: 2D tensor with shape (batch_size, 128)
    """
    hidden_size = int(hidden_states.shape[2])
    # Inside dense layer
    #              hidden_states            dot               W            =>           score_first_part
    # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
    # W is the trainable weight matrix of attention Luong's multiplicative style score

    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
    #            score_first_part           dot        last_hidden_state     => attention_weights
    # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
    score = dot([score_first_part, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('softmax', name='attention_weight')(score)

    # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)

    context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
    pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    return attention_vector



# input_dim = 32
# hidden = 32
#
# #The LSTM  model -  output_shape = (batch, step, hidden)
# model1 = Sequential()
# model1.add(LSTM(input_dim=input_dim, output_dim=hidden, input_length=step, return_sequences=True))
#
# #The weight model  - actual output shape  = (batch, step)
# # after reshape : output_shape = (batch, step,  hidden)
# model2 = Sequential()
# model2.add(Dense(input_dim=input_dim, output_dim=step))
# model2.add(Activation('softmax')) # Learn a probability distribution over each  step.
# #Reshape to match LSTM's output shape, so that we can do element-wise multiplication.
# model2.add(RepeatVector(hidden))
# model2.add(Permute(2, 1))
#
# #The final model which gives the weighted sum:
# model = Sequential()
# model.add(Merge([model1, model2], 'mul'))  # Multiply each element with corresponding weight a[i][j][k] * b[i][j]
# model.add(TimeDistributedMerge('sum')) # Sum the weighted elements.
#
# model.compile(loss='mse', optimizer='sgd')