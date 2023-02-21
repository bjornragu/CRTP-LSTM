# -*- coding: utf-8 -*-


from keras import backend as K
from keras.models import Sequential, Model  
from keras.layers import *
from keras.models import Model#, load_model
from keras.optimizers import Nadam
from keras.utils import Sequence

def get_model(output_dim=25, dropout_rate=0, learning_rate = 0.002, latent_dim=100,
              helper = {}):
    
    print('drop:', dropout_rate)
           
    inp_trace = Input(shape=(output_dim, vocab_size), name='trace') #used to be vocab size +1
    
    inp_timefeat_categorical = Input(shape=(output_dim, 1), name='time_categorical_1')
    inp_timefeat_numberical = Input(shape=(output_dim, 1), name='time_numerical_1')
    inp_casefeat_categorical = Input(shape=(output_dim, 1), name='casefeat_categorical')
    inp_eventfeat_categorical = Input(shape=(output_dim, 1), name='eventfeat_categorical')
            
    emb_timefeat_categorical = Embedding(output_dim = round(1.6 * (helper['time_categorical_1']['size']-1)**0.56), input_dim=helper['time_categorical_1']['size'], name='EmbTime1')(Flatten()(inp_timefeat_categorical))
    emb_casefeat_categorical = Embedding(output_dim =  round(1.6 * (helper['casefeat_categorical']['size']-1)**0.56), input_dim=helper['casefeat_categorical']['size'], name='EmbCF')(Flatten()(inp_casefeat_categorical))
    emb_eventfeat_categorical = Embedding(output_dim =  round(1.6 * (helper['eventfeat_categorical']['size']-1)**0.56), input_dim=helper['eventfeat_categorical']['size'], name='EmbEF')(Flatten()(inp_eventfeat_categorical))
    
    
    merged = concatenate([(inp_trace), 
                          (emb_timefeat_categorical), (inp_timefeat_numberical),
                          (emb_casefeat_categorical), (emb_eventfeat_categorical)], name='concat_input')
        
    
    LSTM1 = Bidirectional(LSTM(latent_dim, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, recurrent_dropout = dropout_rate, dropout = dropout_rate), name='LSTMshared1')
    outputs1 = LSTM1(merged)
    
    b1 = BatchNormalization()(outputs1)
    
    LSTM_trace1 = Bidirectional(LSTM(latent_dim,  implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, recurrent_dropout = dropout_rate, dropout = dropout_rate), name='LSTMtrace1') 
    outputs2_1 = LSTM_trace1(b1)
    
    LSTM_time1 = Bidirectional(LSTM(latent_dim, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, recurrent_dropout = dropout_rate, dropout = dropout_rate), name='LSTMtime1') 
    outputs2_2 = LSTM_time1(b1)
       
    output_trace = TimeDistributed(Dense(vocab_size, kernel_initializer='glorot_uniform', activation='softmax'), name='trace_out')(outputs2_1)
    
    output_time = TimeDistributed(Dense(1, kernel_initializer='he_uniform'), name='time_out_nolr')(outputs2_2)
    output_time = TimeDistributed(PReLU(), name='time_out')(output_time)
        
    model = Model(inputs=[inp_trace, 
                          inp_timefeat_categorical, inp_timefeat_numberical,
                          inp_casefeat_categorical, inp_eventfeat_categorical], 
                  outputs=[output_time, output_trace])
    

    opt = Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
    
    
    model.compile(loss={'trace_out':'categorical_crossentropy', 'time_out':'mae'}, optimizer=opt) 
    return model