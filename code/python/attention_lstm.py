import keras.backend as K
from keras.layers.recurrent import LSTM
from keras.engine import InputSpec
from keras import activations, initializations, regularizers
from keras.layers.recurrent import time_distributed_dense
import numpy as np

class AttentionLSTM(LSTM):

    def step(self, x, states):
        prev_h1 = states[0]
        prev_c1 = states[1]
        proj_z = states[2]
        #alphaz = states[3]
        B_U = states[3]
        B_W = states[4]
        B_Z = states[5]

        proj_state = K.dot(prev_h1, self.Wd_att)
        proj_z = proj_z + proj_state[:, None, :]
        proj_list = []
        proj_list.append(proj_z)
        proj_z = K.tanh(proj_z)

        alpha = K.dot(proj_z, self.U_att ) + self.b2_att
        alpha_shape = alpha.shape
        alpha = K.softmax(alpha.reshape((alpha_shape[0], alpha_shape[1])))
        
        #alphaz = alpha
        #self.alphaz = alpha
        
        z = (self.initial_z * alpha[:, :, None]).sum(1)
        #print(z)
        #print(z.shape)

        x_i = x[:, :self.output_dim]
        x_f = x[:, self.output_dim: 2 * self.output_dim]
        x_c = x[:, 2 * self.output_dim: 3 * self.output_dim]
        x_o = x[:, 3 * self.output_dim:]
        '''
        x_i = K.dot(x * B_W[0], self.W_i) + self.b_i
        x_f = K.dot(x * B_W[1], self.W_f) + self.b_f
        x_c = K.dot(x * B_W[2], self.W_c) + self.b_c
        x_o = K.dot(x * B_W[3], self.W_o) + self.b_o'''
        
        h_i = K.dot(prev_h1 * B_U[0], self.U_i) + self.c_i
        h_f = K.dot(prev_h1 * B_U[1], self.U_f) + self.c_f
        h_c = K.dot(prev_h1 * B_U[2], self.U_c) + self.c_c
        h_o = K.dot(prev_h1 * B_U[3], self.U_o) + self.c_o
        
        z_i = K.dot(z * B_Z[0], self.Z_i) + self.d_i
        z_f = K.dot(z * B_Z[1], self.Z_f) + self.d_f
        z_c = K.dot(z * B_Z[2], self.Z_c) + self.d_c
        z_o = K.dot(z * B_Z[3], self.Z_o) + self.d_o
        
        i = self.inner_activation(x_i + h_i + z_i)
        f = self.inner_activation(x_f + h_f + z_f)
        c = f * prev_c1 + i * self.activation(x_c + h_c + z_c)
        o = self.inner_activation(x_o + h_o + z_o)

        h = o * self.activation(c)
        
        output = None
        if self.ret_alpha:
            output = alpha
        else:
            output = h
        
        return output, [h, c, proj_z]
    
    def get_proj_z(self):
        return K.dot(self.initial_z, self.Wc_att) + self.b_att if self.initial_z is not None else None

    def __init__(self, output_dim, z_dim, ret_alpha=False, initial_h=None, initial_c=None, initial_z=None ,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, Z_regularizer=None, b_regularizer=None, c_regularizer=None, d_regularizer=None,
                 dropout_W=0., dropout_U=0., dropout_Z=0, **kwargs):
        self.output_dim = output_dim
        self.initial_h = initial_h
        self.initial_c = initial_c
        self.initial_z = initial_z
        self.ret_alpha = ret_alpha
        self.z_dim = z_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.Z_regularizer = regularizers.get(Z_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.c_regularizer = regularizers.get(c_regularizer)
        self.d_regularizer = regularizers.get(d_regularizer)
        self.dropout_W, self.dropout_U, self.dropout_Z = dropout_W, dropout_U, dropout_Z

        if self.dropout_W or self.dropout_U or self.dropout_Z:
            self.uses_learning_phase = True
        super(AttentionLSTM, self).__init__(output_dim, **kwargs)

    def build(self, input_shape):
        assert type(input_shape) is list  # must have mutiple input shape tuples
        input_shape_list = input_shape
        input_shape = input_shape[0]
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]

        self.Wc_att = self.init((self.z_dim, self.z_dim),
                                 name='{}_Wc_att'.format(self.name))
        self.Wd_att = self.init((self.output_dim, self.z_dim),
                                 name='{}_Wd_att'.format(self.name))
        self.U_att = self.init((self.z_dim, 1),
                                 name='{}_U_att'.format(self.name))

        self.b_att = K.zeros((self.z_dim,), name='{}_b_att'.format(self.name))
        self.b2_att = K.zeros((1,), name='{}_b2_att'.format(self.name))

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 3 all-zero tensors of shape (output_dim)
            proj_z = self.get_proj_z()
            self.states = [self.initial_h, self.initial_z, proj_z]
        
        '''self.Wc_att = self.init((self.z_dim, self.z_dim),
                                 name='{}_Wc_att'.format(self.name))
        self.Wd_att = self.init((self.output_dim, self.z_dim),
                                 name='{}_Wd_att'.format(self.name))
        self.U_att = self.init((self.z_dim, 1),
                                 name='{}_U_att'.format(self.name))

        self.b_att = K.zeros((self.z_dim,), name='{}_b_att'.format(self.name))
        self.b2_att = K.zeros((1,), name='{}_b2_att'.format(self.name))'''

        self.W_i = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_i'.format(self.name))
        self.U_i = self.inner_init((self.output_dim, self.output_dim),
                                    name='{}_U_i'.format(self.name))
        self.Z_i = self.inner_init((self.z_dim, self.output_dim),
                                    name='{}_Z_i'.format(self.name))
        self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))
        self.c_i = K.zeros((self.output_dim,), name='{}_c_i'.format(self.name))
        self.d_i = K.zeros((self.output_dim,), name='{}_d_i'.format(self.name))

        self.W_f = self.init((self.input_dim, self.output_dim),
                                name='{}_W_f'.format(self.name))
        self.U_f = self.inner_init((self.output_dim, self.output_dim),
                                    name='{}_U_f'.format(self.name))
        self.Z_f = self.inner_init((self.z_dim, self.output_dim),
                                    name='{}_Z_f'.format(self.name))
        self.b_f = self.forget_bias_init((self.output_dim,),
                                            name='{}_b_f'.format(self.name))
        self.c_f = self.forget_bias_init((self.output_dim,),
                                            name='{}_c_f'.format(self.name))
        self.d_f = self.forget_bias_init((self.output_dim,),
                                            name='{}_d_f'.format(self.name))

        self.W_c = self.init((self.input_dim, self.output_dim),
                                name='{}_W_c'.format(self.name))
        self.U_c = self.inner_init((self.output_dim, self.output_dim),
                                    name='{}_U_c'.format(self.name))
        self.Z_c = self.inner_init((self.z_dim, self.output_dim),
                                    name='{}_Z_c'.format(self.name))
        self.b_c = K.zeros((self.output_dim,), name='{}_b_c'.format(self.name))
        self.c_c = K.zeros((self.output_dim,), name='{}_c_c'.format(self.name))
        self.d_c = K.zeros((self.output_dim,), name='{}_d_c'.format(self.name))

        self.W_o = self.init((self.input_dim, self.output_dim),
                                name='{}_W_o'.format(self.name))
        self.U_o = self.inner_init((self.output_dim, self.output_dim),
                                    name='{}_U_o'.format(self.name))
        self.Z_o = self.inner_init((self.z_dim, self.output_dim),
                                    name='{}_Z_o'.format(self.name))
        self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))
        self.c_o = K.zeros((self.output_dim,), name='{}_c_o'.format(self.name))
        self.d_o = K.zeros((self.output_dim,), name='{}_d_o'.format(self.name))

        self.trainable_weights = [self.Wc_att, self.Wd_att, self.U_att, self.b_att, self.b2_att, 
                                      self.W_i, self.U_i, self.Z_i, self.b_i, self.c_i, self.d_i,
                                      self.W_c, self.U_c, self.Z_c, self.b_c, self.c_c, self.d_c,
                                      self.W_f, self.U_f, self.Z_f, self.b_f, self.c_f, self.d_f,
                                      self.W_o, self.U_o, self.Z_o, self.b_o, self.c_o, self.d_o]

        self.W = K.concatenate([self.W_i, self.W_f, self.W_c, self.W_o])
        self.U = K.concatenate([self.U_i, self.U_f, self.U_c, self.U_o])
        self.Z = K.concatenate([self.Z_i, self.Z_f, self.Z_c, self.Z_o])
        self.b = K.concatenate([self.b_i, self.b_f, self.b_c, self.b_o])
        self.c = K.concatenate([self.c_i, self.d_f, self.c_c, self.c_o])
        self.d = K.concatenate([self.d_i, self.d_f, self.d_c, self.d_o])

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.Z_regularizer:
            self.Z_regularizer.set_param(self.Z)
            self.regularizers.append(self.Z_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)
        if self.c_regularizer:
            self.c_regularizer.set_param(self.c)
            self.regularizers.append(self.c_regularizer)
        if self.d_regularizer:
            self.d_regularizer.set_param(self.d)
            self.regularizers.append(self.d_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            #K.set_value(self.states[0],
            #            np.zeros((input_shape[0], self.output_dim)))
            #K.set_value(self.states[1],
            #            np.zeros((input_shape[0], self.output_dim)))
            if self.states[0] is not None:
                K.set_value(self.states[0],
                        self.initial_h)
            else:
                self.states[0] = self.initial_h 
           
            if self.states[1] is not None:
                K.set_value(self.states[1],
                        self.initial_c)
            else:
                self.states[1] = self.initial_c 

            self.states[2] = self.get_proj_z() 
            
            #K.set_value(self.states[3], np.zeros((input_shape[0], 196)))
            '''K.set_value(self.states[1],
                        self.initial_c)
            K.set_value(self.states[2],
                        self.get_proj_z())'''
        else:
            self.states = [self.initial_h,
                           self.initial_c,
                           self.get_proj_z()]
    
    def get_initial_states(self, x):
        input_shape = self.input_spec[0].shape
        if hasattr(self, 'states'):
            #K.set_value(self.states[0],
            #            np.zeros((input_shape[0], self.output_dim)))
            #K.set_value(self.states[1],
            #            np.zeros((input_shape[0], self.output_dim)))
            if self.states[0] is not None:
                K.set_value(self.states[0],
                        self.initial_h)
            else:
                self.states[0] = self.initial_h 
           
            if self.states[1] is not None:
                K.set_value(self.states[1],
                        self.initial_c)
            else:
                self.states[1] = self.initial_c 

            self.states[2] =self.get_proj_z() 

            #K.set_value(self.states[3], np.zeros((1, 196)))
        else:
            self.states = [self.initial_h,
                           self.initial_c,
                           self.get_proj_z()]
        return self.states
    
    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.output_dim))
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(4)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])
        if 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(4)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])
        
        if 0 < self.dropout_Z < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.z_dim))
            B_Z = [K.in_train_phase(K.dropout(ones, self.dropout_Z), ones) for _ in range(4)]
            constants.append(B_Z)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        return constants
    
    def preprocess_input(self, x):
            #x = input
            #print(x)
            #return x
            
            if 0 < self.dropout_W < 1:
                dropout = self.dropout_W
            else:
                dropout = 0

            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2]
            timesteps = input_shape[1]

            x_i = time_distributed_dense(x, self.W_i, self.b_i, dropout,
                                         input_dim, self.output_dim, timesteps)
            x_f = time_distributed_dense(x, self.W_f, self.b_f, dropout,
                                         input_dim, self.output_dim, timesteps)
            x_c = time_distributed_dense(x, self.W_c, self.b_c, dropout,
                                         input_dim, self.output_dim, timesteps)
            x_o = time_distributed_dense(x, self.W_o, self.b_o, dropout,
                                         input_dim, self.output_dim, timesteps)
            return K.concatenate([x_i, x_f, x_c, x_o], axis=2)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                   'initial_h': self.initial_h,
                   'initial_c': self.initial_c,
                  'initial_z': self.initial_z,
                  'z_dim': self.z_dim,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'forget_bias_init': self.forget_bias_init.__name__,
                  'activation': self.activation.__name__,
                  'inner_activation': self.inner_activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'Z_regularizer': self.Z_regularizer.get_config() if self.Z_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'c_regularizer': self.C_regularizer.get_config() if self.c_regularizer else None,
                  'd_regularizer': self.d_regularizer.get_config() if self.d_regularizer else None,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U,
                  'dropout_Z': self.dropout_Z}
        base_config = super(AttentionLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, mask=None):
        if type(inputs) is not list or len(inputs) <= 1:
            raise Exception('Merge must be called on a list of tensors '
                            '(at least 2). Got: ' + str(inputs))
        self.initial_h = inputs[1]
        self.initial_c = inputs[2]
        self.initial_z = inputs[3]

        self.states[2] = self.get_proj_z()
        x = inputs[0]
        '''print(type(x))
        print(x.shape)
        print(x)
        print(self.input_spec[0].shape)
        print(self.input_spec[0].ndim)
        print(self.stateful)'''
        
        result  = super(AttentionLSTM, self).call(x)
        #save alphaz
        #print('Saving Alphaz')
        return result
    
    def get_output_shape_for(self, input_shape):
        assert type(input_shape) is list  # must have mutiple input shape tuples
        print("input shapes %s" %input_shape)
        x_shape = input_shape[0]
        
        output_size = 0
        if self.ret_alpha:
            output_size = 196
        else:
            output_size = self.output_dim

        if self.return_sequences:
            return (x_shape[0], x_shape[1], self.output_dim)
        else:
            return (x_shape[0], self.output_dim)
    
    def compute_mask(self, input, mask):
        if self.return_sequences:
            return mask[0]
        else:
            return None
        