import os 
import pickle 

from tensorflow.keras import Model
from tensorflow.keras import backend as K 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.layers import Input, conv2d,ReLU,BatchNormalization,Flatten,Dense,Reshape,Conv2DTranspose,Activation,Lambda
from tensorflow.keras.losses import MeanSquaredError
import numpy as np 
import tensorflow as tf 

tf.compat.disable_eager_execution()



class VAE:


    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):


        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim 
        self.reconstrucion_loss_weight = 1000

        self.encoder = None
        self.decoder = None
        self.model = None 

        self.num_conv_layers = len(conv_filters)
        self.shape_before_bottleneck = None 
        self._model_input = None 


        self._build()


    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()


    def compile(self,learning_rate = 0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer = optimizer,
        loss = self._calculate_combined_loss,
        metrics = [self._calculate_reconstruction_loss,self._calculate_kl_loss]
        
        )



    def train(self,x_train,batch_size,num_epochs):
        self.model.fit(x_train,x_train,batch_size=batch_size,epochs=num_epochs,shuffle=True)




    def save(self,save_folder = "./"):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self.save_parameters(save_folder)
        self._save_weights(save_folder)



    def load_weights(self,weights_path):
        self.model.load_weights(weights_path)



    #def load_weights


    def reconstruct(self,images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images,latent_representations

@classmethod
def load(cls,save_folder = './'):
    parameters_path = os.path.join(save_folder,"parameters.pkl")
    with open(parameters_path,"rb") as f:
        parameters = pickle.load(f)
    autoencoder = VAE(*parameters)
    weights_path = os.path.join(save_folder,"weights.h5")
    autoencoder.load_weights(weights_path)
    return autoencoder



def _add_bottleneck(self,X):
    self._shape_before_bottleneck = k.int_shape(x)[1:]
    x = Flatten()(X)
    self.mu = Dense(self.latent_space_dim,name="mu")(X)
    self.log_variance = Dense(self.lent_space_dim,name = "log_variance")(X)


    def sample_point_from_normal_distribution(args):
        mu,log_variance = args

        epsilon = K.random_normal(shape = K.shape(self.mu),mean=0.,stddev=1.)

        sampled_point = mu + K.exp(log_variance/2) * epsilon
        return sampled_point

    X = Lambda(sample_point_from_normal_distribution,name="encoder_output")([self.mu,self.log_variance])


    return X





    #@classmethod    

def _calculate_reconstruction_loss(self,y_target,y_predicted):
    error = y_target - y_predicted

    reconstruction_loss = K.mean(K.square(error),axis = [1,2,3])
    return reconstruction_loss



def _calculate_kl_loss(self,y_target,y_predicted):
    # difference between 2 distributions 
    kl_loss = -0.5 * K.sum(1+self.log_variance - k.square(self.mu) - K.exp(self.log_variance),axis=1)
    return kl_loss


def calculate_combined_loss(self,y_target,y_predicted):
    reconstruction_loss = self._calculate_reconstruction_loss(y_target,y_predicted)
    kl_loss = self._calculate_kl_loss(y_target,y_predicted)
    combined_loss = self.reconstruction_loss_weight * reconstruction_loss + kl_loss
    return combined_loss


def _save_parameters(self,save_folder):
    parameters = [
        self.input_shape,
        self.conv_filters,
        self.conv_kernels,
        self.conv_strides,
        self.latent_space_dim
    ]
    save_path = os.path.join(save_folder,"parameters.pkl")
    with open(save_path,"wb") as f:
        pickle.dump(parameters,f)


def _save_weights(self,save_folder):
    save_path = os.path.join(save_folder,"weights.h5")
    self.model.save_weights(save_path)


def _build(self):
    self._build_encoder()
    self._build_decoder()
    self._build_autoencoder()


def _build_autoencoder(self):
    model_input = self._model_input
    model_output = self.decoder(self.encoder(model_input))
    self.model = Model(model_input,model_output,name = "autoencoder")


def _build_decoder(self):
    decoder_input = self._add_decoder_input()
    dense_layer = self._add_dense_layer(decoder_input)
    reshape_layer = self._add_reshape_layer(dense_layer)
    conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
    decoder_output = self._add_decoder_output(conv_transpose_layers)
    self.decoder = Model(decoder_input, decoder_output, name="decoder")






        

        











