#!/usr/bin/env/ python
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

##custom written modules
import data_processing as get
import tf_functions as tfun
import learner_functions as learn
from conv_models import ModularInception
from my_models import RegularizedDense 
from literature_models import *
from timers import tic, toc
from tf_save_load import Saver
from copy import copy
from data_loaders import MixedLoader



#################################################################################
## This script trains the hybrid neural network by being given image data
## and feature data. 
## Inside this script, multiple objects/modules from the other repo are used:
## https://github.com/J-lissner/python_scripts/tree/main/hdf5
## it is recommended to put all of the scripts inside this repo into your
## $PYTHONPATH
## The code is separated into semantic blocks and documented and can be 
## adjusted to the users preferences/data.
#### Specific to the problem presented in the paper ####
## If the 'phase_contrast' variable is adjusted to have a higher length than
## 1, then the variable phase contrast is automatically treated and the extra
## neuron inserted. 
#################################################################################

os.environ["CUDA_VISIBLE_DEVICES"]="0"
### Data and optimizer/training parameters
n_NN             = 3 #how many NN do we want 
## data loader 
problem          = 'heat'
phase_contrast   = [5] 
output_scaletype = None
## learning rate object and training duration
lr_object        = learn.RemoteLR
pretrain_vol     = True
base_lr          = 1e-3
weight_decay     = 5e-4
base_delay       = 100
epochs           = 10000
feedback_interval= 15 #console printout interval
## data batching and trainin duration
n_data           = 2000# 3000 #note that this is the sum of "n" training AND validation data
idx_offset       = 6*1500 #for the data loader
validation_split = 0.20 #20 % for validation
input_scaletype  = 'single_std1'
roll_interval    = 10 #after how many images half of the images should be rolled or translated
roll_part        = 0.5 #percentage of rolled images
batchsize        = 20
## only for variable phase contrast
extra_scaletype  = '0,1' #for variable phase contrast
extra_idx        = 1 #for variable phase contrast, which column contains paramter
get_extra        = lambda x: x[:, extra_idx][:,None]
## model parameters
basename         = 'trained_models/deep_inception'
model_name       = 'ModularInception'
model_kwargs     = dict() 
## other things required for saving/loading
data_path      = '/scratch/lissner/dataverse/2d_mech+heat/' #'/home/lissner/dataverse/' #
model_code     = 'conv_models'#'literature_models' #'conv_models' #'my_models' #
auxiliary_code = [ 'data_processing', 'tf_functions', 'train_hybrid.py', 'literature_layers.py', 'my_layers' ]
#### Automatically computed preallocation
## load data
## data loader: only single contrast but also mechanical
if len(phase_contrast) == 1:
    datafile                 = data_path + '2d_microstructures_results.h5'
    data_loader              = MixedLoader( problems=problem, features=True, targets=True, images=True, data_file=datafile) 
    features, target, images = data_loader[idx_offset:idx_offset+n_data//2]
    data_loader.close()
images   = images.reshape( -1, 400, 400, 1).astype( np.float32) 
## variable phase contrast
n_vol             = min(2, len(phase_contrast) ) #2 features for variable phase contrast
variable_contrast = len(phase_contrast) > 1

### Train multiple ANNs
for j in range( n_NN): 
    #### DATA PRE PROCESSING
    # note: the first column of features is the volume fraction
    x_train, y_train, img_train, x_valid, y_valid, img_valid = get.split_data( features, target, images, split=validation_split )
    x_train, x_valid, input_scaling                          = get.scale_data( x_train, x_valid, scaletype=input_scaletype)
    y_train, y_valid, output_scaling                         = get.scale_data( y_train, y_valid, scaletype=output_scaletype)
    if variable_contrast:
        x_extra, x_vextra, extra_scaling = get.scale_data( get_extra(x_train), get_extra( x_valid), scaletype=extra_scaletype)
        x_train[:,extra_idx]             = x_extra.squeeze()
        x_valid[:,extra_idx]             = x_vextra.squeeze()
    x_train, x_valid, y_train, y_valid, img_train = tfun.to_float32( x_train, x_valid, y_train, y_valid, img_train, arraytype='tf' )
    img_valid                                     = np.concatenate( x_valid.shape[0]//img_valid.shape[0] *[img_valid], axis=0 ).astype( np.float32)
    img_train                                     = tf.Variable( img_train)

    
    ## reset learning rate object
    learning_rate     = lr_object( base_lr=base_lr) #1e-3 #learn.RemoteLR #0.005
    stopping_delay    = base_delay
    plateau_threshold = 0.95 if learn.is_slashable( learning_rate) else 1
    ## Model invocation and storing of constant parameters/code
    savepath     = f'{basename}_nr_{j}'
    model_args   = [y_train.shape[1]]
    model_kwargs = dict( n_vol=n_vol, **model_kwargs)
    ANN = eval( model_name)( *model_args, **model_kwargs)
    try: ANN.enable_vol( pretrain_vol)
    except: print( 'careful, was not able to enable the vol bypass' )
    ANN( img_valid[:2], x_valid[:2], training=False ) #model allocation required for 'pretrain_section'
    ## save the model
    save = Saver( savepath, model_code, model_name )
    save.inputs( *model_args, **model_kwargs)
    save.code( *auxiliary_code) 
    if n_vol > 1:
        save.scaling( input_scaling, output_scaling, extra_scaling=extra_scaling, extra_idx=extra_idx)
    else:
        save.scaling( input_scaling, output_scaling )
    ### Adjustable parameters for ANN optimization (has to be inside the loop)
    loss_metric   = tf.keras.losses.MeanSquaredError() #validate with
    cost_function = tf.function( tfun.relative_mse ) if variable_contrast else tf.keras.losses.MeanSquaredError()#optimize with
    if learn.is_slashable( learning_rate):
        optimizer = tfa.optimizers.AdamW( learning_rate=learning_rate, weight_decay=weight_decay, beta_1=0.8, beta_2=0.85)
        learning_rate.reference_optimizer( optimizer)
        learning_rate.reference_model( ANN)
    else:
        optimizer = tfa.optimizers.AdamW( weight_decay=weight_decay, learning_rate=learning_rate)
    checkpoints        = tf.train.Checkpoint( model=ANN, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager( checkpoints, savepath, max_to_keep=3 )

    ###### Model pretraining of selected layers #####
    tic( 'one entire training', silent=True )
    print( f'Optimizing a total of {len( ANN.trainable_variables)} layers\
        with {sum([np.prod( x.shape) for x in ANN.trainable_variables])} parameters' )
    if pretrain_vol:
        tic( 'pretraining vol', silent=True)
        ANN.freeze_all()
        ANN.freeze_vol( False)
        ANN.pretrain_section( [x_train, y_train], [x_valid, y_valid ],
                              ANN.predict_vol, n_epochs=150, batchsize=500 )
        ANN.freeze_all( False)
        ANN.freeze_vol()
        toc( 'pretraining vol')


    ## variable allocation before training loop
    decline    = 0 #tracked variable for early stop
    best_epoch = 0
    best_loss  = 1e5
    valid_loss = []
    train_loss = [] 
    tic('training model')
    tic('trained for {} epochs'.format( feedback_interval), True)
    ### BODY OF TRAINING AND VARIABLE TRACKING
    print( f'########################## starting training for {basename} ##########################')
    for i in range( epochs):
        if (i+1) % roll_interval == 0:
            tfun.roll_images( img_train, roll_part)  #translate the images
        batch_loss = []
        for x_batch, y_batch, image_batch in tfun.batch_data( batchsize, [ x_train, y_train, img_train]):
            with tf.GradientTape() as tape:
                y_pred   = ANN( image_batch, x_batch, training=True)
                gradient = cost_function( y_batch, y_pred ) 
            gradient = tape.gradient( gradient, ANN.trainable_variables)
            optimizer.apply_gradients( zip(gradient, ANN.trainable_variables) )
            batch_loss.append( loss_metric( y_batch, y_pred) ) 

        # epoch post processing
        y_pred       = ANN.batched_prediction( batchsize, img_valid, x_valid )
        current_loss = np.mean( cost_function( y_pred, y_valid) )
        valid_loss.append( np.mean( loss_metric( y_pred, y_valid) ))
        train_loss.append( np.mean( batch_loss )  )
        if current_loss < best_loss: 
            best_loss  = plateau_threshold * current_loss
            best_epoch = i
            decline    = 0
            checkpoint_manager.save() 
        decline += 1
        if (i+1) % feedback_interval == 0:
            toc( f'trained for {feedback_interval} epochs', auxiliary=f', {i} total epochs', precision=1) 
            print( f'current train loss:   \t{train_loss[-1]:.2e}   vs best: {train_loss[best_epoch]:.3e}' )
            print( f'vs current valid loss:\t{valid_loss[-1]:.2e}   vs best: {valid_loss[best_epoch]:.3e}' )
            tic('trained for {} epochs'.format( feedback_interval), silent=True)
        if decline == stopping_delay:
            if learn.is_slashable( learning_rate) and not learning_rate.allow_stopping:
                learning_rate.slash()
                decline = 0
                if learning_rate.allow_stopping:
                    best_loss         = best_loss/plateau_threshold
                    plateau_threshold = 1
                    stopping_delay    = 5
            else:
                break

    ### POST PROCESSING AND SAVING
    ## recover the checkpoint (is automatically the best ANN, even if it improves till the last epoch))
    checkpoints.restore(checkpoint_manager.latest_checkpoint)
    last_pred   = ANN.batched_prediction( batchsize, img_valid, x_valid)
    true_loss   = get.unscale_data( last_pred, output_scaling)
    true_relmse = 100*tfun.relative_mse( get.unscale_data( y_valid, output_scaling), true_loss )  #requires temporary variable
    rel_mse     = 100*tfun.relative_mse( y_valid, last_pred )
    true_loss   = np.mean( loss_metric( get.unscale_data( y_valid, output_scaling),  true_loss ) )
    toc('training model')
    toc( 'one entire training' )
    print( 'saving model to"{}"'.format( savepath ) )
    print( f"""######################### training finished #########################,
           trained for {i} epochs, best ANN at {best_epoch} epochs, (true scale)
           best valid loss:            {np.mean(valid_loss[best_epoch]):1.4e}, ({true_loss:1.4e}),
           corresponding train loss:   {np.mean(train_loss[best_epoch]):1.4e},
           translates to rel root mse: {rel_mse:1.3f}, ({true_relmse:1.3f}) [\%],
           n_params:                   {sum([np.prod( x.shape) for x in ANN.trainable_variables])}
#####################################################################
           """ )
    ## Save the modified tensroflow objects
    save.model( ANN)
    save.tracked_variables( train_loss=train_loss, valid_loss=valid_loss, 
                            best_epoch=best_epoch, epochs=i, stopping_delay=stopping_delay)
    save.locals( optimizer=optimizer, cost_function=cost_function, learning_rate=learning_rate )
    del ANN, checkpoints, checkpoint_manager, train_loss, valid_loss, cost_function, optimizer, learning_rate

