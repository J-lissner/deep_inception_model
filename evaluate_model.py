import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append( '../scripts' )

import data_processing as get
import feature_importance as features
import plt_templates as plot
import tf_functions as tfun
import scatter_relations
from data_loaders import MixedLoader
from precompute_data import assemble_data
from tf_save_load import Loader
from palette import UniStuttgart as uniS
from timers import tic, toc
from error_processing import binned_errors, r_squared_plot

#tf.config.set_visible_devices([], 'GPU')
plt.rcParams.update( plot.rc_default())
## the idea right now is to load in multiple models and the respective validation data
# and then compute the prediction for the MRE (on true scale) as well as the magnitude of the std
# goal is a lowest possible MRE and the lowest possible std


## temp put all best dense models, non required models remaining
#mech_dense_xi_features_lrschedule_nr_3 
#mech_dense_xi+band_features_lrschedule_nr_2
#mech_dense_all_features_lrschedule_nr_4

#heat_dense_xi_features_lrschedule_nr_0
#heat_dense_xi+band_features_lrschedule_nr_4

problem = 'heat'

## heat models
models = ['deep_inception_heat' ] #can compare mulitple models
## other data related stuff
data_type = 'new'
dset_offset = 3000
mse_precision = 10**2 if problem == 'mech' else 10**6
models = [ x.format( problem) for x in models ]
## other stuff
quantiles =  [0.99, 0.95, 0.90]
phase_contrast = [5]#'all' #

error_lines  = None
save_dir     = 'figures/paper_plots/' #'data_investigation/'
plot_scatter = False#True #True #'png' #False#'pdf'
plot_r_square = True #'pdf' #True #False #'png' #
save_plots = False

## input processing / typo catching
shuffle = False

n_samples = 3000

row = -1
data_loader = MixedLoader( problems=problem, fts=True, target=True, img=True)
testset_offset = 1500
input_raw, target, images = data_loader[testset_offset:testset_offset+n_samples//2]
for ann in models:
    row += 1
    print()
    save_path    = save_dir + ann 
    #model loading
    ANN, shifts = Loader( ann)()
    input_shift  = shifts['input']
    output_shift = shifts['output']


    inputs = input_raw[:, :len( input_shift[0])] if input_raw.shape[1] > len( input_shift[0] ) else input_raw
    inputs = get.scale_with_shifts( inputs[:,:input_shift[0].shape[0]], input_shift)
    if 'extra_scaling' in shifts:
        inputs[:, shifts['extra_idx']] = get.scale_with_shifts( inputs[:, shifts['extra_idx']][:,None], shifts['extra_scaling']).squeeze()
    images, inputs = tfun.to_float32( images, inputs)

    ## model prediction
    tic( '{} prediction'.format( ann ) )
    y_pred = ANN.batched_prediction( 150, images, inputs ).numpy() 
    y_pred = get.unscale_data( y_pred, output_shift)
    toc( '{} prediction'.format( ann) )
    n_params = sum([np.prod( x.shape) for x in ANN.trainable_variables])

    ### Some error evaluation
    mse = ( (y_pred - target)**2)
    mae = ( np.abs(y_pred - target)) 
    mre = 100*( mae[:,:2]/target[:,:2])
    rel_mre = 100*tfun.relative_mse( target, y_pred, axis=1 ).numpy()
    print( '##### ERROR MEASURES (on true scale) for {} set #####'.format( ann) )
    print( 'MAE:', mae.mean( 0), ', total', mae.mean() )
    print( 'MSE:', mse.mean( 0), ', total', mse.mean() )
    print( 'rel mre',  rel_mre.mean() )
    print( 'MRE (11 & 22)', mre.mean( 0) ) 
    print( 'max relative error:', 100*( np.abs(y_pred[:,:2] - target[:,:2])/target[:,:2]).max( 0) )
    print( 'max absolute error:', np.abs(y_pred - target).max( 0) )
    print( 'max squared error: ', ( (y_pred - target)**2 ).max( 0) )
    print( 'total amount of parameters in the model:', n_params )
    print( '##########################################' )
    ## plots of error in each component
    if plot_scatter:
        fig, axes = binned_errors( input_raw[:,0], target, y_pred, line_threshold=error_lines, quantiles=quantiles)
        if save_plots:
            savename = os.path.join( save_dir, f'scatter_component{i}.pdf' )
            fig.savefig( savename) 
    if plot_r_square:
        label_x = [r'$\bar\kappa_{11}$ [-]', r'$\bar\kappa_{22}$ [-]', r'$\sqrt{2}\bar\kappa_{12}$ [-]']
        label_y = [r'$\hat\kappa_{11}$ [-]', r'$\hat\kappa_{22}$ [-]', r'$\sqrt{2}\hat\kappa_{12}$ [-]']
        for i in range( target.shape[-1]):
            fig, axes = plot.fixed_plot(1, 1 ) ###move it to be single plot for each component
            box_xpos= lambda i: 1.00/3*(i+1) - 0.115 #if 3 plots
            box_loc = True #default param
            xlabel = r'true value {}'.format(label_x[i])
            ylabel = r'predicted value {}'.format(label_y[i])
            r_squared_plot( axes, target[:,i], y_pred[:,i], xlabel=xlabel, ylabel=ylabel, box_loc=box_loc )
            if save_plots:
                savename = os.path.join( save_dir, f'rsquare_component{i}.pdf' )
                fig.savefig( savename)
            plt.close()

