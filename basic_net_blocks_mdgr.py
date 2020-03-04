'''
Created on 22Mar.,2019

@author: yifanzuo
'''
import tensorflow as tf
import numpy as np
#define sub-functions for creating variables
#ksize,w_shape,b_shape,strides are all list type, this is very important
def leaky_relu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)

def max_pool_3x3(x,ratio):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],strides=[1, ratio, ratio, 1], padding='SAME')

def weight_variable(w_shape,out_ch=32,name=None):
    std_factor=w_shape[0]*w_shape[1]*out_ch
    std_dev=np.sqrt(2.0/std_factor)
    initial_W = tf.truncated_normal(w_shape, stddev=std_dev)
    if name is None:
        return tf.Variable(initial_W,name="conv_weight",dtype=tf.float32)
    else:
        return tf.get_variable(name,dtype=tf.float32,initializer=initial_W)
    
def bias_variable(b_shape,name=None):
    initial_B = tf.constant(0.0, shape=b_shape)
    if name is None:
        return tf.Variable(initial_B,name="conv_bias",dtype=tf.float32)
    else:
        return tf.get_variable(name,dtype=tf.float32,initializer=initial_B)
    
def Prelu(input_tensor,name=None):
    initial_a=tf.constant(0.25, shape=[input_tensor.get_shape().as_list()[3]])
    if name is None:
        alphas=tf.Variable(initial_a,name="prelu_alpha",dtype=tf.float32)
    else:
        alphas=tf.get_variable(name,dtype=tf.float32,initializer=initial_a)
    pos = tf.nn.relu(input_tensor)
    neg = alphas * (input_tensor - abs(input_tensor)) * 0.5
    return pos+neg

#define batch normalization function
def batch_norm(x, n_out, phase_train,name=None):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    if name is None:
        beta = tf.Variable(tf.constant(0.0, shape=n_out),name="bn_beta",dtype=tf.float32)
        gamma = tf.Variable(tf.truncated_normal(n_out, 1.0, 0.02),name="bn_gamma",dtype=tf.float32)
    else:
        beta=tf.get_variable(name[0],dtype=tf.float32,initializer=tf.constant(0.0, shape=n_out))
        gamma =tf.get_variable(name[1],dtype=tf.float32,initializer=tf.truncated_normal(n_out, 1.0, 0.02))
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(phase_train,mean_var_with_update,lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

#define sub-function for conv layer with prelu
def conv_Prelu_block(input_ten,w_shape,stride,skip_Prelu):
    W=weight_variable(w_shape,w_shape[3])
    conv_ten = tf.nn.conv2d(input_ten, W,stride, padding='SAME')
    B=bias_variable([w_shape[3]])
    if skip_Prelu:
        return conv_ten + B
    else:
        return Prelu(conv_ten + B)
        
#define sub-function for deconv layer with prelu
def deconv_Prelu_block(input_ten,w_shape,out_put_shape,stride,skip_Prelu):
    W=weight_variable(w_shape,w_shape[2])
    deconv_ten= tf.nn.conv2d_transpose(input_ten,W,out_put_shape,stride,padding="SAME")
    B=bias_variable([w_shape[2]])
    if skip_Prelu:
        return deconv_ten + B
    else:
        return Prelu(deconv_ten + B)
        
#from this on is the construction blocks of network
#define sub-functions for the intensity branch
def inten_feature_extraction(input_ten,shape_list=[[7,7,1,64],[7,7,64,32],[5,5,32,32]],stride=[1,1,1,1],skip_Prelu_list=[False,False,False]):
    for index in range(len(skip_Prelu_list)):
        input_ten=conv_Prelu_block(input_ten, shape_list[index], stride, skip_Prelu_list[index])
    return input_ten

def inten_residual_block(input_ten,ratio,shape_list=[5,5,32,32],stride=[1,1,1,1],skip_Prelu_list=False):
    output_ten=conv_Prelu_block(input_ten, shape_list, stride, skip_Prelu_list)
    if ratio>1:
        inten_down_ten=max_pool_3x3(output_ten,ratio)
        return output_ten,inten_down_ten
    else:
        return output_ten

def feature_up_unit(input_ten,iters,shape=[5,5,32,32],skip_Prelu=False):
    output_ten=[]
    input_shape=input_ten.get_shape().as_list()
    for index in range(iters):
        ratio=int(np.exp2(index+1))
        stride=[1,ratio,ratio,1]
        out_shape_list=[input_shape[0],ratio*input_shape[1],ratio*input_shape[2],shape[2]]
        inter_ten=deconv_Prelu_block(input_ten,shape,tf.convert_to_tensor(out_shape_list),stride,skip_Prelu)
        output_ten.append(inter_ten)
    return output_ten

#define sub-functions for the LR depth branch
def LR_dep_feature_extraction(input_ten,shape_list=[[5,5,1,64],[5,5,64,32]],stride=[1,1,1,1],skip_Prelu_list=[False,False]):
    for index in range(len(skip_Prelu_list)):
        input_ten=conv_Prelu_block(input_ten, shape_list[index], stride, skip_Prelu_list[index])
    return input_ten

def RDB_block(input_ten,conv_num=5,kz=3,ch_num=32,stride=[1,1,1,1],skip_Prelu=False):
    mid_ten=input_ten
    for ind in range(conv_num-1):
        in_ch=(ind+1)*ch_num
        pre_ten=conv_Prelu_block(mid_ten, [kz,kz,in_ch,ch_num], stride, skip_Prelu)
        mid_ten=tf.concat([mid_ten,pre_ten],3)
    mid_ten=conv_Prelu_block(mid_ten, [1,1,in_ch+ch_num,ch_num], stride, skip_Prelu=True)
    return mid_ten+input_ten

def LR_dep_fusion(input_dep_ten,input_inten_ten,fusion_stage=1,stride=[1,1,1,1],skip_Prelu_list=False):
    shape_list=[5,5,64*fusion_stage,32]
    shape_dep_list=[1,1,32*fusion_stage,32]
    comp_input_dep=conv_Prelu_block(input_dep_ten, shape_dep_list, stride, skip_Prelu=True)
    concat_input=tf.concat([input_dep_ten,input_inten_ten],3)
    fusion_ten=conv_Prelu_block(concat_input, shape_list, stride, skip_Prelu_list)
    #fusion_final=RDB_block(fusion_ten,5,3,32,[1,1,1,1],skip_Prelu_list)
    for _ in range(3):
        fusion_ten=conv_Prelu_block(fusion_ten, [5,5,32,32], stride, skip_Prelu_list)
    fusion_final=conv_Prelu_block(fusion_ten, [5,5,32,32], stride, True)
    return fusion_final+comp_input_dep

def LR_dep_reconstruction(input_feature,input_fusion_ten,input_coarse_ten,shape_list=[[5,5,32,32],[5,5,32,32],[5,5,32,1]],stride=[1,1,1,1],skip_Prelu_list=[False,True,True]):
    final_ten1=conv_Prelu_block(input_fusion_ten, shape_list[0], stride, skip_Prelu_list[0])
    final_ten2=conv_Prelu_block(final_ten1, shape_list[1], stride, skip_Prelu_list[1])
    final_ten2=final_ten2+input_feature
    final_ten3=conv_Prelu_block(final_ten2, shape_list[2], stride, skip_Prelu_list[2])
    return final_ten3+input_coarse_ten

#define function for reading data from hdf5
def reading_data(train_file,pat_ind_range,HR_batch_dims,LR_batch_dims):
    inten_bat=train_file['inten_patch'][pat_ind_range,:,:]
    inten_bat=inten_bat.reshape(HR_batch_dims)
    gth_dep_bat=train_file['depth_patch'][pat_ind_range,:,:]
    gth_dep_bat=gth_dep_bat.reshape(HR_batch_dims)
    #LR_dep_bat=train_file['LR_noisy_depth_std5'][pat_ind_range,:,:]
    LR_dep_bat=train_file['LR_depth_patch'][pat_ind_range,:,:]
    LR_dep_bat=LR_dep_bat.reshape(LR_batch_dims)
    return inten_bat,gth_dep_bat,LR_dep_bat
