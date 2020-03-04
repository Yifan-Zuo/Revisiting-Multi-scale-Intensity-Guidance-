'''
Created on 22Mar,2019

@author: yizuo
'''
import tensorflow as tf
import basic_net_blocks_mdgr as bnb
import h5py

up_factor=16
batch_sz=16
height=128
width=128

#setting input size and training data addr
epo_range=50
train_h5F_addr="/media/kenny/Data/training_data/gdsr_train_data/16x_data/shuffle_version/16x_training_data.h5"
total_pat=244992
#total_val_pat=0
LR_height=height/up_factor
LR_width=width/up_factor
batch_total=total_pat/batch_sz
#val_batch_total=total_val_pat/batch_sz
HR_patch_size=[height,width]
HR_batch_dims=(batch_sz,height,width,1)
LR_batch_dims=(batch_sz,LR_height,LR_width,1)

#setting input placeholders
HR_depth_batch_input=tf.placeholder(tf.float32,HR_batch_dims)
HR_inten_batch_input=tf.placeholder(tf.float32,HR_batch_dims)
LR_depth_batch_input=tf.placeholder(tf.float32,LR_batch_dims)
coar_inter_dep_batch=tf.image.resize_images(LR_depth_batch_input,tf.constant(HR_patch_size,dtype=tf.int32),tf.image.ResizeMethod.BICUBIC)

#network construction
inten_feature=bnb.inten_feature_extraction(HR_inten_batch_input)
inten_1x_ten=bnb.inten_residual_block(inten_feature,1)
inten_ten,inten_2x_down=bnb.inten_residual_block(inten_1x_ten,2)
inten_2x_down_up=bnb.feature_up_unit(inten_2x_down,1)
inten_ten,inten_4x_down=bnb.inten_residual_block(inten_ten,4)
inten_4x_down_up=bnb.feature_up_unit(inten_4x_down,2)
inten_ten,inten_8x_down=bnb.inten_residual_block(inten_ten,8)
inten_8x_down_up=bnb.feature_up_unit(inten_8x_down,3)
dep_16x_ten=bnb.LR_dep_feature_extraction(LR_depth_batch_input)
dep_16x_ten_up=bnb.feature_up_unit(dep_16x_ten,4)
dep_8x_ten=bnb.LR_dep_fusion(dep_16x_ten_up[0],inten_8x_down,fusion_stage=1)
dep_8x_ten_up=bnb.feature_up_unit(dep_8x_ten,3)
dep_4x_ten=bnb.LR_dep_fusion(tf.concat([dep_16x_ten_up[1],dep_8x_ten_up[0]],3),tf.concat([inten_8x_down_up[0],inten_4x_down],3),fusion_stage=2)
dep_4x_ten_up=bnb.feature_up_unit(dep_4x_ten,2)
dep_2x_ten=bnb.LR_dep_fusion(tf.concat([dep_16x_ten_up[2],dep_8x_ten_up[1],dep_4x_ten_up[0]],3),tf.concat([inten_8x_down_up[1],inten_4x_down_up[0],inten_2x_down],3),fusion_stage=3)
dep_2x_ten_up=bnb.feature_up_unit(dep_2x_ten,1)
dep_ten=bnb.LR_dep_fusion(tf.concat([dep_16x_ten_up[3],dep_8x_ten_up[2],dep_4x_ten_up[1],dep_2x_ten_up[0]],3),tf.concat([inten_8x_down_up[2],inten_4x_down_up[1],inten_2x_down_up[0],inten_1x_ten],3),fusion_stage=4)
HR_gen_dep=bnb.LR_dep_reconstruction(dep_16x_ten_up[3],dep_ten,coar_inter_dep_batch)

#define loss for gen
#loss=tf.reduce_mean(tf.squared_difference(HR_gen_dep,HR_depth_batch_input))
loss=tf.reduce_mean(tf.abs(HR_gen_dep-HR_depth_batch_input))
#loss=tf.reduce_mean(tf.sqrt(tf.squared_difference(HR_gen_dep,HR_depth_batch_input)+1e-3))
train_op_small = tf.train.AdamOptimizer(1e-5).minimize(loss)
train_op_large = tf.train.AdamOptimizer(1e-4).minimize(loss)
saver_full=tf.train.Saver(max_to_keep=1200)
model_ind=0
init_op=tf.global_variables_initializer()

#begin comp_gen training
with h5py.File(train_h5F_addr,"r") as train_file:
    with tf.Session() as sess:
        sess.run(init_op)
        #saver_full.restore(sess, "/media/kenny/Data/trained_models/multi_dense_guide_resnet/noisy/l2loss/16x/full_model1/16x_ny_full_model.ckpt-1")
        for epo in range(epo_range):
            if epo<20:
                train_op=train_op_large
            else:
                train_op=train_op_small
            for ind in range(batch_total):
                gen_pat_ind_range=range(ind*batch_sz,(ind+1)*batch_sz,1)
                gen_inten_bat,gen_gth_dep_bat,gen_LR_dep_bat=bnb.reading_data(train_file, gen_pat_ind_range, HR_batch_dims, LR_batch_dims)
                sess.run(train_op,feed_dict={HR_inten_batch_input:gen_inten_bat,HR_depth_batch_input:gen_gth_dep_bat,LR_depth_batch_input:gen_LR_dep_bat})
                if (ind+1)%1914==0:
                    mae_loss=loss.eval(feed_dict={HR_inten_batch_input:gen_inten_bat,HR_depth_batch_input:gen_gth_dep_bat,LR_depth_batch_input:gen_LR_dep_bat})
                    print("step %d, training loss %g"%(ind, mae_loss))
                if (ind+1)%7656==0:
                    save_path=saver_full.save(sess,"/media/kenny/Data/trained_models/multi_dense_guide_resnet/noise-free/l1loss/16x/full_model1/16x_nf_full_model.ckpt",global_step=model_ind)
                    print("Full Model saved in file: %s" % save_path)
                    #val_mae_loss=0
                    #for val_ind in range(val_batch_total):
                        #gen_pat_ind_range=range(total_pat+val_ind*batch_sz,total_pat+(val_ind+1)*batch_sz,1)
                        #gen_inten_bat,gen_gth_dep_bat,gen_LR_dep_bat=bnb.reading_data(train_file, gen_pat_ind_range, HR_batch_dims, LR_batch_dims)
                        #mae_loss=loss.eval(feed_dict={HR_inten_batch_input:gen_inten_bat,HR_depth_batch_input:gen_gth_dep_bat,LR_depth_batch_input:gen_LR_dep_bat})
                        #val_mae_loss=val_mae_loss+mae_loss
                    #val_mae_loss=val_mae_loss/val_batch_total
                    #print("model %d, validation loss %g"%(model_ind, val_mae_loss))
                    model_ind=model_ind+1
