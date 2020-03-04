'''
Created on 26Mar,2019

@author: yizuo
'''
from PIL import Image
import numpy as np
import tensorflow as tf
import basic_net_blocks_mdgr as bnb

up_factor=16
data_name="dolls"
#read test HR inten img
color_art = Image.open("/media/kenny/Data/test_data/noise_free/middlebury_bicubic_LR_test_pairs/"+data_name+"/color_"+data_name+"_croped.png")#####for noise-free midd2006
#color_art = Image.open("/media/kenny/Data/test_data/noisy/table2/_gth/"+data_name+"_color.png")####mid 2006
#color_art = Image.open("/media/kenny/Data/test_data/noisy/iccv15_test_imgs/gth/"+data_name+"/color_crop.png")####mid 2001
inten_art =color_art.convert("L")
np_inten=np.asarray(inten_art)
val_inten=np_inten.astype(np.float32)/255.0

#read test HR dep img
gth_dep_art=Image.open("/media/kenny/Data/test_data/noise_free/middlebury_bicubic_LR_test_pairs/"+data_name+"/"+data_name+"_croped.png")#####for noise-free midd2006
#gth_dep_art=Image.open("/media/kenny/Data/test_data/noisy/table2/_gth/"+data_name+"_big.png")###mid 2006
#gth_dep_art=Image.open("/media/kenny/Data/test_data/noisy/iccv15_test_imgs/gth/"+data_name+"/depth_crop.png")###mid 2001
np_gth_dep=np.asarray(gth_dep_art)
val_gth_dep=np_gth_dep.astype(np.float32)/255.0

#read test LR dep img
LR_dep_art=Image.open("/media/kenny/Data/test_data/noise_free/middlebury_bicubic_LR_test_pairs/"+data_name+"/"+data_name+"16x_bicubic.png")#####for noise-free midd2006
#LR_dep_art=Image.open("/media/kenny/Data/test_data/noisy/table2/_input/"+data_name+"_big/depth_4_n.png")###mid 2006
#LR_dep_art=Image.open("/media/kenny/Data/test_data/noisy/iccv15_test_imgs/input/"+data_name+"/16_x_dep.png")###mid 2001
np_LR_dep=np.asarray(LR_dep_art)
val_LR_dep=np_LR_dep.astype(np.float32)/255.0

height=val_inten.shape[0]
width=val_inten.shape[1]
LR_height=height/up_factor
LR_width=width/up_factor
val_inten=val_inten.reshape((1,height,width,1))
val_gth_dep=val_gth_dep.reshape(1,height,width,1)
val_LR_dep=val_LR_dep.reshape(1,LR_height,LR_width,1)

#setting input size and training data addr
HR_patch_size=[height,width]
HR_batch_dims=(1,height,width,1)
LR_batch_dims=(1,LR_height,LR_width,1)

#setting input placeholders
HR_depth_batch_input=tf.placeholder(tf.float32,HR_batch_dims)
HR_inten_batch_input=tf.placeholder(tf.float32,HR_batch_dims)
LR_depth_batch_input=tf.placeholder(tf.float32,LR_batch_dims)
coar_inter_dep_batch=tf.image.resize_images(LR_depth_batch_input,tf.constant(HR_patch_size,dtype=tf.int32),tf.image.ResizeMethod.BICUBIC)

#gen_network construction
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
saver_full=tf.train.Saver()

#begin comp_gen testing
with tf.Session() as sess:
    model_path="/media/yifan/Data/trained_models/multi_dense_guide_resnet/noise_free/l1loss/16x/full_model3/16x_nf_full_model.ckpt-75"
    saver_full.restore(sess, model_path)
    ten_fets=sess.run(HR_gen_dep,feed_dict={HR_inten_batch_input:val_inten,HR_depth_batch_input:val_gth_dep,LR_depth_batch_input:val_LR_dep})
    final_array=ten_fets*255.0+0.5
    final_array[final_array>255]=255.0
    final_array[final_array<0]=0.0
    final_array=final_array.astype(np.uint8).reshape((height,width))
    result_img=Image.fromarray(final_array)
    #result_img.show()
    result_img.save("/media/kenny/Data/trained_models/multi_dense_guide_resnet/noise-free/l1loss/16x/results/"+data_name+"16x_result.png")
    ######################computing rmse
    final_array=final_array.astype(np.double)
    np_gth_dep=np_gth_dep.astype(np.double)
    print(np.sqrt(((final_array-np_gth_dep)**2).mean()))
    print((np.absolute(final_array-np_gth_dep)).mean())
    #print("below are evaluated on 1080*1320")
    #print(np.sqrt(((final_array[0:1080,0:1320]-np_gth_dep[0:1080,0:1320])**2).mean()))
    #print((np.absolute(final_array[0:1080,0:1320]-np_gth_dep[0:1080,0:1320])).mean())
    #######################
