import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from TPN import TPN_Model

def contrastive_loss(p1_v,p2_v,cluster,args):

    p1 = tf.math.l2_normalize(p1_v, axis=1)
    p2 = tf.math.l2_normalize(p2_v, axis=1)
    batch_size = len(p1)
    LARGE_NUM=1e9

    logits_ab = tf.matmul(p1,p2,transpose_b=True)/args.tem

    #logints_aa
    pre_class = cluster.fit_predict(p1.numpy())
    masks_aa = tf.convert_to_tensor([i==pre_class for i in pre_class])

    masks_aa = tf.stop_gradient(tf.cast(masks_aa,tf.float32))
    logits_aa = tf.matmul(p1,p1,transpose_b=True)/args.tem
    logits_aa = logits_aa - masks_aa * LARGE_NUM

    masks_ab = masks_aa - tf.one_hot(tf.range(batch_size), batch_size)
    logits_ab = logits_ab - masks_ab* LARGE_NUM
    logits_ba = tf.matmul(p2,p1,transpose_b=True) / args.tem

    #logits_bb
    pre_class = cluster.fit_predict(p2.numpy())
    masks_bb = tf.convert_to_tensor([i == pre_class for i in pre_class])
    masks_bb = tf.stop_gradient(tf.cast(masks_bb, tf.float32))

    logits_bb = tf.matmul(p2, p2, transpose_b=True) / args.tem
    logits_bb = logits_bb - masks_bb * LARGE_NUM
    masks_ba = masks_bb - tf.one_hot(tf.range(batch_size), batch_size)
    logits_ba = logits_ba - masks_ba * LARGE_NUM

    labels = tf.range(batch_size)
    loss_a = keras.losses.sparse_categorical_crossentropy(labels, tf.concat([logits_ab,logits_aa],axis=1), from_logits=True)
    loss_b = keras.losses.sparse_categorical_crossentropy(labels, tf.concat([logits_ba,logits_bb],axis=1), from_logits=True)

    return loss_a+loss_b

def get_backbone(backbone_name,n_timesteps,n_features):
    if backbone_name == 'tpn':
        return TPN_Model((n_timesteps,n_features))
    else:
        raise ValueError("The name is not available.")

def attch_projection_head(backbone,dim1=256,dim2=128,dim3=50):
    return Sequential([backbone,Dense(dim1),ReLU(),Dense(dim2),ReLU(),Dense(dim3)])