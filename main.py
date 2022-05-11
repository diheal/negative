import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from TPN import TPN_Model
from Augment import *
from tensorflow import keras
from sklearn.cluster import KMeans, Birch

batch_size = 1024
epochs = 200
optimizer = tf.keras.optimizers.Adam(0.001)
temperature = 0.1
augmentation = resample_random
feature_dimensions = 96
datasets='UCI'

if datasets=='UCI':
    x_data = np.load('datasets/UCI_X.npy')
    y_data = np.load('datasets/UCI_Y.npy')
np.random.seed(888)
np.random.shuffle(x_data)
np.random.seed(888)
np.random.shuffle(y_data)

n_timesteps, n_features, n_outputs = x_data.shape[1], x_data.shape[2], y_data.shape[1]

def attch_projection_head(backbone,dim1=256,dim2=128,dim3=50):
    return Sequential([backbone,Dense(dim1),ReLU(),Dense(dim2),ReLU(),Dense(dim3)])

backbone = TPN_Model((n_timesteps, n_features))
model_cl = attch_projection_head(backbone,feature_dimensions,feature_dimensions,feature_dimensions)

cluster = Birch(threshold=0.1,n_clusters=n_outputs)
def contrastive_loss(p1_v,p2_v):

    p1 = tf.math.l2_normalize(p1_v, axis=1)
    p2 = tf.math.l2_normalize(p2_v, axis=1)
    batch_size = len(p1)
    LARGE_NUM=1e9

    logits_ab = tf.matmul(p1,p2,transpose_b=True)/temperature

    #logints_aa
    pre_class = cluster.fit_predict(p1.numpy())
    masks_aa = tf.convert_to_tensor([i==pre_class for i in pre_class])

    masks_aa = tf.stop_gradient(tf.cast(masks_aa,tf.float32))
    logits_aa = tf.matmul(p1,p1,transpose_b=True)/temperature
    logits_aa = logits_aa - masks_aa * LARGE_NUM

    masks_ab = masks_aa - tf.one_hot(tf.range(batch_size), batch_size)
    logits_ab = logits_ab - masks_ab* LARGE_NUM
    logits_ba = tf.matmul(p2,p1,transpose_b=True) / temperature

    #logits_bb
    pre_class = cluster.fit_predict(p2.numpy())
    masks_bb = tf.convert_to_tensor([i == pre_class for i in pre_class])
    masks_bb = tf.stop_gradient(tf.cast(masks_bb, tf.float32))

    logits_bb = tf.matmul(p2, p2, transpose_b=True) / temperature
    logits_bb = logits_bb - masks_bb * LARGE_NUM
    masks_ba = masks_bb - tf.one_hot(tf.range(batch_size), batch_size)
    logits_ba = logits_ba - masks_ba * LARGE_NUM

    labels = tf.range(batch_size)
    loss_a = keras.losses.sparse_categorical_crossentropy(labels, tf.concat([logits_ab,logits_aa],axis=1), from_logits=True)
    loss_b = keras.losses.sparse_categorical_crossentropy(labels, tf.concat([logits_ba,logits_bb],axis=1), from_logits=True)

    return loss_a+loss_b


def train_step(xis, xjs, model, optimizer):
    with tf.GradientTape() as tape:
        zis = model(xis)
        zjs = model(xjs)
        loss = contrastive_loss(zis,zjs)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return tf.reduce_mean(loss)

c_loss=1e4
for epoch in range(epochs):
    loss_epoch = []
    train_loss_dataset = tf.data.Dataset.from_tensor_slices(x_data).shuffle(len(y_data),reshuffle_each_iteration=True).batch(batch_size)

    for x in train_loss_dataset:
        xis = augmentation(x)
        xjs = x
        loss = train_step(xis, xjs, model_cl, optimizer)
        loss_epoch.append(loss)
    print("epoch{}===>loss:{}".format(epoch + 1, np.mean(loss_epoch)))
    if np.mean(loss_epoch)<c_loss:
        model_cl.save('contrastive_model/'+datasets+'_'+str(batch_size)+'.h5')
        c_loss=np.mean(loss_epoch)
