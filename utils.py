import numpy as np
import tensorflow as tf
from sklearn.cluster import Birch,KMeans
from sklearnex import patch_sklearn
from Augment import resample_random
from module import contrastive_loss

def get_data(data_name):
    if data_name=='ucihar':
        x_data = np.load('datasets/UCI_X.npy')
        y_data = np.load('datasets/UCI_Y.npy')
    elif data_name=='motion':
        x_data = np.load('datasets/Motion_X.npy')
        y_data = np.load('datasets/Motion_Y.npy')
    elif data_name == 'uschad':
        x_data = np.load('datasets/USCHAD_X.npy')
        y_data = np.load('datasets/USCHAD_Y.npy')
    else:
        raise ValueError("The dataset name is not valid.")
    np.random.seed(888)
    np.random.shuffle(x_data)
    np.random.seed(888)
    np.random.shuffle(y_data)
    return x_data,y_data

def get_cluster(cluster_name,cluster_num):
    if cluster_name == 'birch':
        cluster = Birch(threshold=0.1, n_clusters=cluster_num)
    elif cluster_name == 'kmeans':
        cluster = KMeans(n_clusters=cluster_num)
    return cluster

def train_step(xis, xjs, model, optimizer,cluster,args):
    with tf.GradientTape() as tape:
        zis = model(xis)
        zjs = model(xjs)
        loss = contrastive_loss(zis,zjs,cluster,args)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return tf.reduce_mean(loss)

def train(model,x_data,args):
    optimizer = tf.keras.optimizers.Adam(args.lr)
    epochs = args.epoch
    batch_size = args.batch_size

    patch_sklearn()
    cluster = get_cluster(args.cluster,args.cluster_num)

    cur_loss = 1e9
    seed = len(x_data)
    for epoch in range(epochs):
        loss_epoch = []
        train_loss_dataset = tf.data.Dataset.from_tensor_slices(x_data).shuffle(seed,reshuffle_each_iteration=True).batch(batch_size)
        for x in train_loss_dataset:
            xis = resample_random(x)
            xjs = x
            loss = train_step(xis, xjs, model, optimizer,cluster,args)
            loss_epoch.append(loss)
        print("epoch{}===>loss:{}".format(epoch + 1, np.mean(loss_epoch)))
        if epoch > epochs//2 and np.mean(loss_epoch) < cur_loss:
            tf.keras.models.save_model(model,'contrastive_model/'+'{}_cluster_{}_batchsize_{}_epoch_{}'.format(args.dataset,args.cluster,args.batch_size,args.epoch))
            cur_loss = np.mean(loss_epoch)