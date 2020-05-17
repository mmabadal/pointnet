import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
from model import *


"RUN python train.py --path_data a/b/c --cls 5 --log_dir RUNS/a --batch_size 10 --max_epoch 50"

parser = argparse.ArgumentParser()
parser.add_argument('--path_data', help='folder with train test data')
parser.add_argument('--cls', type=int, help='number of classes')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parsed_args = parser.parse_args()

path_data = parsed_args.path_data
cls = parsed_args.cls
BATCH_SIZE = parsed_args.batch_size
NUM_POINT = parsed_args.num_point
MAX_EPOCH = parsed_args.max_epoch
NUM_POINT = parsed_args.num_point
BASE_LEARNING_RATE = parsed_args.learning_rate
GPU_INDEX = parsed_args.gpu
MOMENTUM = parsed_args.momentum
OPTIMIZER = parsed_args.optimizer
DECAY_STEP = parsed_args.decay_step
DECAY_RATE = parsed_args.decay_rate

LOG_DIR = parsed_args.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp model.py %s' % (LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(parsed_args)+'\n')

NUM_CLASSES = cls
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# ------------------------ OK ------------------------
path_train = os.path.join(path_data, 'train/h5')
files_train = provider.getDataFiles(os.path.join(path_train, 'files.txt'))
filelist_train = provider.getDataFiles(os.path.join(path_train, 'filelist.txt'))

data_batch_list = []
label_batch_list = []

for h5_filename in files_train:
    file_path = os.path.join(path_train, h5_filename)
    data_batch, label_batch = provider.loadDataFile(file_path)
    data_batch_list.append(data_batch)
    label_batch_list.append(label_batch)
train_data = np.concatenate(data_batch_list, 0)
train_label = np.concatenate(label_batch_list, 0)
print(train_data.shape, train_label.shape)


path_test = os.path.join(path_data, 'val/h5')
files_test = provider.getDataFiles(os.path.join(path_test, 'files.txt'))
filelist_test = provider.getDataFiles(os.path.join(path_test, 'filelist.txt'))

data_batch_list = []
label_batch_list = []

for h5_filename in files_test:
    file_path = os.path.join(path_test, h5_filename)
    data_batch, label_batch = provider.loadDataFile(file_path)
    data_batch_list.append(data_batch)
    label_batch_list.append(label_batch)
test_data = np.concatenate(data_batch_list, 0)
test_label = np.concatenate(label_batch_list, 0)
print(test_data.shape, test_label.shape)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred = get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'val'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        loss_t_list = list()
        loss_val_list = list()
        acc_t_list = list()
        acc_val_list = list()

        for epoch in range(MAX_EPOCH):
            
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            loss_t, acc_t = train_one_epoch(sess, ops, train_writer)
            loss_t_list.append(loss_t)
            acc_t_list.append(acc_t)

            loss_val, acc_val = eval_one_epoch(sess, ops, test_writer)
            loss_val_list.append(loss_val)
            acc_val_list.append(acc_val)

            if loss_val == min(loss_val_list):
                best_sess = sess
                best_epoch = epoch

            stop = early_stopping(loss_t_list, loss_val_list, 1)
            if stop:
                log_string('early stopping at epoch %03d' % (best_epoch))
                break

        sess = best_sess
        
        # Save the variables to disk. 
        save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
        log_string("Model saved in file: %s" % save_path)

def early_stopping(t_loss, v_loss, thr):

    stop = False

    if len(t_loss) > 9:

        a = 100 * ((v_loss[-1] / min(v_loss)) - 1)
        b = 1000 * ((sum(t_loss[-10:]) / (10 * min(t_loss[-10:]))) - 1)
        ab = a / b

        if ab > thr:
            stop = True

    return stop


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string('----')
    current_data, current_label, _ = provider.shuffle_data(train_data[:,0:NUM_POINT,:], train_label) 
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    
    for batch_idx in range(num_batches):
        if batch_idx % 100 == 0:
            print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        
        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
    
    mean_loss = loss_sum / float(num_batches)
    accuracy = total_correct / float(total_seen)

    log_string('mean loss: %f' % (mean_loss))
    log_string('accuracy: %f' % (accuracy))

    return mean_loss, accuracy

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    log_string('----')
    current_data = test_data[:,0:NUM_POINT,:]
    current_label = np.squeeze(test_label)
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_label[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx, j] == l)
            
    
    mean_loss = loss_sum / float(total_seen/NUM_POINT)
    accuracy = total_correct / float(total_seen)
    
    
    log_string('eval mean loss: %f' % (mean_loss))
    log_string('eval accuracy: %f'% (accuracy))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class, dtype=np.float))))

    return mean_loss, accuracy
         


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
