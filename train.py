#-*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import os.path
import time
import sys
import numpy as np
import importlib
import itertools
sys.path.append('./src')

import argparse
import tensorflow as tf
import facenet

import cv2
import align.detect_face
import lfw
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
from tensorflow.python.ops import data_flow_ops
from six.moves import xrange  # @UnresolvedImport


#1.人脸检测对齐训练

#2.人脸对齐使用****************************************************************************************
class face_reconition:
    def __init__(self):
        pass

    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y

    # 根据路径获取该文件夹中所有的图片
    def get_image_paths(self, inpath):
        paths = []
        for file in os.listdir(inpath):
            if os.path.isfile(os.path.join(inpath, file)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')) is False:
                    continue

                paths.append(os.path.join(inpath, file))

        return (paths)


    # 将一个文件夹下的所有图片转化为json  方法二  只能是传入文件夹  并存入数据库
    def images_to_vectors(self, inpath, outjson_path, modelpath):
        results = dict()

        with tf.Graph().as_default():
            with tf.Session() as sess:
                facenet.load_model(modelpath)
                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                image_paths = self.get_image_paths(inpath)
                for image_path in image_paths:
                    # 获取图片中的人脸数
                    img = imread(os.path.expanduser(image_path), mode='RGB')
                    images = self.image_array_align_data(img,image_path)
                    #判断是否检测出人脸 检测不出 就跳出此循环
                    if images.shape[0] == 1 : continue
                    
        # 返回图像中所有人脸的向量

    def image_array_align_data(self, image_arr,image_path, image_size=160, margin=32, gpu_memory_fraction=1.0,
                               detect_multiple_faces=True):
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor

        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

        img = image_arr
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]

        nrof_successfully_aligned = 0
        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces > 1:
                if detect_multiple_faces:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))
                else:
                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                    img_center = img_size / 2
                    offsets = np.vstack(
                        [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                    index = np.argmax(
                        bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                    det_arr.append(det[index, :])
            else:
                det_arr.append(np.squeeze(det))

            images = np.zeros((nrof_faces, image_size, image_size, 3))
            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                # 进行图片缩放 cv2.resize(img,(w,h))
                scaled = imresize(cropped, (image_size, image_size), interp='bilinear')
                nrof_successfully_aligned += 1

                # print(scaled)
                # scaled=self.prewhiten(scaled)
                # 保存检测的头像
                filename_base = './img/'
                filename = os.path.basename(image_path)
                filename_name, file_extension = os.path.splitext(filename)
                output_filename_n = "{}/{}_{}{}".format(filename_base, filename_name, i, file_extension)
                imsave(output_filename_n, scaled)


                scaled = facenet.prewhiten(scaled)
                scaled = facenet.crop(scaled, False, 160)
                scaled = facenet.flip(scaled, False)

                images[i] = scaled
        if nrof_faces > 0:
            return images
        else:
            #如果没有检测到人脸  直接返回一个1*3的0矩阵  多少维度都行  只要能和是不是一个图片辨别出来就行
            return np.zeros((1,3))

#3.人脸识别训练**************************************************************************************
def face_train(args):
  
    network = importlib.import_module(args.model_def)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Write arguments to a text file
    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
        
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))    # 读取源文件的文件路径
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))     # 保存版本号

    np.random.seed(seed=args.seed)
    train_set = facenet.get_dataset(args.data_dir)    # 获取数据
    # trainset[0] = {'name': "AJ_cook", "image_path":"----"}

    # 没有预训练模型
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    if args.pretrained_model:
        print('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))

    # 没有该文件夹
    if args.lfw_dir:
        print('LFW directory: %s' % args.lfw_dir)
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
        # Get the paths for the corresponding images
        lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs)
        
    
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)

        # Placeholder for the learning rate
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,3), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int64, shape=(None,3), name='labels')
        
        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                    dtypes=[tf.string, tf.int64],
                                    shapes=[(3,), (3,)],
                                    shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])
        
        nrof_preprocess_threads = 4
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                # 对图片进行解码
                image = tf.image.decode_image(file_contents, channels=3)

                # random crop
                if args.random_crop:
                    image = tf.random_crop(image, [args.image_size, args.image_size, 3])
                else:
                    # resize image
                    image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                if args.random_flip:
                    image = tf.image.random_flip_left_right(image)
    
                #pylint: disable=no-member
                image.set_shape((args.image_size, args.image_size, 3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images, label])
    
        image_batch, labels_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder, 
            shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        # 由于命名的不同，所以op是不一样的
        image_batch = tf.identity(image_batch, 'image_batch')   # identity类似深拷贝， 新开辟内存
        image_batch = tf.identity(image_batch, 'input')
        labels_batch = tf.identity(labels_batch, 'label_batch')

        # Build the inference graph
        prelogits, _ = network.inference(image_batch, args.keep_probability, 
            phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
            weight_decay=args.weight_decay)
        
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        # Split embeddings into anchor, positive and negative and calculate triplet loss
        anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1,3,args.embedding_size]), 3, 1)
        triplet_loss = facenet.triplet_loss(anchor, positive, negative, args.alpha)

        # 学习率指数衰减
        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the total losses   # 获取变量表 ， 获取正则化后的所有变量，可训练的参数
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # 按照元素对tensor进行循环加
        total_loss = tf.add_n([triplet_loss] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = facenet.train(total_loss, global_step, args.optimizer, 
            learning_rate, args.moving_average_decay, tf.global_variables())
        
        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))        

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder:True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder:True})

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)   # 与队列有关的内容

        with sess.as_default():

            if args.pretrained_model:
                print('Restoring pretrained model: %s' % args.pretrained_model)
                saver.restore(sess, os.path.expanduser(args.pretrained_model))

            # Training and validation loop
            epoch = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
                # Train for one epoch
                train(args, sess, train_set, epoch, image_paths_placeholder, labels_placeholder, labels_batch,
                    batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, input_queue, global_step, 
                    embeddings, total_loss, train_op, summary_op, summary_writer, args.learning_rate_schedule_file,
                    args.embedding_size, anchor, positive, negative, triplet_loss, saver, model_dir, subdir, step)

                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

                # Evaluate on LFW
                
                if args.lfw_dir:
                    evaluate(sess, lfw_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder, 
                            batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, actual_issame, args.batch_size, 
                            args.lfw_nrof_folds, log_dir, step, summary_writer, args.embedding_size)
                
    return model_dir

# batchsize=6， 表示2个批次，3张图片
def train(args, sess, dataset, epoch, image_paths_placeholder, labels_placeholder, labels_batch,
          batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, input_queue, global_step, 
          embeddings, loss, train_op, summary_op, summary_writer, learning_rate_schedule_file,
          embedding_size, anchor, positive, negative, triplet_loss, saver, model_dir, subdir, step):
    batch_number = 0
    
    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    while batch_number < args.epoch_size:
        # Sample people randomly from the dataset      # 采样图片，构建三元组
        image_paths, num_per_class = sample_people(dataset, args.people_per_batch, args.images_per_person)
        
        print('Running forward pass on sampled images: ')
        start_time = time.time()
        nrof_examples = args.people_per_batch * args.images_per_person
        labels_array = np.reshape(np.arange(nrof_examples),(-1,3))
        # 文件路径变为三元组
        image_paths_array = np.reshape(np.expand_dims(np.array(image_paths),1), (-1,3))
        # 入列操作
        sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
        emb_array = np.zeros((nrof_examples, embedding_size))
        nrof_batches = int(np.ceil(nrof_examples / args.batch_size))
        for i in range(nrof_batches):
            batch_size = min(nrof_examples-i*args.batch_size, args.batch_size)
            emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size, 
                learning_rate_placeholder: lr, phase_train_placeholder: True})
            emb_array[lab,:] = emb
        print('%.3f' % (time.time()-start_time))

        # Select triplets based on the embeddings
        print('Selecting suitable triplets for training')
        triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array, num_per_class, 
            image_paths, args.people_per_batch, args.alpha)
        selection_time = time.time() - start_time
        print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' % 
            (nrof_random_negs, nrof_triplets, selection_time))

        # Perform training on the selected triplets
        nrof_batches = int(np.ceil(nrof_triplets*3/args.batch_size))
        triplet_paths = list(itertools.chain(*triplets))
        labels_array = np.reshape(np.arange(len(triplet_paths)),(-1,3))
        triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths),1), (-1,3))
        sess.run(enqueue_op, {image_paths_placeholder: triplet_paths_array, labels_placeholder: labels_array})
        nrof_examples = len(triplet_paths)
        train_time = 0
        i = 0
        emb_array = np.zeros((nrof_examples, embedding_size))
        loss_array = np.zeros((nrof_triplets,))
        summary = tf.Summary()
        step = 0
        while i < nrof_batches:
            start_time = time.time()
            batch_size = min(nrof_examples-i*args.batch_size, args.batch_size)
            feed_dict = {batch_size_placeholder: batch_size, learning_rate_placeholder: lr, phase_train_placeholder: True}
            err, _, step, emb, lab = sess.run([loss, train_op, global_step, embeddings, labels_batch], feed_dict=feed_dict)
            emb_array[lab,:] = emb
            loss_array[i] = err
            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
                  (epoch, batch_number+1, args.epoch_size, duration, err))
            batch_number += 1
            i += 1
            train_time += duration
            summary.value.add(tag='loss', simple_value=err)
            
        # Add validation loss and accuracy to summary
        #pylint: disable=maybe-no-member
        summary.value.add(tag='time/selection', simple_value=selection_time)
        summary_writer.add_summary(summary, step)
        if batch_number%20==0:
            save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)
    return step
  
def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    """ Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []
    
    # VGG Face: Choosing good triplets is crucial and should strike a balance between
    #  selecting informative (i.e. challenging) examples and swamping training with examples that
    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    #  the image n at random, but only between the ones that violate the triplet loss margin. The
    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    #  choosing the maximally violating example, as often done in structured output learning.
    # people_per_batch=2 尽在头两个类， 有超过两张图片的类被选中
    for i in xrange(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in xrange(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in xrange(j, nrof_images): # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                #all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                all_neg = np.where(neg_dists_sqr-pos_dist_sqr<alpha)[0] # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    #print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' % 
                    #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    trip_idx += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)

def sample_people(dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person
  
    # Sample classes from the dataset
    nrof_classes = len(dataset)    # class是人数
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)    # 样本进行打乱
    
    i = 0
    # 保存选中的人脸的链表
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths)<nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)  # 将同一个人的图片混排
        # （人时机的照片数，设定每个人的图像数，剩余能保存图片的空间数）
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images-len(image_paths))
        # 从当前选中的人（类别）提取相应张数的图片索引
        idx = image_indices[0:nrof_images_from_class]
        # 根据索引提取图片
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index]*nrof_images_from_class
        # 保存选中图片的路径
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i+=1
  
    return image_paths, num_per_class

def evaluate(sess, image_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder, 
        batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, actual_issame, batch_size, 
        nrof_folds, log_dir, step, summary_writer, embedding_size):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Running forward pass on LFW images: ', end='')
    
    nrof_images = len(actual_issame)*2
    assert(len(image_paths)==nrof_images)
    labels_array = np.reshape(np.arange(nrof_images),(-1,3))
    image_paths_array = np.reshape(np.expand_dims(np.array(image_paths),1), (-1,3))
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    emb_array = np.zeros((nrof_images, embedding_size))
    nrof_batches = int(np.ceil(nrof_images / batch_size))
    label_check_array = np.zeros((nrof_images,))
    for i in xrange(nrof_batches):
        batch_size = min(nrof_images-i*batch_size, batch_size)
        emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
            learning_rate_placeholder: 0.0, phase_train_placeholder: False})
        emb_array[lab,:] = emb
        label_check_array[lab] = 1
    print('%.3f' % (time.time()-start_time))
    
    assert(np.all(label_check_array==1))
    
    _, _, accuracy, val, val_std, far = lfw.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)
    
    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir,'lfw_result.txt'),'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))

def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)
  
  
def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='~/models/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--people_per_batch', type=int,
        help='Number of people per batch.', default=45)
    parser.add_argument('--images_per_person', type=int,
        help='Number of images per person.', default=40)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--alpha', type=float,
        help='Positive to negative triplet distance margin.', default=0.2)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')

    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='')
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)    
    
#4.人脸识别应用*****************************************************************************************
def face_recognition():
    image_size = 160 #don't need equal to real image size, but this value should not small than this
    modeldir = 'src/models/facenet/20170512-110547' #change to your model dir
    image_name1 = './img/2015_0.jpeg' #change to your image name
    image_name2 = './img/2_0.jpg' #change to your image name
    #1.换原始的数据验证
    
    print('建立facenet embedding模型')
    #tf.Graph().as_default()
    #sess = tf.Session()
    with tf.Graph().as_default():
        with tf.Session() as sess:
            #src.facenet.load_model(modelpath)
            facenet.load_model(modeldir)
            #2.解释graph与load_model的关系、META/CPKT、placeholder
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            #3.解释变量名
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            print('facenet embedding模型建立完毕')
            
            scaled_reshape = []
            
            image1 = imread(image_name1, mode='RGB')
            #4.知道每一步的矩阵大小和意义
            print (image1.shape)
            #5.解释插值
            image1 = cv2.resize(image1, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
            #6.归一化
            image1 = facenet.prewhiten(image1)
            scaled_reshape.append(image1.reshape(-1,image_size,image_size,3))
            #7.embedding解释
            emb_array1 = np.zeros((1, embedding_size))
            #8.解释run的对应关系
            emb_array1[0, :] = sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[0], phase_train_placeholder: False })[0]
            
            image2 = imread(image_name2, mode='RGB')
            image2 = cv2.resize(image2, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
            image2 = facenet.prewhiten(image2)
            scaled_reshape.append(image2.reshape(-1,image_size,image_size,3))
            emb_array2 = np.zeros((1, embedding_size))
            emb_array2[0, :] = sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[1], phase_train_placeholder: False })[0]
            
            dist = np.sqrt(np.sum(np.square(emb_array1[0]-emb_array2[0])))
            print("128维特征向量的欧氏距离：%f "%dist)

if __name__ == "__main__":
    #     # 1.人脸检测对齐训练
    #
    # #2.人脸对齐应用
    #
    # face_reconition = face_reconition()
    #
    # images_path = './img'
    # #模型地址
    # modelpath = 'src/models/facenet/20170512-110547'
    # out_path = './img/pic.json'
    # face_reconition.images_to_vectors(inpath=images_path, outjson_path=out_path, modelpath=modelpath)

    # #3.人脸识别训练
    argv = ['src/train_tripletloss.py',
                '--logs_base_dir', 'logs',
                '--models_base_dir', 'models',
                '--data_dir', 'data\lfw',
                '--model_def', 'models.inception_resnet_v1',
                '--pretrained_model', '',
                '--embedding_size', '512',
                '--epoch_size', '1000',
                '--max_nrof_epochs', '100',
                '--batch_size', '6',
                '--people_per_batch', '2',
                '--images_per_person', '3',
                '--lfw_pairs', 'data\pairs.txt',
                '--lfw_dir', 'data\lfw',
                '--lfw_nrof_folds', '2',
                '--optimizer','ADAM']
    train_argv = parse_arguments(argv[1:])
    face_train(train_argv)
    #4.人脸识别应用
    #face_recogniion()



