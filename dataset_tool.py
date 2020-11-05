# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import sys
import glob
import argparse
import threading
import six.moves.queue as Queue
import traceback
import numpy as np
import tensorflow as tf
import PIL.Image

from uv_gan import tfutil
from uv_gan import dataset
import menpo.io as mio
import scipy

#----------------------------------------------------------------------------

def error(msg):
    """
    Print an error message

    Args:
        msg: (str): write your description
    """
    print('Error: ' + msg)
    exit(1)

#----------------------------------------------------------------------------

class TFRecordExporter:
    def __init__(self, tfrecord_dir, expected_images, print_progress=True, progress_interval=10):
        """
        Initialize the tf.

        Args:
            self: (todo): write your description
            tfrecord_dir: (str): write your description
            expected_images: (int): write your description
            print_progress: (bool): write your description
            progress_interval: (int): write your description
        """
        self.tfrecord_dir       = tfrecord_dir
        self.tfr_prefix         = os.path.join(self.tfrecord_dir, os.path.basename(self.tfrecord_dir))
        self.expected_images    = expected_images
        self.cur_images         = 0
        self.shape              = None
        self.resolution_log2    = None
        self.tfr_writers        = []
        self.print_progress     = print_progress
        self.progress_interval  = progress_interval
        if self.print_progress:
            print('Creating dataset "%s"' % tfrecord_dir)
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert(os.path.isdir(self.tfrecord_dir))
        
    def close(self):
        """
        Close the writer.

        Args:
            self: (todo): write your description
        """
        if self.print_progress:
            print('%-40s\r' % 'Flushing data...', end='', flush=True)
        for tfr_writer in self.tfr_writers:
            tfr_writer.close()
        self.tfr_writers = []
        if self.print_progress:
            print('%-40s\r' % '', end='', flush=True)
            print('Added %d images.' % self.cur_images)

    def choose_shuffled_order(self): # Note: Images and labels must be added in shuffled order.
        """
        Chooses a random image.

        Args:
            self: (todo): write your description
        """
        order = np.arange(self.expected_images)
        np.random.RandomState(123).shuffle(order)
        return order

    def add_image(self, img):
        """
        Add image to image.

        Args:
            self: (todo): write your description
            img: (array): write your description
        """
        if self.print_progress and self.cur_images % self.progress_interval == 0:
            print('%d / %d\r' % (self.cur_images, self.expected_images), end='', flush=True)
        if self.shape is None:
            self.shape = img.shape
            self.resolution_log2 = int(np.log2(self.shape[1]))
            assert self.shape[0] in [1, 3]
            assert self.shape[1] == self.shape[2]
            assert self.shape[1] == 2**self.resolution_log2
            tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
            for lod in range(self.resolution_log2 - 1):
                tfr_file = self.tfr_prefix + '-r%02d.tfrecords' % (self.resolution_log2 - lod)
                self.tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))
        assert img.shape == self.shape
        for lod, tfr_writer in enumerate(self.tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[:, 0::2, 0::2] + img[:, 0::2, 1::2] + img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) * 0.25
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1

    def add_shape(self, img):
        """
        Add shape to image.

        Args:
            self: (todo): write your description
            img: (array): write your description
        """
        if self.print_progress and self.cur_images % self.progress_interval == 0:
            print('%d / %d\r' % (self.cur_images, self.expected_images), end='', flush=True)
        if self.shape is None:
            self.shape = img.shape
            self.resolution_log2 = int(np.log2(self.shape[1]))
            assert self.shape[0] in [1, 3]
            assert self.shape[1] == self.shape[2]
            assert self.shape[1] == 2**self.resolution_log2
            tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
            for lod in range(self.resolution_log2 - 1):
                tfr_file = self.tfr_prefix + '-r%02d.tfrecords' % (self.resolution_log2 - lod)
                self.tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))
        assert img.shape == self.shape
        for lod, tfr_writer in enumerate(self.tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[:, 0::2, 0::2] + img[:, 0::2, 1::2] + img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) * 0.25
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=img.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()]))}))
            tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1

    def add_both(self, img):
        """
        Add adversarial image.

        Args:
            self: (todo): write your description
            img: (array): write your description
        """
        if self.print_progress and self.cur_images % self.progress_interval == 0:
            print('%d / %d\r' % (self.cur_images, self.expected_images), end='', flush=True)
        if self.shape is None:
            self.shape = img.shape
            self.resolution_log2 = int(np.log2(self.shape[1]))
            assert self.shape[0] in [1, 6, 9]
            assert self.shape[1] == self.shape[2]
            assert self.shape[1] == 2**self.resolution_log2
            tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
            for lod in range(self.resolution_log2 - 1):
                tfr_file = self.tfr_prefix + '-r%02d.tfrecords' % (self.resolution_log2 - lod)
                self.tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))
        assert img.shape == self.shape
        for lod, tfr_writer in enumerate(self.tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[:, 0::2, 0::2] + img[:, 0::2, 1::2] + img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) * 0.25
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=img.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()]))}))
            tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1

    def add_labels(self, labels):
        """
        Add image labels.

        Args:
            self: (todo): write your description
            labels: (array): write your description
        """
        if self.print_progress:
            print('%-40s\r' % 'Saving labels...', end='', flush=True)
        assert labels.shape[0] == self.cur_images
        with open(self.tfr_prefix + '-rxx.labels', 'wb') as f:
            np.save(f, labels.astype(np.float32))
            
    def __enter__(self):
        """
        Decor function.

        Args:
            self: (todo): write your description
        """
        return self
    
    def __exit__(self, *args):
        """
        Exit the exit.

        Args:
            self: (todo): write your description
        """
        self.close()

#----------------------------------------------------------------------------

class ExceptionInfo(object):
    def __init__(self):
        """
        Initialize the traceback.

        Args:
            self: (todo): write your description
        """
        self.value = sys.exc_info()[1]
        self.traceback = traceback.format_exc()

#----------------------------------------------------------------------------

class WorkerThread(threading.Thread):
    def __init__(self, task_queue):
        """
        Initialize task queue.

        Args:
            self: (todo): write your description
            task_queue: (todo): write your description
        """
        threading.Thread.__init__(self)
        self.task_queue = task_queue

    def run(self):
        """
        Run the task. task.

        Args:
            self: (todo): write your description
        """
        while True:
            func, args, result_queue = self.task_queue.get()
            if func is None:
                break
            try:
                result = func(*args)
            except:
                result = ExceptionInfo()
            result_queue.put((result, args))

#----------------------------------------------------------------------------

class ThreadPool(object):
    def __init__(self, num_threads):
        """
        Initialize the threads.

        Args:
            self: (todo): write your description
            num_threads: (int): write your description
        """
        assert num_threads >= 1
        self.task_queue = Queue.Queue()
        self.result_queues = dict()
        self.num_threads = num_threads
        for idx in range(self.num_threads):
            thread = WorkerThread(self.task_queue)
            thread.daemon = True
            thread.start()

    def add_task(self, func, args=()):
        """
        Add a task to the queue.

        Args:
            self: (todo): write your description
            func: (todo): write your description
        """
        assert hasattr(func, '__call__') # must be a function
        if func not in self.result_queues:
            self.result_queues[func] = Queue.Queue()
        self.task_queue.put((func, args, self.result_queues[func]))

    def get_result(self, func): # returns (result, args)
        """
        Return the result of a function.

        Args:
            self: (todo): write your description
            func: (todo): write your description
        """
        result, args = self.result_queues[func].get()
        if isinstance(result, ExceptionInfo):
            print('\n\nWorker thread caught an exception:\n' + result.traceback)
            raise result.value
        return result, args

    def finish(self):
        """
        Finishes the queue.

        Args:
            self: (todo): write your description
        """
        for idx in range(self.num_threads):
            self.task_queue.put((None, (), None))

    def __enter__(self): # for 'with' statement
        """
        Decor function.

        Args:
            self: (todo): write your description
        """
        return self

    def __exit__(self, *excinfo):
        """
        Finishes the given exception.

        Args:
            self: (todo): write your description
            excinfo: (todo): write your description
        """
        self.finish()

    def process_items_concurrently(self, item_iterator, process_func=lambda x: x, pre_func=lambda x: x, post_func=lambda x: x, max_items_in_flight=None):
        """
        Process items from a list.

        Args:
            self: (todo): write your description
            item_iterator: (todo): write your description
            process_func: (todo): write your description
            x: (todo): write your description
            x: (todo): write your description
            pre_func: (todo): write your description
            x: (todo): write your description
            x: (todo): write your description
            post_func: (todo): write your description
            x: (todo): write your description
            x: (todo): write your description
            max_items_in_flight: (int): write your description
        """
        if max_items_in_flight is None: max_items_in_flight = self.num_threads * 4
        assert max_items_in_flight >= 1
        results = []
        retire_idx = [0]

        def task_func(prepared, idx):
            """
            Return a function for a prepared.

            Args:
                prepared: (todo): write your description
                idx: (str): write your description
            """
            return process_func(prepared)
           
        def retire_result():
            """
            Retire the result of a result.

            Args:
            """
            processed, (prepared, idx) = self.get_result(task_func)
            results[idx] = processed
            while retire_idx[0] < len(results) and results[retire_idx[0]] is not None:
                yield post_func(results[retire_idx[0]])
                results[retire_idx[0]] = None
                retire_idx[0] += 1
    
        for idx, item in enumerate(item_iterator):
            prepared = pre_func(item)
            results.append(None)
            self.add_task(func=task_func, args=(prepared, idx))
            while retire_idx[0] < idx - max_items_in_flight + 2:
                for res in retire_result(): yield res
        while retire_idx[0] < len(results):
            for res in retire_result(): yield res

#----------------------------------------------------------------------------

def display(tfrecord_dir):
    """
    Display tfrecord information.

    Args:
        tfrecord_dir: (str): write your description
    """
    print('Loading dataset "%s"' % tfrecord_dir)
    tfutil.init_tf({'gpu_options.allow_growth': True})
    dset = dataset.TFRecordDataset(tfrecord_dir, max_label_size='full', repeat=False, shuffle_mb=0)
    tfutil.init_uninited_vars()
    
    idx = 0
    while True:
        try:
            images, labels = dset.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            break
        if idx == 0:
            print('Displaying images')
            import cv2 # pip install opencv-python
            cv2.namedWindow('dataset_tool')
            print('Press SPACE or ENTER to advance, ESC to exit')
        print('\nidx = %-8d\nlabel = %s' % (idx, labels[0].tolist()))
        cv2.imshow('dataset_tool', images[0].transpose(1, 2, 0)[:, :, ::-1]) # CHW => HWC, RGB => BGR
        idx += 1
        if cv2.waitKey() == 27:
            break
    print('\nDisplayed %d images.' % idx)

#----------------------------------------------------------------------------

def extract(tfrecord_dir, output_dir):
    """
    Extract tf.

    Args:
        tfrecord_dir: (str): write your description
        output_dir: (str): write your description
    """
    print('Loading dataset "%s"' % tfrecord_dir)
    tfutil.init_tf({'gpu_options.allow_growth': True})
    dset = dataset.TFRecordDataset(tfrecord_dir, max_label_size=0, repeat=False, shuffle_mb=0)
    tfutil.init_uninited_vars()
    
    print('Extracting images to "%s"' % output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    idx = 0
    while True:
        if idx % 10 == 0:
            print('%d\r' % idx, end='', flush=True)
        try:
            images, labels = dset.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            break
        if images.shape[1] == 1:
            img = PIL.Image.fromarray(images[0][0], 'L')
        else:
            img = PIL.Image.fromarray(images[0].transpose(1, 2, 0), 'RGB')
        img.save(os.path.join(output_dir, 'img%08d.png' % idx))
        idx += 1
    print('Extracted %d images.' % idx)

#----------------------------------------------------------------------------

def compare(tfrecord_dir_a, tfrecord_dir_b, ignore_labels):
    """
    Compares two images.

    Args:
        tfrecord_dir_a: (str): write your description
        tfrecord_dir_b: (str): write your description
        ignore_labels: (bool): write your description
    """
    max_label_size = 0 if ignore_labels else 'full'
    print('Loading dataset "%s"' % tfrecord_dir_a)
    tfutil.init_tf({'gpu_options.allow_growth': True})
    dset_a = dataset.TFRecordDataset(tfrecord_dir_a, max_label_size=max_label_size, repeat=False, shuffle_mb=0)
    print('Loading dataset "%s"' % tfrecord_dir_b)
    dset_b = dataset.TFRecordDataset(tfrecord_dir_b, max_label_size=max_label_size, repeat=False, shuffle_mb=0)
    tfutil.init_uninited_vars()
    
    print('Comparing datasets')
    idx = 0
    identical_images = 0
    identical_labels = 0
    while True:
        if idx % 100 == 0:
            print('%d\r' % idx, end='', flush=True)
        try:
            images_a, labels_a = dset_a.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            images_a, labels_a = None, None
        try:
            images_b, labels_b = dset_b.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            images_b, labels_b = None, None
        if images_a is None or images_b is None:
            if images_a is not None or images_b is not None:
                print('Datasets contain different number of images')
            break
        if images_a.shape == images_b.shape and np.all(images_a == images_b):
            identical_images += 1
        else:
            print('Image %d is different' % idx)
        if labels_a.shape == labels_b.shape and np.all(labels_a == labels_b):
            identical_labels += 1
        else:
            print('Label %d is different' % idx)
        idx += 1
    print('Identical images: %d / %d' % (identical_images, idx))
    if not ignore_labels:
        print('Identical labels: %d / %d' % (identical_labels, idx))

#----------------------------------------------------------------------------

def create_mnist(tfrecord_dir, mnist_dir):
    """
    Create mnist dataset.

    Args:
        tfrecord_dir: (str): write your description
        mnist_dir: (str): write your description
    """
    print('Loading MNIST from "%s"' % mnist_dir)
    import gzip
    with gzip.open(os.path.join(mnist_dir, 'train-images-idx3-ubyte.gz'), 'rb') as file:
        images = np.frombuffer(file.read(), np.uint8, offset=16)
    with gzip.open(os.path.join(mnist_dir, 'train-labels-idx1-ubyte.gz'), 'rb') as file:
        labels = np.frombuffer(file.read(), np.uint8, offset=8)
    images = images.reshape(-1, 1, 28, 28)
    images = np.pad(images, [(0,0), (0,0), (2,2), (2,2)], 'constant', constant_values=0)
    assert images.shape == (60000, 1, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (60000,) and labels.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9
    onehot = np.zeros((labels.size, np.max(labels) + 1), dtype=np.float32)
    onehot[np.arange(labels.size), labels] = 1.0
    
    with TFRecordExporter(tfrecord_dir, images.shape[0]) as tfr:
        order = tfr.choose_shuffled_order()
        for idx in range(order.size):
            tfr.add_image(images[order[idx]])
        tfr.add_labels(onehot[order])

#----------------------------------------------------------------------------

def create_mnistrgb(tfrecord_dir, mnist_dir, num_images=1000000, random_seed=123):
    """
    Create adversarial images.

    Args:
        tfrecord_dir: (str): write your description
        mnist_dir: (str): write your description
        num_images: (int): write your description
        random_seed: (int): write your description
    """
    print('Loading MNIST from "%s"' % mnist_dir)
    import gzip
    with gzip.open(os.path.join(mnist_dir, 'train-images-idx3-ubyte.gz'), 'rb') as file:
        images = np.frombuffer(file.read(), np.uint8, offset=16)
    images = images.reshape(-1, 28, 28)
    images = np.pad(images, [(0,0), (2,2), (2,2)], 'constant', constant_values=0)
    assert images.shape == (60000, 32, 32) and images.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    
    with TFRecordExporter(tfrecord_dir, num_images) as tfr:
        rnd = np.random.RandomState(random_seed)
        for idx in range(num_images):
            tfr.add_image(images[rnd.randint(images.shape[0], size=3)])

#----------------------------------------------------------------------------

def create_cifar10(tfrecord_dir, cifar10_dir):
    """
    Create cifar10 dataset.

    Args:
        tfrecord_dir: (str): write your description
        cifar10_dir: (str): write your description
    """
    print('Loading CIFAR-10 from "%s"' % cifar10_dir)
    import pickle
    images = []
    labels = []
    for batch in range(1, 6):
        with open(os.path.join(cifar10_dir, 'data_batch_%d' % batch), 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        images.append(data['data'].reshape(-1, 3, 32, 32))
        labels.append(data['labels'])
    images = np.concatenate(images)
    labels = np.concatenate(labels)
    assert images.shape == (50000, 3, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (50000,) and labels.dtype == np.int32
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9
    onehot = np.zeros((labels.size, np.max(labels) + 1), dtype=np.float32)
    onehot[np.arange(labels.size), labels] = 1.0

    with TFRecordExporter(tfrecord_dir, images.shape[0]) as tfr:
        order = tfr.choose_shuffled_order()
        for idx in range(order.size):
            tfr.add_image(images[order[idx]])
        tfr.add_labels(onehot[order])

#----------------------------------------------------------------------------

def create_cifar100(tfrecord_dir, cifar100_dir):
    """
    Create cifar100.

    Args:
        tfrecord_dir: (str): write your description
        cifar100_dir: (str): write your description
    """
    print('Loading CIFAR-100 from "%s"' % cifar100_dir)
    import pickle
    with open(os.path.join(cifar100_dir, 'train'), 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    images = data['data'].reshape(-1, 3, 32, 32)
    labels = np.array(data['fine_labels'])
    assert images.shape == (50000, 3, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (50000,) and labels.dtype == np.int32
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 99
    onehot = np.zeros((labels.size, np.max(labels) + 1), dtype=np.float32)
    onehot[np.arange(labels.size), labels] = 1.0

    with TFRecordExporter(tfrecord_dir, images.shape[0]) as tfr:
        order = tfr.choose_shuffled_order()
        for idx in range(order.size):
            tfr.add_image(images[order[idx]])
        tfr.add_labels(onehot[order])

#----------------------------------------------------------------------------

def create_svhn(tfrecord_dir, svhn_dir):
    """
    Create tfrecord.

    Args:
        tfrecord_dir: (str): write your description
        svhn_dir: (str): write your description
    """
    print('Loading SVHN from "%s"' % svhn_dir)
    import pickle
    images = []
    labels = []
    for batch in range(1, 4):
        with open(os.path.join(svhn_dir, 'train_%d.pkl' % batch), 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        images.append(data[0])
        labels.append(data[1])
    images = np.concatenate(images)
    labels = np.concatenate(labels)
    assert images.shape == (73257, 3, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (73257,) and labels.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9
    onehot = np.zeros((labels.size, np.max(labels) + 1), dtype=np.float32)
    onehot[np.arange(labels.size), labels] = 1.0

    with TFRecordExporter(tfrecord_dir, images.shape[0]) as tfr:
        order = tfr.choose_shuffled_order()
        for idx in range(order.size):
            tfr.add_image(images[order[idx]])
        tfr.add_labels(onehot[order])

#----------------------------------------------------------------------------

def create_lsun(tfrecord_dir, lmdb_dir, resolution=256, max_images=None):
    """
    Create images from png file.

    Args:
        tfrecord_dir: (str): write your description
        lmdb_dir: (str): write your description
        resolution: (str): write your description
        max_images: (int): write your description
    """
    print('Loading LSUN dataset from "%s"' % lmdb_dir)
    import lmdb # pip install lmdb
    import cv2 # pip install opencv-python
    import io
    with lmdb.open(lmdb_dir, readonly=True).begin(write=False) as txn:
        total_images = txn.stat()['entries']
        if max_images is None:
            max_images = total_images
        with TFRecordExporter(tfrecord_dir, max_images) as tfr:
            for idx, (key, value) in enumerate(txn.cursor()):
                try:
                    try:
                        img = cv2.imdecode(np.fromstring(value, dtype=np.uint8), 1)
                        if img is None:
                            raise IOError('cv2.imdecode failed')
                        img = img[:, :, ::-1] # BGR => RGB
                    except IOError:
                        img = np.asarray(PIL.Image.open(io.BytesIO(value)))
                    crop = np.min(img.shape[:2])
                    img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
                    img = PIL.Image.fromarray(img, 'RGB')
                    img = img.resize((resolution, resolution), PIL.Image.ANTIALIAS)
                    img = np.asarray(img)
                    img = img.transpose(2, 0, 1) # HWC => CHW
                    tfr.add_image(img)
                except:
                    print(sys.exc_info()[1])
                if tfr.cur_images == max_images:
                    break
        
#----------------------------------------------------------------------------

def create_celeba(tfrecord_dir, celeba_dir, cx=89, cy=121):
    """
    Create a png file.

    Args:
        tfrecord_dir: (str): write your description
        celeba_dir: (str): write your description
        cx: (str): write your description
        cy: (str): write your description
    """
    print('Loading CelebA from "%s"' % celeba_dir)
    glob_pattern = os.path.join(celeba_dir, 'img_align_celeba_png', '*.png')
    image_filenames = sorted(glob.glob(glob_pattern))
    expected_images = 202599
    if len(image_filenames) != expected_images:
        error('Expected to find %d images' % expected_images)
    
    with TFRecordExporter(tfrecord_dir, len(image_filenames)) as tfr:
        order = tfr.choose_shuffled_order()
        for idx in range(order.size):
            img = np.asarray(PIL.Image.open(image_filenames[order[idx]]))
            assert img.shape == (218, 178, 3)
            img = img[cy - 64 : cy + 64, cx - 64 : cx + 64]
            img = img.transpose(2, 0, 1) # HWC => CHW
            tfr.add_image(img)

#----------------------------------------------------------------------------

def create_celebahq(tfrecord_dir, celeba_dir, delta_dir, num_threads=4, num_tasks=100):
    """
    Create tfrecord for a tfrecord.

    Args:
        tfrecord_dir: (str): write your description
        celeba_dir: (str): write your description
        delta_dir: (str): write your description
        num_threads: (int): write your description
        num_tasks: (int): write your description
    """
    print('Loading CelebA from "%s"' % celeba_dir)
    expected_images = 202599
    if len(glob.glob(os.path.join(celeba_dir, 'img_celeba', '*.jpg'))) != expected_images:
        error('Expected to find %d images' % expected_images)
    with open(os.path.join(celeba_dir, 'Anno', 'list_landmarks_celeba.txt'), 'rt') as file:
        landmarks = [[float(value) for value in line.split()[1:]] for line in file.readlines()[2:]]
        landmarks = np.float32(landmarks).reshape(-1, 5, 2)
    
    print('Loading CelebA-HQ deltas from "%s"' % delta_dir)
    import scipy.ndimage
    import hashlib
    import bz2
    import zipfile
    import base64
    import cryptography.hazmat.primitives.hashes
    import cryptography.hazmat.backends
    import cryptography.hazmat.primitives.kdf.pbkdf2
    import cryptography.fernet
    expected_zips = 30
    if len(glob.glob(os.path.join(delta_dir, 'delta*.zip'))) != expected_zips:
        error('Expected to find %d zips' % expected_zips)
    with open(os.path.join(delta_dir, 'image_list.txt'), 'rt') as file:
        lines = [line.split() for line in file]
        fields = dict()
        for idx, field in enumerate(lines[0]):
            type = int if field.endswith('idx') else str
            fields[field] = [type(line[idx]) for line in lines[1:]]
    indices = np.array(fields['idx'])

    # Must use pillow version 3.1.1 for everything to work correctly.
    if getattr(PIL, 'PILLOW_VERSION', '') != '3.1.1':
        error('create_celebahq requires pillow version 3.1.1') # conda install pillow=3.1.1
        
    # Must use libjpeg version 8d for everything to work correctly.
    img = np.array(PIL.Image.open(os.path.join(celeba_dir, 'img_celeba', '000001.jpg')))
    md5 = hashlib.md5()
    md5.update(img.tobytes())
    if md5.hexdigest() != '9cad8178d6cb0196b36f7b34bc5eb6d3':
        error('create_celebahq requires libjpeg version 8d') # conda install jpeg=8d

    def rot90(v):
        """
        Rotate the given vector.

        Args:
            v: (array): write your description
        """
        return np.array([-v[1], v[0]])

    def process_func(idx):
        """
        Process a single image

        Args:
            idx: (int): write your description
        """
        # Load original image.
        orig_idx = fields['orig_idx'][idx]
        orig_file = fields['orig_file'][idx]
        orig_path = os.path.join(celeba_dir, 'img_celeba', orig_file)
        img = PIL.Image.open(orig_path)

        # Choose oriented crop rectangle.
        lm = landmarks[orig_idx]
        eye_avg = (lm[0] + lm[1]) * 0.5 + 0.5
        mouth_avg = (lm[3] + lm[4]) * 0.5 + 0.5
        eye_to_eye = lm[1] - lm[0]
        eye_to_mouth = mouth_avg - eye_avg
        x = eye_to_eye - rot90(eye_to_mouth)
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = rot90(x)
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        zoom = 1024 / (np.hypot(*x) * 2)

        # Shrink.
        shrink = int(np.floor(0.5 / zoom))
        if shrink > 1:
            size = (int(np.round(float(img.size[0]) / shrink)), int(np.round(float(img.size[1]) / shrink)))
            img = img.resize(size, PIL.Image.ANTIALIAS)
            quad /= shrink
            zoom *= shrink

        # Crop.
        border = max(int(np.round(1024 * 0.1 / zoom)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Simulate super-resolution.
        superres = int(np.exp2(np.ceil(np.log2(zoom))))
        if superres > 1:
            img = img.resize((img.size[0] * superres, img.size[1] * superres), PIL.Image.ANTIALIAS)
            quad *= superres
            zoom /= superres

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if max(pad) > border - 4:
            pad = np.maximum(pad, int(np.round(1024 * 0.3 / zoom)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.mgrid[:h, :w, :1]
            mask = 1.0 - np.minimum(np.minimum(np.float32(x) / pad[0], np.float32(y) / pad[1]), np.minimum(np.float32(w-1-x) / pad[2], np.float32(h-1-y) / pad[3]))
            blur = 1024 * 0.02 / zoom
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.round(img), 0, 255)), 'RGB')
            quad += pad[0:2]
            
        # Transform.
        img = img.transform((4096, 4096), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        img = img.resize((1024, 1024), PIL.Image.ANTIALIAS)
        img = np.asarray(img).transpose(2, 0, 1)
        
        # Verify MD5.
        md5 = hashlib.md5()
        md5.update(img.tobytes())
        assert md5.hexdigest() == fields['proc_md5'][idx]
        
        # Load delta image and original JPG.
        with zipfile.ZipFile(os.path.join(delta_dir, 'deltas%05d.zip' % (idx - idx % 1000)), 'r') as zip:
            delta_bytes = zip.read('delta%05d.dat' % idx)
        with open(orig_path, 'rb') as file:
            orig_bytes = file.read()
        
        # Decrypt delta image, using original JPG data as decryption key.
        algorithm = cryptography.hazmat.primitives.hashes.SHA256()
        backend = cryptography.hazmat.backends.default_backend()
        salt = bytes(orig_file, 'ascii')
        kdf = cryptography.hazmat.primitives.kdf.pbkdf2.PBKDF2HMAC(algorithm=algorithm, length=32, salt=salt, iterations=100000, backend=backend)
        key = base64.urlsafe_b64encode(kdf.derive(orig_bytes))
        delta = np.frombuffer(bz2.decompress(cryptography.fernet.Fernet(key).decrypt(delta_bytes)), dtype=np.uint8).reshape(3, 1024, 1024)
        
        # Apply delta image.
        img = img + delta
        
        # Verify MD5.
        md5 = hashlib.md5()
        md5.update(img.tobytes())
        assert md5.hexdigest() == fields['final_md5'][idx]
        return img

    with TFRecordExporter(tfrecord_dir, indices.size) as tfr:
        order = tfr.choose_shuffled_order()
        with ThreadPool(num_threads) as pool:
            for img in pool.process_items_concurrently(indices[order].tolist(), process_func=process_func, max_items_in_flight=num_tasks):
                tfr.add_image(img)

#----------------------------------------------------------------------------

def create_from_images(tfrecord_dir, image_dir, shuffle):
    """
    Create a tf.

    Args:
        tfrecord_dir: (str): write your description
        image_dir: (str): write your description
        shuffle: (bool): write your description
    """
    print('Loading images from "%s"' % image_dir)
    image_filenames = sorted(glob.glob(os.path.join(image_dir, '*')))
    if len(image_filenames) == 0:
        error('No input images found')

    good_ids =  mio.import_pickle('/vol/construct3dmm/visualizations/nicp/mein3d/good_ids.pkl')

    img = np.asarray(PIL.Image.open(image_filenames[0]))
    resolution = img.shape[0]
    channels = img.shape[2] if img.ndim == 3 else 1
    # if img.shape[1] != resolution:
    #     error('Input images must have the same width and height')
    # if resolution != 2 ** int(np.floor(np.log2(resolution))):
    #     error('Input image resolution must be a power-of-two')
    if channels not in [1, 3]:
        error('Input images must be stored as RGB or grayscale')
    
    with TFRecordExporter(tfrecord_dir, len(image_filenames)) as tfr:
        order = tfr.choose_shuffled_order() if shuffle else np.arange(len(image_filenames))
        for idx in range(order.size):
                img = PIL.Image.open(image_filenames[order[idx]])
                # img.crop((41, 0, img.size[0]-42, resolution))
                # new_im = PIL.Image.new("RGB", (512, 512),(0, 0, 0))
                # new_im.paste(img, ((512 - img.size[0]) // 2,(512 - img.size[1]) // 2))
                img = np.asarray(img)
                if channels == 1:
                    img = img[np.newaxis, :, :] # HW => CHW
                else:
                    img = img.transpose(2, 0, 1) # HWC => CHW
                tfr.add_image(img)

#----------------------------------------------------------------------------

def create_from_pkl(tfrecord_dir, image_dir, shuffle):
    """
    Create a tf.

    Args:
        tfrecord_dir: (str): write your description
        image_dir: (str): write your description
        shuffle: (bool): write your description
    """
    print('Loading images from "%s"' % image_dir)
    image_filenames = sorted(glob.glob(os.path.join(image_dir, '*')))
    if len(image_filenames) == 0:
        error('No input images found')

    # good_ids =  mio.import_pickle('/vol/construct3dmm/visualizations/nicp/mein3d/good_ids.pkl')

    img = mio.import_pickle(image_filenames[0])
    resolution = img.shape[2]
    channels = img.shape[0] if img.ndim == 3 else 1
    if img.shape[1] != resolution:
        error('Input images must have the same width and height')
    if resolution != 2 ** int(np.floor(np.log2(resolution))):
        error('Input image resolution must be a power-of-two')
    if channels not in [1, 3]:
        error('Input images must be stored as RGB or grayscale')

    with TFRecordExporter(tfrecord_dir, len(image_filenames)) as tfr:
        order = tfr.choose_shuffled_order() if shuffle else np.arange(len(image_filenames))
        for idx in range(order.size):
            img = mio.import_pickle(image_filenames[order[idx]]).astype(np.float32)

            # img[0, :, :] = scipy.ndimage.gaussian_filter(img[0, :, :], 2)
            # img[1, :, :] = scipy.ndimage.gaussian_filter(img[1, :, :], 2)
            # img[2, :, :] = scipy.ndimage.gaussian_filter(img[2, :, :], 2)
            # img_resized = np.stack((cv2.resize(img[0],dsize=(256,256)),cv2.resize(img[1],dsize=(256,256)),cv2.resize(img[2],dsize=(256,256))))
            tfr.add_shape(img)

#----------------------------------------------------------------------------

def create_from_pkl_img(tfrecord_dir, image_dir, pickle_dir, shuffle):
    """
    Create a png from pickle file.

    Args:
        tfrecord_dir: (str): write your description
        image_dir: (str): write your description
        pickle_dir: (str): write your description
        shuffle: (bool): write your description
    """
    print('Loading images from "%s"' % image_dir)

    image_filenames = sorted(glob.glob(os.path.join(image_dir, '*')))
    pickle_filenames = sorted(glob.glob(os.path.join(pickle_dir, '*')))
    if len(image_filenames) == 0:
        error('No input images found')

    # good_ids =  mio.import_pickle('/vol/construct3dmm/visualizations/nicp/mein3d/good_ids.pkl')

    img = mio.import_pickle(pickle_filenames[0])
    resolution = img.shape[2]
    channels = img.shape[0] if img.ndim == 3 else 1
    if img.shape[1] != resolution:
        error('Input images must have the same width and height')
    if resolution != 2 ** int(np.floor(np.log2(resolution))):
        error('Input image resolution must be a power-of-two')
    if channels not in [1, 3]:
        error('Input images must be stored as RGB or grayscale')

    with TFRecordExporter(tfrecord_dir, len(image_filenames)) as tfr:
        order = tfr.choose_shuffled_order() if shuffle else np.arange(len(image_filenames))
        for idx in range(order.size):
            img = mio.import_image(image_filenames[order[idx]]).pixels.astype(np.float32)*2-1
            pkl = mio.import_pickle(pickle_filenames[order[idx]]).astype(np.float32)
            # pkl[0, :, :] = scipy.ndimage.gaussian_filter(pkl[0, :, :], 2)
            # pkl[1, :, :] = scipy.ndimage.gaussian_filter(pkl[1, :, :], 2)
            # pkl[2, :, :] = scipy.ndimage.gaussian_filter(pkl[2, :, :], 2)
                # img_resized = np.stack((cv2.resize(img[0],dsize=(256,256)),cv2.resize(img[1],dsize=(256,256)),cv2.resize(img[2],dsize=(256,256))))
            tfr.add_both(np.concatenate([img,pkl]))

#----------------------------------------------------------------------------

def create_from_pkl_img_norm(tfrecord_dir, image_dir, pickle_dir, normal_dir, shuffle):
    """
    Create tf. pkl.

    Args:
        tfrecord_dir: (str): write your description
        image_dir: (str): write your description
        pickle_dir: (str): write your description
        normal_dir: (str): write your description
        shuffle: (bool): write your description
    """
    print('Loading images from "%s"' % image_dir)

    image_filenames = sorted(glob.glob(os.path.join(image_dir, '*')))
    pickle_filenames = sorted(glob.glob(os.path.join(pickle_dir, '*')))
    normal_filenames = sorted(glob.glob(os.path.join(normal_dir, '*')))
    if len(image_filenames) == 0:
        error('No input images found')

    # good_ids =  mio.import_pickle('/vol/construct3dmm/visualizations/nicp/mein3d/good_ids.pkl')

    img = mio.import_pickle(pickle_filenames[0])
    resolution = img.shape[2]
    channels = img.shape[0] if img.ndim == 3 else 1
    if img.shape[1] != resolution:
        error('Input images must have the same width and height')
    if resolution != 2 ** int(np.floor(np.log2(resolution))):
        error('Input image resolution must be a power-of-two')
    if channels not in [1, 3]:
        error('Input images must be stored as RGB or grayscale')

    with TFRecordExporter(tfrecord_dir, len(image_filenames)) as tfr:
        order = tfr.choose_shuffled_order() if shuffle else np.arange(len(image_filenames))
        for idx in range(order.size):
            img = mio.import_image(image_filenames[order[idx]]).pixels.astype(np.float32)*2-1
            pkl = mio.import_pickle(pickle_filenames[order[idx]]).astype(np.float32)
            normal = mio.import_pickle(normal_filenames[order[idx]]).astype(np.float32)
            # pkl[0, :, :] = scipy.ndimage.gaussian_filter(pkl[0, :, :], 2)
            # pkl[1, :, :] = scipy.ndimage.gaussian_filter(pkl[1, :, :], 2)
            # pkl[2, :, :] = scipy.ndimage.gaussian_filter(pkl[2, :, :], 2)
                # img_resized = np.stack((cv2.resize(img[0],dsize=(256,256)),cv2.resize(img[1],dsize=(256,256)),cv2.resize(img[2],dsize=(256,256))))
            tfr.add_both(np.concatenate([img,pkl,normal]))

#----------------------------------------------------------------------------

def create_from_hdf5(tfrecord_dir, hdf5_filename, shuffle):
    """
    Create a tf. hdf5 files.

    Args:
        tfrecord_dir: (str): write your description
        hdf5_filename: (str): write your description
        shuffle: (bool): write your description
    """
    print('Loading HDF5 archive from "%s"' % hdf5_filename)
    import h5py # conda install h5py
    with h5py.File(hdf5_filename, 'r') as hdf5_file:
        hdf5_data = max([value for key, value in hdf5_file.items() if key.startswith('data')], key=lambda lod: lod.shape[3])
        with TFRecordExporter(tfrecord_dir, hdf5_data.shape[0]) as tfr:
            order = tfr.choose_shuffled_order() if shuffle else np.arange(hdf5_data.shape[0])
            for idx in range(order.size):
                tfr.add_image(hdf5_data[order[idx]])
            npy_filename = os.path.splitext(hdf5_filename)[0] + '-labels.npy'
            if os.path.isfile(npy_filename):
                tfr.add_labels(np.load(npy_filename)[order])

#----------------------------------------------------------------------------

def execute_cmdline(argv):
    """
    Parse command line arguments.

    Args:
        argv: (list): write your description
    """
    prog = argv[0]
    parser = argparse.ArgumentParser(
        prog        = prog,
        description = 'Tool for creating, extracting, and visualizing Progressive GAN datasets.',
        epilog      = 'Type "%s <command> -h" for more information.' % prog)
        
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    def add_command(cmd, desc, example=None):
        """
        Add a sub - command.

        Args:
            cmd: (str): write your description
            desc: (str): write your description
            example: (list): write your description
        """
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p = add_command(    'display',          'Display images in dataset.',
                                            'display datasets/mnist')
    p.add_argument(     'tfrecord_dir',     help='Directory containing dataset')
  
    p = add_command(    'extract',          'Extract images from dataset.',
                                            'extract datasets/mnist mnist-images')
    p.add_argument(     'tfrecord_dir',     help='Directory containing dataset')
    p.add_argument(     'output_dir',       help='Directory to extract the images into')

    p = add_command(    'compare',          'Compare two datasets.',
                                            'compare datasets/mydataset datasets/mnist')
    p.add_argument(     'tfrecord_dir_a',   help='Directory containing first dataset')
    p.add_argument(     'tfrecord_dir_b',   help='Directory containing second dataset')
    p.add_argument(     '--ignore_labels',  help='Ignore labels (default: 0)', type=int, default=0)

    p = add_command(    'create_mnist',     'Create dataset for MNIST.',
                                            'create_mnist datasets/mnist ~/downloads/mnist')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'mnist_dir',        help='Directory containing MNIST')

    p = add_command(    'create_mnistrgb',  'Create dataset for MNIST-RGB.',
                                            'create_mnistrgb datasets/mnistrgb ~/downloads/mnist')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'mnist_dir',        help='Directory containing MNIST')
    p.add_argument(     '--num_images',     help='Number of composite images to create (default: 1000000)', type=int, default=1000000)
    p.add_argument(     '--random_seed',    help='Random seed (default: 123)', type=int, default=123)

    p = add_command(    'create_cifar10',   'Create dataset for CIFAR-10.',
                                            'create_cifar10 datasets/cifar10 ~/downloads/cifar10')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'cifar10_dir',      help='Directory containing CIFAR-10')

    p = add_command(    'create_cifar100',  'Create dataset for CIFAR-100.',
                                            'create_cifar100 datasets/cifar100 ~/downloads/cifar100')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'cifar100_dir',     help='Directory containing CIFAR-100')

    p = add_command(    'create_svhn',      'Create dataset for SVHN.',
                                            'create_svhn datasets/svhn ~/downloads/svhn')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'svhn_dir',         help='Directory containing SVHN')

    p = add_command(    'create_lsun',      'Create dataset for single LSUN category.',
                                            'create_lsun datasets/lsun-car-100k ~/downloads/lsun/car_lmdb --resolution 256 --max_images 100000')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'lmdb_dir',         help='Directory containing LMDB database')
    p.add_argument(     '--resolution',     help='Output resolution (default: 256)', type=int, default=256)
    p.add_argument(     '--max_images',     help='Maximum number of images (default: none)', type=int, default=None)

    p = add_command(    'create_celeba',    'Create dataset for CelebA.',
                                            'create_celeba datasets/celeba ~/downloads/celeba')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'celeba_dir',       help='Directory containing CelebA')
    p.add_argument(     '--cx',             help='Center X coordinate (default: 89)', type=int, default=89)
    p.add_argument(     '--cy',             help='Center Y coordinate (default: 121)', type=int, default=121)

    p = add_command(    'create_celebahq',  'Create dataset for CelebA-HQ.',
                                            'create_celebahq datasets/celebahq ~/downloads/celeba ~/downloads/celeba-hq-deltas')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'celeba_dir',       help='Directory containing CelebA')
    p.add_argument(     'delta_dir',        help='Directory containing CelebA-HQ deltas')
    p.add_argument(     '--num_threads',    help='Number of concurrent threads (default: 4)', type=int, default=4)
    p.add_argument(     '--num_tasks',      help='Number of concurrent processing tasks (default: 100)', type=int, default=100)

    p = add_command(    'create_from_images', 'Create dataset from a directory full of images.',
                                            'create_from_images datasets/mydataset myimagedir')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'image_dir',        help='Directory containing the images')
    p.add_argument(     '--shuffle',        help='Randomize image order (default: 1)', type=int, default=1)

    p = add_command(    'create_from_pkl', 'Create dataset from a directory full of images.',
                                            'create_from_pkl datasets/mydataset myimagedir')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'image_dir',        help='Directory containing the images')
    p.add_argument(     '--shuffle',        help='Randomize image order (default: 1)', type=int, default=1)

    p = add_command(    'create_from_pkl_img', 'Create dataset from a directory full of images.',
                                            'create_from_pkl_img datasets/mydataset myimagedir myshapedir')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'image_dir',        help='Directory containing the images')
    p.add_argument(     'pickle_dir',        help='Directory containing the shape pickles')
    p.add_argument(     '--shuffle',        help='Randomize image order (default: 1)', type=int, default=1)

    p = add_command(    'create_from_pkl_img_norm', 'Create dataset from a directory full of images.',
                                            'create_from_pkl_img_norm datasets/mydataset myimagedir myshapedir')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'image_dir',        help='Directory containing the images')
    p.add_argument(     'pickle_dir',        help='Directory containing the shape pickles')
    p.add_argument(     'normal_dir',        help='Directory containing the normal pickles')
    p.add_argument(     '--shuffle',        help='Randomize image order (default: 1)', type=int, default=1)

    p = add_command(    'create_from_hdf5', 'Create dataset from legacy HDF5 archive.',
                                            'create_from_hdf5 datasets/celebahq ~/downloads/celeba-hq-1024x1024.h5')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'hdf5_filename',    help='HDF5 archive containing the images')
    p.add_argument(     '--shuffle',        help='Randomize image order (default: 1)', type=int, default=1)

    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    execute_cmdline(sys.argv)

#----------------------------------------------------------------------------
