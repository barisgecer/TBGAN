# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import time
import re
import bisect
import numpy as np
import tensorflow as tf
import scipy.ndimage
import scipy.misc
from scipy.spatial.distance import cdist
from sklearn.utils.extmath import softmax
import scipy.ndimage as ndimage

import config_test
import misc
import tfutil
import myutil
import menpo.io as mio

import menpo3d.io as m3io
from menpo.shape import TexturedTriMesh, TriMesh, ColouredTriMesh
from UV_manipulation_2 import from_UV_2_3D
from menpo.image import Image

#----------------------------------------------------------------------------
# Generate random images or image grids using a previously trained network.
# To run, uncomment the appropriate line in config_test.py and launch train.py.

def get_generator(run_id, snapshot=None, image_shrink=1, minibatch_size=8):
    network_pkl = misc.locate_network_pkl(run_id, snapshot)

    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(run_id, snapshot)
    latent = tf.get_variable('latent',shape=(1,512),trainable=True)
    label = tf.get_variable('label',shape=(1,0),trainable=True,initializer=tf.zeros_initializer)
    images = Gs.fit(latent, label, minibatch_size=minibatch_size, num_gpus=config_test.num_gpus, out_mul=0.5, out_add=0.5, out_shrink=image_shrink, out_dtype=np.float32)
    sess = tf.get_default_session()

    sess.run(tf.variables_initializer([latent, label]))

    return images, latent, sess

def fit_real_images(run_id, snapshot=None, num_pngs=1, image_shrink=1, png_prefix=None, random_seed=1000, minibatch_size=8):
    network_pkl = misc.locate_network_pkl(run_id, snapshot)
    if png_prefix is None:
        png_prefix = misc.get_id_string_for_network_pkl(network_pkl) + '-'
    random_state = np.random.RandomState(random_seed)

    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(run_id, snapshot)
    latent = tf.get_variable('latent',shape=(1,512),trainable=True)
    label = tf.get_variable('label',shape=(1,0),trainable=True)
    images = Gs.fit(latent, label, minibatch_size=minibatch_size, num_gpus=config_test.num_gpus)
    sess = tf.get_default_session()

    target = tf.placeholder(tf.float32,name='target')
    lr = tf.placeholder(tf.float32,name='lr')
    #loss = tf.reduce_sum(tf.abs(images[0][0] - target))
    loss = tf.nn.l2_loss(images[0][0] - target)
    with tf.variable_scope('adam'):
        opt = tf.train.AdamOptimizer(lr).minimize(loss,var_list=latent)

    sess.run(tf.variables_initializer([latent, label]))
    sess.run(tf.variables_initializer(tf.global_variables('adam')))


    # real_path = '/vol/phoebe/3DMD_SCIENCE_MUSEUM/Colour_UV_maps'
    # real_path = '/home/baris/data/mein3d_600x600'
    real_path = '/media/gen/pca_alone'
    save_path = '/media/gen/gan-pca'
    #target_im = PIL.Image.open('/media/logs-nvidia/002-fake-images-0/000-pgan-mein3d_tf-preset-v2-2gpus-fp32-VERBOSE-HIST-network-final-000001.png')

    for ind, real in enumerate(myutil.files(real_path)):
        target_im = myutil.crop_im(PIL.Image.open(os.path.join(real_path,real)))
        for j in [0.1,0.01,0.001]:
            for i in range(500):
                l2,_ = sess.run([loss,opt],{target: myutil.rgb2tf(target_im),lr:j})
                if i % 100 == 0:
                    print(l2)

        myutil.concat_image(np.asarray(target_im),myutil.tf2rgb(sess.run(images))).save(os.path.join(save_path,real))

    sess.close()

def generate_fake_images_glob(run_id, snapshot=None, grid_size=[1,1], num_pngs=1, image_shrink=1, png_prefix=None, random_seed=1000, minibatch_size=8):
    network_pkl = misc.locate_network_pkl(run_id, snapshot)
    if png_prefix is None:
        png_prefix = misc.get_id_string_for_network_pkl(network_pkl) + '-'
    random_state = np.random.RandomState(random_seed)

    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(run_id, snapshot)
    latents = random_state.randn(num_pngs, *G.input_shape[1:]).astype(np.float32)
    dist = cdist(latents,latents)
    np.fill_diagonal(dist,100)
    result_subdir = misc.create_result_subdir(config_test.result_dir, config_test.desc)
    for png_idx in range(num_pngs):
        print('Generating png %d / %d...' % (png_idx, num_pngs))
        latents = misc.random_latents(np.prod(grid_size), Gs, random_state=random_state)
        labels = np.zeros([latents.shape[0], 0], np.float32)
        images = Gs.run(latents, labels, minibatch_size=minibatch_size, num_gpus=config_test.num_gpus, out_mul=127.5, out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)
        misc.save_image_grid(images, os.path.join(result_subdir, '%s%06d.png' % (png_prefix, png_idx)), [0,255], grid_size)
    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()

def generate_fake_images(run_id, snapshot=None, grid_size=[1,1],batch_size=8, num_pngs=1, image_shrink=1, png_prefix=None, random_seed=1000, minibatch_size=8):
    network_pkl = misc.locate_network_pkl(run_id, snapshot)
    if png_prefix is None:
        png_prefix = misc.get_id_string_for_network_pkl(network_pkl) + '-'
    random_state = np.random.RandomState(random_seed)

    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(run_id, snapshot)

    result_subdir = misc.create_result_subdir(config_test.result_dir, config_test.desc)
    for png_idx in range(int(num_pngs/batch_size)):
        start = time.time()
        print('Generating png %d-%d / %d... in ' % (png_idx*batch_size,(png_idx+1)*batch_size, num_pngs),end='')
        latents = misc.random_latents(np.prod(grid_size)*batch_size, Gs, random_state=random_state)
        labels = np.zeros([latents.shape[0], 7], np.float32)
        images = Gs.run(latents, labels, minibatch_size=minibatch_size, num_gpus=config_test.num_gpus, out_shrink=image_shrink)
        for i in range(batch_size):
            if images.shape[1]==3:
                mio.export_pickle(images[i],os.path.join(result_subdir, '%s%06d.pkl' % (png_prefix, png_idx*batch_size+i)))
                # misc.save_image(images[i], os.path.join(result_subdir, '%s%06d.png' % (png_prefix, png_idx*batch_size+i)), [0,255], grid_size)
            elif images.shape[1]==6:
                mio.export_pickle(images[i][3:6],
                                  os.path.join(result_subdir, '%s%06d.pkl' % (png_prefix, png_idx * batch_size + i)),overwrite=True)
                misc.save_image(images[i][0:3], os.path.join(result_subdir, '%s%06d.png' % (png_prefix, png_idx*batch_size+i)), [-1,1], grid_size)
            elif images.shape[1]==9:
                mio.export_pickle(images[i][3:6],
                                  os.path.join(result_subdir, '%s%06d_shp.pkl' % (png_prefix, png_idx * batch_size + i)),overwrite=True)
                mio.export_pickle(images[i][6:9],
                                  os.path.join(result_subdir, '%s%06d_nor.pkl' % (png_prefix, png_idx * batch_size + i)),overwrite=True)
                misc.save_image(images[i][0:3], os.path.join(result_subdir, '%s%06d.png' % (png_prefix, png_idx*batch_size+i)), [-1,1], grid_size)
        print('%0.2f seconds' % (time.time() - start))

    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()

#----------------------------------------------------------------------------
# Generate MP4 video of random interpolations using a previously trained network.
# To run, uncomment the appropriate line in config_test.py and launch train.py.

def generate_interpolation_video(run_id, snapshot=None, grid_size=[1,1], image_shrink=1, image_zoom=1, duration_sec=60.0, smoothing_sec=1.0, mp4=None, mp4_fps=30, mp4_codec='libx265', mp4_bitrate='16M', random_seed=1000, minibatch_size=8):
    network_pkl = misc.locate_network_pkl(run_id, snapshot)
    if mp4 is None:
        mp4 = misc.get_id_string_for_network_pkl(network_pkl) + '-lerp.mp4'
    num_frames = int(np.rint(duration_sec * mp4_fps))
    random_state = np.random.RandomState(random_seed)

    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(run_id, snapshot)

    print('Generating latent vectors...')
    shape = [num_frames, np.prod(grid_size)] + Gs.input_shape[1:] # [frame, image, channel, component]
    all_latents = random_state.randn(*shape).astype(np.float32)
    all_latents = scipy.ndimage.gaussian_filter(all_latents, [smoothing_sec * mp4_fps] + [0] * len(Gs.input_shape), mode='wrap')
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    # Frame generation func for moviepy.
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        latents = all_latents[frame_idx]
        labels = np.zeros([latents.shape[0], 0], np.float32)
        images = Gs.run(latents, labels, minibatch_size=minibatch_size, num_gpus=config_test.num_gpus, out_mul=127.5, out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)
        grid = misc.create_image_grid(images, grid_size).transpose(1, 2, 0) # HWC
        if image_zoom > 1:
            grid = scipy.ndimage.zoom(grid, [image_zoom, image_zoom, 1], order=0)
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2) # grayscale => RGB
        return grid

    # Generate video.
    import moviepy.editor # pip install moviepy
    result_subdir = misc.create_result_subdir(config_test.result_dir, config_test.desc)
    moviepy.editor.VideoClip(make_frame, duration=duration_sec).write_videofile(os.path.join(result_subdir, mp4), fps=mp4_fps, codec='libx264', bitrate=mp4_bitrate)
    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()

#----------------------------------------------------------------------------
# Generate MP4 video of random interpolations using a previously trained network.
# To run, uncomment the appropriate line in config_test.py and launch train.py.

def generate_interpolation_images(run_id, snapshot=None, grid_size=[1,1], image_shrink=1, image_zoom=1, duration_sec=60.0, smoothing_sec=1.0, mp4=None, mp4_fps=30, mp4_codec='libx265', mp4_bitrate='16M', random_seed=1000, minibatch_size=8):

    network_pkl = misc.locate_network_pkl(run_id, snapshot)
    if mp4 is None:
        mp4 = misc.get_id_string_for_network_pkl(network_pkl) + '-lerp.mp4'
    num_frames = int(np.rint(duration_sec * mp4_fps))
    random_state = np.random.RandomState(random_seed)

    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(run_id, snapshot)

    print('Generating latent vectors...')
    shape = [num_frames, np.prod(grid_size)] + [Gs.input_shape[1:][0]+Gs.input_shapes[1][1:][0]] # [frame, image, channel, component]
    all_latents = random_state.randn(*shape).astype(np.float32)
    all_latents = scipy.ndimage.gaussian_filter(all_latents, [smoothing_sec * mp4_fps] + [0] * len(Gs.input_shape), mode='wrap')
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    #10 10 10 10 5 3 10
    # model = mio.import_pickle('../models/lsfm_shape_model_fw.pkl')
    # facesoft_model = mio.import_pickle('../models/facesoft_id_and_exp_3d_face_model.pkl')['shape_model']
    # lsfm_model = m3io.import_lsfm_model('/home/baris/Projects/faceganhd/models/all_all_all.mat')
    # model_mean = lsfm_model.mean().copy()
    # mask = mio.import_pickle('../UV_spaces_V2/mask_full_2_crop.pkl')
    lsfm_tcoords = \
    mio.import_pickle('512_UV_dict.pkl')['tcoords']
    lsfm_params = []
    result_subdir = misc.create_result_subdir(config_test.result_dir, config_test.desc)
    for png_idx in range(int(num_frames/minibatch_size)):
        start = time.time()
        print('Generating png %d-%d / %d... in ' % (png_idx*minibatch_size,(png_idx+1)*minibatch_size, num_frames),end='')
        latents = all_latents[png_idx*minibatch_size:(png_idx+1)*minibatch_size,0,:Gs.input_shape[1:][0]]
        labels = all_latents[png_idx*minibatch_size:(png_idx+1)*minibatch_size,0,Gs.input_shape[1:][0]:]
        labels_softmax = softmax(labels) *np.array([10,10,10,10,5,3,10])
        images = Gs.run(latents, labels_softmax, minibatch_size=minibatch_size, num_gpus=config_test.num_gpus, out_shrink=image_shrink)
        for i in range(minibatch_size):
            texture = Image(np.clip(images[i,0:3]/2+0.5,0,1))
            img_shape = ndimage.gaussian_filter(images[i,3:6], sigma=(0, 3, 3), order=0)
            mesh_raw = from_UV_2_3D(Image(img_shape),topology='full',uv_layout='oval')
            # model_mean.points[mask,:] = mesh_raw.points
            normals = images[i,6:9]
            normals_norm = (normals - normals.min()) / (normals.max() - normals.min())
            mesh = mesh_raw#facesoft_model.reconstruct(model_mean).from_mask(mask)
            # lsfm_params.append(lsfm_model.project(mesh_raw))
            t_mesh = TexturedTriMesh(mesh.points, lsfm_tcoords.points, texture, mesh.trilist)
            m3io.export_textured_mesh(t_mesh, os.path.join(result_subdir, '%06d.obj' % (png_idx * minibatch_size + i)),texture_extension='.png')
            fix_obj(os.path.join(result_subdir, '%06d.obj' % (png_idx * minibatch_size + i)))
            mio.export_image(Image(normals_norm), os.path.join(result_subdir, '%06d_nor.png' % (png_idx * minibatch_size + i)))
        print('%0.2f seconds' % (time.time() - start))
    mio.export_pickle(lsfm_params,os.path.join(result_subdir, 'lsfm_params.pkl'))
    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()

def generate_interpolation_video_bydim(run_id, snapshot=None, grid_size=[1,1], image_shrink=1, image_zoom=1, duration_sec=60.0, smoothing_sec=1.0, mp4=None, mp4_fps=30, mp4_codec='libx265', mp4_bitrate='16M', random_seed=1000, minibatch_size=8, dim=0):
    network_pkl = misc.locate_network_pkl(run_id, snapshot)
    if mp4 is None:
        mp4 = misc.get_id_string_for_network_pkl(network_pkl) + '-lerp.mp4'
    num_frames = int(np.rint(duration_sec * mp4_fps))
    random_state = np.random.RandomState(random_seed)

    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(run_id, snapshot)

    print('Generating latent vectors...')
    shape = [num_frames, np.prod(grid_size)] + Gs.input_shape[1:] # [frame, image, channel, component]
    all_latents = np.tile(random_state.randn(*shape[1:3]).astype(np.float32),[shape[0],1,1])
    #all_latents = random_state.randn(*shape).astype(np.float32)
    #all_latents = scipy.ndimage.gaussian_filter(all_latents, [smoothing_sec * mp4_fps] + [0] * len(Gs.input_shape), mode='wrap')
    all_latents[:,0,dim]=np.linspace(-4.0,4.0,shape[0])
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    # Frame generation func for moviepy.
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        latents = all_latents[frame_idx]
        labels = np.zeros([latents.shape[0], 0], np.float32)
        images = Gs.run(latents, labels, minibatch_size=minibatch_size, num_gpus=config_test.num_gpus, out_mul=127.5, out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)
        grid = misc.create_image_grid(images, grid_size).transpose(1, 2, 0) # HWC
        if image_zoom > 1:
            grid = scipy.ndimage.zoom(grid, [image_zoom, image_zoom, 1], order=0)
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2) # grayscale => RGB
        return grid

    # Generate video.
    import moviepy.editor # pip install moviepy
    result_subdir = misc.create_result_subdir(config_test.result_dir, config_test.desc)
    moviepy.editor.VideoClip(make_frame, duration=duration_sec).write_videofile(os.path.join(result_subdir, mp4), fps=mp4_fps, codec='libx264', bitrate=mp4_bitrate)
    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()

#----------------------------------------------------------------------------
# Generate MP4 video of training progress for a previous training run.
# To run, uncomment the appropriate line in config_test.py and launch train.py.

def generate_training_video(run_id, duration_sec=20.0, time_warp=1.5, mp4=None, mp4_fps=30, mp4_codec='libx265', mp4_bitrate='16M'):
    src_result_subdir = misc.locate_result_subdir(run_id)
    if mp4 is None:
        mp4 = os.path.basename(src_result_subdir) + '-train.mp4'

    # Parse log.
    times = []
    snaps = [] # [(png, kimg, lod), ...]
    with open(os.path.join(src_result_subdir, 'log.txt'), 'rt') as log:
        for line in log:
            k = re.search(r'kimg ([\d\.]+) ', line)
            l = re.search(r'lod ([\d\.]+) ', line)
            t = re.search(r'time (\d+d)? *(\d+h)? *(\d+m)? *(\d+s)? ', line)
            if k and l and t:
                k = float(k.group(1))
                l = float(l.group(1))
                t = [int(t.group(i)[:-1]) if t.group(i) else 0 for i in range(1, 5)]
                t = t[0] * 24*60*60 + t[1] * 60*60 + t[2] * 60 + t[3]
                png = os.path.join(src_result_subdir, 'fakes%06d.png' % int(np.floor(k)))
                if os.path.isfile(png):
                    times.append(t)
                    snaps.append((png, k, l))
    assert len(times)

    # Frame generation func for moviepy.
    png_cache = [None, None] # [png, img]
    def make_frame(t):
        wallclock = ((t / duration_sec) ** time_warp) * times[-1]
        png, kimg, lod = snaps[max(bisect.bisect(times, wallclock) - 1, 0)]
        if png_cache[0] == png:
            img = png_cache[1]
        else:
            img = scipy.misc.imread(png)
            while img.shape[1] > 1920 or img.shape[0] > 1080:
                img = img.astype(np.float32).reshape(img.shape[0]//2, 2, img.shape[1]//2, 2, -1).mean(axis=(1,3))
            png_cache[:] = [png, img]
        img = misc.draw_text_label(img, 'lod %.2f' % lod, 16, img.shape[0]-4, alignx=0.0, aligny=1.0)
        img = misc.draw_text_label(img, misc.format_time(int(np.rint(wallclock))), img.shape[1]//2, img.shape[0]-4, alignx=0.5, aligny=1.0)
        img = misc.draw_text_label(img, '%.0f kimg' % kimg, img.shape[1]-16, img.shape[0]-4, alignx=1.0, aligny=1.0)
        return img

    # Generate video.
    import moviepy.editor # pip install moviepy
    result_subdir = misc.create_result_subdir(config_test.result_dir, config_test.desc)
    moviepy.editor.VideoClip(make_frame, duration=duration_sec).write_videofile(os.path.join(result_subdir, mp4), fps=mp4_fps, codec='libx264', bitrate=mp4_bitrate)
    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()

#----------------------------------------------------------------------------
# Evaluate one or more metrics for a previous training run.
# To run, uncomment one of the appropriate lines in config_test.py and launch train.py.

def evaluate_metrics(run_id, log, metrics, num_images, real_passes, minibatch_size=None):
    metric_class_names = {
        'swd':      'metrics.sliced_wasserstein.API',
        'fid':      'metrics.frechet_inception_distance.API',
        'is':       'metrics.inception_score.API',
        'msssim':   'metrics.ms_ssim.API',
    }

    # Locate training run and initialize logging.
    result_subdir = misc.locate_result_subdir(run_id)
    snapshot_pkls = misc.list_network_pkls(result_subdir, include_final=False)
    assert len(snapshot_pkls) >= 1
    log_file = os.path.join(result_subdir, log)
    print('Logging output to', log_file)
    misc.set_output_log_file(log_file)

    # Initialize dataset and select minibatch size.
    dataset_obj, mirror_augment = misc.load_dataset_for_previous_run(result_subdir, verbose=True, shuffle_mb=0)
    if minibatch_size is None:
        minibatch_size = np.clip(8192 // dataset_obj.shape[1], 4, 256)

    # Initialize metrics.
    metric_objs = []
    for name in metrics:
        class_name = metric_class_names.get(name, name)
        print('Initializing %s...' % class_name)
        class_def = tfutil.import_obj(class_name)
        image_shape = [3] + dataset_obj.shape[1:]
        obj = class_def(num_images=num_images, image_shape=image_shape, image_dtype=np.uint8, minibatch_size=minibatch_size)
        tfutil.init_uninited_vars()
        mode = 'warmup'
        obj.begin(mode)
        for idx in range(10):
            obj.feed(mode, np.random.randint(0, 256, size=[minibatch_size]+image_shape, dtype=np.uint8))
        obj.end(mode)
        metric_objs.append(obj)

    # Print table header.
    print()
    print('%-10s%-12s' % ('Snapshot', 'Time_eval'), end='')
    for obj in metric_objs:
        for name, fmt in zip(obj.get_metric_names(), obj.get_metric_formatting()):
            print('%-*s' % (len(fmt % 0), name), end='')
    print()
    print('%-10s%-12s' % ('---', '---'), end='')
    for obj in metric_objs:
        for fmt in obj.get_metric_formatting():
            print('%-*s' % (len(fmt % 0), '---'), end='')
    print()

    # Feed in reals.
    for title, mode in [('Reals', 'reals'), ('Reals2', 'fakes')][:real_passes]:
        print('%-10s' % title, end='')
        time_begin = time.time()
        labels = np.zeros([num_images, dataset_obj.label_size], dtype=np.float32)
        [obj.begin(mode) for obj in metric_objs]
        for begin in range(0, num_images, minibatch_size):
            end = min(begin + minibatch_size, num_images)
            images, labels[begin:end] = dataset_obj.get_minibatch_np(end - begin)
            if mirror_augment:
                images = misc.apply_mirror_augment(images)
            if images.shape[1] == 1:
                images = np.tile(images, [1, 3, 1, 1]) # grayscale => RGB
            [obj.feed(mode, images) for obj in metric_objs]
        results = [obj.end(mode) for obj in metric_objs]
        print('%-12s' % misc.format_time(time.time() - time_begin), end='')
        for obj, vals in zip(metric_objs, results):
            for val, fmt in zip(vals, obj.get_metric_formatting()):
                print(fmt % val, end='')
        print()

    # Evaluate each network snapshot.
    for snapshot_idx, snapshot_pkl in enumerate(reversed(snapshot_pkls)):
        prefix = 'network-snapshot-'; postfix = '.pkl'
        snapshot_name = os.path.basename(snapshot_pkl)
        assert snapshot_name.startswith(prefix) and snapshot_name.endswith(postfix)
        snapshot_kimg = int(snapshot_name[len(prefix) : -len(postfix)])

        print('%-10d' % snapshot_kimg, end='')
        mode ='fakes'
        [obj.begin(mode) for obj in metric_objs]
        time_begin = time.time()
        with tf.Graph().as_default(), tfutil.create_session(config_test.tf_config).as_default():
            G, D, Gs = misc.load_pkl(snapshot_pkl)
            for begin in range(0, num_images, minibatch_size):
                end = min(begin + minibatch_size, num_images)
                latents = misc.random_latents(end - begin, Gs)
                images = Gs.run(latents, labels[begin:end], num_gpus=config_test.num_gpus, out_mul=127.5, out_add=127.5, out_dtype=np.uint8)
                if images.shape[1] == 1:
                    images = np.tile(images, [1, 3, 1, 1]) # grayscale => RGB
                [obj.feed(mode, images) for obj in metric_objs]
        results = [obj.end(mode) for obj in metric_objs]
        print('%-12s' % misc.format_time(time.time() - time_begin), end='')
        for obj, vals in zip(metric_objs, results):
            for val, fmt in zip(vals, obj.get_metric_formatting()):
                print(fmt % val, end='')
        print()
    print()


def fix_obj(fp):
    os.path.dirname(fp)
    template = """# Produced by Dimensional Imaging OBJ exporter
# http://www.di3d.com
#
#
newmtl merged_material
Ka  0.5 0.5 0.5
Kd  0.5 0.5 0.5
Ks  0.47 0.47 0.47
d 1
Ns 0
illum 2
map_Kd {}.png
#
#
# EOF""".format(os.path.splitext(os.path.basename(fp))[0])
    with open(os.path.join(os.path.dirname(fp), os.path.splitext(os.path.basename(fp))[0] + '.mtl'), 'w') as f:
        f.write(template)

    with open(fp, 'r+')  as f:
        content = f.read()
        f.seek(0, 0)
        f.write('mtllib ' + os.path.splitext(os.path.basename(fp))[0] + '.mtl' + '\n' + content)


#----------------------------------------------------------------------------
