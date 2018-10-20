import numpy as np
import menpo.io as mio
import menpo3d.io as m3io
from pathlib import Path
import scipy.io as sio
import matplotlib
import tensorflow as tf
matplotlib.use('Qt4Agg')

from menpo.shape import PointCloud, TriMesh, ColouredTriMesh, TexturedTriMesh

def uv2mesh(uv, tcoords_path='/homes/bg2715/utils/tcoords.pkl'):

    tcoords = mio.import_pickle(tcoords_path)

    sample_points = tcoords[:, [1, 0]]
    sample_points[:, 0] = 1 - sample_points[:, 0]
    sample_points *= [377, 595]
    sample_points = np.round(sample_points).astype(np.int)
    sample_points = tf.constant(sample_points)

    pad_crop = lambda img : tf.manip.gather_nd(tf.image.pad_to_bounding_box(tf.transpose(img[:,67:444,:],[1,2,0]),0,41,377,595), sample_points)

    return tf.map_fn(pad_crop,uv)


