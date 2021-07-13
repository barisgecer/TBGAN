import os

import menpo3d.io as m3io
from menpo3d import io
import menpo.io as mio
import numpy as np
from menpo.shape import TriMesh, ColouredTriMesh, PointCloud, TexturedTriMesh
from menpo.image import Image
from scipy.interpolate import NearestNDInterpolator
from menpo.image import MaskedImage
from functools import lru_cache
from menpo.transform import AlignmentSimilarity
#-------------------------------------------------------------------#
#Full face load dictionaries
@lru_cache()
def load_512_ifo_dict():
    return mio.import_pickle('512_UV_dict.pkl')

#-------------------------------------------------------------------#
@lru_cache()
def load_mean():
    return mio.import_pickle('./pkls/all_all_all_mean.pkl'),mio.import_pickle('./pkls/all_all_all_lands_ids.pkl')
#-------------------------------------------------------------------#

def alignment(mesh):
    if mesh.n_points==53215:
        template, idxs= load_mean()

        
    alignment = AlignmentSimilarity(PointCloud(mesh.points[idxs]), PointCloud(template.points[idxs]))
    aligned_mesh = alignment.apply(mesh)
    return aligned_mesh


def import_uv_info(instance, res, uv_layout='oval', topology='full'):
    if np.logical_or(type(instance).__name__=='TriMesh',type(instance).__name__=='TexturedTriMesh'):
        if instance.n_points == 53215:
            topology = 'full'
        else:
            raise ValueError('Unknown topology')

        if topology == 'full':
            if res==512:
                if uv_layout=='oval':
                    info_dict = load_512_ifo_dict()
                elif uv_layout=='stretch':
                    info_dict = load_512_ifo_dict_strech()
            else:
                raise ValueError('Wrong resolution')
    elif type(instance).__name__=='Image':
        if topology == 'full':
            if res==512:
                if uv_layout=='oval':
                    info_dict = load_512_ifo_dict()
                elif uv_layout=='stretch':
                    info_dict = load_512_ifo_dict_strech()
            else:
                raise ValueError('Wrong resolution')
    return info_dict  

def from_UV_2_3D(uv, uv_layout='oval', topology='full', plot=False):
    res = uv.shape[0]
    info_dict = import_uv_info(uv,res,uv_layout=uv_layout,topology=topology)
        
    tmask = info_dict['tmask']
    tc_ps = info_dict['tcoords_pixel_scaled']
    tmask_im =  info_dict['tmask_image']
    trilist = info_dict['trilist']
    
    #uv = interpolaton_of_uv_xyz(uv,tmask).as_unmasked()
    x = uv.pixels[0][(tc_ps.points.astype(int).T[0,:], tc_ps.points.astype(int).T[1,:])]
    y = uv.pixels[1][(tc_ps.points.astype(int).T[0,:], tc_ps.points.astype(int).T[1,:])] 
    z = uv.pixels[2][(tc_ps.points.astype(int).T[0,:], tc_ps.points.astype(int).T[1,:])]
    points = np.hstack((x.T[:,None],y.T[:,None],z.T[:,None]))
    if plot is True:
        TriMesh(points,trilist).view()
    return TriMesh(points,trilist)