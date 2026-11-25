"""Box-to-collision-mesh distances, as in the original support audit."""
import os
import ctypes as C
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
root=Path(__file__).resolve().parent.parent/'forearm_rotation'
xml=ET.parse(root/'forearm.xml')
lib=C.CDLL(os.environ['ORCA_GAP_LIBRARY'])
lib.create.argtypes=[C.c_void_p,C.c_int,C.c_void_p,C.c_int];lib.create.restype=C.c_void_p
lib.distances.argtypes=[C.c_void_p,C.c_int,C.c_void_p,C.c_void_p,C.c_void_p];lib.distances.restype=None
models={m.get('name'):m for m in xml.findall('.//asset/mesh')}
geometry={}
mount=xml.find(".//body[@name='right_forearm_mount']/geom")
mount_offset=np.fromstring(mount.get('pos'),sep=' ')
mount_radius=float(mount.get('size'))
for body in xml.findall('.//worldbody//body'):
    for geom in body.findall('geom'):
        if geom.get('mesh') is None:continue
        spec=models[geom.get('mesh')]
        stl=np.fromfile(root/spec.get('file'),dtype=[('n','<f4',3),('v','<f4',(3,3)),('extra','<u2')],offset=84)
        vertices=np.unique(stl['v'].reshape(-1,3),axis=0)*np.fromstring(spec.get('scale'),sep=' ')
        rot=R.from_quat(np.roll(np.fromstring(geom.get('quat','1 0 0 0'),sep=' '),-1))
        vertices=np.ascontiguousarray(rot.apply(vertices)+np.fromstring(geom.get('pos','0 0 0'),sep=' '))
        triangles=np.ascontiguousarray(ConvexHull(vertices).simplices,dtype='i4')
        h=lib.create(vertices.ctypes.data,len(vertices),triangles.ctypes.data,len(triangles))
        geometry.setdefault(body.get('name'),[]).append((h,vertices.min(0),vertices.max(0)))

def gap(name,bodies,objects):
    inv=R.from_quat(bodies[:,3:]).inv()
    rot=inv*R.from_quat(objects[:,3:])
    pos=inv.apply(objects[:,:3]-bodies[:,:3])
    if name=='right_forearm_mount':
        local=rot.inv().apply(mount_offset-pos)
        return np.maximum(0,np.linalg.norm(np.maximum(np.abs(local)-.03,0),axis=1)-mount_radius)*1000
    # Broad phase only omits mesh pairs strictly beyond the largest tested distance.
    q=rot.as_quat()
    radius=.03*np.abs(rot.as_matrix()).sum(axis=2)
    dist=np.full(len(pos),np.inf,dtype='f4')
    for handle,lower,upper in geometry[name]:
        near=np.all((pos-radius <= upper+.006001)&(pos+radius >= lower-.006001),axis=1)
        if not np.any(near):continue
        xyz=np.ascontiguousarray(pos[near]);xyzw=np.ascontiguousarray(q[near])
        g=np.empty(len(xyz),dtype='f4')
        lib.distances(handle,len(g),xyz.ctypes.data,xyzw.ctypes.data,g.ctypes.data)
        dist[near]=np.minimum(dist[near],g)
    return dist

