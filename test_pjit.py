import jax
from jax.experimental import maps
from jax.experimental import PartitionSpec
from jax.experimental.pjit import pjit
import numpy as np

mesh_shape = (4, 2)
devices = np.asarray(jax.devices()).reshape(*mesh_shape)
# 'x', 'y' axis names are used here for simplicity
mesh = maps.Mesh(devices, ('x', 'y'))
mesh

input_data = np.arange(8 * 2).reshape(8, 2)
input_data

in_axis_resources=None
out_axis_resources=PartitionSpec('x', 'y')

f = pjit(
  lambda x: x,
  in_axis_resources=None,
  out_axis_resources=PartitionSpec('x', 'y'))
 
# Sends data to accelerators based on partition_spec
with maps.Mesh(mesh.devices, mesh.axis_names):
 data = f(input_data)

data

data.device_buffers

f = pjit(
  lambda x: x,
  in_axis_resources=None,
  out_axis_resources=PartitionSpec('x', None))
 
with maps.Mesh(mesh.devices, mesh.axis_names):
 data = f(input_data)

data.device_buffers

f = pjit(
  lambda x: x,
  in_axis_resources=None,
  out_axis_resources=PartitionSpec(('x', 'y'), None))
 
with maps.Mesh(mesh.devices, mesh.axis_names):
 data = f(input_data)

data.device_buffers

