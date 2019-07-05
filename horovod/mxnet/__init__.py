# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from horovod.common import check_extension

check_extension('horovod.mxnet', 'HOROVOD_WITH_MXNET',
                __file__, 'mpi_lib')

from horovod.mxnet.mpi_ops import allgather
from horovod.mxnet.mpi_ops import allreduce, allreduce_
from horovod.mxnet.mpi_ops import broadcast, broadcast_
from horovod.mxnet.mpi_ops import init, shutdown
from horovod.mxnet.mpi_ops import size, local_size, rank, local_rank
from horovod.mxnet.mpi_ops import mpi_threads_supported

import mxnet as mx


# This is where Horovod's DistributedOptimizer wrapper for MXNet goes
class DistributedOptimizer(mx.optimizer.Optimizer):
    def __init__(self, optimizer):
        self._optimizer = optimizer

    def __getattr__(self, item):
        return getattr(self._optimizer, item)

    def create_state_multi_precision(self, index, weight):
        return self._optimizer.create_state_multi_precision(index, weight)

    def _do_allreduce(self, index, grad):
        if isinstance(index, (tuple, list)):
            for i in range(len(index)):
                allreduce_(grad[i], average=True,
                           name=str(index[i]), priority=-i)
        else:
            allreduce_(grad, average=True, name=str(index))

    def update(self, index, weight, grad, state):
        self._do_allreduce(index, grad)
        self._optimizer.update(index, weight, grad, state)

    def update_multi_precision(self, index, weight, grad, state):
        self._do_allreduce(index, grad)
        self._optimizer.update_multi_precision(index, weight, grad, state)

    def set_learning_rate(self, lr):
        self._optimizer.set_learning_rate(lr)

    def set_lr_mult(self, args_lr_mult):
        self._optimizer.set_lr_mult(args_lr_mult)

    def set_wd_mult(self, args_wd_mult):
        self._optimizer.set_wd_mult(args_wd_mult)


def ResizeEvalDataIter(data_iter):
    try:
        from mpi4py import MPI
    except ImportError as e:
        raise ImportError(
            "ResizeEvalDataIter must be used with mpi4py now, run 'pip install mpi4py' to install it")
    batch_num = 0
    for _ in data_iter:
        batch_num += 1
    data_iter.reset()
    comm = MPI.COMM_WORLD
    batch_num = comm.gather(batch_num, root=0)
    if rank() == 0:
        max_batch_num = max(batch_num)
    else:
        max_batch_num = 0
    max_batch_num = comm.bcast(max_batch_num, root=0)
    return mx.io.ResizeIter(data_iter, max_batch_num)


def DistributedEvalMetric(base):
    assert(issubclass(base, mx.metric.EvalMetric))

    try:
        from mpi4py import MPI
    except ImportError as e:
        raise ImportError(
            "DistributedEvalMetric must be used with mpi4py now, run 'pip install mpi4py' to install it")
    class _DistributedEvalMetric(base):
        def __init__(self, *args, **kwargs):
            self._size = size()
            self._rank = rank()
            return super().__init__(*args, **kwargs)

        def update(self, labels, preds):
            labels = MPI.COMM_WORLD.gather(labels, root=0)
            preds = MPI.COMM_WORLD.gather(preds, root=0)
            if self._rank == 0:
                for i in range(self._size):
                    super().update(labels[i], preds[i])

    return _DistributedEvalMetric


# DistributedTrainer, a subclass of MXNet gluon.Trainer.
# There are two differences between DistributedTrainer and Trainer:
# 1. DistributedTrainer calculates gradients using Horovod allreduce
#    API while Trainer does it using kvstore push/pull APIs;
# 2. DistributedTrainer performs allreduce(summation) and average
#    while Trainer only performs allreduce(summation).
class DistributedTrainer(mx.gluon.Trainer):
    def __init__(self, params, optimizer, optimizer_params=None):
        if isinstance(optimizer, DistributedOptimizer):
            optimizer = optimizer._optimizer
            warnings.warn("DistributedTrainer does not take DistributedOptimizer "
                          "as its optimizer. We have unwrapped it for you.")

        super(DistributedTrainer, self).__init__(
            params, optimizer, optimizer_params=optimizer_params, kvstore=None)

        # _scale is used to check and set rescale_grad for optimizer in Trainer.step()
        # function. Normalizing it by Horovod size, which is equivalent to performing
        # average in allreduce, has better performance. 
        self._scale /= size()

    def _allreduce_grads(self):
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                allreduce_(param.list_grad()[0], average=False,
                           name=str(i), priority=-i)

def broadcast_parameters(params, root_rank=0):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the `Module.get_params()` or the
    `Block.collect_params()`.

    Arguments:
        params: One of the following:
            - dict of parameters to broadcast
            - ParameterDict to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    tensors = []
    if isinstance(params, dict):
        tensors = [p for _, p in sorted(params.items())]
    elif isinstance(params, mx.gluon.parameter.ParameterDict):
        for _, p in sorted(params.items()):
            try:
                tensors.append(p.data())
            except mx.gluon.parameter.DeferredInitializationError:
                # skip broadcasting deferred init param
                pass
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run broadcasts.
    for i, tensor in enumerate(tensors):
        broadcast_(tensor, root_rank, str(i))

    # Make sure tensors pushed to MXNet engine get processed such that all
    # workers are synced before starting training.
    for tensor in tensors:
        tensor.wait_to_read()
