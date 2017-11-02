# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


# Tests require MXNet built with Intel MKLDNN library
import mxnet as mx
import numpy as np
from numpy.testing import assert_allclose

def check_rnn_consistency(cell1, cell2):
    dshape = (32, 5, 200)
    data = mx.sym.Variable('data')

    sym1, _ = cell1.unroll(5, data, merge_outputs=True)
    mod1 = mx.mod.Module(sym1, label_names=None, context=mx.cpu(0))
    mod1.bind(data_shapes=[('data', dshape)], label_shapes=None)

    sym2, _ = cell2.unroll(5, data, merge_outputs=True)
    mod2 = mx.mod.Module(sym2, label_names=None, context=mx.cpu(0))
    mod2.bind(data_shapes=[('data', dshape)], label_shapes=None)

    mod1.init_params()
    args, auxs = mod1.get_params()
    args = cell1.unpack_weights(args)
    args = cell2.pack_weights(args)
    mod2.set_params(args, auxs)

    batch = mx.io.DataBatch(data=[mx.random.uniform(shape=dshape)], label=[])
    mod1.forward(batch, is_train=False)
    mod2.forward(batch, is_train=False)

    assert_allclose(mod1.get_outputs()[0].asnumpy(), mod2.get_outputs()[0].asnumpy(), rtol=1e-2, atol=1e-4)

def test_fused_rnn():
    fused = mx.rnn.FusedRNNCell(100, num_layers=2, mode='rnn_relu', prefix='')

    stack = mx.rnn.SequentialRNNCell()
    stack.add(mx.rnn.RNNCell(100, activation='relu', prefix='l0_'))
    stack.add(mx.rnn.RNNCell(100, activation='relu', prefix='l1_'))

    check_rnn_consistency(fused, stack)
    check_rnn_consistency(stack, fused)

def test_fused_rnn_tanh():
    fused = mx.rnn.FusedRNNCell(100, num_layers=2, mode='rnn_tanh', prefix='')

    stack = mx.rnn.SequentialRNNCell()
    stack.add(mx.rnn.RNNCell(100, activation='tanh', prefix='l0_'))
    stack.add(mx.rnn.RNNCell(100, activation='tanh', prefix='l1_'))

    check_rnn_consistency(fused, stack)
    check_rnn_consistency(stack, fused)


def test_fused_lstm():
    fused = mx.rnn.FusedRNNCell(100, num_layers=2, mode='lstm', prefix='')

    stack = mx.rnn.SequentialRNNCell()
    stack.add(mx.rnn.LSTMCell(100, prefix='l0_'))
    stack.add(mx.rnn.LSTMCell(100, prefix='l1_'))

    check_rnn_consistency(fused, stack)
    check_rnn_consistency(stack, fused)


def test_fused_lstm_forget_bias():
    forget_bias = 2.0
    fused = mx.rnn.FusedRNNCell(10, forget_bias=forget_bias, num_layers=2, mode='lstm', prefix='')

    dshape = (32, 1, 20)
    data = mx.sym.Variable('data')

    sym, _ = fused.unroll(1, data, merge_outputs=True)
    mod = mx.mod.Module(sym, label_names=None, context=mx.cpu(0))
    mod.bind(data_shapes=[('data', dshape)], label_shapes=None)

    mod.init_params()

    args, auxs = mod.get_params()
    args = fused.unpack_weights(args)

    bias_name = next(x for x in args if x.endswith('f_bias'))
    expected_bias = forget_bias * np.ones(10, )
    assert_allclose(args[bias_name].asnumpy(), expected_bias)

def test_residual_fused():
    cell = mx.rnn.ResidualCell(
            mx.rnn.FusedRNNCell(50, num_layers=3, mode='lstm',
                               prefix='rnn_', dropout=0.5))

    inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(2)]
    outputs, _ = cell.unroll(2, inputs, merge_outputs=None)
    assert sorted(cell.params._params.keys()) == \
           ['rnn_parameters']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10, 50), rnn_t1_data=(10, 50))
    assert outs == [(10, 2, 50)]
    outputs = outputs.eval(ctx=mx.cpu(0),
                           rnn_t0_data=mx.nd.ones((10, 50), ctx=mx.cpu(0))+5,
                           rnn_t1_data=mx.nd.ones((10, 50), ctx=mx.cpu(0))+5,
                           rnn_parameters=mx.nd.zeros((61200,), ctx=mx.cpu(0)))
    expected_outputs = np.ones((10, 2, 50))+5
    assert np.array_equal(outputs[0].asnumpy(), expected_outputs)

def test_bidirectional():
    fused = mx.rnn.FusedRNNCell(100, num_layers=2, mode='lstm', prefix='',
            bidirectional=True)

    stack = mx.rnn.SequentialRNNCell()
    stack.add(mx.rnn.BidirectionalCell(
                mx.rnn.LSTMCell(100, prefix='l0_'),
                mx.rnn.LSTMCell(100, prefix='r0_'),
                output_prefix='bi_lstm_0_'))
    stack.add(mx.rnn.BidirectionalCell(
                mx.rnn.LSTMCell(100, prefix='l1_'),
                mx.rnn.LSTMCell(100, prefix='r1_'),
                output_prefix='bi_lstm_1_'))

    check_rnn_consistency(fused, stack)
    check_rnn_consistency(stack, fused)

def test_unfuse():
    for mode in ['rnn_tanh', 'rnn_relu', 'lstm']:
        fused = mx.rnn.FusedRNNCell(
            100, num_layers=2, mode=mode,
            prefix='test_%s'%mode,
            bidirectional=True,
            dropout=0.5)

        stack = fused.unfuse()

        check_rnn_consistency(fused, stack)
        check_rnn_consistency(stack, fused)

###################################
# GRU NOT SUPPORTED
###################################
# def test_fused_gru():
#     fused = mx.rnn.FusedRNNCell(100, num_layers=2, mode='gru', prefix='')

#     stack = mx.rnn.SequentialRNNCell()
#     stack.add(mx.rnn.GRUCell(100, prefix='l0_'))
#     stack.add(mx.rnn.GRUCell(100, prefix='l1_'))

#     check_rnn_consistency(fused, stack)
#     check_rnn_consistency(stack, fused)


# def test_fused_bidirectional():
#     fused = mx.rnn.FusedRNNCell(100, num_layers=2, mode='gru', prefix='',
#             bidirectional=True)

#     stack = mx.rnn.SequentialRNNCell()
#     stack.add(mx.rnn.BidirectionalCell(
#                 mx.rnn.GRUCell(100, prefix='l0_'),
#                 mx.rnn.GRUCell(100, prefix='r0_'),
#                 output_prefix='bi_gru_0_'))
#     stack.add(mx.rnn.BidirectionalCell(
#                 mx.rnn.GRUCell(100, prefix='l1_'),
#                 mx.rnn.GRUCell(100, prefix='r1_'),
#                 output_prefix='bi_gru_1_'))

#     check_rnn_consistency(fused, stack)
#     check_rnn_consistency(stack, fused)

if __name__ == '__main__':
    import nose
    nose.runmodule()
