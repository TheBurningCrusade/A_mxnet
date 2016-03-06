#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import os
import mxnet as mx
from common import models
import pickle as pkl

def test_symbol_basic():
    print "======================test_symbol_basic========================"
    """models.mlp2　会产生一个含有两个隐藏层的mlp
    list_arguments: ['data', 'fc1_weight', 'fc1_bias', 'fc2_weight', 'fc2_bias']
    list_outputs: ['fc2_output']"""
    mlist = []
    mlist.append(models.mlp2())
    for m in mlist:
        m.list_arguments()
        print 'list_argumens'
        print m.list_arguments()
        m.list_outputs()
        print 'list_outputs'
        print m.list_outputs()
    print "======================test_symbol_basic========================"

def test_symbol_compose():
    """这个测试表示我们可以将两个网络join起来然后就胡形成一个网络，如果
    我们想要截取一部分网络作为输出，那么可以使用Group保存一段网络的输出.
    这里定义网络拓扑时候有一个relu 类型的activation但并不是作为最后的输出
    层这个还不清楚是干嘛的"""
    data = mx.symbol.Variable('data')
    net1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=10)
    net1 = mx.symbol.FullyConnected(data=net1, name='fc2', num_hidden=100)
    net1.list_arguments() == ['data',
                              'fc1_weight', 'fc1_bias',
                              'fc2_weight', 'fc2_bias']

    net2 = mx.symbol.FullyConnected(name='fc3', num_hidden=10)
    net2 = mx.symbol.Activation(data=net2, act_type='relu')
    net2 = mx.symbol.FullyConnected(data=net2, name='fc4', num_hidden=20)
    """print 'net2 debug'
    print(net2.debug_str())
    print 'compos debug' """
    composed = net2(fc3_data=net1, name='composed')
    #print(composed.debug_str())
    multi_out = mx.symbol.Group([composed, net1])
    print "multi_out arguments"
    print multi_out.list_arguments()
    assert len(multi_out.list_outputs()) == 2
    print "multi_out outputs"
    print multi_out.list_outputs()


def test_symbol_copy():
    data = mx.symbol.Variable('data')
    data_2 = copy.deepcopy(data)
    data_3 = copy.copy(data)
    assert data.tojson() == data_2.tojson()
    assert data.tojson() == data_3.tojson()


def test_symbol_internal():
    print "===================test_symbol_internal=========================="
    """get_internals回从原来的symbol中得到一个symobol这个symbol中包含所有
    隐藏层的输出,可以用list_outputs打印出来"""
    data = mx.symbol.Variable('data')
    oldfc = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=10)
    net1 = mx.symbol.FullyConnected(data=oldfc, name='fc2', num_hidden=100)
    print 'net1.list_arguments()'
    print net1.list_arguments()
    net1.list_arguments() == ['data',
                              'fc1_weight', 'fc1_bias',
                              'fc2_weight', 'fc2_bias']
    internal =  net1.get_internals()
    print internal.list_arguments()
    print internal.list_outputs()
    fc1 = internal['fc1_output']
    print fc1.list_arguments()
    print oldfc.list_arguments()
    assert fc1.list_arguments() == oldfc.list_arguments()
    print "===================test_symbol_inernal==========================="


def test_symbol_pickle():
    mlist = [models.mlp2(), models.conv()]
    data = pkl.dumps(mlist)
    mlist2 = pkl.loads(data)
    for x, y  in zip(mlist, mlist2):
        assert x.tojson() == y.tojson()


def test_symbol_saveload():
    sym = models.mlp2()
    fname = 'tmp_sym.json'
    sym.save(fname)
    data2 = mx.symbol.load(fname)
    # save because of order
    assert sym.tojson() == data2.tojson()
    os.remove(fname)


if __name__ == '__main__':
    #test_symbol_internal()
    #test_symbol_basic()
    #test_symbol_compose()
    #test_symbol_saveload()
    test_symbol_pickle()
