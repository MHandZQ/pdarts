import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
'''
用于参数 α 的更新
'''

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    # 用来更新α的优化器Adam
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  """
     我们更新梯度就是theta = theta + v + weight_decay * theta 
       1.theta就是我们要更新的参数
       2.weight_decay*theta为正则化项用来防止过拟合
       3.v的值我们分带momentum和不带momentum：
         普通的梯度下降：v = -dtheta * lr 其中lr是学习率，dx是目标函数对x的一阶导数
         带momentum的梯度下降：v = lr*(-dtheta + v * momentum)
  """

  # 【完全复制外面的Network更新w的过程】，对应公式6第一项的w − ξ*dwLtrain(w, α)
  # 不直接用外面的optimizer来进行w的更新，而是自己新建一个unrolled_model展开，主要是因为我们这里的更新不能对Network的w进行更新
  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    loss = self.model._loss(input, target) # 公式(6)中的Ltrain
    theta = _concat(self.model.parameters()).data
    try:
      # momentum*v,用的就是Network进行w更新的momentum
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    # 前面的是loss对参数theta求梯度，后面的self.network_weight_decay*theta就是正则项
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
    # 对参数进行更新，等价于optimizer.step()
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))#unrolled_model =  w − ξ*dwLtrain(w, α)
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
    self.optimizer.zero_grad()# 清除上一步的残余更新参数值
    if unrolled:#用论文的提出的方法,eta是学习率
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:#不用论文提出的bilevel optimization，只是简单的对α求导
        self._backward_step(input_valid, target_valid)
    self.optimizer.step() # 更新架构参数α

  def _backward_step(self, input_valid, target_valid):
    # 直接进行反向传播
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    # unrolled_model就是实现了权重一次更新 w' = w − ξ*dwLtrain(w, α) 的模型
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    # Lval(w',α)
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)#对做了一次更新后的w的unrolled_model求验证集的损失，Lval，以用来对α进行更新

    unrolled_loss.backward()
    # dαLval(w',α)
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]# 对α求梯度,unrolled_model.arch_parameters()调用model_search.py中的arch_parameters函数,返回架构参数α
    # dw'Lval(w',α)
    vector = [v.grad.data for v in unrolled_model.parameters()]# 对w'求导,unrolled_model.parameters()得到w'
    # 计算公式八(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)
    # 其中w+=w + dw'Lval(w',α)*epsilon，w- = w - dw'Lval(w',α)*epsilon
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)
    # 计算公式（7）,第二项由公式（8）近似代替,即dαLval(w',α)-(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)
    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)
    # 到此获得了 公式(6)
    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    # 不是直接使用model_search.py中构建的模型,而是调用了其中的new()函数,返回一个复制的新模型
    # 目的在于在架构参数更新的时候,对权重参数进行一次更新是不影响原模型的权重参数的
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params) # 新模型中的参数已经更新为做一次反向传播后的值
    model_new.load_state_dict(model_dict)# 将刚更新的参数加载到新模型中,就实现了对模型中权重的一次更新
    return model_new.cuda()              # 即实现了 w − ξ*dwLtrain(w, α)

  # #计算公式八(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)
  # 其中w+=w+dw'Lval(w',α)*epsilon w- = w-dw'Lval(w',α)*epsilon
  def _hessian_vector_product(self, vector, input, target, r=1e-2): # vector = dw'Lval(w',α)
    R = r / _concat(vector).norm()# epsilon

    # dαLtrain(w+,α)
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)# 把模型中所有的w'更新成w+=w+dw'Lval(w',α)*epsilon
    loss = self.model._loss(input, target)# Ltrain(w+,α)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())# dαLtrain(w+,α)

    # dαLtrain(w-,α)
    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)# 把模型中所有的w'更新成w-=w-dw'Lval(w',α)*epsilon
    loss = self.model._loss(input, target)# Ltrain(w-,α)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())# dαLtrain(w-,α)

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

