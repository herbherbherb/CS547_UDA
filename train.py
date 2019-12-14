import os
import json
import tensorflow as tf

import numpy as np
from copy import deepcopy
from typing import NamedTuple
from tqdm import tqdm

import torch
import torch.nn as nn

# from utils.logger import Logger
from tensorboardX import SummaryWriter
from utils.utils import output_logging


class Trainer(object):
    """Training Helper class"""
    def __init__(self, cfg, model, data_iter, optimizer, device,args):
        self.args = args
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.device = device

        # data iter
        if len(data_iter) == 1:
            self.sup_iter = data_iter[0]
        elif len(data_iter) == 2:
            self.sup_iter = self.repeat_dataloader(data_iter[0])
            self.unsup_iter = self.repeat_dataloader(data_iter[1])
        elif len(data_iter) == 3:
            self.sup_iter = self.repeat_dataloader(data_iter[0])
            self.unsup_iter = self.repeat_dataloader(data_iter[1])
            self.eval_iter = data_iter[2]

    def train(self, get_loss, get_acc, model_file, pretrain_file):
        """ train uda"""

        # tensorboardX logging
        if self.cfg.results_dir:
            logger = SummaryWriter(log_dir=os.path.join(self.cfg.results_dir, 'logs'))

        self.model.train()
        self.load(model_file, pretrain_file)    # between model_file and pretrain_file, only one model will be loaded
        model = self.model.to(self.device)
        if self.cfg.data_parallel:                       # Parallel GPU mode
            model = nn.DataParallel(model)

        global_step = 0
        loss_sum = 0.
        max_acc = [0., 0]   # acc, step

        # Progress bar is set by unsup or sup data
        # uda_mode == True --> sup_iter is repeated
        # uda_mode == False --> sup_iter is not repeated
        if self.args.debug:
            self.args.total_steps = 1
            print('^ ^ You are in Debug mode ^ ^')

        iter_bar = tqdm(self.unsup_iter, total=self.args.total_steps) if self.cfg.uda_mode \
              else tqdm(self.sup_iter, total=self.args.total_steps)
        for i, batch in enumerate(iter_bar):
                
            # Device assignment
            if self.cfg.uda_mode:
                sup_batch = [t.to(self.device) for t in next(self.sup_iter)]
                unsup_batch = [t.to(self.device) for t in batch]
            else:
                sup_batch = [t.to(self.device) for t in batch]
                unsup_batch = None

            # update
            self.optimizer.zero_grad()
            final_loss, sup_loss, unsup_loss = get_loss(model, sup_batch, unsup_batch, global_step,self.args)
            final_loss.backward()
            self.optimizer.step()

            # print loss
            global_step += 1
            loss_sum += final_loss.item()
            if self.cfg.uda_mode:
                iter_bar.set_description('final=%5.3f unsup=%5.3f sup=%5.3f'\
                        % (final_loss.item(), unsup_loss.item(), sup_loss.item()))
            else:
                iter_bar.set_description('loss=%5.3f' % (final_loss.item()))

            # logging            
            if self.cfg.uda_mode:
                logger.add_scalars('data/scalar_group',
                                    {'final_loss': final_loss.item(),
                                     'sup_loss': sup_loss.item(),
                                     'unsup_loss': unsup_loss.item(),
                                     'lr': self.optimizer.get_lr()[0]
                                    }, global_step)
            else:
                logger.add_scalars('data/scalar_group',
                                    {'sup_loss': final_loss.item()}, global_step)

            if global_step % self.cfg.save_steps == 0:
                self.save(global_step)

            if get_acc and global_step % self.cfg.check_steps == 0 and global_step > 4999:
                results = self.eval(get_acc, None, model)
                total_accuracy = torch.cat(results).mean().item()
                logger.add_scalars('data/scalar_group', {'eval_acc' : total_accuracy}, global_step)
                if max_acc[0] < total_accuracy:
                    self.save(global_step)
                    max_acc = total_accuracy, global_step
                print('Accuracy : %5.3f' % total_accuracy)
                print('Max Accuracy : %5.3f Max global_steps : %d Cur global_steps : %d' %(max_acc[0], max_acc[1], global_step), end='\n\n')

            if self.cfg.total_steps and self.cfg.total_steps < global_step:
                print('The total steps have been reached')
                print('Average Loss %5.3f' % (loss_sum/(i+1)))
                if get_acc:
                    results = self.eval(get_acc, None, model)
                    total_accuracy = torch.cat(results).mean().item()
                    logger.add_scalars('data/scalar_group', {'eval_acc' : total_accuracy}, global_step)
                    if max_acc[0] < total_accuracy:
                        max_acc = total_accuracy, global_step                
                    print('Accuracy :', total_accuracy)
                    print('Max Accuracy : %5.3f Max global_steps : %d Cur global_steps : %d' %(max_acc[0], max_acc[1], global_step), end='\n\n')
                self.save(global_step)
                return
        return global_step

    def eval(self, evaluate, model_file, model):
        """ evaluation function """
        if model_file:
            self.model.eval()
            self.load(model_file, None)
            model = self.model.to(self.device)
            if self.cfg.data_parallel:
                model = nn.DataParallel(model)

        results = []
        iter_bar = tqdm(self.sup_iter) if model_file \
            else tqdm(deepcopy(self.eval_iter))
        for batch in iter_bar:
            batch = [t.to(self.device) for t in batch]

            with torch.no_grad():
                accuracy, result = evaluate(model, batch)
            results.append(result)

            iter_bar.set_description('Eval Acc=%5.3f' % accuracy)
        return results
            
    def load(self, model_file, pretrain_file):

        if model_file:
            print('Loading the model from', model_file)
            if torch.cuda.is_available():
                self.model.load_state_dict(torch.load(model_file))
            else:
                self.model.load_state_dict(torch.load(model_file, map_location='cpu'))

        elif pretrain_file:
            print('Loading the pretrained model from', pretrain_file)
            self.load_model(self.model, pretrain_file)
            print('Loading is done !')
   
    ### Load Bert model ### 
    def load_model(self,model, checkpoint_file):
        ### Only load pretrained model in tensorflow ###
        def load_param(checkpoint_file, conversion_table):
            for pyt_param, tf_param_name in conversion_table.items():
                tf_param = tf.train.load_variable(checkpoint_file, tf_param_name)

                # for weight(kernel), we should do transpose --> pytorch, tensorflow 다름
                if tf_param_name.endswith('kernel'):
                    tf_param = np.transpose(tf_param)

                assert pyt_param.size() == tf_param.shape, \
                    'Dim Mismatch: %s vs %s ; %s' % (tuple(pyt_param.size()), tf_param.shape, tf_param_name)

                # assign pytorch tensor from tensorflow param
                pyt_param.data = torch.from_numpy(tf_param)

        

        ### Embedding layer ###
        e, p = model, 'bert/embeddings/'
        load_param(checkpoint_file, {
            e.tok_embed.weight: p+'word_embeddings',
            e.pos_embed.weight: p+'position_embeddings',
            e.seg_embed.weight: p+'token_type_embeddings',
            e.norm.gamma:       p+'LayerNorm/gamma',
            e.norm.beta:        p+'LayerNorm/beta'
        })

        ### Transformer blocks ###
        for i in range(len(model.blocks)):
            b, p = model.blocks[i], "bert/encoder/layer_%d/"%i
            load_param(checkpoint_file, {
                b.attn.proj_q.weight:   p+"attention/self/query/kernel",
                b.attn.proj_q.bias:     p+"attention/self/query/bias",
                b.attn.proj_k.weight:   p+"attention/self/key/kernel",
                b.attn.proj_k.bias:     p+"attention/self/key/bias",
                b.attn.proj_v.weight:   p+"attention/self/value/kernel",
                b.attn.proj_v.bias:     p+"attention/self/value/bias",
                b.proj.weight:          p+"attention/output/dense/kernel",
                b.proj.bias:            p+"attention/output/dense/bias",
                b.fc1.weight:           p+"intermediate/dense/kernel",
                b.fc1.bias:             p+"intermediate/dense/bias",
                b.fc2.weight:           p+"output/dense/kernel",
                b.fc2.bias:             p+"output/dense/bias",
                b.norm1.gamma:          p+"attention/output/LayerNorm/gamma",
                b.norm1.beta:           p+"attention/output/LayerNorm/beta",
                b.norm2.gamma:          p+"output/LayerNorm/gamma",
                b.norm2.beta:           p+"output/LayerNorm/beta",
            })





    def save(self, i):
        """ save model """
        if not os.path.isdir(os.path.join(self.cfg.results_dir, 'save')):
            os.makedirs(os.path.join(self.cfg.results_dir, 'save'))
        torch.save(self.model.state_dict(),
                        os.path.join(self.cfg.results_dir, 'save', 'model_steps_'+str(i)+'.pt'))

    def repeat_dataloader(self, iteralbe):
        """ repeat dataloader """
        while True:
            for x in iteralbe:
                yield x
