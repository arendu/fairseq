#!/usr/bin/env python
__author__ = 'arenduchintala'
import math
import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('adv_cross_entropy')
class AdversarialCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.lam = args.adv_lambda
        self.ease_in = args.adv_ease
        tgt_dict = task.target_dictionary
        self.adv_classes = (tgt_dict.index('<en>'), tgt_dict.index('<de>'))
        self.adv_classnames = ['en', 'de']
        self.use_continue_tags = False
        if tgt_dict.index('<cde>') == 3 or tgt_dict.index('<cen>') == 3:
            ##print('not detected use of continued language labels')
            self.use_continue_tags = False
        else:
            ##print('detected use of continued language labels')
            self.continue_adv_classes = (tgt_dict.index('<cen>'), tgt_dict.index('<cde>'))
            self.continue_adv_classenames = ['cen', 'cde']
            self.use_continue_tags = True

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--adv-lambda', default=1.0, type=float, metavar='D',
                            help='lambda for adversarial loss, 0 means no adv loss is applied')
        parser.add_argument('--adv-ease', default=1, type=int, metavar='D', choices=[0, 1],
                            help='decides if lambda should be eased in or constant')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        ##print('\nhere', sample.get('epoch_num', None))
        epoch_scale = sample.get('epoch_num', None) 

        lprobs, adv_lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(lprobs, target, size_average=False, ignore_index=self.padding_idx,
                          reduce=reduce)
        assert not self.args.sentence_avg, "not supporting sentence_avg"
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        l1_sample_size = 0
        l2_sample_size = 0
        l1_loss = torch.tensor(0).type_as(loss)
        l2_loss = torch.tensor(0).type_as(loss)
        if sample.get('target_labels', None) is not None:
            tar = sample['target']
            tar_lang = sample['target_labels']
            tar = tar.view(-1)
            tar_lang = tar_lang.squeeze(2).view(-1)

            l1_probs = lprobs[tar_lang == self.adv_classes[0], :]   # we count EOS as l1
            l1_tar = tar[tar_lang == self.adv_classes[0]]
            l2_probs = lprobs[tar_lang == self.adv_classes[1], :]
            l2_tar = tar[tar_lang == self.adv_classes[1]]

            ##l3_probs = lprobs[tar_lang == 2]
            ##l3_tar = tar[tar_lang == 2]
            if l1_tar.size(0) > 0:
                l1_loss = F.nll_loss(l1_probs, l1_tar, size_average=False, reduce=reduce)
            else:
                l1_loss = torch.tensor(0.0).type_as(lprobs)
            if l2_tar.size(0) > 0:
                l2_loss = F.nll_loss(l2_probs, l2_tar, size_average=False, reduce=reduce)
            else:
                l2_loss = torch.tensor(0.0).type_as(lprobs)

            ##l3_loss = F.nll_loss(l3_probs, l3_tar, size_average=False, reduce=reduce) if l3_tar.size(0) > 0 else torch.tensor(0.0)
            l1_sample_size = l1_tar.size(0)
            l2_sample_size = l2_tar.size(0)
            if self.use_continue_tags:
                l1_continue_probs = lprobs[tar_lang == self.continue_adv_classes[0], :]   # we count EOS as l1
                l1_continue_tar = tar[tar_lang == self.continue_adv_classes[0]]
                l2_continue_probs = lprobs[tar_lang == self.continue_adv_classes[1], :]
                l2_continue_tar = tar[tar_lang == self.continue_adv_classes[1]]

                if l1_continue_tar.size(0) > 0:
                    l1_continue_loss = F.nll_loss(l1_continue_probs, l1_continue_tar, size_average=False, reduce=reduce)
                else:
                    l1_continue_loss = torch.tensor(0.0).type_as(lprobs)
                if l2_continue_tar.size(0) > 0:
                    l2_continue_loss = F.nll_loss(l2_continue_probs, l2_continue_tar, size_average=False, reduce=reduce)
                else:
                    l2_continue_loss = torch.tensor(0.0).type_as(lprobs)
                l1_loss += l1_continue_loss
                l2_loss += l2_continue_loss

                l1_sample_size += l1_continue_tar.size(0)
                l2_sample_size += l2_continue_tar.size(0)
            #assert abs(l1_loss.item() + l2_loss.item() + l3_loss.item() - loss.item()) < 1e-2, \
            #    "l1 and l2 not matching loss" + str(l1_loss.item()) + " " + str(l2_loss.item()) + \
            #    " " + str(l3_loss.item()) + " " + str(loss.item())
        adv_sample_size = 0
        if sample.get('target_labels', None) is not None:
            adv_target = sample['target_labels']
            adv_target = adv_target.squeeze(2)
            adv_mask = (adv_target == self.adv_classes[0]) + (adv_target == self.adv_classes[1])
            adv_masked_target = adv_target[adv_mask]
            adv_masked_target[adv_masked_target == self.adv_classes[0]] = 0
            adv_masked_target[adv_masked_target == self.adv_classes[1]] = 1
            adv_masked_lprobs = adv_lprobs[adv_mask, :]
            adv_sample_size = adv_masked_target.size(0)
            adv_loss = F.nll_loss(adv_masked_lprobs, adv_masked_target, size_average=False, reduce=reduce)
            if model.training:
                if epoch_scale is not None and self.ease_in:
                    e_scale = min((float(epoch_scale) - 1) / 10.0, 1.0)
                    e_scale = 0.0 if e_scale < 0.0 else e_scale
                else:
                    e_scale = 1.0
                loss = loss + (self.lam * e_scale * adv_loss)
            else:
                loss = loss + (0.0 * adv_loss)
                #when this was set to 0.0 the performance dropped so im putting it back on
        else:
            adv_loss = torch.tensor(0.0)
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'adv_loss': utils.item(adv_loss.data) if reduce else adv_loss.data,
            'l1_loss': utils.item(l1_loss.data) if reduce else l1_loss.data,
            'l2_loss': utils.item(l2_loss.data) if reduce else l2_loss.data,
            'l1_ntokens': l1_sample_size,
            'l2_ntokens': l2_sample_size,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'l1_sample_size': l1_sample_size,
            'l2_sample_size': l2_sample_size,
            'adv_sample_size': adv_sample_size,
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        adv_loss_sum = sum(log.get('adv_loss', 0) for log in logging_outputs)
        l1_loss_sum = sum(log.get('l1_loss', 0) for log in logging_outputs)
        l2_loss_sum = sum(log.get('l2_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        l1_sample_size = sum(log.get('l1_sample_size', 0) for log in logging_outputs)
        l2_sample_size = sum(log.get('l2_sample_size', 0) for log in logging_outputs)
        adv_sample_size = sum(log.get('adv_sample_size', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        if l1_sample_size == 0:
            assert l1_loss_sum == 0, "should be zero but got " + str(l1_loss_sum)
            l1_loss_avg = 0.
        else:
            l1_loss_avg = l1_loss_sum / l1_sample_size
        if l2_sample_size == 0:
            assert l2_loss_sum == 0, "should be zero but got " + str(l2_loss_sum)
            l2_loss_avg = 0.
        else:
            l2_loss_avg = l2_loss_sum / l2_sample_size

        if adv_sample_size == 0:
            adv_loss_avg = 0
        else:
            adv_loss_avg = adv_loss_sum / adv_sample_size

        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'adv_loss': adv_loss_avg / math.log(2),
            'l1_loss': l1_loss_avg / math.log(2),
            'l2_loss': l2_loss_avg / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
