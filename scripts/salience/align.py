# -*- coding: utf-8 -*-
#
# Copyright © 2019 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2019-02-19
#
# Distributed under terms of the MIT license.

import numpy as np
import pdb
import sys
import torch

from fairseq import data, options, progress_bar, tasks, tokenizer, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_generator import SequenceGenerator
from fairseq.sequence_scorer import SequenceScorer
from fairseq.models import SaliencyManager
from fairseq.models.lstm import LSTMEncoder

class NLForwardValueManager:
    """
    The forward switch is added to take care of the case where
    multiple backward is done for one forward.

    TODO: We may no loner need this functionality while
    refactoring the smoothgrad part.
    """
    forward_switch = True
    nl_forward_values = []
    buf = []

    @classmethod
    def switch_forword(cls):
        if not cls.forward_switch:
            cls.buf = []
            cls.nl_forward_values = []
            cls.forward_switch = True

    @classmethod
    def switch_backward(cls):
        if cls.forward_switch:
            cls.buf = cls.nl_forward_values.copy()
            cls.forward_switch = False

    @classmethod
    def append_value(cls, val):
        cls.nl_forward_values.append(val)

    @classmethod
    def pop_value(cls):
        val = cls.buf[-1]
        del cls.buf[-1]
        if len(cls.buf) == 0:
            cls.buf = cls.nl_forward_values.copy()
        return val


def parallel_buffered_read(src_stream, tgt_stream, buffer_size):
    buffer = []
    for src_str, tgt_str in zip(src_stream, tgt_stream):
        buffer.append((src_str.strip(), tgt_str.strip()))
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer

def make_batches(src_lines, tgt_lines, args, task, max_positions):
    src_tokens = [
        tokenizer.Tokenizer.tokenize(src_str, task.source_dictionary, add_if_not_exist=False).long()
        for src_str in src_lines
    ]
    tgt_tokens = [
        tokenizer.Tokenizer.tokenize(tgt_str, task.target_dictionary, add_if_not_exist=False).long()
        for tgt_str in tgt_lines
    ]

    src_lengths = np.array([t.numel() for t in src_tokens])
    tgt_lengths = np.array([t.numel() for t in tgt_tokens])

    itr = task.get_batch_iterator(
        dataset=data.LanguagePairDataset(src_tokens, src_lengths, task.source_dictionary, \
                                         tgt_tokens, tgt_lengths, task.target_dictionary),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    return itr


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _ = utils.load_ensemble_for_inference(args.path.split(':'), task, model_arg_overrides=eval(args.model_overrides))
    model = models[0]  # assume only one model for now
    if type(model.encoder) == LSTMEncoder:
        model.encoder.training = False
        model.decoder.training = False
    else:
        model.eval()  # turn off dropout, will kill cudnn rnn backward
    if use_cuda:
        model.cuda()
    saliencies = []
    attns = []

    def retain_nl_forward(module, input, output):
        assert(len(input) == 1)
        NLForwardValueManager.switch_forword()
        NLForwardValueManager.append_value(input[0])

    def guided_hook(module, grad_in, grad_out):

        assert(len(grad_in) == 1)
        assert(len(grad_out) == 1)

        grad_in = grad_in[0]
        grad_out = grad_out[0]

        NLForwardValueManager.switch_backward()
        forward_value = NLForwardValueManager.pop_value()
        fw_mask_p = torch.zeros_like(forward_value)
        fw_mask_p[forward_value > 0] = 1
        fw_mask_n = torch.zeros_like(forward_value)
        fw_mask_n[forward_value < 0] = 1
        bw_mask_p = torch.zeros_like(grad_in)
        bw_mask_p[grad_in > 0] = 1
        bw_mask_n = torch.zeros_like(grad_in)
        bw_mask_n[grad_in < 0] = 1
        grad_in = grad_in * (fw_mask_p * bw_mask_p + fw_mask_n * bw_mask_n)
        return (grad_in,)

    if args.saliency == "guided":
        for module in model.modules():
            if type(module) == torch.nn.ReLU or type(module) == torch.nn.GLU:
                module.register_forward_hook(retain_nl_forward)
                module.register_backward_hook(guided_hook)

    # TODO: eliminate white baseline for the moment
    # used only when saliency == "integral"
    # background = torch.mean(model.encoder.embed_tokens.weights, dim=0)  # (emb_dim,)
    # if args.baseline == "b":
    #     background = background * 0.0
    # model.encoder.background = background

    def process_batch(batch):
        if use_cuda:
            batch = utils.move_to_cuda(batch)
        net_input = {}
        for key in batch['net_input'].keys():
            orig_size = batch['net_input'][key].size()
            orig_size = list(orig_size)
            new_size = tuple([orig_size[0] * args.n_samples] + orig_size[1:])
            net_input[key] = batch['net_input'][key].expand(*new_size)  # TODO: this will not work for batch_size != 1
        net_input['smoothing_factor'] = args.smoothing_factor

        if args.abs:
            net_input['abs_saliency'] = True

        # TODO: assuming batch_size = 1 for this if
        # clean up later
        if args.saliency == "integral":
            alpha = torch.pow(torch.arange(0, 1, 1 / args.n_samples) + 1 / args.n_samples, 2)  # (0, 1] rather than [0, 1)
            batch_size = batch['net_input']['src_tokens'].size(0)
            alpha = alpha.unsqueeze(0).expand(batch_size, args.n_samples)
            alpha = alpha.view(batch_size * args.n_samples, -1).squeeze()  # TODO: make sure alpha and the batched input aligns
            alpha = alpha.cuda() if use_cuda else alpha
            net_input['alpha'] = alpha  # (batch_size * n_samples)

        target = batch['target']
        bsz, tlen = target.size()
        target = target.unsqueeze(2).expand(bsz * args.n_samples, tlen, 1)

        decoder_out = model(**net_input)
        # sample argument is only used for adaptive softmax, so don't worry about it
        # we don't want *negative* log likelihood -- all the setting is to maximize objective
        probs = model.get_normalized_probs(decoder_out, log_probs=True, sample=None)  # (batch_size * n_samples, target_len, vocab)
        target_probs = torch.gather(probs, -1, target).view(bsz, args.n_samples, tlen)  # (batch_size, n_samples, target_len)
        # this mean is taken mainly for speed reason
        # otherwise, we would have to iterate through n_samples as well, which is not necessary
        # as gradient will be 0 for prediction score that does not correspond to the input sample
        target_probs = torch.mean(target_probs, dim=1)  # (batch_size, target_len)
        for i in range(bsz):
            for j in range(tlen):
                target_probs[i, j].backward(retain_graph=True)
                model.zero_grad()

        # single sentence saliency will be a list with (tgt * n_samples) of (bsz, src)
        saliency = torch.stack(SaliencyManager.single_sentence_saliency, dim=1)  # (bsz * n_samples, tgt, src)
        saliency = saliency.view(bsz, args.n_samples, tlen, -1)  # (bsz, n_samples, tgt, src)
        saliency = torch.mean(saliency, dim=1)  # (bsz, tgt, src)
        # we don't need to multiply the  x - x' term for integrated because it's always 1

        if type(decoder_out[1]) == dict:
            attn = decoder_out[1]['attn']
        else:
            attn = decoder_out[1]
        SaliencyManager.clear_saliency()

        saliency = saliency.detach().cpu()
        attn = attn.detach().cpu()
        return saliency, attn

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )
    num_batches = 0
    for inputs in parallel_buffered_read( \
            open(args.data[0] + "/" + args.source_lang), \
            open(args.data[0] + "/" + args.target_lang), \
            args.buffer_size
        ):
        src_inputs, tgt_inputs = zip(*inputs)
        for batch in make_batches(src_inputs, tgt_inputs, args, task, max_positions):
            saliency, attn = process_batch(batch)
            saliencies.append(saliency)
            attns.append(attn)
            num_batches += 1
            if num_batches % 10 == 0:
                sys.stderr.write(".")
                sys.stderr.flush()

    torch.save(saliencies, open(args.out + ".sa", 'wb'))
    torch.save(attns, open(args.out + ".at", 'wb'))


if __name__ == '__main__':
    parser = options.get_generation_parser(True)
    parser.add_argument("--saliency", choices=["plain", "guided", "integral"], help="")
    parser.add_argument("--out", metavar="PATH", help="")
    parser.add_argument("--smoothing-factor", "-sf", type=float, default=0.0, help="")
    parser.add_argument("--n-samples", "-sn", type=int, default=1, help="")
    parser.add_argument("--abs", action='store_true', default=False, help="")
    parser.add_argument("--baseline", "-bl", choices=["b", "w"], help="black or white baseline")
    args = options.parse_args_and_arch(parser)
    main(args)
