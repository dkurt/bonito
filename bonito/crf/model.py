"""
Bonito CTC-CRF Model.
"""

import torch
import numpy as np
from bonito.nn import Permute, Scale, activations, rnns
from torch.nn import Sequential, Module, Linear, Tanh, Conv1d

if torch.cuda.is_available():
    import seqdist.sparse
    from seqdist.ctc_simple import logZ_cupy
else:
    from scipy import special
from seqdist.core import SequenceDist, Max, Log, semiring


if torch.cuda.is_available():
    def sparse_logZ(Ms, idx, alpha_0, beta_T, S):
        return seqdist.sparse.logZ(Ms, idx, alpha_0, beta_T, S)
else:
    def logZ_fwd_cpu(Ms, idx, v0, vT, S):
        T, N, C, NZ = Ms.shape
        Ms_grad = torch.zeros(T, N, C, NZ)

        a = v0
        for t in range(T):
            s = a[:, idx] + Ms[t]
            a = torch.logsumexp(s, -1)
            Ms_grad[t, :] = s
        return S.sum(a + vT, dim=1), Ms_grad

    def logZ_bwd_cpu(Ms, idx, vT, S):
        T, N, C, NZ = Ms.shape
        Ms = Ms.reshape(T, N, -1)
        idx_T = idx.flatten().argsort().to(dtype=torch.long).reshape(C, NZ)

        betas = torch.ones(T + 1, N, C)

        a = vT
        betas[T, :, :] = a[:, :]

        for t in reversed(range(T)):
            s = a[:, idx_T // NZ] + Ms[t, :, idx_T]
            a = torch.logsumexp(s, -1)
            betas[t, :] = a[:]
        return betas

    class _LogZ(torch.autograd.Function):
        @staticmethod
        def forward(ctx, Ms, idx, v0, vT, S:semiring):
            idx = idx.to(dtype=torch.long, device=Ms.device)
            logZ, Ms_grad = logZ_fwd_cpu(Ms, idx, v0, vT, S)
            ctx.save_for_backward(Ms_grad, Ms, idx, vT)
            ctx.semiring = S
            return logZ

        @staticmethod
        def backward(ctx, grad):
            Ms_grad, Ms, idx, vT = ctx.saved_tensors
            S = ctx.semiring
            T, N, C, NZ = Ms.shape
            betas = logZ_bwd_cpu(Ms, idx, vT, S)
            Ms_grad = S.mul(Ms_grad, betas[1:,:,:,None])
            Ms_grad = S.dsum(Ms_grad.reshape(T, N, -1), dim=2).reshape(T, N, C, NZ)
            return grad[None, :, None, None] * Ms_grad, None, None, None, None, None

    def sparse_logZ(Ms, idx, v0, vT, S:semiring=Log):
        return _LogZ.apply(Ms, idx, v0, vT, S)


class Model(Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.stride = config['encoder']['stride']
        self.alphabet = config['labels']['labels']
        self.seqdist = CTC_CRF(config['global_norm']['state_len'], self.alphabet)

        insize = config['input']['features']
        winlen = config['encoder']['winlen']
        activation = activations[config['encoder']['activation']]()

        rnn = rnns[config['encoder']['rnn_type']]
        size = config['encoder']['features']

        self.encoder = Sequential(
            conv(insize, 4, ks=5, bias=True), activation,
            conv(4, 16, ks=5, bias=True), activation,
            conv(16, size, ks=winlen, stride=self.stride, bias=True), activation,
            Permute(2, 0, 1),
            rnn(size, size, reverse=True), rnn(size, size),
            rnn(size, size, reverse=True), rnn(size, size),
            rnn(size, size, reverse=True),
            Linear(size, self.seqdist.n_score(), bias=True),
            Tanh(),
            Scale(config['encoder']['scale']),
        )
        self.global_norm = GlobalNorm(self.seqdist)

    def forward(self, x):
        return self.global_norm(self.encoder(x))

    def decode(self, x, beamsize=5, threshold=1e-3, qscores=False, return_path=False):
        scores = self.seqdist.posteriors(x.to(torch.float32).unsqueeze(1)) + 1e-8
        tracebacks = self.seqdist.viterbi(scores.log()).to(torch.int16).T
        return self.seqdist.path_to_str(tracebacks.cpu().numpy())


def conv(c_in, c_out, ks, stride=1, bias=False, dilation=1, groups=1):
    if stride > 1 and dilation > 1:
        raise ValueError("Dilation and stride can not both be greater than 1")
    return Conv1d(
        c_in, c_out, ks, stride=stride, padding=(ks // 2) * dilation,
        bias=bias, dilation=dilation, groups=groups
    )


class GlobalNorm(Module):

    def __init__(self, seq_dist):
        super().__init__()
        self.seq_dist = seq_dist

    def forward(self, x):
        scores = x.to(torch.float32)
        return (scores - self.seq_dist.logZ(scores)[:, None] / len(scores)).to(x.dtype)


class CTC_CRF(SequenceDist):

    def __init__(self, state_len, alphabet):
        super().__init__()
        self.alphabet = alphabet
        self.state_len = state_len
        self.n_base = len(alphabet[1:])
        self.idx = torch.cat([
            torch.arange(self.n_base**(self.state_len))[:, None],
            torch.arange(
                self.n_base**(self.state_len)
            ).repeat_interleave(self.n_base).reshape(self.n_base, -1).T
        ], dim=1).to(torch.int32)

    def n_score(self):
        return len(self.alphabet) * self.n_base**(self.state_len)

    def logZ(self, scores, S:semiring=Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, len(self.alphabet))
        alpha_0 = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        beta_T = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        return sparse_logZ(Ms, self.idx, alpha_0, beta_T, S)

    def backward_scores(self, scores, S: semiring=Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        beta_T = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        return seqdist.sparse.logZ_bwd_cupy(Ms, self.idx, beta_T, S, K=1)

    def viterbi(self, scores):
        traceback = self.posteriors(scores, Max)
        paths = traceback.argmax(2) % len(self.alphabet)
        return paths

    def path_to_str(self, path):
        alphabet = np.frombuffer(''.join(self.alphabet).encode(), dtype='u1')
        seq = alphabet[path[path != 0]]
        return seq.tobytes().decode()

    def ctc_loss(self, scores, targets, target_lengths):
        # convert from CTC targets (with blank=0) to zero indexed
        targets = torch.clamp(targets - 1, 0)

        T, N, C = scores.shape
        scores = scores.to(torch.float32)
        n = targets.size(1) - (self.state_len - 1)
        stay_indices = sum(
            targets[:, i:n + i] * self.n_base ** (self.state_len - i - 1)
            for i in range(self.state_len)
        ) * len(self.alphabet)
        move_indices = stay_indices[:, 1:] + targets[:, :n - 1] + 1
        stay_scores = scores.gather(2, stay_indices.expand(T, -1, -1))
        move_scores = scores.gather(2, move_indices.expand(T, -1, -1))
        logz = logZ_cupy(stay_scores, move_scores, target_lengths + 1 - self.state_len)
        return - (logz / target_lengths).mean()
