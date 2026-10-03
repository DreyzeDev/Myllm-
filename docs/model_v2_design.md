# MyLLM V2 architecture and training plan

MyLLM V1 is closed as an experimental research release. Its base checkpoint
`step_32000` and the selected SFT V3 checkpoint are preserved. No further V1
training is planned. V2 starts from newly initialized random weights and does
not load any pretrained checkpoint.

## Proposed architecture

| Item | Value |
| --- | ---: |
| Architecture | Decoder-only Transformer |
| Layers | 12 |
| Hidden size | 768 |
| Attention heads | 12 |
| Head dimension | 64 |
| SwiGLU intermediate size | 2048 |
| Vocabulary | 32,000 (existing tokenizer) |
| Context | 1,024 |
| Position encoding | RoPE, theta 10,000 |
| Normalization | RMSNorm |
| Embeddings | Tied input and output |
| Parameters | 109,529,856 |

Parameter calculation for hidden size `H`, layers `L`, intermediate `I`, and
vocabulary `V`, with tied embeddings and biasless linear layers:

```text
P = V*H + L*(4*H^2 + 3*H*I + 2*H) + H
  = 32,000*768 + 12*(4*768^2 + 3*768*2,048 + 2*768) + 768
  = 109,529,856
```

## Training resource estimate (RTX 5060 8 GB)

Planning estimate: about **5–6.5 GiB total device use** with BF16, micro-batch
1, sequence length 1,024, Flash SDPA, and gradient accumulation 16. This is
estimated from the V1 training log (3.67 GB PyTorch peak allocated with
micro-batch 4 and context 1,024), then scaling persistent optimizer state and
activation shapes for V2. FP32 V2 weights, gradients, and AdamW first/second
moments account for about 1.63 GiB before activations and CUDA/runtime
workspaces. RTX 5060 currently reports BF16 and Flash SDPA support; this is a
planning estimate, not a V2 benchmark. Keep V1 training stopped and start V2
only from random initialization. The config uses micro-batch 1 with accumulation
16; keep other GPU-heavy applications closed during the first authorized run.

At roughly 20 training tokens per parameter, a compute-oriented first-pass
pretraining reference is about **2.19 billion tokens**. The existing V1 train
split contains 2,077,200,374 tokens (about 19 tokens per V2 parameter), which
is the current config target. With 16,384 input tokens per optimizer step, the
config schedules 126,783 steps and about **2.077 billion training tokens**.

The initial full-run config uses AdamW with a conservative peak LR of `2e-4`,
about 4% warmup, cosine decay to 10% of peak LR, gradient clipping at 1.0,
validation every 2,000 steps, prompt evaluation every 8,000 steps, and resumable
checkpoints every 4,000 steps. Probe weights are diagnostic only; full
pretraining starts from a separate random initialization.

## Compatibility and status

`configs/model_v2.yaml` follows the same `model` / `tokenizer` / `data` /
`training` schema consumed by `scripts/train_v1.py` and `src.train`. It points
to the existing compatible V1 pretraining token blocks and a separate V2
checkpoint directory. It retains the current tokenizer and data-block format.
The test checks the config,
parameter formula, instantiated parameter count, and tied embedding identity.

The training config starts from the model's normal random initialization and
does not refer to a pretrained checkpoint. The hardware probe uses a separate
diagnostic output directory; its weights are not used to initialize the full
pretraining run. V1 checkpoints and SFT datasets are outside the V2 training
paths.
