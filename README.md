# microgpt.c

> *"The most atomic way to train and run inference for a GPT in pure, dependency-free Python. This file is the complete algorithm. Everything else is just efficiency."* — [Andrej Karpathy](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)

This is the efficiency part, in C.

## Build and run

```
cc -O3 -march=native -o microgpt microgpt.c -lm
./microgpt
```

The dataset downloads automatically on first run.

```
num docs: 32033
vocab size: 27
num params: 4192
step 1000 / 1000 | loss 2.3277
Training time: 0.651 seconds

--- inference (new, hallucinated names) ---
sample  1: karira
sample  2: kaleni
sample  3: kaalely
sample  4: kenlel
sample  5: alrela
...
```

## Files

**microgpt.c** — the algorithm, in C. Same autograd, same computation graph, same backward pass as the Python. ~465 lines.

**microgpt_fast.c** — the efficiency. Hand-written gradients, SIMD (NEON, AVX2, AVX-512). Same model, same output. ~675 lines.

```
cc -O3 -march=native -o microgpt_fast microgpt_fast.c -lm
```

## Performance

1000 training steps, Python 3.14, Clang 21 / GCC 15, `-O3 -march=native`:

| | Python | C | C (fast) |
|--|--:|--:|--:|
| Apple M1 Max | 60.9s | 0.61s | 0.009s |
| AMD 7950X3D | 52.9s | 0.36s | 0.007s |
| AMD 9950X | 37.0s | 0.27s | 0.005s |
