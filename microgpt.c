/*
 * microgpt.c — Karpathy's microgpt.py, in C
 * https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
 *
 * cc -O3 -march=native -o microgpt microgpt.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/* ── Hyperparameters ────────────────────────────────────────────────── */
#define N_LAYER    1
#define N_EMBD     16
#define N_HEAD     4
#define HEAD_DIM   (N_EMBD / N_HEAD)
#define MLP_DIM    (N_EMBD * 4)
#define BLOCK_SIZE 16
#define MAX_VOCAB  64
#define MAX_DOCS   40000
#define MAX_DOCLEN 20

/* ── RNG (xorshift64) ──────────────────────────────────────────────── */
static uint64_t rng_s = 42;
static uint32_t rng32(void) {
    rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17;
    return (uint32_t)(rng_s >> 16);
}
static float randf(void) { return (float)(rng32() & 0xFFFFFF) / 16777216.0f; }
static float randgauss(void) {
    float u, v;
    do { u = randf(); } while (u < 1e-30f);
    v = randf();
    return sqrtf(-2.0f * logf(u)) * cosf(6.283185307f * v);
}
static void shuffle_ints(int *a, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(rng32() % (unsigned)(i + 1));
        int t = a[i]; a[i] = a[j]; a[j] = t;
    }
}

/* ── Dataset ────────────────────────────────────────────────────────── */
static char docs[MAX_DOCS][MAX_DOCLEN];
static int  doc_order[MAX_DOCS];
static int  ndocs, nchars, vocab_size;
static char uchars[MAX_VOCAB];

static int char_to_tok(char c) {
    for (int i = 0; i < nchars; i++) if (uchars[i] == c) return i;
    return -1;
}
static void load_data(void) {
    FILE *f = fopen("input.txt", "r");
    if (!f) {
        if (system("curl -sL https://raw.githubusercontent.com/karpathy/makemore/"
                    "988aa59/names.txt -o input.txt") != 0)
            { fprintf(stderr, "Failed to download input.txt\n"); exit(1); }
        f = fopen("input.txt", "r");
        if (!f) { fprintf(stderr, "Cannot open input.txt\n"); exit(1); }
    }
    int seen[256] = {0};
    char line[64];
    ndocs = 0;
    while (fgets(line, sizeof(line), f) && ndocs < MAX_DOCS) {
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = 0;
        if (!len) continue;
        if (len >= MAX_DOCLEN) len = MAX_DOCLEN - 1;
        line[len] = '\0';
        memcpy(docs[ndocs], line, (size_t)len + 1);
        for (int i = 0; i < len; i++) seen[(unsigned char)line[i]] = 1;
        ndocs++;
    }
    fclose(f);
    nchars = 0;
    for (int i = 0; i < 256; i++) {
        if (seen[i]) {
            if (nchars >= MAX_VOCAB - 1) { fprintf(stderr, "Too many unique characters\n"); exit(1); }
            uchars[nchars++] = (char)i;
        }
    }
    vocab_size = nchars + 1;
    if (ndocs == 0) { fprintf(stderr, "No documents in input.txt\n"); exit(1); }
    for (int i = 0; i < ndocs; i++) doc_order[i] = i;
    shuffle_ints(doc_order, ndocs);
    printf("num docs: %d\n", ndocs);
    printf("vocab size: %d\n", vocab_size);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Autograd
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct Value {
    float data;           /* scalar value (forward pass) */
    float grad;           /* dLoss/dThis (backward pass) */
    struct Value *ch[2];  /* children in computation graph */
    float lg[2];          /* local gradients w.r.t. children */
    int nch;              /* number of children (0, 1, or 2) */
} Value;

/* Pool allocator — avoids malloc/free per node */
#define MAX_POOL 500000
static Value pool[MAX_POOL];
static int pool_n;
static int param_end;  /* pool index after persistent parameter Values */

static Value *V(float data) {
    if (pool_n >= MAX_POOL) { fprintf(stderr, "Value pool exhausted\n"); exit(1); }
    Value *v = &pool[pool_n++];
    v->data = data; v->grad = 0; v->nch = 0;
    return v;
}

/* Binary operation */
static Value *Vop2(float data, Value *a, Value *b, float lga, float lgb) {
    Value *v = V(data);
    v->nch = 2; v->ch[0] = a; v->ch[1] = b; v->lg[0] = lga; v->lg[1] = lgb;
    return v;
}

/* Unary operation */
static Value *Vop1(float data, Value *a, float lga) {
    Value *v = V(data);
    v->nch = 1; v->ch[0] = a; v->lg[0] = lga;
    return v;
}

/* ── Value operations ───────────────────────────────────────────────── */
static Value *vadd(Value *a, Value *b) {
    return Vop2(a->data + b->data, a, b, 1, 1);
}
static Value *vmul(Value *a, Value *b) {
    return Vop2(a->data * b->data, a, b, b->data, a->data);
}
static Value *vpow_f(Value *a, float e) {
    return Vop1(powf(a->data, e), a, e * powf(a->data, e - 1));
}
static Value *vlog(Value *a) {
    return Vop1(logf(a->data), a, 1.0f / a->data);
}
static Value *vexp(Value *a) {
    float e = expf(a->data);
    return Vop1(e, a, e);
}
static Value *vrelu(Value *a) {
    float d = a->data;
    return Vop1(d > 0 ? d : 0, a, d > 0 ? 1.0f : 0);
}

/* Value/float ops */
static Value *vadd_f(Value *a, float b) { return vadd(a, V(b)); }
static Value *vmul_f(Value *a, float b) { return vmul(a, V(b)); }
static Value *vdiv(Value *a, Value *b) { return vmul(a, vpow_f(b, -1)); }

/* ── Backward pass (topological sort + chain rule) ──────────────────── */
static Value *topo[MAX_POOL];
static int topo_n;
static int visited[MAX_POOL]; /* indexed by pool offset */
static int visit_gen;         /* generation counter (avoids clearing) */

static void build_topo(Value *v) {
    int idx = (int)(v - pool);
    if (visited[idx] == visit_gen) return;
    visited[idx] = visit_gen;
    for (int i = 0; i < v->nch; i++)
        build_topo(v->ch[i]);
    topo[topo_n++] = v;
}

static void backward(Value *root) {
    topo_n = 0;
    visit_gen++;
    build_topo(root);
    root->grad = 1;
    for (int i = topo_n - 1; i >= 0; i--) {
        Value *v = topo[i];
        for (int j = 0; j < v->nch; j++)
            v->ch[j]->grad += v->lg[j] * v->grad;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Model
 * ═══════════════════════════════════════════════════════════════════════ */
static Value *wte[MAX_VOCAB * N_EMBD];
static Value *wpe[BLOCK_SIZE * N_EMBD];
static Value *lm_head[MAX_VOCAB * N_EMBD];
static Value *attn_wq[N_LAYER][N_EMBD * N_EMBD];
static Value *attn_wk[N_LAYER][N_EMBD * N_EMBD];
static Value *attn_wv[N_LAYER][N_EMBD * N_EMBD];
static Value *attn_wo[N_LAYER][N_EMBD * N_EMBD];
static Value *mlp_fc1[N_LAYER][MLP_DIM * N_EMBD];
static Value *mlp_fc2[N_LAYER][N_EMBD * MLP_DIM];

static float *adam_m, *adam_v;

static void init_matrix(Value **mat, int nout, int nin) {
    for (int i = 0; i < nout * nin; i++)
        mat[i] = V(0.08f * randgauss());
}

static void init_model(void) {
    init_matrix(wte, vocab_size, N_EMBD);
    init_matrix(wpe, BLOCK_SIZE, N_EMBD);
    init_matrix(lm_head, vocab_size, N_EMBD);
    for (int l = 0; l < N_LAYER; l++) {
        init_matrix(attn_wq[l], N_EMBD, N_EMBD);
        init_matrix(attn_wk[l], N_EMBD, N_EMBD);
        init_matrix(attn_wv[l], N_EMBD, N_EMBD);
        init_matrix(attn_wo[l], N_EMBD, N_EMBD);
        init_matrix(mlp_fc1[l], MLP_DIM, N_EMBD);
        init_matrix(mlp_fc2[l], N_EMBD, MLP_DIM);
    }
    param_end = pool_n;

    adam_m = calloc((size_t)param_end, sizeof(float));
    adam_v = calloc((size_t)param_end, sizeof(float));
    if (!adam_m || !adam_v) { fprintf(stderr, "Out of memory\n"); exit(1); }
    printf("num params: %d\n", param_end);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Forward pass
 * ═══════════════════════════════════════════════════════════════════════ */

/* linear(x, w) — y[i] = sum(w[i][j] * x[j]) */
static void linear(Value **y, Value **w, Value **x, int nout, int nin) {
    for (int i = 0; i < nout; i++) {
        Value *s = V(0);
        for (int j = 0; j < nin; j++)
            s = vadd(s, vmul(w[i * nin + j], x[j]));
        y[i] = s;
    }
}

/* softmax(logits) */
static void softmax(Value **out, Value **logits, int n) {
    float max_val = logits[0]->data;
    for (int i = 1; i < n; i++)
        if (logits[i]->data > max_val) max_val = logits[i]->data;
    Value *exps[MAX_VOCAB];
    for (int i = 0; i < n; i++)
        exps[i] = vexp(vadd_f(logits[i], -max_val));
    Value *total = V(0);
    for (int i = 0; i < n; i++)
        total = vadd(total, exps[i]);
    for (int i = 0; i < n; i++)
        out[i] = vdiv(exps[i], total);
}

static void rmsnorm(Value **y, Value **x, int n) {
    /* ms = sum(xi * xi) / len(x) */
    Value *ms = V(0);
    for (int i = 0; i < n; i++)
        ms = vadd(ms, vmul(x[i], x[i]));
    ms = vmul_f(ms, 1.0f / n);
    /* scale = (ms + 1e-5) ** -0.5 */
    Value *scale = vpow_f(vadd_f(ms, 1e-5f), -0.5f);
    for (int i = 0; i < n; i++)
        y[i] = vmul(x[i], scale);
}

/* gpt(token_id, pos_id, keys, values) — one token through the model */
static void gpt(Value **logits, int token_id, int pos_id,
                Value *keys[][BLOCK_SIZE][N_EMBD],
                Value *values[][BLOCK_SIZE][N_EMBD]) {
    Value *x[N_EMBD], *tmp[N_EMBD];

    /* x = tok_emb + pos_emb */
    for (int i = 0; i < N_EMBD; i++)
        x[i] = vadd(wte[token_id * N_EMBD + i], wpe[pos_id * N_EMBD + i]);

    /* x = rmsnorm(x) */
    rmsnorm(tmp, x, N_EMBD);
    memcpy(x, tmp, sizeof(Value*) * N_EMBD);

    for (int li = 0; li < N_LAYER; li++) {
        /* Save residual */
        Value *x_residual[N_EMBD];
        memcpy(x_residual, x, sizeof(Value*) * N_EMBD);

        /* x = rmsnorm(x) */
        rmsnorm(tmp, x, N_EMBD);
        memcpy(x, tmp, sizeof(Value*) * N_EMBD);

        /* q, k, v projections */
        Value *q[N_EMBD], *k[N_EMBD], *v[N_EMBD];
        linear(q, attn_wq[li], x, N_EMBD, N_EMBD);
        linear(k, attn_wk[li], x, N_EMBD, N_EMBD);
        linear(v, attn_wv[li], x, N_EMBD, N_EMBD);

        /* Store k, v in cache */
        for (int i = 0; i < N_EMBD; i++) {
            keys[li][pos_id][i] = k[i];
            values[li][pos_id][i] = v[i];
        }

        /* Multi-head attention */
        Value *x_attn[N_EMBD];
        for (int h = 0; h < N_HEAD; h++) {
            int hs = h * HEAD_DIM;

            /* Attention logits: dot(q_h, k_h[t]) / sqrt(head_dim) */
            Value *attn_logits[BLOCK_SIZE];
            for (int t = 0; t <= pos_id; t++) {
                Value *dot = V(0);
                for (int j = 0; j < HEAD_DIM; j++)
                    dot = vadd(dot, vmul(q[hs + j], keys[li][t][hs + j]));
                attn_logits[t] = vmul_f(dot, 1.0f / sqrtf((float)HEAD_DIM));
            }

            /* Softmax over attention logits */
            Value *attn_weights[BLOCK_SIZE];
            softmax(attn_weights, attn_logits, pos_id + 1);

            /* Weighted sum of values */
            for (int j = 0; j < HEAD_DIM; j++) {
                Value *s = V(0);
                for (int t = 0; t <= pos_id; t++)
                    s = vadd(s, vmul(attn_weights[t], values[li][t][hs + j]));
                x_attn[h * HEAD_DIM + j] = s;
            }
        }

        /* Output projection */
        linear(x, attn_wo[li], x_attn, N_EMBD, N_EMBD);

        /* Residual connection */
        for (int i = 0; i < N_EMBD; i++)
            x[i] = vadd(x[i], x_residual[i]);

        /* MLP block */
        memcpy(x_residual, x, sizeof(Value*) * N_EMBD);
        rmsnorm(tmp, x, N_EMBD);
        memcpy(x, tmp, sizeof(Value*) * N_EMBD);

        Value *h1[MLP_DIM];
        linear(h1, mlp_fc1[li], x, MLP_DIM, N_EMBD);
        for (int i = 0; i < MLP_DIM; i++)
            h1[i] = vrelu(h1[i]);

        linear(x, mlp_fc2[li], h1, N_EMBD, MLP_DIM);
        for (int i = 0; i < N_EMBD; i++)
            x[i] = vadd(x[i], x_residual[i]);
    }

    linear(logits, lm_head, x, vocab_size, N_EMBD);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Training + Inference
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    load_data();
    init_model();

    float lr0 = 0.01f, b1 = 0.85f, b2 = 0.99f, eps = 1e-8f;
    int nsteps = 1000;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int step = 0; step < nsteps; step++) {
        /* Reset transient Values (keep parameters) */
        pool_n = param_end;

        const char *doc = docs[doc_order[step % ndocs]];
        int toks[MAX_DOCLEN + 2];
        int tlen = (int)strlen(doc);
        toks[0] = nchars;
        for (int i = 0; i < tlen; i++) {
            int t = char_to_tok(doc[i]);
            if (t < 0) { fprintf(stderr, "Unknown character: %c\n", doc[i]); exit(1); }
            toks[i + 1] = t;
        }
        toks[tlen + 1] = nchars;
        int n = tlen + 1;
        if (n > BLOCK_SIZE) n = BLOCK_SIZE;

        /* Forward: build computation graph */
        Value *kv_keys[N_LAYER][BLOCK_SIZE][N_EMBD];
        Value *kv_vals[N_LAYER][BLOCK_SIZE][N_EMBD];
        Value *losses[BLOCK_SIZE];

        for (int pos = 0; pos < n; pos++) {
            Value *logits[MAX_VOCAB], *probs[MAX_VOCAB];
            gpt(logits, toks[pos], pos, kv_keys, kv_vals);
            softmax(probs, logits, vocab_size);
            /* loss_t = -log(probs[target]) */
            losses[pos] = vmul_f(vlog(probs[toks[pos + 1]]), -1.0f);
        }

        /* loss = (1/n) * sum(losses) */
        Value *loss_sum = V(0);
        for (int i = 0; i < n; i++)
            loss_sum = vadd(loss_sum, losses[i]);
        Value *loss = vmul_f(loss_sum, 1.0f / n);

        /* Backward */
        backward(loss);

        /* Adam optimizer */
        float lr = lr0 * (1.0f - (float)step / nsteps);
        float b1c = 1.0f - powf(b1, (float)(step + 1));
        float b2c = 1.0f - powf(b2, (float)(step + 1));
        for (int i = 0; i < param_end; i++) {
            Value *p = &pool[i];
            adam_m[i] = b1 * adam_m[i] + (1 - b1) * p->grad;
            adam_v[i] = b2 * adam_v[i] + (1 - b2) * p->grad * p->grad;
            float mh = adam_m[i] / b1c;
            float vh = adam_v[i] / b2c;
            p->data -= lr * mh / (sqrtf(vh) + eps);
            p->grad = 0;
        }

        printf("\rstep %4d / %4d | loss %.4f", step + 1, nsteps, loss->data);
        fflush(stdout);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (double)(t1.tv_sec - t0.tv_sec)
                   + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("\nTraining time: %.3f seconds\n", elapsed);

    /* Inference */
    float temperature = 0.5f;
    printf("\n--- inference (new, hallucinated names) ---\n");
    for (int s = 0; s < 20; s++) {
        pool_n = param_end;  /* reset for each sample */
        Value *kv_keys[N_LAYER][BLOCK_SIZE][N_EMBD];
        Value *kv_vals[N_LAYER][BLOCK_SIZE][N_EMBD];
        int token = nchars;
        char name[BLOCK_SIZE + 1];
        int len = 0;

        for (int pos = 0; pos < BLOCK_SIZE; pos++) {
            Value *logits[MAX_VOCAB], *scaled[MAX_VOCAB], *probs[MAX_VOCAB];
            gpt(logits, token, pos, kv_keys, kv_vals);
            for (int i = 0; i < vocab_size; i++)
                scaled[i] = vmul_f(logits[i], 1.0f / temperature);
            softmax(probs, scaled, vocab_size);

            /* Sample from distribution */
            float r = randf(), cum = 0;
            int next = vocab_size - 1;
            for (int i = 0; i < vocab_size; i++) {
                cum += probs[i]->data;
                if (cum > r) { next = i; break; }
            }
            if (next == nchars) break;
            name[len++] = uchars[next];
            token = next;
        }
        name[len] = 0;
        printf("sample %2d: %s\n", s + 1, name);
    }

    free(adam_m); free(adam_v);
    return 0;
}
