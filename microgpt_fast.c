/*
 * microgpt_fast.c — Optimised microgpt with manual gradients and SIMD
 * Same model, same training, same results — just faster
 * https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
 *
 * cc -O3 -march=native -o microgpt_fast microgpt_fast.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

/* ── Hyperparameters ────────────────────────────────────────────────── */
#ifndef N_LAYER
#define N_LAYER    1
#endif
#ifndef N_EMBD
#define N_EMBD     16
#endif
#ifndef N_HEAD
#define N_HEAD     4
#endif
#define HEAD_DIM   (N_EMBD / N_HEAD)
#define MLP_DIM    (N_EMBD * 4)
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif
#define MAX_VOCAB  64
#define MAX_DOCS   40000
#define MAX_DOCLEN 20
/* Computed once — compiler constant-folds sqrtf on compile-time HEAD_DIM */
static float attn_scale;  /* set in init_model() */

/* ── RNG (same xorshift64) ──────────────────────────────────────────── */
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

/* ── Model parameters ───────────────────────────────────────────────── */
static int   NP;
static float *P, *G, *AM, *AV;
static int   o_wte, o_wpe, o_lmh;
static int   o_wq[N_LAYER], o_wk[N_LAYER], o_wv[N_LAYER], o_wo[N_LAYER];
static int   o_f1[N_LAYER], o_f2[N_LAYER];

static void init_model(void) {
    int n = 0;
    o_wte = n; n += vocab_size * N_EMBD;
    o_wpe = n; n += BLOCK_SIZE * N_EMBD;
    o_lmh = n; n += vocab_size * N_EMBD;
    for (int l = 0; l < N_LAYER; l++) {
        o_wq[l] = n; n += N_EMBD * N_EMBD;
        o_wk[l] = n; n += N_EMBD * N_EMBD;
        o_wv[l] = n; n += N_EMBD * N_EMBD;
        o_wo[l] = n; n += N_EMBD * N_EMBD;
        o_f1[l] = n; n += MLP_DIM * N_EMBD;
        o_f2[l] = n; n += N_EMBD * MLP_DIM;
    }
    NP = n;
    attn_scale = 1.0f / sqrtf((float)HEAD_DIM);
    printf("num params: %d\n", NP);
    /* Aligned allocation for SIMD */
    size_t sz = (size_t)n * sizeof(float);
    size_t asz = (sz + 63) & ~(size_t)63;
    P  = aligned_alloc(64, asz); G  = aligned_alloc(64, asz);
    AM = aligned_alloc(64, asz); AV = aligned_alloc(64, asz);
    if (!P || !G || !AM || !AV) { fprintf(stderr, "Out of memory\n"); exit(1); }
    memset(P, 0, asz); memset(G, 0, asz); memset(AM, 0, asz); memset(AV, 0, asz);
    for (int i = 0; i < n; i++) P[i] = 0.08f * randgauss();
}

/* ── Activation cache ───────────────────────────────────────────────── */
static struct {
    float emb[BLOCK_SIZE][N_EMBD];
    float rn0[BLOCK_SIZE][N_EMBD]; float rn0s[BLOCK_SIZE];
    struct {
        float ra[BLOCK_SIZE][N_EMBD];
        float na[BLOCK_SIZE][N_EMBD]; float nas[BLOCK_SIZE];
        float q[BLOCK_SIZE][N_EMBD];
        float k[BLOCK_SIZE][N_EMBD];
        float v[BLOCK_SIZE][N_EMBD];
        float aw[BLOCK_SIZE][N_HEAD][BLOCK_SIZE];
        float ao[BLOCK_SIZE][N_EMBD];
        float rm[BLOCK_SIZE][N_EMBD];
        float nm[BLOCK_SIZE][N_EMBD]; float nms[BLOCK_SIZE];
        float h1[BLOCK_SIZE][MLP_DIM];
        float hr[BLOCK_SIZE][MLP_DIM];
    } L[N_LAYER];
    float xo[BLOCK_SIZE][N_EMBD];
    float pr[BLOCK_SIZE][MAX_VOCAB];
} C;

/* ═══════════════════════════════════════════════════════════════════════
 * SIMD-optimised primitives
 * ═══════════════════════════════════════════════════════════════════════ */

/* y[R] = W[R,C] @ x[C] */
static inline void lin_f(float * restrict y, const float * restrict W,
                          const float * restrict x, int R, int Ci) {
    for (int i = 0; i < R; i++) {
        const float *w = W + i * Ci;
#if defined(__ARM_NEON)
        float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
        float32x4_t s2 = vdupq_n_f32(0), s3 = vdupq_n_f32(0);
        int j = 0;
        for (; j + 15 < Ci; j += 16) {
            s0 = vfmaq_f32(s0, vld1q_f32(w+j),    vld1q_f32(x+j));
            s1 = vfmaq_f32(s1, vld1q_f32(w+j+4),  vld1q_f32(x+j+4));
            s2 = vfmaq_f32(s2, vld1q_f32(w+j+8),  vld1q_f32(x+j+8));
            s3 = vfmaq_f32(s3, vld1q_f32(w+j+12), vld1q_f32(x+j+12));
        }
        s0 = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
        for (; j + 3 < Ci; j += 4)
            s0 = vfmaq_f32(s0, vld1q_f32(w+j), vld1q_f32(x+j));
        float s = vaddvq_f32(s0);
        for (; j < Ci; j++) s += w[j] * x[j];
        y[i] = s;
#elif defined(__AVX512F__)
        __m512 s0 = _mm512_setzero_ps();
        int j = 0;
        for (; j + 15 < Ci; j += 16)
            s0 = _mm512_fmadd_ps(_mm512_loadu_ps(w+j), _mm512_loadu_ps(x+j), s0);
        float s = _mm512_reduce_add_ps(s0);
        for (; j < Ci; j++) s += w[j] * x[j];
        y[i] = s;
#elif defined(__AVX2__) && defined(__FMA__)
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        int j = 0;
        for (; j + 15 < Ci; j += 16) {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(w+j),   _mm256_loadu_ps(x+j),   s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(w+j+8), _mm256_loadu_ps(x+j+8), s1);
        }
        s0 = _mm256_add_ps(s0, s1);
        for (; j + 7 < Ci; j += 8)
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(w+j), _mm256_loadu_ps(x+j), s0);
        __m128 hi = _mm256_extractf128_ps(s0, 1);
        __m128 lo = _mm256_castps256_ps128(s0);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float s = _mm_cvtss_f32(sum128);
        for (; j < Ci; j++) s += w[j] * x[j];
        y[i] = s;
#else
        float s = 0;
        for (int j = 0; j < Ci; j++) s += w[j] * x[j];
        y[i] = s;
#endif
    }
}

/* dx[C] += W^T @ dy;  dW[R,C] += dy outer x */
static inline void lin_b(float * restrict dx, float * restrict dW, const float * restrict dy,
                          const float * restrict W, const float * restrict x, int R, int Ci) {
    /* dW[i,j] += dy[i] * x[j] — outer product */
    for (int i = 0; i < R; i++) {
        float d = dy[i];
        float *dw = dW + i * Ci;
#if defined(__ARM_NEON)
        float32x4_t dv = vdupq_n_f32(d);
        int j = 0;
        for (; j + 15 < Ci; j += 16) {
            vst1q_f32(dw+j,    vfmaq_f32(vld1q_f32(dw+j),    dv, vld1q_f32(x+j)));
            vst1q_f32(dw+j+4,  vfmaq_f32(vld1q_f32(dw+j+4),  dv, vld1q_f32(x+j+4)));
            vst1q_f32(dw+j+8,  vfmaq_f32(vld1q_f32(dw+j+8),  dv, vld1q_f32(x+j+8)));
            vst1q_f32(dw+j+12, vfmaq_f32(vld1q_f32(dw+j+12), dv, vld1q_f32(x+j+12)));
        }
        for (; j + 3 < Ci; j += 4)
            vst1q_f32(dw+j, vfmaq_f32(vld1q_f32(dw+j), dv, vld1q_f32(x+j)));
        for (; j < Ci; j++) dw[j] += d * x[j];
#elif defined(__AVX512F__)
        __m512 dv = _mm512_set1_ps(d);
        int j = 0;
        for (; j + 15 < Ci; j += 16)
            _mm512_storeu_ps(dw+j, _mm512_fmadd_ps(dv, _mm512_loadu_ps(x+j), _mm512_loadu_ps(dw+j)));
        for (; j < Ci; j++) dw[j] += d * x[j];
#elif defined(__AVX2__) && defined(__FMA__)
        __m256 dv = _mm256_set1_ps(d);
        int j = 0;
        for (; j + 15 < Ci; j += 16) {
            _mm256_storeu_ps(dw+j,   _mm256_fmadd_ps(dv, _mm256_loadu_ps(x+j),   _mm256_loadu_ps(dw+j)));
            _mm256_storeu_ps(dw+j+8, _mm256_fmadd_ps(dv, _mm256_loadu_ps(x+j+8), _mm256_loadu_ps(dw+j+8)));
        }
        for (; j + 7 < Ci; j += 8)
            _mm256_storeu_ps(dw+j, _mm256_fmadd_ps(dv, _mm256_loadu_ps(x+j), _mm256_loadu_ps(dw+j)));
        for (; j < Ci; j++) dw[j] += d * x[j];
#else
        for (int j = 0; j < Ci; j++) dw[j] += d * x[j];
#endif
    }
    /* dx[j] += W[i,j] * dy[i] — transpose multiply */
    if (dx)
        for (int i = 0; i < R; i++) {
            const float *w = W + i * Ci;
            float d = dy[i];
#if defined(__ARM_NEON)
            float32x4_t dv = vdupq_n_f32(d);
            int j = 0;
            for (; j + 15 < Ci; j += 16) {
                vst1q_f32(dx+j,    vfmaq_f32(vld1q_f32(dx+j),    dv, vld1q_f32(w+j)));
                vst1q_f32(dx+j+4,  vfmaq_f32(vld1q_f32(dx+j+4),  dv, vld1q_f32(w+j+4)));
                vst1q_f32(dx+j+8,  vfmaq_f32(vld1q_f32(dx+j+8),  dv, vld1q_f32(w+j+8)));
                vst1q_f32(dx+j+12, vfmaq_f32(vld1q_f32(dx+j+12), dv, vld1q_f32(w+j+12)));
            }
            for (; j + 3 < Ci; j += 4)
                vst1q_f32(dx+j, vfmaq_f32(vld1q_f32(dx+j), dv, vld1q_f32(w+j)));
            for (; j < Ci; j++) dx[j] += d * w[j];
#elif defined(__AVX512F__)
            __m512 dv = _mm512_set1_ps(d);
            int j = 0;
            for (; j + 15 < Ci; j += 16)
                _mm512_storeu_ps(dx+j, _mm512_fmadd_ps(dv, _mm512_loadu_ps(w+j), _mm512_loadu_ps(dx+j)));
            for (; j < Ci; j++) dx[j] += d * w[j];
#elif defined(__AVX2__) && defined(__FMA__)
            __m256 dv = _mm256_set1_ps(d);
            int j = 0;
            for (; j + 15 < Ci; j += 16) {
                _mm256_storeu_ps(dx+j,   _mm256_fmadd_ps(dv, _mm256_loadu_ps(w+j),   _mm256_loadu_ps(dx+j)));
                _mm256_storeu_ps(dx+j+8, _mm256_fmadd_ps(dv, _mm256_loadu_ps(w+j+8), _mm256_loadu_ps(dx+j+8)));
            }
            for (; j + 7 < Ci; j += 8)
                _mm256_storeu_ps(dx+j, _mm256_fmadd_ps(dv, _mm256_loadu_ps(w+j), _mm256_loadu_ps(dx+j)));
            for (; j < Ci; j++) dx[j] += d * w[j];
#else
            for (int j = 0; j < Ci; j++) dx[j] += d * w[j];
#endif
        }
}

static inline float rms_f(float * restrict y, const float * restrict x, int n) {
    float ms = 0;
    for (int i = 0; i < n; i++) ms += x[i] * x[i];
    float s = 1.0f / sqrtf(ms / n + 1e-5f);
    for (int i = 0; i < n; i++) y[i] = x[i] * s;
    return s;
}

static inline void rms_b(float * restrict dx, const float * restrict dy,
                          const float * restrict y, float s, int n) {
    float d = 0;
    for (int i = 0; i < n; i++) d += dy[i] * y[i];
    d /= n;
    for (int i = 0; i < n; i++) dx[i] += s * (dy[i] - y[i] * d);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Forward + Backward (manual gradients)
 * ═══════════════════════════════════════════════════════════════════════ */
static float train_step(const int *tok, int n) {
    /* G is zeroed by Adam after each use, not here */
    float loss = 0;

    /* Forward */
    for (int p = 0; p < n; p++) {
        const float *te = P + o_wte + tok[p] * N_EMBD;
        const float *pe = P + o_wpe + p * N_EMBD;
        for (int i = 0; i < N_EMBD; i++) C.emb[p][i] = te[i] + pe[i];
        C.rn0s[p] = rms_f(C.rn0[p], C.emb[p], N_EMBD);
        float *x = C.rn0[p];

        for (int l = 0; l < N_LAYER; l++) {
            memcpy(C.L[l].ra[p], x, sizeof(float) * N_EMBD);
            C.L[l].nas[p] = rms_f(C.L[l].na[p], x, N_EMBD);
            lin_f(C.L[l].q[p], P + o_wq[l], C.L[l].na[p], N_EMBD, N_EMBD);
            lin_f(C.L[l].k[p], P + o_wk[l], C.L[l].na[p], N_EMBD, N_EMBD);
            lin_f(C.L[l].v[p], P + o_wv[l], C.L[l].na[p], N_EMBD, N_EMBD);

            float xa[N_EMBD];
            for (int h = 0; h < N_HEAD; h++) {
                int hs = h * HEAD_DIM;
                float mx = -1e30f;
                for (int t = 0; t <= p; t++) {
                    float sc = 0;
                    for (int j = 0; j < HEAD_DIM; j++)
                        sc += C.L[l].q[p][hs+j] * C.L[l].k[t][hs+j];
                    sc *= attn_scale;
                    C.L[l].aw[p][h][t] = sc;
                    if (sc > mx) mx = sc;
                }
                float se = 0;
                for (int t = 0; t <= p; t++) {
                    C.L[l].aw[p][h][t] = expf(C.L[l].aw[p][h][t] - mx);
                    se += C.L[l].aw[p][h][t];
                }
                for (int t = 0; t <= p; t++) C.L[l].aw[p][h][t] /= se;
                for (int j = 0; j < HEAD_DIM; j++) {
                    float s = 0;
                    for (int t = 0; t <= p; t++)
                        s += C.L[l].aw[p][h][t] * C.L[l].v[t][hs+j];
                    xa[hs+j] = s;
                }
            }
            memcpy(C.L[l].ao[p], xa, sizeof(float) * N_EMBD);

            float xp[N_EMBD];
            lin_f(xp, P + o_wo[l], xa, N_EMBD, N_EMBD);
            for (int i = 0; i < N_EMBD; i++)
                C.L[l].rm[p][i] = C.L[l].ra[p][i] + xp[i];

            C.L[l].nms[p] = rms_f(C.L[l].nm[p], C.L[l].rm[p], N_EMBD);
            lin_f(C.L[l].h1[p], P + o_f1[l], C.L[l].nm[p], MLP_DIM, N_EMBD);
            for (int i = 0; i < MLP_DIM; i++)
                C.L[l].hr[p][i] = C.L[l].h1[p][i] > 0 ? C.L[l].h1[p][i] : 0;
            float xm[N_EMBD];
            lin_f(xm, P + o_f2[l], C.L[l].hr[p], N_EMBD, MLP_DIM);
            for (int i = 0; i < N_EMBD; i++)
                C.xo[p][i] = C.L[l].rm[p][i] + xm[i];
            x = C.xo[p];
        }

        float logits[MAX_VOCAB];
        lin_f(logits, P + o_lmh, x, vocab_size, N_EMBD);
        float mx = logits[0];
        for (int i = 1; i < vocab_size; i++) if (logits[i] > mx) mx = logits[i];
        float se = 0;
        for (int i = 0; i < vocab_size; i++) {
            C.pr[p][i] = expf(logits[i] - mx);
            se += C.pr[p][i];
        }
        for (int i = 0; i < vocab_size; i++) C.pr[p][i] /= se;
        loss += -logf(C.pr[p][tok[p+1]]);
    }
    loss /= n;

    /* Backward */
    float dx[BLOCK_SIZE][N_EMBD];
    memset(dx, 0, sizeof(dx));

    for (int p = 0; p < n; p++) {
        float dl[MAX_VOCAB];
        for (int i = 0; i < vocab_size; i++) dl[i] = C.pr[p][i] / n;
        dl[tok[p+1]] -= 1.0f / n;
        lin_b(dx[p], G + o_lmh, dl, P + o_lmh, C.xo[p], vocab_size, N_EMBD);
    }

    for (int l = N_LAYER - 1; l >= 0; l--) {
        float dxm[BLOCK_SIZE][N_EMBD];
        memset(dxm, 0, sizeof(dxm));
        for (int p = 0; p < n; p++) {
            float dhr[MLP_DIM] = {0};
            lin_b(dhr, G + o_f2[l], dx[p], P + o_f2[l], C.L[l].hr[p], N_EMBD, MLP_DIM);
            float dh1[MLP_DIM];
            for (int i = 0; i < MLP_DIM; i++)
                dh1[i] = C.L[l].h1[p][i] > 0 ? dhr[i] : 0;
            float dnm[N_EMBD] = {0};
            lin_b(dnm, G + o_f1[l], dh1, P + o_f1[l], C.L[l].nm[p], MLP_DIM, N_EMBD);
            float drm[N_EMBD] = {0};
            rms_b(drm, dnm, C.L[l].nm[p], C.L[l].nms[p], N_EMBD);
            for (int i = 0; i < N_EMBD; i++) dxm[p][i] = drm[i] + dx[p][i];
        }

        float dao[BLOCK_SIZE][N_EMBD];
        memset(dao, 0, sizeof(dao));
        for (int p = 0; p < n; p++)
            lin_b(dao[p], G + o_wo[l], dxm[p], P + o_wo[l], C.L[l].ao[p], N_EMBD, N_EMBD);

        float dq[BLOCK_SIZE][N_EMBD]; memset(dq, 0, sizeof(dq));
        float dk[BLOCK_SIZE][N_EMBD]; memset(dk, 0, sizeof(dk));
        float dv[BLOCK_SIZE][N_EMBD]; memset(dv, 0, sizeof(dv));
        for (int p = 0; p < n; p++) {
            for (int h = 0; h < N_HEAD; h++) {
                int hs = h * HEAD_DIM;
                float daw[BLOCK_SIZE] = {0};
                for (int j = 0; j < HEAD_DIM; j++)
                    for (int t = 0; t <= p; t++) {
                        daw[t] += dao[p][hs+j] * C.L[l].v[t][hs+j];
                        dv[t][hs+j] += C.L[l].aw[p][h][t] * dao[p][hs+j];
                    }
                float dot = 0;
                for (int t = 0; t <= p; t++)
                    dot += daw[t] * C.L[l].aw[p][h][t];
                float sc = attn_scale;
                for (int t = 0; t <= p; t++) {
                    float ds = C.L[l].aw[p][h][t] * (daw[t] - dot);
                    for (int j = 0; j < HEAD_DIM; j++) {
                        dq[p][hs+j] += ds * C.L[l].k[t][hs+j] * sc;
                        dk[t][hs+j] += ds * C.L[l].q[p][hs+j] * sc;
                    }
                }
            }
        }

        float dna[BLOCK_SIZE][N_EMBD]; memset(dna, 0, sizeof(dna));
        for (int p = 0; p < n; p++) {
            float tmp[N_EMBD];
            memset(tmp, 0, sizeof(tmp));
            lin_b(tmp, G + o_wq[l], dq[p], P + o_wq[l], C.L[l].na[p], N_EMBD, N_EMBD);
            for (int i = 0; i < N_EMBD; i++) dna[p][i] += tmp[i];
            memset(tmp, 0, sizeof(tmp));
            lin_b(tmp, G + o_wk[l], dk[p], P + o_wk[l], C.L[l].na[p], N_EMBD, N_EMBD);
            for (int i = 0; i < N_EMBD; i++) dna[p][i] += tmp[i];
            memset(tmp, 0, sizeof(tmp));
            lin_b(tmp, G + o_wv[l], dv[p], P + o_wv[l], C.L[l].na[p], N_EMBD, N_EMBD);
            for (int i = 0; i < N_EMBD; i++) dna[p][i] += tmp[i];
        }

        for (int p = 0; p < n; p++) {
            float dra[N_EMBD] = {0};
            rms_b(dra, dna[p], C.L[l].na[p], C.L[l].nas[p], N_EMBD);
            for (int i = 0; i < N_EMBD; i++)
                dx[p][i] = dra[i] + dxm[p][i];
        }
    }

    for (int p = 0; p < n; p++) {
        float de[N_EMBD] = {0};
        rms_b(de, dx[p], C.rn0[p], C.rn0s[p], N_EMBD);
        int t = tok[p];
        for (int i = 0; i < N_EMBD; i++) {
            G[o_wte + t * N_EMBD + i] += de[i];
            G[o_wpe + p * N_EMBD + i] += de[i];
        }
    }

    return loss;
}

/* ── SIMD-optimised Adam ────────────────────────────────────────────── */
static void adam(float lr, float b1_pow, float b2_pow) {
    const float b1 = 0.85f, b2 = 0.99f, eps = 1e-8f;
    float b1c = 1.0f - b1_pow;
    float b2c = 1.0f - b2_pow;
    float lr_b1c = lr / b1c;
    int i = 0;
#if defined(__AVX512F__)
    __m512 vb1 = _mm512_set1_ps(b1), vb1m = _mm512_set1_ps(1-b1);
    __m512 vb2 = _mm512_set1_ps(b2), vb2m = _mm512_set1_ps(1-b2);
    __m512 vlr = _mm512_set1_ps(lr_b1c);
    __m512 vb2c = _mm512_set1_ps(b2c), veps = _mm512_set1_ps(eps);
    for (; i + 15 < NP; i += 16) {
        __m512 g = _mm512_loadu_ps(G+i);
        _mm512_storeu_ps(G+i, _mm512_setzero_ps()); /* zero after read */
        __m512 m = _mm512_fmadd_ps(vb1m, g, _mm512_mul_ps(vb1, _mm512_loadu_ps(AM+i)));
        __m512 v = _mm512_fmadd_ps(vb2m, _mm512_mul_ps(g, g), _mm512_mul_ps(vb2, _mm512_loadu_ps(AV+i)));
        _mm512_storeu_ps(AM+i, m);
        _mm512_storeu_ps(AV+i, v);
        __m512 p = _mm512_loadu_ps(P+i);
        __m512 vh = _mm512_div_ps(v, vb2c);
        __m512 denom = _mm512_add_ps(_mm512_sqrt_ps(vh), veps);
        p = _mm512_fnmadd_ps(vlr, _mm512_div_ps(m, denom), p);
        _mm512_storeu_ps(P+i, p);
    }
#elif defined(__ARM_NEON)
    float32x4_t vb1 = vdupq_n_f32(b1), vb1m = vdupq_n_f32(1-b1);
    float32x4_t vb2 = vdupq_n_f32(b2), vb2m = vdupq_n_f32(1-b2);
    float32x4_t vlr = vdupq_n_f32(lr_b1c);
    float32x4_t vb2c = vdupq_n_f32(b2c), veps = vdupq_n_f32(eps);
    for (; i + 3 < NP; i += 4) {
        float32x4_t g = vld1q_f32(G+i);
        vst1q_f32(G+i, vdupq_n_f32(0)); /* zero after read */
        float32x4_t m = vfmaq_f32(vmulq_f32(vb1, vld1q_f32(AM+i)), vb1m, g);
        float32x4_t v = vfmaq_f32(vmulq_f32(vb2, vld1q_f32(AV+i)), vb2m, vmulq_f32(g, g));
        vst1q_f32(AM+i, m);
        vst1q_f32(AV+i, v);
        float32x4_t p = vld1q_f32(P+i);
        float32x4_t vh = vdivq_f32(v, vb2c);
        float32x4_t denom = vaddq_f32(vsqrtq_f32(vh), veps);
        p = vfmsq_f32(p, vlr, vdivq_f32(m, denom));
        vst1q_f32(P+i, p);
    }
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 vb1 = _mm256_set1_ps(b1), vb1m = _mm256_set1_ps(1-b1);
    __m256 vb2 = _mm256_set1_ps(b2), vb2m = _mm256_set1_ps(1-b2);
    __m256 vlr = _mm256_set1_ps(lr_b1c);
    __m256 vb2c = _mm256_set1_ps(b2c), veps = _mm256_set1_ps(eps);
    for (; i + 7 < NP; i += 8) {
        __m256 g = _mm256_loadu_ps(G+i);
        _mm256_storeu_ps(G+i, _mm256_setzero_ps()); /* zero after read */
        __m256 m = _mm256_fmadd_ps(vb1m, g, _mm256_mul_ps(vb1, _mm256_loadu_ps(AM+i)));
        __m256 v = _mm256_fmadd_ps(vb2m, _mm256_mul_ps(g, g), _mm256_mul_ps(vb2, _mm256_loadu_ps(AV+i)));
        _mm256_storeu_ps(AM+i, m);
        _mm256_storeu_ps(AV+i, v);
        __m256 p = _mm256_loadu_ps(P+i);
        __m256 vh = _mm256_div_ps(v, vb2c);
        __m256 denom = _mm256_add_ps(_mm256_sqrt_ps(vh), veps);
        p = _mm256_fnmadd_ps(vlr, _mm256_div_ps(m, denom), p);
        _mm256_storeu_ps(P+i, p);
    }
#endif
    for (; i < NP; i++) {
        float g = G[i]; G[i] = 0;
        AM[i] = b1 * AM[i] + (1 - b1) * g;
        AV[i] = b2 * AV[i] + (1 - b2) * g * g;
        P[i] -= lr_b1c * AM[i] / (sqrtf(AV[i] / b2c) + eps);
    }
}

/* ── Inference ──────────────────────────────────────────────────────── */
static void generate(int ns, float temp) {
    printf("\n--- inference (new, hallucinated names) ---\n");
    for (int s = 0; s < ns; s++) {
        float keys[N_LAYER][BLOCK_SIZE][N_EMBD];
        float vals[N_LAYER][BLOCK_SIZE][N_EMBD];
        int token = nchars;
        char name[BLOCK_SIZE + 1];
        int len = 0;
        for (int pos = 0; pos < BLOCK_SIZE; pos++) {
            float raw[N_EMBD], cur[N_EMBD];
            const float *te = P + o_wte + token * N_EMBD;
            const float *pe = P + o_wpe + pos * N_EMBD;
            for (int i = 0; i < N_EMBD; i++) raw[i] = te[i] + pe[i];
            rms_f(cur, raw, N_EMBD);
            for (int l = 0; l < N_LAYER; l++) {
                float xr[N_EMBD], xnn[N_EMBD];
                memcpy(xr, cur, sizeof(xr));
                rms_f(xnn, cur, N_EMBD);
                float q[N_EMBD];
                lin_f(q, P + o_wq[l], xnn, N_EMBD, N_EMBD);
                lin_f(keys[l][pos], P + o_wk[l], xnn, N_EMBD, N_EMBD);
                lin_f(vals[l][pos], P + o_wv[l], xnn, N_EMBD, N_EMBD);
                float ao[N_EMBD] = {0};
                for (int h = 0; h < N_HEAD; h++) {
                    int hs = h * HEAD_DIM;
                    float sc[BLOCK_SIZE], mx = -1e30f;
                    for (int t = 0; t <= pos; t++) {
                        float ss = 0;
                        for (int j = 0; j < HEAD_DIM; j++)
                            ss += q[hs+j] * keys[l][t][hs+j];
                        sc[t] = ss * attn_scale;
                        if (sc[t] > mx) mx = sc[t];
                    }
                    float se = 0;
                    for (int t = 0; t <= pos; t++) { sc[t] = expf(sc[t]-mx); se += sc[t]; }
                    for (int t = 0; t <= pos; t++) sc[t] /= se;
                    for (int j = 0; j < HEAD_DIM; j++) {
                        float ss = 0;
                        for (int t = 0; t <= pos; t++) ss += sc[t] * vals[l][t][hs+j];
                        ao[hs+j] = ss;
                    }
                }
                float xp[N_EMBD], xam[N_EMBD];
                lin_f(xp, P + o_wo[l], ao, N_EMBD, N_EMBD);
                for (int i = 0; i < N_EMBD; i++) xam[i] = xr[i] + xp[i];
                float xmn[N_EMBD], h1[MLP_DIM], h2[N_EMBD];
                rms_f(xmn, xam, N_EMBD);
                lin_f(h1, P + o_f1[l], xmn, MLP_DIM, N_EMBD);
                for (int i = 0; i < MLP_DIM; i++) if (h1[i] < 0) h1[i] = 0;
                lin_f(h2, P + o_f2[l], h1, N_EMBD, MLP_DIM);
                for (int i = 0; i < N_EMBD; i++) cur[i] = xam[i] + h2[i];
            }
            float logits[MAX_VOCAB], mx = -1e30f;
            lin_f(logits, P + o_lmh, cur, vocab_size, N_EMBD);
            for (int i = 0; i < vocab_size; i++) { logits[i] /= temp; if (logits[i] > mx) mx = logits[i]; }
            float se = 0;
            for (int i = 0; i < vocab_size; i++) { logits[i] = expf(logits[i]-mx); se += logits[i]; }
            float r = randf() * se, cum = 0;
            int next = vocab_size - 1;
            for (int i = 0; i < vocab_size; i++) { cum += logits[i]; if (cum > r) { next = i; break; } }
            if (next == nchars) break;
            name[len++] = uchars[next];
            token = next;
        }
        name[len] = 0;
        printf("sample %2d: %s\n", s + 1, name);
    }
}

/* ── Main ────────────────────────────────────────────────────────────── */
int main(void) {
    load_data();
    init_model();

    const int nsteps = 1000;
    float b1_pow = 1.0f, b2_pow = 1.0f; /* incremental bias correction */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int step = 0; step < nsteps; step++) {
        const char *doc = docs[doc_order[step % ndocs]];
        int toks[MAX_DOCLEN + 2];
        int tlen = (int)strlen(doc);
        toks[0] = nchars;
        for (int i = 0; i < tlen; i++) {
            int t = char_to_tok(doc[i]);
            if (t < 0) { fprintf(stderr, "Unknown character: %c\n", doc[i]); exit(1); }
            toks[i + 1] = t;
        }
        toks[tlen+1] = nchars;
        int n = tlen + 1;
        if (n > BLOCK_SIZE) n = BLOCK_SIZE;

        float loss = train_step(toks, n);
        float lr = 0.01f * (1.0f - (float)step / nsteps);
        b1_pow *= 0.85f; b2_pow *= 0.99f;
        adam(lr, b1_pow, b2_pow);

        printf("\rstep %4d / %4d | loss %.4f", step + 1, nsteps, loss);
        fflush(stdout);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (double)(t1.tv_sec - t0.tv_sec)
                   + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("\nTraining time: %.3f seconds\n", elapsed);

    generate(20, 0.5f);

    free(P); free(G); free(AM); free(AV);
    return 0;
}
