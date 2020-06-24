/* Copyright (c) 2018 Gregor Richards
 * Copyright (c) 2017 Mozilla */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#define DEBUG

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "kiss_fft.h"
#include "common.h"
#include <math.h>
#include "rnnoise.h"
#include "pitch.h"
#include "arch.h"
#include "rnn.h"
#include "rnn_data.h"

#define FRAME_SIZE (160)
#define WINDOW_SIZE (2*FRAME_SIZE)
#define FREQ_SIZE (FRAME_SIZE + 1)

#define PITCH_MIN_PERIOD 32
#define PITCH_MAX_PERIOD 256
#define PITCH_FRAME_SIZE 320
#define PITCH_BUF_SIZE (PITCH_MAX_PERIOD+PITCH_FRAME_SIZE)

#define SQUARE(x) ((x)*(x))

#define NB_BANDS 36 

#define CEPS_MEM 8
#define NB_DELTA_CEPS 6

#define NB_FEATURES (NB_BANDS+3*NB_DELTA_CEPS+2)


#ifndef TRAINING
#define TRAINING 0
#endif


/* The built-in model, used if no file is given as input */
extern const struct RNNModel rnnoise_model_orig;


static const opus_int16 eband5ms[] = {
/*0  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9  1k  1.2 1.4 1.6 1.8 2k  2.4k 2.8 3.2 3.4 3.6 3.8 4.0 4.2 4.4 4.6  4.8 5.0  5.2  5.6  6.0  6.4  6.8  7.2k 7.6 8k */
  0,  2,  4,  6, 8,  10, 12, 14, 16, 18,  20, 24, 28, 32, 36, 40, 48,  56, 64, 68, 72, 76, 80, 84, 88, 92,  96, 100, 104, 112, 120, 128, 136, 142, 152, 160
};

//static const opus_int16 eband5ms[] = {
///*0  200 400 600 800  1k 1.2 1.4 1.6  2k 2.4 2.8 3.2  4k 4.8 5.6 6.8  8k 9.6 12k 15.6 20k*/
//  0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100
//};

typedef struct {
  int init;
  kiss_fft_state *kfft;
  float half_window[FRAME_SIZE];
  float dct_table[NB_BANDS*NB_BANDS];
} CommonState;

struct DenoiseState {
  float analysis_mem[FRAME_SIZE];
  float cepstral_mem[CEPS_MEM][NB_BANDS];
  int memid;
  float synthesis_mem[FRAME_SIZE];
  float pitch_buf[PITCH_BUF_SIZE];
  float pitch_enh_buf[PITCH_BUF_SIZE];
  float last_gain;
  int last_period;
  float mem_hp_x[2];
  float lastg[NB_BANDS];
  RNNState rnn;
};

void compute_band_energy(float *bandE, const kiss_fft_cpx *X) {
  int i;
  float sum[NB_BANDS] = {0};
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    int band_size;
    band_size = (eband5ms[i+1]-eband5ms[i]);
    for (j=0;j<band_size;j++) {
      float tmp;
      float frac = (float)j/band_size;
      tmp = SQUARE(X[eband5ms[i] + j].r);
      tmp += SQUARE(X[eband5ms[i] + j].i);
      sum[i] += (1-frac)*tmp;
      sum[i+1] += frac*tmp;
    }
  }
  sum[0] *= 2;
  sum[NB_BANDS-1] *= 2;
  for (i=0;i<NB_BANDS;i++)
  {
    bandE[i] = sum[i];
  }
}

void compute_band_corr(float *bandE, const kiss_fft_cpx *X, const kiss_fft_cpx *P) {
  int i;
  float sum[NB_BANDS] = {0};
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    int band_size;
    band_size = eband5ms[i+1]-eband5ms[i];
    for (j=0;j<band_size;j++) {
      float tmp;
      float frac = (float)j/band_size;
      tmp = X[eband5ms[i] + j].r * P[eband5ms[i] + j].r;
      tmp += X[eband5ms[i] + j].i * P[eband5ms[i] + j].i;
      sum[i] += (1-frac)*tmp;
      sum[i+1] += frac*tmp;
    }
  }
  sum[0] *= 2;
  sum[NB_BANDS-1] *= 2;
  for (i=0;i<NB_BANDS;i++)
  {
    bandE[i] = sum[i];
  }
}

void interp_band_gain(float *g, const float *bandE) {
  int i;
  memset(g, 0, FREQ_SIZE);
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    int band_size;
    band_size = eband5ms[i+1]-eband5ms[i];
    for (j=0;j<band_size;j++) {
      float frac = (float)j/band_size;
      g[eband5ms[i] + j] = (1-frac)*bandE[i] + frac*bandE[i+1];
    }
  }
}


CommonState common;

static void check_init() {
  int i;
  if (common.init) return;
  common.kfft = opus_fft_alloc_twiddles(2*FRAME_SIZE, NULL, NULL, NULL, 0);
  for (i=0;i<FRAME_SIZE;i++)
    common.half_window[i] = sin(.5*M_PI*sin(.5*M_PI*(i+.5)/FRAME_SIZE) * sin(.5*M_PI*(i+.5)/FRAME_SIZE));
  for (i=0;i<NB_BANDS;i++) {
    int j;
    for (j=0;j<NB_BANDS;j++) {
      common.dct_table[i*NB_BANDS + j] = cos((i+.5)*j*M_PI/NB_BANDS);
      if (j==0) common.dct_table[i*NB_BANDS + j] *= sqrt(.5);
    }
  }
  common.init = 1;
}

static void dct(float *out, const float *in) {
  int i;
  check_init();
  for (i=0;i<NB_BANDS;i++) {
    int j;
    float sum = 0;
    for (j=0;j<NB_BANDS;j++) {
      sum += in[j] * common.dct_table[j*NB_BANDS + i];
    }
    out[i] = sum*sqrt(2./22);
  }
}

#if 0
static void idct(float *out, const float *in) {
  int i;
  check_init();
  for (i=0;i<NB_BANDS;i++) {
    int j;
    float sum = 0;
    for (j=0;j<NB_BANDS;j++) {
      sum += in[j] * common.dct_table[i*NB_BANDS + j];
    }
    out[i] = sum*sqrt(2./22);
  }
}
#endif

static void forward_transform(kiss_fft_cpx *out, const float *in) {
  int i;
  kiss_fft_cpx x[WINDOW_SIZE];
  kiss_fft_cpx y[WINDOW_SIZE];
  check_init();
  for (i=0;i<WINDOW_SIZE;i++) {
    x[i].r = in[i];
    x[i].i = 0;
  }
  opus_fft(common.kfft, x, y, 0);
  for (i=0;i<FREQ_SIZE;i++) {
    out[i] = y[i];
  }
}

static void inverse_transform(float *out, const kiss_fft_cpx *in) {
  int i;
  kiss_fft_cpx x[WINDOW_SIZE];
  kiss_fft_cpx y[WINDOW_SIZE];
  check_init();
  for (i=0;i<FREQ_SIZE;i++) {
    x[i] = in[i];
  }
  for (;i<WINDOW_SIZE;i++) {
    x[i].r = x[WINDOW_SIZE - i].r;
    x[i].i = -x[WINDOW_SIZE - i].i;
  }
  opus_fft(common.kfft, x, y, 0);
  /* output in reverse order for IFFT. */
  out[0] = WINDOW_SIZE*y[0].r;
  for (i=1;i<WINDOW_SIZE;i++) {
    out[i] = WINDOW_SIZE*y[WINDOW_SIZE - i].r;
  }
}

static void apply_window(float *x) {
  int i;
  check_init();
  for (i=0;i<FRAME_SIZE;i++) {
    x[i] *= common.half_window[i];
    x[WINDOW_SIZE - 1 - i] *= common.half_window[i];
  }
}

int rnnoise_get_size() {
  return sizeof(DenoiseState);
}

int rnnoise_init(DenoiseState *st, RNNModel *model) {
  memset(st, 0, sizeof(*st));
  if (model)
    st->rnn.model = model;
  else
    st->rnn.model = &rnnoise_model_orig;
  st->rnn.vad_gru_state = calloc(sizeof(float), st->rnn.model->vad_gru_size);
  st->rnn.vad_gru2_state = calloc(sizeof(float), st->rnn.model->vad_gru2_size);
  st->rnn.noise_gru_state = calloc(sizeof(float), st->rnn.model->noise_gru_size);
  st->rnn.denoise_gru_state = calloc(sizeof(float), st->rnn.model->denoise_gru_size);
  return 0;
}

DenoiseState *rnnoise_create(RNNModel *model) {
  DenoiseState *st;
  st = malloc(rnnoise_get_size());
  rnnoise_init(st, model);
  return st;
}

void rnnoise_destroy(DenoiseState *st) {
  free(st->rnn.vad_gru_state);
  free(st->rnn.noise_gru_state);
  free(st->rnn.denoise_gru_state);
  free(st);
}

#if TRAINING
int lowpass = FREQ_SIZE;
int band_lp = NB_BANDS;
#endif

static void frame_analysis(DenoiseState *st, kiss_fft_cpx *X, float *Ex, const float *in) {
  int i;
  float x[WINDOW_SIZE];
  RNN_COPY(x, st->analysis_mem, FRAME_SIZE);
  for (i=0;i<FRAME_SIZE;i++) x[FRAME_SIZE + i] = in[i];
  RNN_COPY(st->analysis_mem, in, FRAME_SIZE);
  apply_window(x);
  forward_transform(X, x);
#if TRAINING
  for (i=lowpass;i<FREQ_SIZE;i++)
    X[i].r = X[i].i = 0;
#endif
  compute_band_energy(Ex, X);
}

static int compute_frame_features(DenoiseState *st, kiss_fft_cpx *X, kiss_fft_cpx *P,
                                  float *Ex, float *Ep, float *Exp, float *features, const float *in) {
  int i;
  float E = 0;
  float *ceps_0, *ceps_1, *ceps_2;
  float spec_variability = 0;
  float Ly[NB_BANDS];
  float p[WINDOW_SIZE];
  float pitch_buf[PITCH_BUF_SIZE];
  int pitch_index;
  float gain;
  float tmp[NB_BANDS];
  float follow, logMax;
  frame_analysis(st, X, Ex, in);
  RNN_MOVE(st->pitch_buf, &st->pitch_buf[FRAME_SIZE], PITCH_BUF_SIZE-FRAME_SIZE);
  RNN_COPY(&st->pitch_buf[PITCH_BUF_SIZE-FRAME_SIZE], in, FRAME_SIZE);
 // pitch_downsample(pitch_buf, PITCH_BUF_SIZE);
  pitch_search(st->pitch_buf+(PITCH_MAX_PERIOD>>1), st->pitch_buf, PITCH_FRAME_SIZE,
               PITCH_MAX_PERIOD-3*PITCH_MIN_PERIOD, &pitch_index);
  pitch_index = PITCH_MAX_PERIOD-pitch_index;

  gain = remove_doubling(st->pitch_buf, PITCH_MAX_PERIOD, PITCH_MIN_PERIOD,
          PITCH_FRAME_SIZE, &pitch_index, st->last_period, st->last_gain);
  st->last_period = pitch_index;
  st->last_gain = gain;
  for (i=0;i<WINDOW_SIZE;i++)
    p[i] = st->pitch_buf[PITCH_BUF_SIZE-WINDOW_SIZE-pitch_index+i];
  apply_window(p);
  forward_transform(P, p);
  compute_band_energy(Ep, P);
  compute_band_corr(Exp, X, P);
  for (i=0;i<NB_BANDS;i++) Exp[i] = Exp[i]/sqrt(.001+Ex[i]*Ep[i]);
  dct(tmp, Exp);
  for (i=0;i<NB_DELTA_CEPS;i++) features[NB_BANDS+2*NB_DELTA_CEPS+i] = tmp[i];
  features[NB_BANDS+2*NB_DELTA_CEPS] -= 1.3;
  features[NB_BANDS+2*NB_DELTA_CEPS+1] -= 0.9;
  features[NB_BANDS+3*NB_DELTA_CEPS] = .01*(pitch_index-300);
  logMax = -2;
  follow = -2;
  for (i=0;i<NB_BANDS;i++) {
    Ly[i] = log10(1e-2+Ex[i]);
    Ly[i] = MAX16(logMax-7, MAX16(follow-1.5, Ly[i]));
    logMax = MAX16(logMax, Ly[i]);
    follow = MAX16(follow-1.5, Ly[i]);
    E += Ex[i];
  }
  if (!TRAINING && E < 0.04) {
    /* If there's no audio, avoid messing up the state. */
    RNN_CLEAR(features, NB_FEATURES);
    return 1;
  }
  dct(features, Ly);
  features[0] -= 12;
  features[1] -= 4;
  ceps_0 = st->cepstral_mem[st->memid];
  ceps_1 = (st->memid < 1) ? st->cepstral_mem[CEPS_MEM+st->memid-1] : st->cepstral_mem[st->memid-1];
  ceps_2 = (st->memid < 2) ? st->cepstral_mem[CEPS_MEM+st->memid-2] : st->cepstral_mem[st->memid-2];
  for (i=0;i<NB_BANDS;i++) ceps_0[i] = features[i];
  st->memid++;
  for (i=0;i<NB_DELTA_CEPS;i++) {
    features[i] = ceps_0[i] + ceps_1[i] + ceps_2[i];
    features[NB_BANDS+i] = ceps_0[i] - ceps_2[i];
    features[NB_BANDS+NB_DELTA_CEPS+i] =  ceps_0[i] - 2*ceps_1[i] + ceps_2[i];
  }
  /* Spectral variability features. */
  if (st->memid == CEPS_MEM) st->memid = 0;
  for (i=0;i<CEPS_MEM;i++)
  {
    int j;
    float mindist = 1e15f;
    for (j=0;j<CEPS_MEM;j++)
    {
      int k;
      float dist=0;
      for (k=0;k<NB_BANDS;k++)
      {
        float tmp;
        tmp = st->cepstral_mem[i][k] - st->cepstral_mem[j][k];
        dist += tmp*tmp;
      }
      if (j!=i)
        mindist = MIN32(mindist, dist);
    }
    spec_variability += mindist;
  }
  features[NB_BANDS+3*NB_DELTA_CEPS+1] = spec_variability/CEPS_MEM-2.1;
  return TRAINING && E < 0.1;
}

static void frame_synthesis(DenoiseState *st, float *out, const kiss_fft_cpx *y) {
  float x[WINDOW_SIZE];
  int i;
  inverse_transform(x, y);
  apply_window(x);
  for (i=0;i<FRAME_SIZE;i++) out[i] = x[i] + st->synthesis_mem[i];
  RNN_COPY(st->synthesis_mem, &x[FRAME_SIZE], FRAME_SIZE);
}

static void biquad(float *y, float mem[2], const float *x, const float *b, const float *a, int N) {
  int i;
  for (i=0;i<N;i++) {
    float xi, yi;
    xi = x[i];
    yi = x[i] + mem[0];
    mem[0] = mem[1] + (b[0]*(double)xi - a[0]*(double)yi);
    mem[1] = (b[1]*(double)xi - a[1]*(double)yi);
    y[i] = yi;
  }
}

void pitch_filter(kiss_fft_cpx *X, const kiss_fft_cpx *P, const float *Ex, const float *Ep,
                  const float *Exp, const float *g) {
  int i;
  float r[NB_BANDS];
  float rf[FREQ_SIZE] = {0};
  for (i=0;i<NB_BANDS;i++) {
#if 0
    if (Exp[i]>g[i]) r[i] = 1;
    else r[i] = Exp[i]*(1-g[i])/(.001 + g[i]*(1-Exp[i]));
    r[i] = MIN16(1, MAX16(0, r[i]));
#else
    if (Exp[i]>g[i]) r[i] = 1;
    else r[i] = SQUARE(Exp[i])*(1-SQUARE(g[i]))/(.001 + SQUARE(g[i])*(1-SQUARE(Exp[i])));
    r[i] = sqrt(MIN16(1, MAX16(0, r[i])));
#endif
    r[i] *= sqrt(Ex[i]/(1e-8+Ep[i]));
  }
  interp_band_gain(rf, r);
  for (i=0;i<FREQ_SIZE;i++) {
    X[i].r += rf[i]*P[i].r;
    X[i].i += rf[i]*P[i].i;
  }
  float newE[NB_BANDS];
  compute_band_energy(newE, X);
  float norm[NB_BANDS];
  float normf[FREQ_SIZE]={0};
  for (i=0;i<NB_BANDS;i++) {
    norm[i] = sqrt(Ex[i]/(1e-8+newE[i]));
  }
  interp_band_gain(normf, norm);
  for (i=0;i<FREQ_SIZE;i++) {
    X[i].r *= normf[i];
    X[i].i *= normf[i];
  }
}

void gain_pitch_filter(const float *Ex_near, const float *Ex_far, const float *Exp_near, const float *Exp_far, float *g) {
  int i;
  float r[NB_BANDS];
  float rf[FREQ_SIZE] = {0};
  for (i=0;i<NB_BANDS;i++) {
        g[i] = MIN16(1.0, Exp_near[i]/(1e-8+Exp_far[i])-0.3);
        g[i] = MAX16(0.0, g[i]);
    }
}

float rnnoise_process_frame(DenoiseState *st_near, DenoiseState *st_far, const float *near, const float *far, float *out) {
  int i;
  kiss_fft_cpx X_near[FREQ_SIZE];
  kiss_fft_cpx P_near[WINDOW_SIZE];
  kiss_fft_cpx X_far[FREQ_SIZE];
  kiss_fft_cpx P_far[WINDOW_SIZE];
  float x_near[FRAME_SIZE];
  float x_far[FRAME_SIZE];
  float Ex_near[NB_BANDS], Ep_near[NB_BANDS];
  float Exp_near[NB_BANDS];
  float Ex_far[NB_BANDS], Ep_far[NB_BANDS];
  float Exp_far[NB_BANDS];
  float features_near[NB_FEATURES];
  float features_far[NB_FEATURES];
  float g[NB_BANDS]={0};
  float gf[FREQ_SIZE]={1};
  float delay_value = 0;
  int silence;
  static const float a_hp[2] = {-1.99599, 0.99600};
  static const float b_hp[2] = {-2, 1};
  biquad(x_near, st_near->mem_hp_x, near, b_hp, a_hp, FRAME_SIZE);
  biquad(x_far, st_far->mem_hp_x, far, b_hp, a_hp, FRAME_SIZE);
  silence = compute_frame_features(st_near, X_near, P_near, Ex_near, Ep_near, Exp_near, features_near, x_near);
  silence = compute_frame_features(st_far, X_far, P_far, Ex_far, Ep_far, Exp_far, features_far, x_far);

  if (!silence) {
    compute_rnn(&st_near->rnn, g, &delay_value, features_near, features_far);
    gain_pitch_filter(Ex_near, Ex_far, Exp_near, Exp_far, g);
    for (i=0;i<NB_BANDS;i++) {
      float alpha = .6f;
      g[i] = MAX16(g[i], alpha*st_near->lastg[i]);
      st_near->lastg[i] = g[i];
    }
    interp_band_gain(gf, g);
#if 1
    for (i=0;i<FREQ_SIZE;i++) {
      X_near[i].r *= gf[i];
      X_near[i].i *= gf[i];
    }
#endif
  }

  frame_synthesis(st_near, out, X_near);
  return delay_value;
}

#if TRAINING

static float uni_rand() {
  return rand()/(double)RAND_MAX-.5;
}

static void rand_resp(float *a, float *b) {
  a[0] = .75*uni_rand();
  a[1] = .75*uni_rand();
  b[0] = .75*uni_rand();
  b[1] = .75*uni_rand();
}

int main(int argc, char **argv) {
  int i;
  int count=0;
  static const float a_hp[2] = {-1.99599, 0.99600};
  static const float b_hp[2] = {-2, 1};

  float far_a_noise[2] = {0};
  float far_b_noise[2] = {0};
  float far_a_speech[2] = {0};
  float far_b_speech[2] = {0};

  float delayed_far_a_noise[2] = {0};
  float delayed_far_b_noise[2] = {0};
  float delayed_far_a_speech[2] = {0};
  float delayed_far_b_speech[2] = {0};

  float near_a_noise[2] = {0};
  float near_b_noise[2] = {0};
  float near_a_speech[2] = {0};
  float near_b_speech[2] = {0};

  float mem_hp_far_speech[2]={0};
  float mem_resp_far_speech[2]={0};
  float mem_hp_far_noise[2]={0};
  float mem_resp_far_noise[2]={0};

  float mem_hp_delayed_far_speech[2]={0};
  float mem_resp_delayed_far_speech[2]={0};
  float mem_hp_delayed_far_noise[2]={0};
  float mem_resp_delayed_far_noise[2]={0};

  float mem_hp_near_speech[2]={0};
  float mem_resp_near_speech[2]={0};
  float mem_hp_near_noise[2]={0};
  float mem_resp_near_noise[2]={0};

  float far_speech[FRAME_SIZE];
  float far_noise[FRAME_SIZE];
  float far_noisy[FRAME_SIZE];
  float delayed_far_speech[FRAME_SIZE];
  float delayed_far_noise[FRAME_SIZE];
  float delayed_far_noisy[FRAME_SIZE];
  float near_speech[FRAME_SIZE];
  float near_noise[FRAME_SIZE];
  float near_noisy[FRAME_SIZE];
  float mic_captured[FRAME_SIZE];
  
  short far_speech_int16[FRAME_SIZE];
  short far_noisy_int16[FRAME_SIZE];
  short delayed_far_noisy_int16[FRAME_SIZE];
  short mic_captured_int16[FRAME_SIZE];

//Maximum delay is 500ms
  const int delay_den = 500;
//Max frame length is 30s
  const int len_den = 300;
  int vad_cnt=0;
  int gain_change_count=0;
  float far_speech_gain = 1, delayed_far_speech_gain=1, near_speech_gain=1, far_noise_gain = 1, delayed_far_noise_gain=1, near_noise_gain=1;
  FILE *f_far_speech, *f_near_speech, *f_far_noise, *f_near_noise, *f_out;
#ifdef DEBUG
  FILE *f_far_noisy, *f_delayed_far_noisy, *f_near_noisy, *f_mic_record;
  f_far_noisy = fopen("far_noisy.pcm", "wb");
  f_delayed_far_noisy = fopen("delayed_far_noisy.pcm", "wb");
  //f_near_noisy = fopen("near_noisy.pcm", "wb");
  f_mic_record = fopen("mic_record.pcm", "wb");
#endif
  int maxCount;
  DenoiseState *st_near_speech;
  DenoiseState *st_near_noisy;
  DenoiseState *st_mic_captured;
  DenoiseState *st_far_noisy;
  st_near_speech = rnnoise_create(NULL);
  st_mic_captured = rnnoise_create(NULL);
  st_far_noisy = rnnoise_create(NULL);
  if (argc!=7) {
    fprintf(stderr, "usage: %s <far speech> <near speech> <far noise> <near noise> <count> <outputfile>\n", argv[0]);
    return 1;
  }
  f_far_speech = fopen(argv[1], "r");
  f_near_speech = fopen(argv[2], "r");
  f_far_noise = fopen(argv[3], "r");
  f_near_noise = fopen(argv[4], "r");
  f_out = fopen(argv[6], "wb");
  maxCount = atoi(argv[5]);
  float delayed_gain_factor = 1.0+ (4.0 - rand()%9)/10;


  short tmp_speech_slide[FRAME_SIZE*100] = {0};
  short tmp_noise_slide[FRAME_SIZE*100] = {0};
  while (1) {
    int delay = rand()%delay_den;
    int len = rand()%len_den + delay/10;
    if (count>=maxCount) break;
    if ((count%1000)==0) fprintf(stderr, "%d\r", count);
    if (++gain_change_count > 2821) {
       far_speech_gain = pow(10., (-40+(rand()%60))/20.);
       far_noise_gain = pow(10., (-30+(rand()%50))/20.);
       near_speech_gain = pow(10., (-40+(rand()%60))/20.);
       near_noise_gain = pow(10., (-30+(rand()%50))/20.);
       if (rand()%10==0) { far_noise_gain = 0; near_noise_gain = 0;}
       far_noise_gain *= far_speech_gain;
       delayed_far_noise_gain *= delayed_far_speech_gain;
       near_noise_gain *= near_speech_gain;
       if (rand()%10==0) { far_speech_gain = 0; near_speech_gain = 0;}
       gain_change_count = 0;
       rand_resp(far_a_noise, far_b_noise);
       rand_resp(far_a_speech, far_b_speech);
       rand_resp(delayed_far_a_noise, delayed_far_b_noise);
       rand_resp(delayed_far_a_speech, delayed_far_b_speech);
       rand_resp(near_a_noise, near_b_noise);
       rand_resp(near_a_speech, near_b_speech);

       lowpass = FREQ_SIZE * 3000./24000. * pow(50., rand()/(double)RAND_MAX);
       for (i=0;i<NB_BANDS;i++) {
         if (eband5ms[i] > lowpass) {
           band_lp = i;
           break;
         }
       }
    }

    for(int tot=0; tot< len; tot++){
        kiss_fft_cpx X_near_speech[FREQ_SIZE], Y_near_speech[FREQ_SIZE], N_near_speech[FREQ_SIZE], P_near_speech[WINDOW_SIZE];
        float Ex_near_speech[NB_BANDS], Ey_near_speech[NB_BANDS], En_near_speech[NB_BANDS], Ep_near_speech[NB_BANDS];

        kiss_fft_cpx X_near_noisy[FREQ_SIZE], Y_near_noisy[FREQ_SIZE], N_near_noisy[FREQ_SIZE], P_near_noisy[WINDOW_SIZE];
        float Ex_near_noisy[NB_BANDS], Ey_near_noisy[NB_BANDS], En_near_noisy[NB_BANDS], Ep_near_noisy[NB_BANDS];

        kiss_fft_cpx X_far_noisy[FREQ_SIZE], Y_far_noisy[FREQ_SIZE], N_far_noisy[FREQ_SIZE], P_far_noisy[WINDOW_SIZE];
        float Ex_far_noisy[NB_BANDS], Ey_far_noisy[NB_BANDS], En_far_noisy[NB_BANDS], Ep_far_noisy[NB_BANDS];
	float Exp_far_noisy[NB_BANDS];

	kiss_fft_cpx X_mic_captured[FREQ_SIZE], Y_mic_captured[FREQ_SIZE], N_mic_captured[FREQ_SIZE], P_mic_captured[WINDOW_SIZE];
	float Ex_mic_captured[NB_BANDS], Ey_mic_captured[NB_BANDS], En_mic_captured[NB_BANDS], Ep_mic_captured[NB_BANDS];
        float Exp_mic_captured[NB_BANDS];
        float Ln[NB_BANDS];
        float mic_captured_features[NB_FEATURES];
	float far_noisy_features[NB_FEATURES];
	float g[NB_BANDS];
        float g_mic[NB_BANDS];
	float g_far_noisy[NB_BANDS];
        short tmp[FRAME_SIZE];
        short tmp_delayed_far[FRAME_SIZE];
        float vad=0;
        float E_far_speech=0, E_near_speech=0, E_mic_captured=0, E_far_noisy=0;
        //printf("far_speech_gain:%f\n", far_speech_gain);

        memmove(tmp_speech_slide, tmp_speech_slide + FRAME_SIZE, 99*sizeof(short)*FRAME_SIZE);
        memmove(tmp_noise_slide, tmp_noise_slide + FRAME_SIZE, 99*sizeof(short)*FRAME_SIZE);

        if (far_speech_gain != 0) {
          fread(tmp, sizeof(short), FRAME_SIZE, f_far_speech);
          if (feof(f_far_speech)) {
            rewind(f_far_speech);
            fread(tmp, sizeof(short), FRAME_SIZE, f_far_speech);
          }
          for (i=0;i<FRAME_SIZE;i++) {
              far_speech[i] = far_speech_gain*tmp[i];
          }
          for (i=0;i<FRAME_SIZE;i++) E_far_speech += tmp[i]*(float)tmp[i];
        } else {
          for (i=0;i<FRAME_SIZE;i++) {
              far_speech[i] = 0;
              delayed_far_speech[i] = 0;
          }
          E_far_speech = 0;
        }
        memcpy(tmp_speech_slide+FRAME_SIZE*99, tmp, sizeof(short)*FRAME_SIZE);

        for (i=0; i<FRAME_SIZE; i++) delayed_far_speech[i] = (far_speech_gain*delayed_gain_factor)*tmp_speech_slide[FRAME_SIZE*99-delay*16+i];

        if (far_noise_gain != 0) {
          fread(tmp, sizeof(short), FRAME_SIZE, f_far_noise);
          if (feof(f_far_noise)) {
            rewind(f_far_noise);
            fread(tmp, sizeof(short), FRAME_SIZE, f_far_noise);
          }
          for (i=0;i<FRAME_SIZE;i++) {
              far_noise[i] = far_noise_gain*tmp[i];
          }
        } else {
          for (i=0;i<FRAME_SIZE;i++) {
              far_noise[i] = 0;
              delayed_far_noise[i] = 0;
          }
        }
        memcpy(tmp_noise_slide+FRAME_SIZE*99, tmp, sizeof(short)*FRAME_SIZE);
        for (i=0; i<FRAME_SIZE; i++) delayed_far_noise[i] = far_noise_gain*delayed_gain_factor*tmp_noise_slide[FRAME_SIZE*99-delay*16+i];

        if (near_speech_gain != 0) {
          fread(tmp, sizeof(short), FRAME_SIZE, f_near_speech);
          if (feof(f_near_speech)) {
            rewind(f_near_speech);
            fread(tmp, sizeof(short), FRAME_SIZE, f_near_speech);
          }
          for (i=0;i<FRAME_SIZE; i++) {
              near_speech[i] = near_speech_gain*tmp[i];
          }
          for (i=0;i<FRAME_SIZE;i++) E_near_speech += tmp[i]*(float)tmp[i];
        } else {
          for (i=0;i<FRAME_SIZE;i++) {
              near_speech[i] = 0;
          }
          E_near_speech = 0;
        }
    
        if (near_noise_gain != 0) {
          fread(tmp, sizeof(short), FRAME_SIZE, f_near_noise);
          if (feof(f_near_noise)) {
            rewind(f_near_noise);
            fread(tmp, sizeof(short), FRAME_SIZE, f_near_noise);
          }
          for (i=0;i<FRAME_SIZE;i++) {
              near_noise[i] = near_noise_gain*tmp[i];
          }
        } else {
          for (i=0;i<FRAME_SIZE;i++) {
              near_noise[i] = 0;
          }
        }

        biquad(far_speech, mem_hp_far_speech, far_speech, b_hp, a_hp, FRAME_SIZE);
        biquad(far_speech, mem_resp_far_speech, far_speech, far_b_speech, far_a_speech, FRAME_SIZE);
        biquad(far_noise, mem_hp_far_noise, far_noise, b_hp, a_hp, FRAME_SIZE);
        biquad(far_noise, mem_resp_far_noise, far_noise, far_b_noise, far_a_noise, FRAME_SIZE);
    
        biquad(delayed_far_speech, mem_hp_delayed_far_speech, delayed_far_speech, b_hp, a_hp, FRAME_SIZE);
        biquad(delayed_far_speech, mem_resp_delayed_far_speech, delayed_far_speech, delayed_far_b_speech, delayed_far_a_speech, FRAME_SIZE);
        biquad(delayed_far_noise, mem_hp_delayed_far_noise, delayed_far_noise, b_hp, a_hp, FRAME_SIZE);
        biquad(delayed_far_noise, mem_resp_delayed_far_noise, delayed_far_noise, delayed_far_b_noise, delayed_far_a_noise, FRAME_SIZE);
    
        biquad(near_speech, mem_hp_near_speech, near_speech, b_hp, a_hp, FRAME_SIZE);
        biquad(near_speech, mem_resp_near_speech, near_speech, near_b_speech, near_a_speech, FRAME_SIZE);
        biquad(near_noise, mem_hp_near_noise, near_noise, b_hp, a_hp, FRAME_SIZE);
        biquad(near_noise, mem_resp_near_noise, near_noise, near_b_noise, near_a_noise, FRAME_SIZE);
    
        for (i=0;i<FRAME_SIZE;i++){
            far_noisy[i] = far_speech[i] + far_noise[i];
            delayed_far_noisy[i] = delayed_far_speech[i] + delayed_far_noise[i];
            near_noisy[i] = near_noise[i] + near_speech[i];
            far_noisy_int16[i] = (short)far_noisy[i];
            delayed_far_noisy_int16[i] = (short)delayed_far_noisy[i];
        }
    
        for (i=0;i<FRAME_SIZE;i++){
            mic_captured[i] = near_speech[i] + near_noise[i] + delayed_far_noisy[i];
            mic_captured_int16[i] = (short)mic_captured[i];
        }

#ifdef DEBUG
        fwrite(far_noisy_int16, sizeof(short), FRAME_SIZE, f_far_noisy);
        fwrite(delayed_far_noisy_int16, sizeof(short), FRAME_SIZE, f_delayed_far_noisy);
        //fwrite(delayed_far_noisy, sizeof(float), FRAME_SIZE, f_near_noisy);
        fwrite(mic_captured_int16, sizeof(short), FRAME_SIZE, f_mic_record);
#endif
        if (E_near_speech > 1e9f) {
        	vad_cnt=0;
        } else if (E_near_speech > 1e8f) {
        	vad_cnt -= 5;
        } else if (E_near_speech > 1e7f) {
        	vad_cnt++;
        } else {
        	vad_cnt+=2;
        }
        if (vad_cnt < 0) vad_cnt = 0;
        if (vad_cnt > 15) vad_cnt = 15;
        
        if (vad_cnt >= 10) vad = 0;
        else if (vad_cnt > 0) vad = 0.5f;
        else vad = 1.f;
    
        frame_analysis(st_near_speech, Y_near_speech, Ey_near_speech, near_speech);
        frame_analysis(st_near_noisy, Y_near_noisy, Ey_near_noisy, near_noisy);
        frame_analysis(st_mic_captured, Y_mic_captured, Ey_mic_captured, mic_captured);
        frame_analysis(st_far_noisy, Y_far_noisy, Ey_far_noisy, far_noisy);
        for (i=0;i<NB_BANDS;i++) Ln[i] = log10(1e-2+En_mic_captured[i]);
        int silence = compute_frame_features(st_far_noisy, X_far_noisy, P_far_noisy, Ex_far_noisy, Ep_far_noisy, Exp_far_noisy, far_noisy_features, far_noisy);
        silence = compute_frame_features(st_mic_captured, X_mic_captured, P_mic_captured, Ex_mic_captured, Ep_mic_captured, Exp_mic_captured, mic_captured_features, mic_captured);
        //pitch_filter(X_mic_captured, P_mic_captured, Ex_mic_captured, Ep_mic_captured, Exp_mic_captured, g_mic);
	//pitch_filter(X_far_noisy, P_far_noisy, Ex_far_noisy, Ep_far_noisy, Exp_far_noisy, g_far_noisy);
		
        //printf("%f %d\n", noisy->last_gain, noisy->last_period);
        for (i=0;i<NB_BANDS;i++) {
          g[i] = sqrt((Ey_near_noisy[i]+1e-3)/(Ex_mic_captured[i]+1e-3));
          if (g[i] > 1) g[i] = 1;
          if (silence || i > band_lp) g[i] = -1;
          if (Ey_near_speech[i] < 5e-2 && Ex_mic_captured[i] < 5e-2) g[i] = -1;
          if (near_noise_gain==0) g[i] = -1;
        }
    #if 1
        fwrite(mic_captured_features, sizeof(float), NB_FEATURES, f_out);
        fwrite(far_noisy_features, sizeof(float), NB_FEATURES, f_out);
        fwrite(g, sizeof(float), NB_BANDS, f_out);
        fwrite(Ln, sizeof(float), NB_BANDS, f_out);
        fwrite(&vad, sizeof(float), 1, f_out);
        fwrite(&delay, sizeof(float), 1, f_out);
    #endif
        count++;
    }
  }
  fprintf(stderr, "matrix size: %d x %d\n", count, NB_FEATURES*2 + 2*NB_BANDS + 1+1);
  fclose(f_far_speech);
  fclose(f_near_speech);
  fclose(f_far_noise);
  fclose(f_near_noise);
  fclose(f_out);
#ifdef DEBUG
  fclose(f_far_noisy);
  fclose(f_delayed_far_noisy);
//  fclose(f_near_noisy);
  fclose(f_mic_record);
#endif
  return 0;
}

#endif
