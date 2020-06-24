#!/bin/sh

gcc -DTRAINING=1 -Wall -W -O3 -g -I../include deecho.c kiss_fft.c pitch.c celt_lpc.c rnn.c rnn_data.c -o deecho_training -lm
