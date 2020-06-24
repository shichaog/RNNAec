RNNAec is a echo suppression library based on a recurrent neural network.
I refered from RNNnoise Open Source Project.

To compile, just type:
% ./autogen.sh
% ./configure
% make

Optionally:
% make install

While it is meant to be used as a library, a simple command-line tool is
provided as an example. It operates on RAW 16-bit (machine endian) mono
PCM files sampled at 16 kHz. It can be used as:

./examples/rnnaec_demo <mono near speech> <mono far speech> <denoised && aeced output>

The output is also a 16-bit raw PCM file.

To Training for your own model, see next steps:
%cd src && ./compile && cd -
%./src/deecho_training dataset/rnn_far.pcm dataset/rnn_near.pcm dataset/aec_noise_far.pcm dataset/aec_noise_near.pcm 10000000 out.f32

%python ./training/bin2hdf5.py out.f32 XXXXX 138 training.h5

%python ./training/rnn_train.py training.h5

%python training/dump_rnn.py final_weights.hdf5 ./src/rnn_data.c rnn_data.h orig

%make clean & make
%./examples/rnnoise_demo speech_noise.pcm denoised_speech_noise.pcm

Next to research:
1) Use this RNNAec as a NLP(Non-Linear Processing) module, LP(Linear Processing) module can use (speex or webrtc).
2) NLP-RNNAec used to improve double-talk performance. Model based method LP(signal process) are more general for various devices and environment.
3) Training dataset can add real captured, instead of just use clean speech and noise data(As above training example)
4) This RNNAec can have denoise effect, Network layer may change as needed.
5) Band may change as need.(I think should change)
6) Pitch filter(I think use full band can replace this module, but computational complexity also rise)