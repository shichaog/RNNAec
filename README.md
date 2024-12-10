![My book](https://github.com/shichaog/WebRTC-audio-processing/blob/master/book.png)

[天猫购买链接](https://detail.tmall.com/item.htm?spm=a220m.1000858.1000725.6.3a8e144cSO3Gp9&id=616382027158&areaId=330100&user_id=1932014659&cat_id=2&is_b=1&rn=919b763eb3051be569c91f85996e73eb)

[京东购买链接](https://item.jd.com/12838726.html)

关于效果问题，这里我解释下：

关于效果不好，我这里主要的原因是训练数据多样性不足，导致模型泛化比较差， 所以如果我提供的数据集（后文有下载链接）效果是可以的，那么我建议扩充数据集（或者数据下载链接整理后发我邮箱，里头应该还有些小细节以及和基于模型的算法配合需要尝试），至少涵盖你测试case的数据情况，应该也是可以的，但要普适性基于建模的方法应该是要的。

如果想着拿去直接商业化可能要失望，如果您的数据集效果不好（不论是算法生成的或是真实场景的），都可以给笔者传份，下载方式发送邮箱shichaog@126.com，汇总后数据量够多的话，我会更新个更准确的模型，数据不足的话就算了（贡献数据的各位朋友也会按需在readme里一一感谢），另外如果对这个模型或者算法有兴趣的，比如增加信号处理的LP支持等，也可以申请这个库的开发者权限（也会按需readme里列出），欢迎各位朋友一起提升性能，谢谢~！


为了方便重现，这里把生成的训练数据和结果传了上来，应该比较大，只截取的数据含结果12G（这12G包含的训练集生成算法一样的，也许生成两三个G的结果训练集效果和这差不多），如果你只用我附带的部分数据training，我不清楚训练的结果是否能和我一样。
链接: https://pan.baidu.com/s/1mchtEFtFzKurR2Weum8IDA 提取码: kqm2 复制这段内容后打开百度网盘手机App，操作更方便哦

上传的数据是使用如下命令测试的，其中rnn_near.pcm是近端语音数据，rnn_far.pcm是远端语音数据，aec_out.pcm是aec结果数据，所有的数据均是16kHz，mono，raw 数据；
```
~/RNNAec/examples/rnnaec_demo dataset/rnn_near.pcm dataset/rnn_far.pcm aec_out.pcm
```
result.png是前几秒的效果图。
I also generated result.png for view.


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
%./examples/rnnaec_demo <near speech> <far speech> out.pcm

Next to research:
1) Use this RNNAec as a NLP(Non-Linear Processing) module, LP(Linear Processing) module can use (speex or webrtc).
2) NLP-RNNAec used to improve double-talk performance. Model based method LP(signal process) are more general for various devices and environment.
3) Training dataset can add real captured, instead of just use clean speech and noise data(As above training example)
4) This RNNAec can have denoise effect, Network layer may change as needed.
5) Band may change as need.(I think should change)
6) Pitch filter(I think use full band can replace this module, but computational complexity also rise)

Note: To get beset result, pls use clean raw data to train and test; use no delay for test
Note: This repo just for demo, I just test a limited cases, If you have problems, pls describe detail training process && attach testing datas(pcm, wav)

If you like this repo, Please click star!!!

UPdate WebRTC AEC3 VS2019 soultion test module as a standalone module. May have time integrate as a linear processing module as Next to research part noted.
