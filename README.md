![My book](https://github.com/shichaog/WebRTC-audio-processing/blob/master/book.png)

[天猫购买链接](https://detail.tmall.com/item.htm?spm=a220m.1000858.1000725.6.3a8e144cSO3Gp9&id=616382027158&areaId=330100&user_id=1932014659&cat_id=2&is_b=1&rn=919b763eb3051be569c91f85996e73eb)

[京东购买链接](https://item.jd.com/12838726.html)

关于效果问题，这里我解释下：

关于效果比较差不太好回复，举个简单例子，我录个几分钟数据，然后就用这几分钟数据去训练模型，然后其他人不知道哪里搞了段数据，来测试，“然后说不行”，我只能说基于传统建模和数学公式推导的算法普适性更好，这个是有物理意义的，比如基于谐振腔的发声模型一定是有基频和谐波存在，但是能不能根据几分钟数据训练一个模型学习到发声模型的基频和谐波特性，这个就难说了，也许能，也许部分能，也许完全不能，但是如果数据是完备的，那我想应该是能的。 所以如果用我提供的数据（后文有下载链接）效果是可以的，那么我建议扩充数据集（或者数据下载链接整理后发我邮箱），至少涵盖你测试case的数据情况，应该也是可以的（类似于图片识别，训练集中没有人像，然后你拿个人的照片灌入模型；但是基于信号处理建模的方法，不需要照片应该就可以，毕竟人的五官特征和相对位置关系具有物理学上的意义，结论就是正常也不正常，我这里的模型不是信号处理建模，建模的过程靠数据，答案根据数据集差异而差异，也许有些人用的数据恰好和我的模型训练结果涵盖了，他那就是行的）


很多朋友对这个算法非常感兴趣，在此做个说明，请一定先阅读完中文说明，谢谢~！

为了方便重现，这里把生成的训练数据和结果传了上来，应该比较大，只截取的数据，含结果12G，所以，如果你只用我附带的部分数据training，我不清楚训练的结果是否能和我一样。
链接: https://pan.baidu.com/s/1mchtEFtFzKurR2Weum8IDA 提取码: kqm2 复制这段内容后打开百度网盘手机App，操作更方便哦

上传的数据是使用如下命令测试的，其中rnn_near.pcm是近端语音数据，rnn_far.pcm是远端语音数据，aec_out.pcm是aec结果数据，所有的数据均是16kHz，mono，raw 数据；
```
~/RNNAec/examples/rnnaec_demo dataset/rnn_near.pcm dataset/rnn_far.pcm aec_out.pcm
```
result.png是前几秒的效果图。
I also generated result.png for view.

这一效果是基于算法生成数据作为集训练的情况（主要是笔者真实数据不足），如果想着拿去直接商业化可能要失望，如果您的数据集效果不好（不论是算法生成的或是真实场景的），都可以给笔者传份，下载方式发送邮箱shichaog@126.com，汇总后数据量够多的话，我会更新个更准确的模型，数据不足的话就算了（贡献数据的各位朋友也会在这个readme里一一感谢），另外如果对这个模型或者算法有兴趣的，比如增加信号处理的LP支持等，也可以申请这个库的开发者权限（也会readme里列出），欢迎各位朋友一起提升性能，谢谢~！

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
