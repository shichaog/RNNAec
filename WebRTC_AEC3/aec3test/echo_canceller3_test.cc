/*
 *  Copyright (c) 2016 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "echo_canceller3.h"

#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>

#include "aec3_common.h"
#include "block_processor.h"
#include "frame_blocker.h"
//#include "mock/mock_block_processor.h"
#include "audio_buffer.h"
#include "string_builder.h"
#include "gmock.h"
#include "gtest.h"
using namespace webrtc;
#define PARTLEN 160
#define PARTLEN2 (PARTLEN*2)
#define SAMPLE_RATE_HZ 16000
#define NUM_CHANNELS 1
//namespace webrtc {
void usage() {
	printf("Usage: aec3_test [options] [-ir REVERSE_FILE] [-i PRIMARY_FILE]\n");
	printf("  [-o OUT_FILE]\n");

}
int main(int argc, char*argv[]) {
	if(argc > 1 && strcmp(argv[1], "--help") == 0) {
	   usage();
	   return 0;
	}
	if (argc < 2) {
	   printf("Run without arguments?\n");
	}
	const char* far_filename = NULL;
	const char* near_filename = NULL;
	const char* out_filename = NULL;
    char far_file_default[] = "aecfar.pcm";
	char near_file_default[] = "aecnear.pcm";
	char out_file_default[] = "aecout.pcm";
	FILE *fp_far = NULL;
	FILE *fp_near = NULL;
	FILE *fp_out = NULL;
	int16_t InBlock[PARTLEN];
	int16_t RefBlock[PARTLEN];
	int16_t OutBlock[PARTLEN];
	int sample_rate_hz = SAMPLE_RATE_HZ;
	size_t num_bands = (sample_rate_hz == 8000 ? 1 : sample_rate_hz / 16000);;
	size_t frame_length = sample_rate_hz == 8000 ? 80 : 160;
	int fullband_frame_length = sample_rate_hz / 100;
	bool use_highpass_filter = true;
	int num_channels = NUM_CHANNELS;
	int framecount = 0;
	int i, j, k,f;
	AudioBuffer capture_buffer(fullband_frame_length, num_channels, fullband_frame_length, num_channels, fullband_frame_length);
	AudioBuffer render_buffer(fullband_frame_length, num_channels, fullband_frame_length, num_channels, fullband_frame_length);
	EchoCanceller3 AEC3(EchoCanceller3Config(), sample_rate_hz, use_highpass_filter);
	StreamConfig streamconfig(sample_rate_hz, num_channels, false);
	float **NearIn = (float**)malloc(sizeof(float *)*num_channels);
	for (i = 0; i < num_channels; i++) {
		NearIn[i] = (float *)malloc(sizeof(float)*PARTLEN);
		memset(NearIn[i], 0, sizeof(float)*PARTLEN);
	}
	float **NearOut = (float**)malloc(sizeof(float *)*num_channels);
	for (i = 0; i < num_channels; i++) {
		NearOut[i] = (float *)malloc(sizeof(float)*PARTLEN);
		memset(NearOut[i], 0, sizeof(float)*PARTLEN);
	}

	float **FarIn = (float **)malloc(sizeof(float*)*num_channels);
	for (i = 0; i < num_channels; i++) {
		FarIn[i] = (float *)malloc(sizeof(float)*PARTLEN);
		memset(FarIn[i], 0,sizeof(float)*PARTLEN);
	}

	
	if (argc > 1) {
		for (int i = 1; i < argc; i++) {
			if (strcmp(argv[i], "-ir") == 0) {
				i++;
				if (i > argc) printf("Specify filename after -ir\n");
				far_filename = argv[i];
			}
			else if (strcmp(argv[i], "-i") == 0) {
				i++;
				if (i > argc) printf("Specify filename after -i\n");
				near_filename = argv[i];

			}
			else if (strcmp(argv[i], "-o") == 0) {
				i++;
				if (i > argc) printf("Specify filename after -o\n");
				out_filename = argv[i];
			}
			else {
				printf("Unrecognized argument\n");
			}
		}
	}else {
		printf("Run aec3test with default file names!\n");
		if(far_filename == NULL)  far_filename = far_file_default;
		if(near_filename == NULL) near_filename = near_file_default;
		if(out_filename == NULL) out_filename = out_file_default;
	}

	fp_out = fopen(out_filename, "wb+");
	if (fp_out == NULL) {
		printf("Error Create outfile\n");
		return -1;
	}


	fp_far = fopen(far_filename, "rb");
	if (fp_far == NULL) {
		printf("Error Open far file\n");
		return -1;
	}	
	fp_near = fopen(near_filename, "rb");
	if (fp_near == NULL) {
		printf("Error Open near file\n");
		return -1;
	}
	
	for (f = 0; ; f++) {
		framecount++;
		if (feof(fp_far) || feof(fp_near)) break;
		fread(InBlock, sizeof(int16_t), PARTLEN, fp_near);
		fread(RefBlock, sizeof(int16_t), PARTLEN, fp_far);
		for (j = 0; j < PARTLEN; j++) {
			NearIn[0][j] = (float) InBlock[j];
			FarIn[0][j] = (float) RefBlock[j];
			
		}
		capture_buffer.CopyFrom(NearIn, streamconfig);
		render_buffer.CopyFrom(FarIn, streamconfig);
		//printf("Frame:%d \n", framecount);
		AEC3.AnalyzeRender(&render_buffer);
		AEC3.AnalyzeCapture(&capture_buffer);
		//AEC3.SetAudioBufferDelay(43);
		AEC3.ProcessCapture(&capture_buffer,false);
	    capture_buffer.CopyTo(streamconfig, NearOut);
		for (j = 0; j < PARTLEN; j++) {
			int16_t tmp;
			if (NearOut[0][j] > 32767) tmp = 32767;
			else if (NearOut[0][j] < -32768) tmp = -32768;
			else tmp = (int16_t)NearOut[0][j];
			OutBlock[j] = tmp;
		}   
		fwrite(OutBlock, sizeof(int16_t), PARTLEN, fp_out);
	}

	fclose(fp_far);
	fclose(fp_near);
	fclose(fp_out);

	if (NearIn) {
		for (i = 0; i < num_channels; i++)
		{
			if (NearIn) free(NearIn[i]);
		}
		if (NearIn) free(NearIn);
	}
	if (NearOut) {
		for (i = 0; i < num_channels; i++)
		{
			if (NearOut) free(NearOut[i]);
		}
		if (NearOut) free(NearOut);
	}
	if (FarIn) {
		for (i = 0; i < num_channels; i++)
		{
			if (FarIn) free(FarIn[i]);
		}
		if (FarIn) free(FarIn);
	}
	return 0;
}

//}  // namespace webrtc
