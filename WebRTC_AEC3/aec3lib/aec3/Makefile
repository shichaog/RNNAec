LOCAL_PATH := $(patsubst %/,%,$(dir $(shell pwd)/$(lastword $(MAKEFILE_LIST))))

include $(BUILD_DEF)
#LIB_C_SRC := \
    $(LOCAL_PATH)/fft4g.c \
    $(LOCAL_PATH)/real_fft.c \
    $(LOCAL_PATH)/common.c \
    $(LOCAL_PATH)/complex_bit_reverse.c \
    $(LOCAL_PATH)/complex_fft.c  \
   

LOCAL_CPP_EXTENSION :=.cc
LIB_CPP_SRC := \
    $(LOCAL_PATH)/adaptive_fir_filter.cc \
	$(LOCAL_PATH)/aec3_common.cc \
	$(LOCAL_PATH)/aec3_fft.cc \
    $(LOCAL_PATH)/aec_state.cc \
    $(LOCAL_PATH)/api_call_jitter_metrics.cc  \
    $(LOCAL_PATH)/block_delay_buffer.cc \
    $(LOCAL_PATH)/block_framer.cc \
    $(LOCAL_PATH)/block_processor.cc \
    $(LOCAL_PATH)/block_processor_metrics.cc \
    $(LOCAL_PATH)/cascaded_biquad_filter.cc \
    $(LOCAL_PATH)/clockdrift_detector.cc  \
    $(LOCAL_PATH)/comfort_noise_generator.cc \
    $(LOCAL_PATH)/decimator.cc \
    $(LOCAL_PATH)/downsampled_render_buffer.cc \
    $(LOCAL_PATH)/echo_audibility.cc \
    $(LOCAL_PATH)/echo_canceller3.cc \
    $(LOCAL_PATH)/echo_path_delay_estimator.cc \
    $(LOCAL_PATH)/echo_path_variability.cc \
    $(LOCAL_PATH)/echo_remover.cc \
    $(LOCAL_PATH)/echo_remover_metrics.cc \
    $(LOCAL_PATH)/erl_estimator.cc \
    $(LOCAL_PATH)/erle_estimator.cc \
    $(LOCAL_PATH)/fft_buffer.cc \
    $(LOCAL_PATH)/filter_analyzer.cc \
    $(LOCAL_PATH)/frame_blocker.cc  \
    $(LOCAL_PATH)/fullband_erle_estimator.cc \
    $(LOCAL_PATH)/main_filter_update_gain.cc \
    $(LOCAL_PATH)/matched_filter.cc \
    $(LOCAL_PATH)/matched_filter_lag_aggregator.cc \
    $(LOCAL_PATH)/matrix_buffer.cc \
    $(LOCAL_PATH)/moving_average.cc \
    $(LOCAL_PATH)/render_buffer.cc \
    $(LOCAL_PATH)/render_delay_buffer.cc \
    $(LOCAL_PATH)/render_delay_controller.cc \
    $(LOCAL_PATH)/render_delay_controller_metrics.cc \
    $(LOCAL_PATH)/render_reverb_model.cc \
    $(LOCAL_PATH)/render_signal_analyzer.cc \
    $(LOCAL_PATH)/residual_echo_estimator.cc \
    $(LOCAL_PATH)/reverb_decay_estimator.cc \
    $(LOCAL_PATH)/reverb_frequency_response.cc \
    $(LOCAL_PATH)/reverb_model.cc \
    $(LOCAL_PATH)/reverb_model_estimator.cc \
    $(LOCAL_PATH)/reverb_model_fallback.cc  \
    $(LOCAL_PATH)/shadow_filter_update_gain.cc \
    $(LOCAL_PATH)/signal_dependent_erle_estimator.cc \
    $(LOCAL_PATH)/skew_estimator.cc \
    $(LOCAL_PATH)/stationarity_estimator.cc \
    $(LOCAL_PATH)/subband_erle_estimator.cc \
    $(LOCAL_PATH)/subtractor.cc \
    $(LOCAL_PATH)/subtractor_output.cc \
    $(LOCAL_PATH)/subtractor_output_analyzer.cc \
    $(LOCAL_PATH)/suppression_gain.cc \
    $(LOCAL_PATH)/vector_buffer.cc  \
	$(LoCAL_PATH)/aligned_malloc.cc \
    $(LoCAL_PATH)/audio_buffer.cc \
	$(LoCAL_PATH)/audio_frame.cc  \
	$(LoCAL_PATH)/audio_processing_statistics.cc  \
	$(LoCAL_PATH)/channel_buffer.cc  \
	$(LoCAL_PATH)/checks.cc \
	$(LoCAL_PATH)/config.cc  \
	$(LoCAL_PATH)/cpu_features.cc  \
	$(LoCAL_PATH)/cpu_info.cc  \
	$(LoCAL_PATH)/critical_section.cc \
	$(LoCAL_PATH)/dot_product_with_scale.cc \
	$(LoCAL_PATH)/echo_canceller_test_tools.cc  \
    $(LoCAL_PATH)/field_trial.cc  \
    $(LoCAL_PATH)/logging.cc \
    $(LoCAL_PATH)/metrics.cc \
    $(LoCAL_PATH)/ooura_fft.cc \
    $(LoCAL_PATH)/platform_file.cc \
    $(LoCAL_PATH)/push_resampler.cc \
    $(LoCAL_PATH)/push_sinc_resampler.cc \
    $(LoCAL_PATH)/race_checker.cc \
    $(LoCAL_PATH)/random.cc \
    $(LoCAL_PATH)/resampler.cc \
    $(LoCAL_PATH)/sinc_resampler.cc \
    $(LoCAL_PATH)/sparse_fir_filter.cc \
    $(LoCAL_PATH)/splitting_filter.cc \
    $(LoCAL_PATH)/string_builder.cc \
    $(LoCAL_PATH)/string_encode.cc \
    $(LoCAL_PATH)/string_to_number.cc \
    $(LoCAL_PATH)/string_utils.cc \
    $(LoCAL_PATH)/three_band_filter_bank.cc \
    $(LoCAL_PATH)/time_utils.cc \
    $(LoCAL_PATH)/wav_file.cc \
    $(LoCAL_PATH)/wav_header.cc \
    $(LoCAL_PATH)/platform_thread_types.cc 

   #$(LoCAL_PATH)/json.cc \
   $(LoCAL_PATH)/ooura_fft_mips.cc \
    $(LoCAL_PATH)/ooura_fft_neon.cc \
    $(LoCAL_PATH)/ooura_fft_sse2.cc \
    $(LOCAL_PATH)/adaptive_fir_filter_unittest.cc \
    $(LOCAL_PATH)/aec3_fft_unittest.cc \
    $(LOCAL_PATH)/aec_state_unittest.cc \
    $(LOCAL_PATH)/api_call_jitter_metrics_unittest.cc \
	$(LOCAL_PATH)/block_delay_buffer_unittest.cc \
    $(LOCAL_PATH)/block_framer_unittest.cc \
    $(LOCAL_PATH)/block_processor_metrics_unittest.cc \
    $(LOCAL_PATH)/block_processor_unittest.cc \
    $(LOCAL_PATH)/cascaded_biquad_filter_unittest.cc \
    $(LOCAL_PATH)/clockdrift_detector_unittest.cc \
    $(LOCAL_PATH)/comfort_noise_generator_unittest.cc \
    $(LOCAL_PATH)/decimator_unittest.cc \
    $(LOCAL_PATH)/echo_canceller3_unittest.cc \
    $(LOCAL_PATH)/echo_path_delay_estimator_unittest.cc \
    $(LOCAL_PATH)/echo_path_variability_unittest.cc \
    $(LOCAL_PATH)/echo_remover_metrics_unittest.cc \
    $(LOCAL_PATH)/echo_remover_unittest.cc \
    $(LOCAL_PATH)/erl_estimator_unittest.cc \
    $(LOCAL_PATH)/erle_estimator_unittest.cc \
    $(LOCAL_PATH)/fft_data_unittest.cc \
    $(LOCAL_PATH)/filter_analyzer_unittest.cc \
    $(LOCAL_PATH)/frame_blocker_unittest.cc \
    $(LOCAL_PATH)/main_filter_update_gain_unittest.cc \
    $(LOCAL_PATH)/matched_filter_lag_aggregator_unittest.cc \
    $(LOCAL_PATH)/matched_filter_unittest.cc \
    $(LOCAL_PATH)/moving_average_unittest.cc \
    $(LOCAL_PATH)/render_buffer_unittest.cc \
    $(LOCAL_PATH)/render_delay_buffer_unittest.cc \
    $(LOCAL_PATH)/render_delay_controller_metrics_unittest.cc \
    $(LOCAL_PATH)/render_delay_controller_unittest.cc \
    $(LOCAL_PATH)/render_signal_analyzer_unittest.cc \
    $(LOCAL_PATH)/reverb_model_estimator_unittest.cc \
    $(LOCAL_PATH)/residual_echo_estimator_unittest.cc \
    $(LOCAL_PATH)/shadow_filter_update_gain_unittest.cc \
    $(LOCAL_PATH)/signal_dependent_erle_estimator_unittest.cc \
    $(LOCAL_PATH)/vector_math_unittest.cc \
    $(LOCAL_PATH)/suppression_filter_unittest.cc \
    $(LOCAL_PATH)/suppression_gain_unittest.cc \
    $(LOCAL_PATH)/subtractor_unittest.cc \
    $(LOCAL_PATH)/suppression_filter.cc \
    $(LOCAL_PATH)/skew_estimator_unittest.cc \


LIB_INC_PATH += \
    $(LOCAL_PATH) \
	$(LOCAL_PATH)/include \
	$(LOCAL_PATH)/logging \
	$(LOCAL_PATH)/audio \
    $(LOCAL_PATH)/.. \
	$(LOCAL_PATH)/third_party/absl_include 

    #$(TOP_PATH)/utility \
    $(TOP_PATH)/include/internal \
    $(TOP_PATH)/libckits/include \
    $(TOP_PATH) \
    $(TOP_PATH)/system_wrappers/interface \
    $(TOP_PATH)/common_audio/signal_processing/include \
    $(TOP_PATH)/common_audio/include 

CFLAGS += -O3 -fno-omit-frame-pointer -Wno-multichar -D__STDC_CONSTANT_MACROS -Wno-sign-compare  -DUSE_NBNEW -DWEBRTC_POSIX -D_CRT_SECURE_NO_WARNINGS -D_ITERATOR_DEBUG_LEVEL=0 
CXXFLAGS += -O3 -fno-omit-frame-pointer -Wno-multichar -D__STDC_CONSTANT_MACROS -Wno-sign-compare  -DUSE_NBNEW -DWEBRTC_POSIX -D_CRT_SECURE_NO_WARNINGS -D_ITERATOR_DEBUG_LEVEL=0 
CXXFLAGS += -std=c++11
THREE_PARTY_LIB_PATH += \
    $(LOCAL_PATH)/third_party/absl_lib 

LOCAL_STATIC_LIBRARIES += \
    $(LOCAL_PATH)/third_party/absl_include


TARGET_STATIC_LIB = newaec3

include $(BUILD_RULE)

