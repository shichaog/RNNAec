/*
 *  Copyright (c) 2017 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef MODULES_AUDIO_PROCESSING_AEC3_MOCK_MOCK_RENDER_DELAY_CONTROLLER_H_
#define MODULES_AUDIO_PROCESSING_AEC3_MOCK_MOCK_RENDER_DELAY_CONTROLLER_H_

#include "absl/types/optional.h"
#include "array_view.h"
#include "downsampled_render_buffer.h"
#include "render_delay_controller.h"
#include "gmock.h"

namespace webrtc {
namespace test {

class MockRenderDelayController : public RenderDelayController {
 public:
  MockRenderDelayController();
  virtual ~MockRenderDelayController();

  MOCK_METHOD1(Reset, void(bool reset_delay_statistics));
  MOCK_METHOD0(LogRenderCall, void());
  MOCK_METHOD3(GetDelay,
               absl::optional<DelayEstimate>(
                   const DownsampledRenderBuffer& render_buffer,
                   size_t render_delay_buffer_delay,
                   rtc::ArrayView<const float> capture));
  MOCK_CONST_METHOD0(HasClockdrift, bool());
};

}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_MOCK_MOCK_RENDER_DELAY_CONTROLLER_H_
