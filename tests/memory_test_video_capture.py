#!/usr/bin/python
#  ================================================================
#  Created by Gregory Kramida on 7/20/18.
#  Copyright (c) 2018 Gregory Kramida
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ================================================================

import cv2
import numpy as np
import pbcvt


# Used to test the code for memory leaks: requires an OpenCV-compatible webcam to be connected to the system
# If there is a memory leak in the conversion, memory used by the program should grow to eventually overwhelm
# the system; memory usage monitors may be used to check the behavior
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    can_capture = True
    while can_capture:
        can_capture, frame = cap.read()
        cv2.imshow('video', frame)
        ch = 0xFF & cv2.waitKey(1)
        frame_copy = frame.copy()
        if ch == 27:
            break
        pbcvt.increment_elements_by_one(frame)
        sum = (np.sum(frame - frame_copy) / (frame.size/3))
        #print(sum)
    print("exiting...")
