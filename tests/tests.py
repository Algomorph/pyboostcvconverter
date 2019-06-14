#  ================================================================
#  Created by Gregory Kramida on 6/11/19.
#  Copyright (c) 2019 Gregory Kramida
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

import numpy as np
import pytest
import test_project as tp


def test_dot_vec():
    input_a = np.array([1.0, 3.4, 30.2]).reshape(-1, 1)
    input_b = np.array([33.0, 49.0, 1928.23]).reshape(1, 3)
    expected_output = input_a.dot(input_b)
    assert np.allclose(tp.dot(input_a, input_b), expected_output)


def test_dot_mat():
    input_a = np.random.rand(3, 4)
    input_b = np.random.rand(4, 3)
    expected_output = input_a.dot(input_b)
    assert np.allclose(tp.dot(input_a, input_b), expected_output)


def test_dot2_mat():
    input_a = np.random.rand(3, 4)
    input_b = np.random.rand(4, 3)
    expected_output = input_a.dot(input_b)
    assert np.allclose(tp.dot2(input_a, input_b), expected_output)


def test_increment_elements_by_one_int():
    input = np.arange(-20, 20)
    expected_output = np.arange(-19, 21)
    assert np.allclose(tp.increment_elements_by_one(input), expected_output)


def test_increment_elements_by_one_float():
    input = np.arange(0.0, 25.0, dtype=np.float32)
    expected_output = np.arange(1.0, 26.0, dtype=np.float32)
    assert np.allclose(tp.increment_elements_by_one(input), expected_output)


def test_increment_elements_by_one_double():
    input = np.arange(0.0, 25.0, dtype=np.float64)
    expected_output = np.arange(1.0, 26.0, dtype=np.float64)
    assert np.allclose(tp.increment_elements_by_one(input), expected_output)
