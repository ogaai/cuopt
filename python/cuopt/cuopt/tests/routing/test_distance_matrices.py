# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from cuopt import routing


def test_distance_matrix_records_host_input():
    model = routing.DataModel(2, 1, 1)
    matrix = np.array([[0, 5], [7, 0]], dtype=np.float32)
    model.add_distance_matrix(matrix)
    stored, vehicle_type = model._recorded("add_distance_matrix")[0]
    np.testing.assert_array_equal(stored, matrix)
    assert vehicle_type == 0


def test_duplicate_distance_matrix_is_rejected():
    model = routing.DataModel(2, 1)
    matrix = np.zeros((2, 2), dtype=np.float32)
    model.add_distance_matrix(matrix)
    with pytest.raises(ValueError, match="already been added"):
        model.add_distance_matrix(matrix)


@pytest.mark.parametrize("vehicle_type", [-1, 256, 1.5])
def test_distance_matrix_rejects_invalid_vehicle_type(vehicle_type):
    model = routing.DataModel(2, 1)
    with pytest.raises((ValueError, TypeError)):
        model.add_distance_matrix(
            np.zeros((2, 2), dtype=np.float32), vehicle_type
        )


@pytest.mark.parametrize(
    "matrix",
    [
        np.zeros((2, 3)),
        np.zeros((3, 3)),
        [[0, -1], [1, 0]],
        [[0, np.nan], [1, 0]],
        [[0, -np.inf], [1, 0]],
        [[0, float(np.finfo(np.float32).max) * 2.0], [1, 0]],
    ],
)
def test_distance_matrix_rejects_invalid_values_or_shape(matrix):
    model = routing.DataModel(2, 1)
    with pytest.raises((ValueError, TypeError)):
        model.add_distance_matrix(np.asarray(matrix, dtype=np.float64))


def test_distance_matrix_accepts_unreachable_arc():
    model = routing.DataModel(2, 1)
    model.add_distance_matrix(
        np.array([[0, np.inf], [1, 0]], dtype=np.float32)
    )
    assert len(model._recorded("add_distance_matrix")) == 1


def test_auxiliary_distance_matrix_preserves_cost_and_time():
    model = routing.DataModel(2, 1, 1)
    model.add_cost_matrix(np.array([[0, 3], [5, 0]], dtype=np.float32))
    model.add_distance_matrix(np.array([[0, 100], [200, 0]], dtype=np.float32))
    model.add_transit_time_matrix(
        np.array([[0, 11], [17, 0]], dtype=np.float32)
    )
    model.set_order_locations(np.array([1], dtype=np.int32))
    settings = routing.SolverSettings()
    settings.set_time_limit(0.1)
    solution = routing.Solve(model, settings)
    assert solution.get_status() == 0
    np.testing.assert_allclose(
        solution.get_objective_values()[routing.Objective.COST], 8
    )
    route = solution.get_route().to_pandas()
    np.testing.assert_allclose(route["arrival_stamp"].to_numpy(), [0, 11, 28])
