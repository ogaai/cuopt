# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from fastapi import HTTPException

from cuopt_server.utils.routing import conversion
from cuopt_server.utils.routing.data_definition import (
    CostMatrices,
    DistanceMatrices,
    FleetData,
    SolverSettingsConfig,
    TaskData,
)
from cuopt_server.utils.routing.host_optimization_data_model import (
    HostOptimizationDataModel,
)
from cuopt_server.utils.routing.optimization_data_model import (
    OptimizationDataModel,
)
from cuopt_server.utils.routing.validation_distance_matrix import (
    validate_distance_matrix,
)


@pytest.mark.parametrize(
    "data",
    [
        None,
        {},
        {0: []},
        {0: [[0, 1], [1]]},
        {0: [[0, 1, 2], [1, 0, 3]]},
        {0: [[0, -1], [1, 0]]},
        {0: [[0, np.nan], [1, 0]]},
        {0: [[0, np.inf], [1, 0]]},
        {256: [[0]]},
        {0: [[0, float(np.finfo(np.float32).max) * 2.0], [1, 0]]},
    ],
)
def test_invalid_distance_inputs(data):
    assert not validate_distance_matrix(data)[0]


def test_distance_requires_matching_cost_matrix():
    costs = {0: np.zeros((2, 2))}
    assert not validate_distance_matrix({1: [[0, 1], [1, 0]]}, costs)[0]
    assert not validate_distance_matrix({0: [[0]]}, costs)[0]
    assert validate_distance_matrix({0: [[0, 1e30], [1, 0]]}, costs)[0]


def test_host_conversion_preserves_distance_matrix(monkeypatch):
    import cuopt_server.utils.routing.optimization_data_model as model_module

    def fail(*args, **kwargs):
        raise AssertionError(
            "host distance conversion allocated a cudf object"
        )

    monkeypatch.setattr(model_module.cudf, "DataFrame", fail)
    monkeypatch.setattr(model_module.cudf, "Series", fail)
    optimization = conversion.populate_optimization_data(
        cost_matrix_data=CostMatrices(data={0: [[0, 3], [5, 0]]}),
        distance_matrix_data=DistanceMatrices(data={0: [[0, 100], [200, 0]]}),
        fleet_data=FleetData(vehicle_locations=[[0, 0]]),
        task_data=TaskData(task_locations=[1]),
        solver_config=SolverSettingsConfig(time_limit=0.1),
    )
    prepared, costs, times, _ = conversion.prep_optimization_data(optimization)
    _, model = conversion.create_data_model(
        prepared, cost_matrix=costs, travel_time_matrix=times
    )
    distance, vehicle_type = model._recorded("add_distance_matrix")[0]
    np.testing.assert_array_equal(distance, [[0, 100], [200, 0]])
    assert vehicle_type == 0
    assert distance.dtype == np.float32
    assert optimization.get_distance_matrix() == {0: [[0, 100], [200, 0]]}


@pytest.mark.parametrize(
    "model_class", [HostOptimizationDataModel, OptimizationDataModel]
)
def test_distance_update_behavior(model_class):
    model = model_class()
    assert model.set_cost_matrix({0: [[0, 3], [5, 0]]})[0]
    assert model.set_distance_matrix({0: [[0, 100], [200, 0]]})[0]
    if model_class is HostOptimizationDataModel:
        with pytest.raises(NotImplementedError):
            model.update_distance_matrix({0: [[0, 2], [2, 0]]})
    else:
        assert model.update_distance_matrix({0: [[0, 2], [2, 0]]})[0]
        assert model.get_distance_matrix() == {0: [[0, 2], [2, 0]]}


def test_distance_shape_rechecked_after_cost_preparation():
    optimization = conversion.populate_optimization_data(
        cost_matrix_data=CostMatrices(data={0: [[0, 1], [1, 0]]}),
        distance_matrix_data=DistanceMatrices(data={0: [[0, 5], [5, 0]]}),
        fleet_data=FleetData(vehicle_locations=[[0, 0]]),
        task_data=TaskData(task_locations=[1]),
        solver_config=SolverSettingsConfig(time_limit=0.1),
    )
    optimization.distance_matrix[0] = optimization.distance_matrix[0].iloc[
        :1, :1
    ]
    with pytest.raises(HTTPException, match="shape must match"):
        conversion.prep_optimization_data(optimization)
