/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>

#include <routing/dimensions.cuh>
#include <routing/routing_details.hpp>

#include "../vehicle_info.hpp"

#include <algorithm>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class cost_node_t {
 public:
  //! Cost distance gathered to node
  double cost_forward = 0.0;
  //! Cost distance gathered after node
  double cost_backward = 0.0;
  //! Physical travel distance gathered to node
  double distance_forward = 0.0;
  //! Physical travel distance gathered after node
  double distance_backward = 0.0;

  /*! \brief { Calculate next node forward gathered distance data based on actual node} */
  void HDI calculate_forward(cost_node_t& next,
                             double cost_between,
                             double distance_between) const noexcept
  {
    next.cost_forward     = cost_forward + cost_between;
    next.distance_forward = distance_forward + distance_between;
  }

  /*! \brief { Calculate prev node gathered distance backward data based on actual node} */
  void HDI calculate_backward(cost_node_t& prev,
                              double cost_between,
                              double distance_between) const noexcept
  {
    prev.cost_backward     = cost_backward + cost_between;
    prev.distance_backward = distance_backward + distance_between;
  }

  HDI double forward_excess(const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    const double objective_cost =
      vehicle_info.compute_distance_cost(distance_forward, cost_forward);
    return max(max(0., distance_forward - vehicle_info.max_distance),
               max(0., objective_cost - vehicle_info.max_cost));
  }

  HDI double backward_excess(const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    const double objective_cost =
      vehicle_info.compute_distance_cost(distance_backward, cost_backward);
    return max(max(0., distance_backward - vehicle_info.max_distance),
               max(0., objective_cost - vehicle_info.max_cost));
  }

  HDI bool forward_feasible(const VehicleInfo<f_t>& vehicle_info,
                            const double weight    = 1.,
                            const f_t excess_limit = 0.) const noexcept
  {
    return forward_excess(vehicle_info) * weight <= excess_limit;
  }

  /*! \brief  { Combine information from begining and ending fragments.}
      \return { Distance excess of route represented by nodes prev and next }*/
  template <bool is_device = true>
  static HDI double combine(const cost_node_t& prev,
                            const cost_node_t& next,
                            const VehicleInfo<f_t, is_device>& vehicle_info,
                            f_t cost_between,
                            f_t distance_between) noexcept
  {
    double total_distance_cost = prev.cost_forward + next.cost_backward + cost_between;
    double total_distance      = prev.distance_forward + next.distance_backward + distance_between;
    const double objective_cost =
      vehicle_info.compute_distance_cost(total_distance, total_distance_cost);
    return max(max(0., total_distance - vehicle_info.max_distance),
               max(0., objective_cost - vehicle_info.max_cost));
  }

  HDI bool backward_feasible(const VehicleInfo<f_t>& vehicle_info,
                             const double weight    = 1.,
                             const f_t excess_limit = 0.) const noexcept
  {
    return backward_excess(vehicle_info) * weight <= excess_limit;
  }

  template <bool is_device = true>
  HDI void get_cost([[maybe_unused]] const cost_node_t& prev_node,
                    const VehicleInfo<f_t, is_device>& vehicle_info,
                    const cost_dimension_info_t& dim_info,
                    objective_cost_t& obj_cost,
                    infeasible_cost_t& inf_cost) const noexcept
  {
    double total_distance_cost = ((double)cost_forward + (double)cost_backward);
    double total_distance      = ((double)distance_forward + (double)distance_backward);

    obj_cost[objective_t::COST] =
      vehicle_info.compute_distance_cost(total_distance, total_distance_cost);
    if (dim_info.has_max_constraint) {
      inf_cost[dim_t::COST] = max(0., total_distance - vehicle_info.max_distance) +
                              max(0., obj_cost[objective_t::COST] - vehicle_info.max_cost);
    }
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
