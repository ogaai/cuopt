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
class distance_node_t {
 public:
  //! Cost distance gathered to node
  double distance_forward = 0.0;
  //! Cost distance gathered after node
  double distance_backward = 0.0;
  //! Physical travel distance gathered to node
  double travel_distance_forward = 0.0;
  //! Physical travel distance gathered after node
  double travel_distance_backward = 0.0;

  /*! \brief { Calculate next node forward gathered distance data based on actual node} */
  void HDI calculate_forward(distance_node_t& next,
                             double distance_between,
                             double travel_distance_between) const noexcept
  {
    next.distance_forward = distance_forward + distance_between;
    next.travel_distance_forward = travel_distance_forward + travel_distance_between;
  }

  /*! \brief { Calculate prev node gathered distance backward data based on actual node} */
  void HDI calculate_backward(distance_node_t& prev,
                              double distance_between,
                              double travel_distance_between) const noexcept
  {
    prev.distance_backward = distance_backward + distance_between;
    prev.travel_distance_backward = travel_distance_backward + travel_distance_between;
  }

  HDI double forward_excess(const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    const double objective_cost =
      vehicle_info.compute_distance_cost(travel_distance_forward, distance_forward);
    return max(max(0., travel_distance_forward - vehicle_info.max_distance),
               max(0., objective_cost - vehicle_info.max_cost));
  }

  HDI double backward_excess(const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    const double objective_cost =
      vehicle_info.compute_distance_cost(travel_distance_backward, distance_backward);
    return max(max(0., travel_distance_backward - vehicle_info.max_distance),
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
  static HDI double combine(const distance_node_t& prev,
                            const distance_node_t& next,
                            const VehicleInfo<f_t, is_device>& vehicle_info,
                            f_t distance_between,
                            f_t travel_distance_between) noexcept
  {
    double total_distance_cost =
      prev.distance_forward + next.distance_backward + distance_between;
    double total_travel_distance =
      prev.travel_distance_forward + next.travel_distance_backward + travel_distance_between;
    const double objective_cost =
      vehicle_info.compute_distance_cost(total_travel_distance, total_distance_cost);
    return max(max(0., total_travel_distance - vehicle_info.max_distance),
               max(0., objective_cost - vehicle_info.max_cost));
  }

  HDI bool backward_feasible(const VehicleInfo<f_t>& vehicle_info,
                             const double weight    = 1.,
                             const f_t excess_limit = 0.) const noexcept
  {
    return backward_excess(vehicle_info) * weight <= excess_limit;
  }

  template <bool is_device = true>
  HDI void get_cost([[maybe_unused]] const distance_node_t& prev_node,
                    const VehicleInfo<f_t, is_device>& vehicle_info,
                    const cost_dimension_info_t& dim_info,
                    objective_cost_t& obj_cost,
                    infeasible_cost_t& inf_cost) const noexcept
  {
    double total_distance_cost = ((double)distance_forward + (double)distance_backward);
    double total_travel_distance =
      ((double)travel_distance_forward + (double)travel_distance_backward);

    obj_cost[objective_t::COST] =
      vehicle_info.compute_distance_cost(total_travel_distance, total_distance_cost);
    if (dim_info.has_max_constraint) {
      inf_cost[dim_t::DIST] = max(0., total_travel_distance - vehicle_info.max_distance) +
                              max(0., obj_cost[objective_t::COST] - vehicle_info.max_cost);
    }
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
