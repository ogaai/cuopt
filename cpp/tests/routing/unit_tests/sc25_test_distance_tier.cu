// sc25_distance_tiers_test.cu
#include <gtest/gtest.h>

#include <cuopt/routing/data_model_view.hpp>
#include <cuopt/routing/routing_structures.hpp>
#include <cuopt/routing/solve.hpp>
#include <cuopt/routing/solver_settings.hpp>
#include <utilities/copy_helpers.hpp>

#include <raft/core/copy.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

class SC25DistanceTiersTest : public ::testing::Test {
 protected:
  void SetUp() override { handle = std::make_unique<raft::handle_t>(); }
  std::unique_ptr<raft::handle_t> handle;
};

static std::vector<std::string> split_semis(const std::string& s)
{
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string x;
  while (std::getline(ss, x, ';'))
    out.push_back(x);
  return out;
}

static int time_to_minutes(const std::string& s)
{
  if (s.empty()) return 0;
  std::stringstream ss(s);
  std::string H, M, S;
  std::getline(ss, H, ':');
  std::getline(ss, M, ':');
  std::getline(ss, S, ':');
  if (H.empty()) return 0;
  return std::stoi(H) * 60 + (M.empty() ? 0 : std::stoi(M));
}

static float parse_float(std::string s)
{
  auto l = s.find_first_not_of(" \t\r\n\"'");
  auto r = s.find_last_not_of(" \t\r\n\"'");
  if (l == std::string::npos) return 0.f;
  s = s.substr(l, r - l + 1);
  if (s == "nan" || s == "NaN") return 0.f;
  for (char& c : s)
    if (c == ',') c = '.';
  std::string t;
  bool dot = false;
  for (char c : s) {
    if ((c >= '0' && c <= '9') || c == '-' || c == '+') {
      t.push_back(c);
      continue;
    }
    if (c == '.' && !dot) {
      t.push_back(c);
      dot = true;
    }
  }
  if (t.empty() || t == "-" || t == "+") return 0.f;
  return std::strtof(t.c_str(), nullptr);
}

TEST_F(SC25DistanceTiersTest, Turno2_WithDistanceTiers)
{
  std::cout << "🚛 === SC25 TEST CON DISTANCE TIERS ===\n";
  std::cout << "Este test implementa costos escalonados basados en distancia\n\n";

  // 1) Leer nodos turno 2
  std::ifstream fnodes("../../../../datasets/SC25/nodes_df.csv");
  ASSERT_TRUE(fnodes.is_open()) << "No se pudo abrir nodes_df.csv";
  std::string line;
  std::getline(fnodes, line);  // header
  struct Row {
    std::vector<std::string> v;
  };
  std::vector<Row> rows2;
  while (std::getline(fnodes, line)) {
    auto f = split_semis(line);
    if (f.size() > 10 && f[10] == "2") rows2.push_back({std::move(f)});
  }
  fnodes.close();
  ASSERT_FALSE(rows2.empty()) << "No hay órdenes turno 2";

  // 2) Atributos
  std::vector<std::string> order_ids;
  std::vector<int> earliest, latest, service, demand;
  order_ids.reserve(rows2.size());
  earliest.reserve(rows2.size());
  latest.reserve(rows2.size());
  service.reserve(rows2.size());
  demand.reserve(rows2.size());

  for (auto& r : rows2) {
    const auto& v = r.v;
    order_ids.push_back(v[1]);
    demand.push_back(v.size() > 11 ? std::stoi(v[11]) : 0);
    earliest.push_back(v.size() > 12 ? time_to_minutes(v[12]) : 0);
    latest.push_back(v.size() > 13 ? time_to_minutes(v[13]) : 24 * 60);
    service.push_back(v.size() > 14 ? std::stoi(v[14]) : 0);
  }

  for (size_t i = 0; i < earliest.size(); ++i) {
    if (latest[i] < earliest[i]) std::swap(latest[i], earliest[i]);
    if (latest[i] == earliest[i]) latest[i] = earliest[i] + 1;
  }

  int n_orders    = (int)order_ids.size();
  int n_vehicles  = 20;
  int n_locations = n_orders + 1;

  // 3) Matriz de costos y tiempos
  std::ifstream fmat("../../../../datasets/SC25/matrix_df.csv");
  ASSERT_TRUE(fmat.is_open()) << "No se pudo abrir matrix_df.csv";
  std::getline(fmat, line);  // header

  std::map<std::string, int> node2loc;
  node2loc["SC25"]       = 0;
  node2loc["DEPOT"]      = 0;
  node2loc["SC25_DEPOT"] = 0;
  for (int i = 0; i < n_orders; ++i)
    node2loc[order_ids[i]] = i + 1;

  const float BIG = 1e6f;
  std::vector<std::vector<float>> timeM(n_locations, std::vector<float>(n_locations, BIG));
  std::vector<std::vector<float>> costM(n_locations, std::vector<float>(n_locations, BIG));
  for (int i = 0; i < n_locations; ++i) {
    timeM[i][i] = 0.f;
    costM[i][i] = 0.f;
  }

  int lines = 0, used = 0;
  while (std::getline(fmat, line)) {
    ++lines;
    auto f = split_semis(line);
    if (f.size() < 5) continue;
    auto itO = node2loc.find(f[0]);
    auto itD = node2loc.find(f[1]);
    if (itO == node2loc.end() || itD == node2loc.end()) continue;
    const int oi = itO->second, di = itD->second;

    float dist = parse_float(f[3]);
    float mins = parse_float(f[4]);
    if (!std::isfinite(dist) || dist < 0) dist = BIG;
    if (!std::isfinite(mins) || mins < 0) mins = BIG;

    if (oi == di) {
      dist = 0.f;
      mins = 0.f;
    }
    costM[oi][di] = dist;
    timeM[oi][di] = mins;
    ++used;
  }
  fmat.close();

  // 3.a) Filtrar órdenes sin conectividad
  std::vector<int> keep_idx;
  keep_idx.reserve(n_orders);
  for (int i = 0; i < n_orders; ++i) {
    int loc     = i + 1;
    bool dep_to = (timeM[0][loc] < BIG);
    bool to_dep = (timeM[loc][0] < BIG);
    if (dep_to && to_dep) keep_idx.push_back(i);
  }

  if ((int)keep_idx.size() != n_orders) {
    auto compact = [&](auto& vec) {
      using T = typename std::decay<decltype(vec[0])>::type;
      std::vector<T> tmp;
      tmp.reserve(keep_idx.size());
      for (int k : keep_idx)
        tmp.push_back(vec[k]);
      vec.swap(tmp);
    };
    compact(order_ids);
    compact(earliest);
    compact(latest);
    compact(service);
    compact(demand);

    n_orders    = (int)order_ids.size();
    n_locations = n_orders + 1;

    std::map<std::string, int> node2loc2;
    node2loc2["SC25"] = 0;
    for (int i = 0; i < n_orders; ++i)
      node2loc2[order_ids[i]] = i + 1;

    std::vector<std::vector<float>> time2(n_locations, std::vector<float>(n_locations, BIG));
    std::vector<std::vector<float>> cost2(n_locations, std::vector<float>(n_locations, BIG));
    for (int i = 0; i < n_locations; ++i) {
      time2[i][i] = 0.f;
      cost2[i][i] = 0.f;
    }
    for (int i = 0; i < n_orders; ++i) {
      int oldi        = keep_idx[i] + 1;
      time2[0][i + 1] = timeM[0][oldi];
      time2[i + 1][0] = timeM[oldi][0];
      cost2[0][i + 1] = costM[0][oldi];
      cost2[i + 1][0] = costM[oldi][0];
    }
    for (int i = 0; i < n_orders; ++i) {
      int oldi = keep_idx[i] + 1;
      for (int j = 0; j < n_orders; ++j) {
        int oldj            = keep_idx[j] + 1;
        time2[i + 1][j + 1] = timeM[oldi][oldj];
        cost2[i + 1][j + 1] = costM[oldi][oldj];
      }
    }
    node2loc.swap(node2loc2);
    timeM.swap(time2);
    costM.swap(cost2);
  }

  // 4) Data Model
  cuopt::routing::data_model_view_t<int, float> dm(handle.get(), n_locations, n_vehicles, n_orders);

  // 5) Matrices aplanadas
  std::vector<float> h_cost(n_locations * n_locations), h_time(n_locations * n_locations);
  for (int i = 0; i < n_locations; ++i) {
    for (int j = 0; j < n_locations; ++j) {
      float c = costM[i][j];
      float t = timeM[i][j];
      if (!std::isfinite(c)) c = BIG;
      if (!std::isfinite(t)) t = BIG;
      if (i == j) {
        c = 0.f;
        t = 0.f;
      }
      h_cost[i * n_locations + j] = c;
      h_time[i * n_locations + j] = t;
    }
  }

  rmm::device_uvector<float> d_cost(h_cost.size(), handle->get_stream());
  rmm::device_uvector<float> d_time(h_time.size(), handle->get_stream());
  raft::copy(d_cost.data(), h_cost.data(), h_cost.size(), handle->get_stream());
  raft::copy(d_time.data(), h_time.data(), h_time.size(), handle->get_stream());
  dm.add_cost_matrix(d_cost.data(), 0);
  dm.add_transit_time_matrix(d_time.data(), 0);

  // 6) order_locations
  std::vector<int> h_loc(n_orders);
  std::iota(h_loc.begin(), h_loc.end(), 1);
  rmm::device_uvector<int> d_loc(n_orders, handle->get_stream());
  raft::copy(d_loc.data(), h_loc.data(), n_orders, handle->get_stream());
  dm.set_order_locations(d_loc.data());

  // 7) Time Windows
  rmm::device_uvector<int> d_e(n_orders, handle->get_stream());
  rmm::device_uvector<int> d_l(n_orders, handle->get_stream());
  raft::copy(d_e.data(), earliest.data(), n_orders, handle->get_stream());
  raft::copy(d_l.data(), latest.data(), n_orders, handle->get_stream());
  dm.set_order_time_windows(d_e.data(), d_l.data());

  // 8) Service Times
  rmm::device_uvector<int> d_srv(n_orders, handle->get_stream());
  raft::copy(d_srv.data(), service.data(), n_orders, handle->get_stream());
  dm.set_order_service_times(d_srv.data(), -1);

  // 9) Vehicle Time Windows
  std::vector<int> veh_earliest(n_vehicles, 15 * 60);  // 15:00 = 900 minutos
  std::vector<int> veh_latest(n_vehicles, 21 * 60);    // 21:00 = 1260 minutos
  rmm::device_uvector<int> d_veh_earliest(n_vehicles, handle->get_stream());
  rmm::device_uvector<int> d_veh_latest(n_vehicles, handle->get_stream());
  raft::copy(d_veh_earliest.data(), veh_earliest.data(), n_vehicles, handle->get_stream());
  raft::copy(d_veh_latest.data(), veh_latest.data(), n_vehicles, handle->get_stream());
  dm.set_vehicle_time_windows(d_veh_earliest.data(), d_veh_latest.data());

  // 10) Capacities
  std::vector<int> veh_caps(n_vehicles, 120);
  rmm::device_uvector<int> d_caps(n_vehicles, handle->get_stream());
  rmm::device_uvector<int> d_dem(n_orders, handle->get_stream());
  raft::copy(d_caps.data(), veh_caps.data(), n_vehicles, handle->get_stream());
  raft::copy(d_dem.data(), demand.data(), n_orders, handle->get_stream());
  dm.add_capacity_dimension("capacity", d_dem.data(), d_caps.data());

  // ============================================================================
  // 11) DISTANCE TIERS - NUEVA FUNCIONALIDAD
  // ============================================================================

  std::cout << "🎯 Configurando Distance Tiers para vehículos...\n\n";

  // Definir tramos de distancia para diferentes vehículos
  // Escenario:
  // - Vehículos 0-9: Rutas cortas (< 50 km) = coste fijo 30,
  //                  Rutas medias (50-100 km) = 0.2/km,
  //                  Rutas largas (> 100 km) = 0.5/km
  // - Vehículos 10-19: Rutas cortas (< 75 km) = coste fijo 50,
  //                    Rutas largas (> 75 km) = 0.3/km

  // Formato: Para cada vehículo, lista de tramos con:
  // {threshold, fixed_cost, cost_per_unit}

  std::vector<float> all_thresholds;
  std::vector<float> all_fixed_costs;
  std::vector<float> all_costs_per_unit;
  std::vector<int> tier_offsets;  // Offset para cada vehículo

  tier_offsets.push_back(0);  // Offset inicial

  for (int v = 0; v < n_vehicles; ++v) {
    if (v < 10) {
      // Vehículos 0-9: 3 tramos
      all_thresholds.push_back(50.0f);
      all_fixed_costs.push_back(30.0f);
      all_costs_per_unit.push_back(0.0f);

      all_thresholds.push_back(100.0f);
      all_fixed_costs.push_back(0.0f);
      all_costs_per_unit.push_back(0.2f);

      all_thresholds.push_back(1e9f);  // INF
      all_fixed_costs.push_back(0.0f);
      all_costs_per_unit.push_back(0.5f);

      tier_offsets.push_back(tier_offsets.back() + 3);
    } else {
      // Vehículos 10-19: 2 tramos
      all_thresholds.push_back(75.0f);
      all_fixed_costs.push_back(50.0f);
      all_costs_per_unit.push_back(0.0f);

      all_thresholds.push_back(1e9f);  // INF
      all_fixed_costs.push_back(0.0f);
      all_costs_per_unit.push_back(0.3f);

      tier_offsets.push_back(tier_offsets.back() + 2);
    }
  }

  std::cout << "📊 Distance Tiers configurados:\n";
  std::cout << "   - Vehículos 0-9:\n";
  std::cout << "     • < 50 km: Coste fijo 30\n";
  std::cout << "     • 50-100 km: 0.2 por km\n";
  std::cout << "     • > 100 km: 0.5 por km\n";
  std::cout << "   - Vehículos 10-19:\n";
  std::cout << "     • < 75 km: Coste fijo 50\n";
  std::cout << "     • > 75 km: 0.3 por km\n\n";

  // Copiar a device memory
  int total_tiers = (int)all_thresholds.size();
  rmm::device_uvector<float> d_thresholds(total_tiers, handle->get_stream());
  rmm::device_uvector<float> d_fixed_costs(total_tiers, handle->get_stream());
  rmm::device_uvector<float> d_costs_per_unit(total_tiers, handle->get_stream());
  rmm::device_uvector<int> d_tier_offsets(tier_offsets.size(), handle->get_stream());

  raft::copy(d_thresholds.data(), all_thresholds.data(), total_tiers, handle->get_stream());
  raft::copy(d_fixed_costs.data(), all_fixed_costs.data(), total_tiers, handle->get_stream());
  raft::copy(d_costs_per_unit.data(), all_costs_per_unit.data(), total_tiers, handle->get_stream());
  raft::copy(
    d_tier_offsets.data(), tier_offsets.data(), (int)tier_offsets.size(), handle->get_stream());

  // ⚠️ IMPORTANTE: Esta función aún no está implementada en data_model_view_t
  // Una vez implementada, se usaría así:
  /*
  dm.set_vehicle_distance_tiers(
    d_thresholds.data(),
    d_fixed_costs.data(),
    d_costs_per_unit.data(),
    d_tier_offsets.data(),
    total_tiers
  );
  */

  std::cout << "⚠️  NOTA: set_vehicle_distance_tiers() pendiente de implementación en C++\n";
  std::cout << "    Los datos están preparados y se pasarían al solver cuando esté disponible.\n\n";

  // ============================================================================
  // 12) Objectives
  // ============================================================================

  std::vector<cuopt::routing::objective_t> objs = {
    cuopt::routing::objective_t::COST,
    cuopt::routing::objective_t::TRAVEL_TIME,
  };
  std::vector<float> w = {1.f, 1.f};
  rmm::device_uvector<cuopt::routing::objective_t> d_objs(objs.size(), handle->get_stream());
  rmm::device_uvector<float> d_w(w.size(), handle->get_stream());
  raft::copy(d_objs.data(), objs.data(), objs.size(), handle->get_stream());
  raft::copy(d_w.data(), w.data(), w.size(), handle->get_stream());
  dm.set_objective_function(d_objs.data(), d_w.data(), (int)objs.size());

  // 13) Solver Settings
  cuopt::routing::solver_settings_t<int, float> set;
  set.set_time_limit(60.0f);
  set.set_verbose_mode(true);

  std::cout << "🚛 Vehículos configurados con turno de 15:00 a 21:00 (900-1260 min)\n";
  std::cout << "🚀 Ejecutando solver...\n\n";

  auto sol = cuopt::routing::solve(dm, set);

  // ============================================================================
  // 14) ANALYSIS WITH DISTANCE TIERS
  // ============================================================================

  std::cout << "\n===== SOLUTION WITH DISTANCE TIERS =====\n";
  std::cout << "status_string: " << sol.get_status_string() << "\n";
  std::cout << "total_objective: " << sol.get_total_objective() << "\n";

  auto objectives = sol.get_objectives();
  std::cout << "objectives_map_size: " << objectives.size() << "\n";
  for (auto& kv : objectives) {
    std::cout << "  objective_key_int=" << static_cast<int>(kv.first) << " value=" << kv.second
              << "\n";
  }

  rmm::cuda_stream_view stream = rmm::cuda_stream_default;
  auto route_host              = cuopt::host_copy(sol.get_route(), stream);
  auto node_types_host         = cuopt::host_copy(sol.get_node_types(), stream);
  auto truck_id_host           = cuopt::host_copy(sol.get_truck_id(), stream);
  auto order_locs_host         = cuopt::host_copy(sol.get_order_locations(), stream);
  auto arrival_host            = cuopt::host_copy(sol.get_arrival_stamp(), stream);

  auto is_depot = [&](size_t i) -> bool {
    return (i < node_types_host.size()) &&
           (node_types_host[i] == (int)cuopt::routing::node_type_t::DEPOT);
  };

  // Calcular distancias por vehículo y aplicar tiers
  std::map<int, std::vector<size_t>> visits_by_vehicle;
  std::map<int, float> distance_by_vehicle;

  for (size_t i = 0; i < route_host.size(); ++i) {
    int truck = (i < truck_id_host.size()) ? truck_id_host[i] : 0;
    if (!is_depot(i)) { visits_by_vehicle[truck].push_back(i); }
  }

  // Calcular distancia total por vehículo
  std::cout << "\n===== VEHICLE ROUTES & DISTANCE TIERS =====\n";
  for (auto& kv : visits_by_vehicle) {
    int truck    = kv.first;
    auto& visits = kv.second;

    if (visits.empty()) continue;

    float total_distance = 0.0f;
    std::vector<int> route_locs;
    route_locs.push_back(0);  // Start at depot

    for (size_t idx : visits) {
      int loc = (idx < order_locs_host.size()) ? order_locs_host[idx] : 0;
      if (loc > 0) route_locs.push_back(loc);
    }
    route_locs.push_back(0);  // Return to depot

    // Calcular distancia total
    for (size_t i = 0; i + 1 < route_locs.size(); ++i) {
      int from   = route_locs[i];
      int to     = route_locs[i + 1];
      float dist = h_cost[from * n_locations + to];
      if (std::isfinite(dist) && dist < BIG) { total_distance += dist; }
    }

    distance_by_vehicle[truck] = total_distance;

    // Determinar qué tier se aplica
    int tier_start = tier_offsets[truck];
    int tier_end   = tier_offsets[truck + 1];

    float applied_cost = total_distance;  // Default: raw distance
    int applied_tier   = -1;

    for (int t = tier_start; t < tier_end; ++t) {
      if (total_distance < all_thresholds[t]) {
        applied_tier = t - tier_start;
        if (all_fixed_costs[t] > 0) {
          applied_cost = all_fixed_costs[t];
        } else {
          applied_cost = total_distance * all_costs_per_unit[t];
        }
        break;
      }
    }

    std::cout << "\n🚛 Vehículo " << truck << ":\n";
    std::cout << "   Ruta: ";
    for (size_t i = 0; i < route_locs.size(); ++i) {
      if (i > 0) std::cout << " -> ";
      std::cout << route_locs[i];
    }
    std::cout << "\n";
    std::cout << "   Distancia total: " << total_distance << " km\n";
    std::cout << "   Tier aplicado: " << applied_tier << "\n";
    std::cout << "   Coste con tier: " << applied_cost << "\n";
    std::cout << "   Número de órdenes: " << visits.size() << "\n";
  }

  std::cout << "\n✅ Test completado. Distance tiers preparados y análisis realizado.\n";
  std::cout << "   Una vez implementado set_vehicle_distance_tiers(), el solver\n";
  std::cout << "   optimizará automáticamente considerando los costos escalonados.\n";

  SUCCEED() << "Test con distance_tiers ejecutado correctamente.";
}
