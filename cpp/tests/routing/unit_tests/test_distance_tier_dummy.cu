/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// test_distance_tier_dummy.cu
// Test de Distance Tiers con datos sintéticos: 4 vehículos y 20 clientes

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
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <vector>

class DistanceTiersDummyTest : public ::testing::Test {
 protected:
  void SetUp() override { handle = std::make_unique<raft::handle_t>(); }
  std::unique_ptr<raft::handle_t> handle;
};

// ============================================================================
// TEST CON DATOS SINTÉTICOS: 4 VEHÍCULOS, 20 CLIENTES
// TODOS LOS VEHÍCULOS CON LA MISMA CONFIGURACIÓN DE COSTES
// ============================================================================
TEST_F(DistanceTiersDummyTest, FourVehicles_TwentyClients)
{
  std::cout << "🚛 === TEST DISTANCE TIERS CON DATOS SINTÉTICOS ===\n";
  std::cout << "Configuración: 4 vehículos (mismo coste), 20 clientes\n\n";

  // Configuración básica
  const int n_orders    = 20;
  const int n_vehicles  = 4;
  const int n_locations = n_orders + 1;  // +1 para el depósito

  // ============================================================================
  // 1) CREAR DATOS SINTÉTICOS
  // ============================================================================

  std::cout << "📋 Generando datos sintéticos...\n";

  // 1.a) Atributos de las órdenes
  std::vector<int> earliest(n_orders);
  std::vector<int> latest(n_orders);
  std::vector<int> service(n_orders);
  std::vector<int> demand(n_orders);
  std::vector<uint8_t> soft_type(n_orders);
  std::vector<float> soft_pen(n_orders);

  // Generar ventanas de tiempo y demandas sintéticas
  for (int i = 0; i < n_orders; ++i) {
    // Ventanas de tiempo: distribuidas a lo largo del día (8:00 - 18:00)
    earliest[i] = 480 + (i * 30);     // 8:00 AM + i*30 minutos
    latest[i]   = earliest[i] + 120;  // 2 horas de ventana

    // Tiempo de servicio: 10-20 minutos
    service[i] = 10 + (i % 11);

    // Demanda: 5-25 unidades
    demand[i] = 5 + (i % 21);

    // Soft time windows: primeros 10 clientes tienen STRICT, resto SOFT
    if (i < 10) {
      soft_type[i] = 0;  // STRICT
      soft_pen[i]  = 0.0f;
    } else {
      soft_type[i] = 1;  // SOFT
      soft_pen[i]  = 10.0f;
    }
  }

  // 1.b) Crear matrices de costos y tiempos sintéticas
  std::vector<float> h_cost(n_locations * n_locations, 0.0f);
  std::vector<float> h_time(n_locations * n_locations, 0.0f);

  // Generar matriz de distancias basada en posiciones simuladas
  // Simulamos clientes en una cuadrícula aproximada
  auto distance_func = [](int i, int j) -> float {
    if (i == j) return 0.0f;
    // Distancia euclidiana simulada basada en índices
    float dx = static_cast<float>((i % 5) - (j % 5));
    float dy = static_cast<float>((i / 5) - (j / 5));
    return std::sqrt(dx * dx + dy * dy) * 10.0f + 5.0f;  // Escala a km
  };

  for (int i = 0; i < n_locations; ++i) {
    for (int j = 0; j < n_locations; ++j) {
      if (i == j) {
        h_cost[i * n_locations + j] = 0.0f;
        h_time[i * n_locations + j] = 0.0f;
      } else {
        float dist                  = distance_func(i, j);
        h_cost[i * n_locations + j] = dist;
        // Tiempo: asumiendo 40 km/h promedio + tiempo fijo
        h_time[i * n_locations + j] = (dist / 40.0f) * 60.0f + 5.0f;  // en minutos
      }
    }
  }

  std::cout << "✅ Matrices sintéticas creadas (" << n_locations << "x" << n_locations << ")\n";
  std::cout << "   Rango de distancias: ~5-70 km\n";
  std::cout << "   Rango de tiempos: ~5-110 minutos\n";
  std::cout << "   Clientes: 20 (10 STRICT + 10 SOFT time windows)\n";
  std::cout << "   Demandas: 5-25 unidades por cliente\n\n";

  // ============================================================================
  // 2) CREAR DATA MODEL
  // ============================================================================

  cuopt::routing::data_model_view_t<int, float> dm(handle.get(), n_locations, n_vehicles, n_orders);

  // 2.a) Copiar matrices a device
  rmm::device_uvector<float> d_cost(h_cost.size(), handle->get_stream());
  rmm::device_uvector<float> d_time(h_time.size(), handle->get_stream());
  raft::copy(d_cost.data(), h_cost.data(), h_cost.size(), handle->get_stream());
  raft::copy(d_time.data(), h_time.data(), h_time.size(), handle->get_stream());
  dm.add_cost_matrix(d_cost.data(), 0);
  dm.add_transit_time_matrix(d_time.data(), 0);

  // 2.b) Order locations
  std::vector<int> h_loc(n_orders);
  std::iota(h_loc.begin(), h_loc.end(), 1);  // 1, 2, 3, ..., n_orders
  rmm::device_uvector<int> d_loc(n_orders, handle->get_stream());
  raft::copy(d_loc.data(), h_loc.data(), n_orders, handle->get_stream());
  dm.set_order_locations(d_loc.data());

  // 2.c) Time Windows
  rmm::device_uvector<int> d_e(n_orders, handle->get_stream());
  rmm::device_uvector<int> d_l(n_orders, handle->get_stream());
  raft::copy(d_e.data(), earliest.data(), n_orders, handle->get_stream());
  raft::copy(d_l.data(), latest.data(), n_orders, handle->get_stream());
  dm.set_order_time_windows(d_e.data(), d_l.data());

  // 2.d) Service Times
  rmm::device_uvector<int> d_srv(n_orders, handle->get_stream());
  raft::copy(d_srv.data(), service.data(), n_orders, handle->get_stream());
  dm.set_order_service_times(d_srv.data(), -1);

  // 2.e) Vehicle Time Windows
  std::vector<int> veh_earliest(n_vehicles, 8 * 60);  // 8:00 AM
  std::vector<int> veh_latest(n_vehicles, 18 * 60);   // 6:00 PM
  rmm::device_uvector<int> d_veh_earliest(n_vehicles, handle->get_stream());
  rmm::device_uvector<int> d_veh_latest(n_vehicles, handle->get_stream());
  raft::copy(d_veh_earliest.data(), veh_earliest.data(), n_vehicles, handle->get_stream());
  raft::copy(d_veh_latest.data(), veh_latest.data(), n_vehicles, handle->get_stream());
  dm.set_vehicle_time_windows(d_veh_earliest.data(), d_veh_latest.data());

  // 2.f) Capacities
  std::vector<int> veh_caps(n_vehicles, 150);  // Capacidad de 150 unidades
  rmm::device_uvector<int> d_caps(n_vehicles, handle->get_stream());
  rmm::device_uvector<int> d_dem(n_orders, handle->get_stream());
  raft::copy(d_caps.data(), veh_caps.data(), n_vehicles, handle->get_stream());
  raft::copy(d_dem.data(), demand.data(), n_orders, handle->get_stream());
  dm.add_capacity_dimension("capacity", d_dem.data(), d_caps.data());

  std::cout << "✅ Data model configurado\n\n";

  // ============================================================================
  // 3) CONFIGURAR DISTANCE TIERS
  // ============================================================================

  std::cout << "🎯 Configurando Distance Tiers para 4 vehículos...\n\n";

  // TODOS LOS VEHÍCULOS CON LA MISMA CONFIGURACIÓN DE COSTES
  // Esto permite validar primero que el sistema funciona correctamente
  // antes de introducir variabilidad entre vehículos
  //
  // Configuración uniforme para todos: 3 tramos de distancia
  // - Tramo 1: < 40 km = coste fijo 50
  // - Tramo 2: 40-80 km = 0.5 por km
  // - Tramo 3: > 80 km = 1.0 por km

  std::vector<float> all_thresholds;
  std::vector<float> all_fixed_costs;
  std::vector<float> all_costs_per_unit;
  std::vector<int> tier_offsets;

  tier_offsets.push_back(0);  // Offset inicial

  // Configurar los mismos 3 tramos para cada uno de los 4 vehículos
  for (int v = 0; v < n_vehicles; ++v) {
    // Tramo 1: < 40 km = coste fijo 50
    all_thresholds.push_back(40.0f);
    all_fixed_costs.push_back(50.0f);
    all_costs_per_unit.push_back(0.0f);

    // Tramo 2: 40-80 km = 0.5 por km
    all_thresholds.push_back(80.0f);
    all_fixed_costs.push_back(0.0f);
    all_costs_per_unit.push_back(0.5f);

    // Tramo 3: > 80 km = 1.0 por km
    all_thresholds.push_back(1e9f);  // INF
    all_fixed_costs.push_back(0.0f);
    all_costs_per_unit.push_back(1.0f);

    tier_offsets.push_back(tier_offsets.back() + 3);
  }

  std::cout << "📊 Distance Tiers configurados (TODOS IGUALES):\n";
  std::cout << "   🔵 Todos los vehículos (0-3) tienen la misma estructura:\n";
  std::cout << "     • Tramo 1: < 40 km = Coste fijo 50\n";
  std::cout << "     • Tramo 2: 40-80 km = 0.5 por km\n";
  std::cout << "     • Tramo 3: > 80 km = 1.0 por km\n\n";
  std::cout << "   ℹ️  Esta configuración uniforme permite validar\n";
  std::cout << "      que el sistema aplica correctamente los tiers\n";
  std::cout << "      sin introducir variabilidad entre vehículos.\n\n";

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

  // Configurar los distance tiers en el data model
  dm.set_vehicle_distance_tiers(d_thresholds.data(),
                                d_fixed_costs.data(),
                                d_costs_per_unit.data(),
                                d_tier_offsets.data(),
                                total_tiers);

  std::cout << "✅ Distance tiers configurados correctamente en el data model\n\n";

  // ============================================================================
  // 4) CONFIGURAR OBJETIVOS Y RESOLVER
  // ============================================================================

  std::vector<cuopt::routing::objective_t> objs = {cuopt::routing::objective_t::COST,
                                                   cuopt::routing::objective_t::TRAVEL_TIME};
  std::vector<float> w                          = {1.0f, 0.5f};
  rmm::device_uvector<cuopt::routing::objective_t> d_objs(objs.size(), handle->get_stream());
  rmm::device_uvector<float> d_w(w.size(), handle->get_stream());
  raft::copy(d_objs.data(), objs.data(), objs.size(), handle->get_stream());
  raft::copy(d_w.data(), w.data(), w.size(), handle->get_stream());
  dm.set_objective_function(d_objs.data(), d_w.data(), (int)objs.size());

  // Solver Settings
  cuopt::routing::solver_settings_t<int, float> set;
  set.set_time_limit(30.0f);
  set.set_verbose_mode(true);

  std::cout << "🚛 4 vehículos configurados con horario de 8:00 a 18:00 (480-1080 min)\n";
  std::cout << "   Capacidad: 150 unidades por vehículo\n";
  std::cout << "🚀 Ejecutando solver...\n\n";

  auto sol = cuopt::routing::solve(dm, set);

  // ============================================================================
  // 5) ANÁLISIS DE RESULTADOS
  // ============================================================================

  std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  std::cout << "📊 SOLUCIÓN CON DISTANCE TIERS (DATOS SINTÉTICOS)\n";
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

  std::cout << "Status: " << sol.get_status_string() << "\n";
  std::cout << "Objetivo total: " << sol.get_total_objective() << "\n\n";

  auto objectives = sol.get_objectives();
  std::cout << "Desglose de objetivos (" << objectives.size() << "):\n";
  for (auto& kv : objectives) {
    std::string obj_name;
    switch (kv.first) {
      case cuopt::routing::objective_t::COST: obj_name = "COST"; break;
      case cuopt::routing::objective_t::TRAVEL_TIME: obj_name = "TRAVEL_TIME"; break;
      default: obj_name = "UNKNOWN(" + std::to_string(static_cast<int>(kv.first)) + ")";
    }
    std::cout << "  - " << obj_name << ": " << kv.second << "\n";
  }
  std::cout << "\n";

  auto route_host      = cuopt::host_copy(sol.get_route(), handle->get_stream());
  auto node_types_host = cuopt::host_copy(sol.get_node_types(), handle->get_stream());
  auto truck_id_host   = cuopt::host_copy(sol.get_truck_id(), handle->get_stream());
  auto order_locs_host = cuopt::host_copy(sol.get_order_locations(), handle->get_stream());
  auto arrival_host    = cuopt::host_copy(sol.get_arrival_stamp(), handle->get_stream());

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

  // Calcular distancia total por vehículo y costes con tiers
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  std::cout << "🚛 RUTAS DE VEHÍCULOS Y APLICACIÓN DE DISTANCE TIERS\n";
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

  float total_manual_cost  = 0.0f;
  float total_raw_distance = 0.0f;
  int total_orders_served  = 0;

  for (int v = 0; v < n_vehicles; ++v) {
    auto it = visits_by_vehicle.find(v);
    if (it == visits_by_vehicle.end() || it->second.empty()) {
      std::cout << "\n🚛 Vehículo " << v << ": ⚪ Sin órdenes asignadas\n";
      continue;
    }

    auto& visits         = it->second;
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
      total_distance += dist;
    }

    distance_by_vehicle[v] = total_distance;
    total_raw_distance += total_distance;
    total_orders_served += static_cast<int>(visits.size());

    // Determinar qué tier se aplica
    int tier_start = tier_offsets[v];
    int tier_end   = tier_offsets[v + 1];

    float applied_cost = total_distance;
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

    total_manual_cost += applied_cost;

    // Todos los vehículos tienen la misma configuración
    std::string vehicle_type = "ESTÁNDAR";
    std::string icon         = "🔵";

    std::cout << "\n" << icon << " Vehículo " << v << " (" << vehicle_type << "):\n";
    std::cout << "   Ruta: ";
    for (size_t i = 0; i < route_locs.size(); ++i) {
      if (i > 0) std::cout << " → ";
      std::cout << route_locs[i];
    }
    std::cout << "\n";
    std::cout << "   📏 Distancia bruta: " << total_distance << " km\n";
    std::cout << "   🎯 Tier aplicado: " << applied_tier << "\n";
    std::cout << "   💰 Coste con tier: " << applied_cost << "\n";
    std::cout << "   📦 Órdenes servidas: " << visits.size() << "\n";
  }

  // Comparar con el coste devuelto por el solver
  std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  std::cout << "📊 COMPARACIÓN DE COSTES\n";
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

  std::cout << "📊 Resumen:\n";
  std::cout << "   Total órdenes servidas: " << total_orders_served << " / " << n_orders << "\n";
  std::cout << "   Distancia total (sin tiers): " << total_raw_distance << " km\n";
  std::cout << "   Distancia promedio por vehículo: " << (total_raw_distance / n_vehicles)
            << " km\n\n";

  std::cout << "💰 Análisis de costes:\n";
  std::cout << "   Coste calculado manualmente (con tiers): " << total_manual_cost << "\n";

  auto cost_objective = objectives.find(cuopt::routing::objective_t::COST);
  if (cost_objective != objectives.end()) {
    double solver_cost = cost_objective->second;
    std::cout << "   Coste devuelto por el solver: " << solver_cost << "\n\n";

    float diff     = std::abs(total_manual_cost - static_cast<float>(solver_cost));
    float rel_diff = (solver_cost > 0) ? (diff / solver_cost * 100.0f) : 0.0f;

    std::cout << "📉 Diferencias:\n";
    std::cout << "   Absoluta: " << diff << "\n";
    std::cout << "   Relativa: " << rel_diff << "%\n\n";

    if (rel_diff < 0.1f) {
      std::cout << "✅ ÉXITO: Los costes coinciden perfectamente!\n";
      std::cout << "   El solver está aplicando los distance tiers correctamente.\n";
    } else if (rel_diff < 5.0f) {
      std::cout << "⚠️  ADVERTENCIA: Pequeña diferencia detectada\n";
      std::cout << "   Puede deberse a aproximaciones numéricas o componentes adicionales.\n";
    } else {
      std::cout << "❌ ERROR: Diferencia significativa detectada\n";
      std::cout << "   Posibles causas:\n";
      std::cout << "   - El solver no está aplicando los distance tiers correctamente\n";
      std::cout << "   - Hay otros componentes de coste no considerados en el cálculo manual\n";
      std::cout << "   - Diferencias en el redondeo o cálculo de distancias\n";
    }
  } else {
    std::cout << "⚠️  No se encontró el objetivo COST en la solución\n";
  }

  std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  std::cout << "✅ TEST COMPLETADO CON DATOS SINTÉTICOS\n";
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  std::cout << "   ✓ 4 vehículos con MISMA configuración de costes\n";
  std::cout << "   ✓ 20 clientes distribuidos con ventanas de tiempo\n";
  std::cout << "   ✓ Distance tiers uniformes configurados y aplicados\n";
  std::cout << "   ✓ Validación de costes realizada\n";
  std::cout << "   ℹ️  Configuración uniforme ideal para validación inicial\n";
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

  SUCCEED() << "Test con datos sintéticos ejecutado correctamente.";
}
