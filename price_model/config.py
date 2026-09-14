"""Configuration for the two commodity price models.

Each commodity maps a **demand** input to a **price** output, learned per bidding zone
from Project 3's own LP economic-dispatch output (``inputs/hourly_balance_{elec,h2}.csv``
-- the 20-zone Central-European CORE region, NT2030 scenario; see
``price_model/extract.py``):

* electricity -- ``hourly_balance_elec.csv``, price = Marginal Price (EUR/MWh)
* hydrogen    -- ``hourly_balance_h2.csv``,   price = Marginal Price (EUR/MWhH2)

``demand`` is the primary input; the remaining ``features`` are supporting context (they
default to each zone's median when a caller supplies only demand). ``target`` is the
price column the model predicts. ``adjacency`` names the JSON file (built by
``build_dataset.py`` from Project 3's own ``networks_2030.parquet`` line topology)
mapping each zone to its directly interconnected neighbours -- used to add per-zone
``neighbor_net_demand_<N>`` / ``neighbor_net_demand_total`` / ``net_demand_system_total``
features (see
``price_model/neighbors.py``). Because neighbour counts vary a lot per zone (median ~3,
up to 9 for a hub like DE00), each zone ends up with its own feature list, stored per
zone in the trained bundle rather than as one shared list.

``net_demand_col``, if set, both (a) is added to the zone's own base ``features`` as its
own net demand, and (b) is the column used for the neighbour/system-total *net* features
-- built *alongside* (not instead of) the raw-demand versions of each, since both are
wanted. "Net demand" means demand net of renewables (electricity's existing
``residual_load = demand - wind - solar``), not raw demand. Hydrogen has no wind/solar
equivalent tied to H2 zones, so it's left unset there -- only raw-demand neighbour/
system-total features exist for hydrogen, and there's no separate "own net demand" for
it either (h2_demand is the only demand quantity available).
"""
from __future__ import annotations

COMMODITIES = {
    "electricity": {
        "unit": "EUR/MWh",
        "demand": "demand",
        "target": "price_eur_mwh",
        # demand + net demand (residual_load = demand - wind - solar) for the zone
        # itself, + weather drivers + calendar/time context. `ens`/`dumped` (energy
        # not served / curtailed) were dropped 2026-08-04: like thermal/hydro/
        # balance/dsr, they're dispatch OUTCOMES -- not known ahead of solving a
        # network LP -- so keeping them in made this model's own CV R^2 look better
        # than a genuinely exogenous-features proxy could ever score (see the
        # exogenous-only-features experiment in conversation history: dropping just
        # the price/dispatch-outcome features already cost ~0.06 mean R^2 on top of
        # this). `residual_load` was briefly dropped 2026-08-04 too (as an ablation:
        # it's a linear combination of demand/wind/solar already in the list, so the
        # test was whether the model could recover the same merit-order signal
        # without being handed the derived column directly) then RESTORED the same
        # day at user request -- the ablation showed real, if modest, cost
        # concentrated in zones without a strong correlated-neighbour price to lean
        # on instead (FR15 -0.077 CV R^2, PL00/RO00/CZ00/HR00 -0.002 to -0.008; every
        # neighbour-price-dominated zone was unaffected). Neighbour/system-total
        # demand features (both raw and net) are added per zone on top of this (see
        # `adjacency`/`net_demand_col`).
        # wind_capacity_mw/pv_capacity_mw (added 2026-09-14): each zone's installed
        # capacity for these 2 technologies, constant across all hours of one
        # capacity-uncertainty scenario (see run_capacity_scenarios.py) -- distinct
        # from `wind`/`solar` above (actual hourly dispatched generation, an existing
        # feature). Only present/varying when inputs/scenarios/ has been generated and
        # pooled by build_dataset.py; a single-scenario dataset still trains fine with
        # these held constant (they just contribute nothing). `electrolyser_capacity_mw`
        # was a 3rd feature here too, briefly, until dropped same-day at user request
        # ("just wind and pv") along with its `ely70` scenario -- see
        # run_capacity_scenarios.py's module docstring.
        "features": ["demand", "residual_load", "wind", "solar", "month", "season", "hour",
                    "wind_capacity_mw", "pv_capacity_mw"],
        "samples": "elec_samples.parquet",
        "adjacency": "elec_adjacency.json",
        "net_demand_col": "residual_load",  # demand - wind - solar, already in the parquet
        "max_price": 500,  # hours where this zone's own price exceeds this are dropped
                            # entirely, from both training and CV scoring
        "model": "electricity_model.joblib",
        "metrics": "electricity_metrics.csv",
    },
    "hydrogen": {
        "unit": "EUR/MWhH2",
        "demand": "h2_demand",
        "target": "h2_price",
        # H2 demand (primary) + calendar/time context. `dumped`/`hns` (curtailed H2 /
        # hydrogen not served) were dropped 2026-08-04 for the same reason as
        # electricity's `ens`/`dumped` above -- dispatch outcomes, not exogenous
        # inputs. `elec_price` (electrolysis feedstock cost) was ALSO dropped
        # 2026-08-04: ablation showed it does essentially nothing for 11/13 zones
        # (already dominated by a correlated neighbour's own H2 price) but costs
        # SI00/HR00 ~0.06 mean CV R^2 each -- the two zones with no strong
        # correlated-neighbour fallback. Dropped anyway, at user request, accepting
        # that SI00/HR00 cost, in favour of a smaller, more defensibly-exogenous
        # feature list (elec_price is itself only exogenous under a price-taker
        # framing, see conversation history). `smr` (Steam Methane Reformer
        # generation) ADDED 2026-08-04, at user request, reversing the earlier
        # blanket exclusion of hydrogen supply-mix features -- unlike
        # electrolyser_gen/storage/balance/h2_net_trade (still excluded), SMR
        # capacity/output is comparatively slow-moving/plannable rather than a
        # pure real-time dispatch outcome, so it's a more defensible exception; the
        # same "is this really exogenous" caveat as elec_price still applies to it
        # though. Neighbour/system-total demand features are added per zone on top
        # of this (see `adjacency` above).
        # wind_capacity_mw/pv_capacity_mw: see electricity's own comment above -- same
        # 2 capacity-scenario features, added here too since wind/solar capacity
        # affects the elec price the electrolyser pays, even though elec_price itself
        # was dropped as a feature -- see "Key findings" in CLAUDE.md.
        # `electrolyser_capacity_mw` was a 3rd feature here too, briefly, until dropped
        # same-day at user request -- see electricity's own comment above.
        "features": ["h2_demand", "smr", "month", "season", "hour",
                    "wind_capacity_mw", "pv_capacity_mw"],
        "samples": "h2_samples.parquet",
        "adjacency": "h2_adjacency.json",
        "net_demand_col": None,  # no renewables column for H2 zones -- falls back to h2_demand
        "model": "hydrogen_model.joblib",
        "metrics": "hydrogen_metrics.csv",
    },
}
