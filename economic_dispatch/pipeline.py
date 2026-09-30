"""End-to-end scenario runner: load data, build and solve the dispatch LP."""
from __future__ import annotations

from .config import RunConfig
from . import data_loader, network_loader, model, solve


def solve_scenario(cfg: RunConfig, return_zdata: bool = False):
    """Load data, build and solve the dispatch, attach prices; return the result (and zdata if requested)."""
    h0, h1 = cfg.hour_slice()
    zdata = data_loader.load_zones_from_db(cfg.zones, cfg.zones_db, h0, h1)
    if cfg.capacity_scale:
        zdata = data_loader.apply_capacity_scale(zdata, cfg.capacity_scale)
    net = network_loader.load_networks(cfg.zones, cfg.networks_db)
    build = model.build_model(zdata, net, cfg)
    solve.solve(build)
    startup_cost_eur = 0.0
    if build.uc_gens:
        fixed_profile, startup_cost_eur = model.uc_fixed_profile_and_cost(build)
        build = model.build_model(zdata, net, cfg, fixed_uc_profile=fixed_profile)
        solve.solve(build)
    build.price_e, build.price_h = model.marginal_prices(build)
    build.startup_cost_eur = startup_cost_eur
    return (build, zdata) if return_zdata else build
