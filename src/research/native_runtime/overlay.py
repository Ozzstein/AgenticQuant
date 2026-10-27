"""Version-checked adaptation of native financial experiments to explicit US data."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

UPSTREAM_REVISION = "484776c211e4fbbeef03e0ec00d6bbee7362a4f4"


def prepare_overlay(source: Path, destination: Path, scenario: dict) -> dict:
    """Copy pinned native code, then patch only reviewed, exact input templates."""
    pins = json.loads(Path(__file__).with_name("upstream_files.json").read_text())
    for name, digest in pins.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"native upstream file differs from pinned version: {name}")
    if destination.exists():
        raise ValueError("native overlay output must be fresh")
    if scenario["region"] != "us" or scenario["calendar"] != "XNYS" or not scenario["universe"]:
        raise ValueError("native finance requires explicit US scenario")
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    updates = {}
    for name in pins:
        path = destination/name
        text = path.read_text()
        if name.endswith('.yaml') and '/experiment/' in name and 'template/' in name:
            replacements = {
                'provider_uri: "~/.qlib/qlib_data/cn_data"': 'provider_uri: '+json.dumps(scenario["provider_uri"]),
                'region: cn': 'region: us',
                'market: &market csi300': 'market: &market '+json.dumps(scenario["universe"]),
                'benchmark: &benchmark SH000300': 'benchmark: &benchmark '+json.dumps(scenario["benchmark"]),
                'limit_threshold: 0.095': 'limit_threshold: null',
                'deal_price: close': 'deal_price: open',
                'open_cost: 0.0005': 'open_cost: '+str(scenario["equity_cost_bps"]/10000),
                'close_cost: 0.0015': 'close_cost: '+str(scenario["equity_cost_bps"]/10000),
                'min_cost: 5': 'min_cost: 0',
                'Ref($close, -2)/Ref($close, -1) - 1': f'Ref($close, -{scenario["label_horizon"]})/$close-1',
            }
            for old,new in replacements.items():
                if old not in text:
                    raise ValueError(f"expected native template token missing: {name}: {old}")
                text = text.replace(old,new)
            text = text.replace('GPU: 0', 'GPU: -1').replace('n_jobs: 20', 'n_jobs: 2').replace('num_threads: 20', 'num_threads: 2')
        elif name.endswith('prompts.yaml'):
            text = text.replace('CSI300','explicit archived US universe').replace('China A-share market','US equity market')
        elif name.endswith('generate.py'):
            # Full/debug data both derive from this one versioned US bundle.
            text = '''import json
from pathlib import Path
import qlib
from qlib.data import D
scenario = json.loads(Path('/work/scenario.json').read_text())
qlib.init(provider_uri=scenario['provider_uri'], region='us', kernels=1,
          expression_cache=None, dataset_cache=None)
fields = ['$open', '$close', '$high', '$low', '$volume', '$factor']
data = D.features(scenario['universe'], fields, start_time=scenario['train'][0],
                  end_time=scenario['test'][1], freq='day').sort_index()
# Qlib returns instrument/datetime; factor coder's contract is datetime/instrument.
data = data.reorder_levels(['datetime','instrument']).sort_index()
if data.empty:
    raise ValueError('declared US bundle produced no factor data')
data.to_hdf('daily_pv_all.h5', key='data')
cutoff = scenario['valid'][1]
debug = data.loc[:cutoff]
if debug.empty:
    raise ValueError('declared development slice produced no debug data')
debug.to_hdf('daily_pv_debug.h5', key='data')
'''
        path.write_text(text)
        updates[name] = {"upstream": pins[name], "adapted": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"upstream_revision": UPSTREAM_REVISION, "files": updates,
            "scenario_hash": hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest(),
            "native_performance_authority": "discovery only; independent chronological replay required"}
