"""Offline packaging probe; never executes a native experiment or provider health check."""
from __future__ import annotations

import importlib
import importlib.metadata as metadata
import json
import sys
import sysconfig
import tempfile
from pathlib import Path


def check_templates(source: Path, destination: Path) -> dict:
    """Render the existing pinned overlay with inert paths, without loading market data."""
    import yaml
    from jinja2 import Environment, StrictUndefined

    from src.research.native_runtime.overlay import prepare_overlay

    scenario = dict(
        provider_uri='/nonexistent/offline-smoke-data', universe=['offline_one', 'offline_two'],
        benchmark='offline_one', region='us', calendar='XNYS', label_horizon=1,
        equity_cost_bps=10, train=['2019-01-01', '2019-12-27'],
        valid=['2020-01-02', '2020-12-28'], test=['2021-01-04', '2021-12-31'],
    )
    manifest = prepare_overlay(source, destination, scenario)
    context = {f'{segment}_{position}': scenario[segment][i]
               for segment in ('train', 'valid', 'test')
               for i, position in enumerate(('start', 'end'))}
    context.update(feature_expressions=['$close'], feature_names=['CLOSE'], model_code='',
                   n_epochs=1, lr=.01, early_stop=1, batch_size=8, weight_decay=0,
                   num_timesteps=None, num_features=1, step_len=None)
    rendered = []
    for path in sorted(destination.glob('scenarios/qlib/experiment/*template/*.yaml')):
        content = Environment(undefined=StrictUndefined).from_string(path.read_text()).render(**context)
        result = yaml.safe_load(content)
        assert result['qlib_init'] == dict(provider_uri=scenario['provider_uri'], region='us')
        assert result['market'] == scenario['universe']
        assert result['benchmark'] == scenario['benchmark']
        costs = result['port_analysis_config']['backtest']['exchange_kwargs']
        assert costs['limit_threshold'] is None and costs['deal_price'] == 'open'
        assert costs['open_cost'] == costs['close_cost'] == .001 and costs['min_cost'] == 0
        assert 'cn_data' not in content and 'csi300' not in content
        rendered.append(str(path.relative_to(destination)))
    assert len(rendered) == 5
    generated = destination/'scenarios/qlib/experiment/factor_data_template/generate.py'
    compile(generated.read_text(), str(generated), 'exec')  # Compile only; never execute.
    return dict(upstream_revision=manifest['upstream_revision'],
                checked_upstream_files=len(manifest['files']), rendered_templates=rendered)


def main() -> None:
    """Check installed imports and template packaging inside the isolated image."""
    import src

    assert Path(src.__file__).resolve() == Path('/opt/quantagent/src/__init__.py')
    site = Path(sysconfig.get_paths()['purelib']).resolve()
    assert site.is_relative_to(Path('/opt/native/.venv'))
    identities = {}
    for module_name, distribution in (
        ('rdagent', 'rdagent'), ('qlib', 'pyqlib'), ('torch', 'torch'), ('numpy', 'numpy'),
        ('pandas', 'pandas'), ('pyarrow', 'pyarrow'), ('tables', 'tables'),
        ('rdagent.utils.env', 'rdagent'),
    ):
        module = importlib.import_module(module_name)
        locations = ([Path(module.__file__).resolve()] if module.__file__ else
                     [Path(p).resolve() for p in module.__path__])
        assert locations and all(p.is_relative_to(site) for p in locations)
        identities[module_name] = dict(version=metadata.version(distribution),
                                       paths=[str(p) for p in locations])
    direct = json.loads(metadata.distribution('rdagent').read_text('direct_url.json'))
    from src.research.native_runtime.overlay import UPSTREAM_REVISION
    assert direct['vcs_info']['commit_id'] == UPSTREAM_REVISION
    assert metadata.version('pyqlib') == '0.9.7'
    assert metadata.version('torch') == '2.14.1+cpu'
    import rdagent
    with tempfile.TemporaryDirectory(prefix='rd-offline-') as temporary:
        templates = check_templates(Path(next(iter(rdagent.__path__))), Path(temporary)/'rdagent')
    print(json.dumps(dict(scope='offline packaging only', runtime_acceptance='unverified',
                          python=sys.version, imports=identities, templates=templates), indent=2))


if __name__ == '__main__':
    main()
