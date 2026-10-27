"""Data/quant entry points usable before canonical CLI integration.

All inputs are supplied artifacts; there are no runtime sample-data generators.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.data.evidence.store import EvidenceStore
from src.domain.common import EvidenceRecord, Instrument, RunContext
from src.domain.market import FeatureSpec, MarketSnapshot


def read_json(path):
    return json.loads(Path(path).read_text())


def execute(args) -> dict:
    store = EvidenceStore(args.store)
    if args.command == "import-archive":
        manifest_path = Path(args.manifest).resolve()
        root = manifest_path.parent

        def local_file(name):
            path = (root / name).resolve()
            if not path.is_relative_to(root):
                raise ValueError("archive file must reside under its manifest directory")
            return path

        records = []
        for item in read_json(manifest_path)["records"]:
            record = EvidenceRecord.model_validate(item["record"])
            envelope = store.put(
                record,
                local_file(item["raw_file"]).read_bytes(),
                kind=item["kind"],
                logical_key=item["logical_key"],
                normalized=item.get("normalized"),
                document=local_file(item["document_file"]).read_text()
                if item.get("document_file")
                else None,
                published_at=item.get("published_at"),
                revised_at=item.get("revised_at"),
                supersedes_id=item.get("supersedes_id"),
            )
            records.append(envelope.record.evidence_id)
        return {
            "imported": records,
            "provenance_status": "manifest-asserted; provider acceptance separate",
        }
    if args.command == "query":
        return store.query(
            as_of=datetime.fromisoformat(args.as_of),
            kind=args.kind,
            instrument_id=args.instrument_id,
        ).model_dump(mode="json")
    if args.command == "document":
        passages = store.retrieve(
            args.evidence_id,
            as_of=datetime.fromisoformat(args.as_of),
            query=args.query,
            max_chars=args.max_chars,
        )
        return {"passages": [p.model_dump(mode="json") for p in passages]}
    if args.command == "snapshot":
        from src.data.evidence.service import EvidenceService

        context = RunContext.model_validate(read_json(args.context))
        instruments = tuple(Instrument.model_validate(item) for item in read_json(args.instruments))
        snapshot = EvidenceService(store).market_snapshot(
            instruments,
            context,
            start=datetime.fromisoformat(args.start),
            market_type=args.market_type,
        )
        key = store.save_snapshot(snapshot.model_dump(mode="json"))
        return {
            "snapshot_id": key,
            "bars": len(snapshot.bars),
            "gaps": [g.model_dump(mode="json") for g in snapshot.gaps],
        }
    if args.command == "train":
        from src.core.ensemble import create_model
        from src.data.features.dataset import feature_dataset

        snapshot = MarketSnapshot.model_validate(store.snapshot(args.snapshot))
        specs = tuple(FeatureSpec.model_validate(item) for item in read_json(args.features))
        context = RunContext.model_validate(read_json(args.context))
        dataset = feature_dataset(snapshot, specs, horizon=args.horizon).eligible_training(
            context.as_of
        )
        model = create_model(args.model)
        model.train(
            dataset.features,
            dataset.labels,
            context=context,
            feature_available_at=dataset.feature_available_at,
            label_available_at=dataset.label_available_at,
            feature_spec_hash=dataset.feature_spec_hash,
            data_snapshot_hash=dataset.snapshot_hash,
        )
        model.save(args.output)
        return {
            "model": args.model,
            "artifact": str(Path(args.output).resolve()),
            "training_rows": len(dataset.features),
            "snapshot_hash": dataset.snapshot_hash,
            "feature_spec_hash": dataset.feature_spec_hash,
        }
    if args.command == "predict":
        from src.core.ensemble import create_model
        from src.data.features.dataset import feature_dataset

        snapshot = MarketSnapshot.model_validate(store.snapshot(args.snapshot))
        specs = tuple(FeatureSpec.model_validate(item) for item in read_json(args.features))
        context = RunContext.model_validate(read_json(args.context))
        dataset = feature_dataset(snapshot, specs)
        model = create_model(args.model).load(args.artifact)
        dates = dataset.features.index.get_level_values("datetime")
        selected = dates == dates.max()
        predictions = model.predict(
            dataset.features.loc[selected],
            context=context,
            feature_available_at=dataset.feature_available_at.loc[selected],
        )
        return {
            "predictions": [
                {"datetime": index[0].isoformat(), "instrument_id": index[1], "score": float(value)}
                for index, value in predictions.items()
            ],
            "context": context.model_dump(mode="json"),
        }
    if args.command == "qlib-export":
        from src.data.qlib_us import export_bundle

        result = export_bundle(
            MarketSnapshot.model_validate(store.snapshot(args.snapshot)), args.output
        )
        return {
            "bundle_hash": result["bundle_hash"],
            "instrument_map": result["instrument_map"],
            "provider_uri": args.output,
        }
    if args.command == "qlib-alpha158":
        from src.data.qlib_us import USQlibScenario, alpha158_dataset

        scenario = USQlibScenario.model_validate(read_json(args.scenario))
        dataset = alpha158_dataset(scenario)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        counts = {}
        for segment in ("train", "valid", "test"):
            frame = dataset.prepare(
                segment,
                col_set=["feature", "label"],
                data_key="learn" if segment == "train" else "infer",
            )
            frame.to_parquet(output / f"{segment}.parquet")
            counts[segment] = len(frame)
        return {"rows": counts, "features": 158, "scenario_hash": scenario.content_hash()}
    if args.command == "fmp-transcript":
        from src.data.adapters.fundamentals import FMPAdapter

        context = RunContext.model_validate(read_json(args.context))
        instrument = Instrument.model_validate(read_json(args.instrument))
        adapter = FMPAdapter(store, os.environ.get("AIQUANT_FMP_API_KEY"))
        try:
            result = adapter.fetch(
                "transcript",
                instrument,
                context,
                ingested_at=datetime.now(timezone.utc),
                year=args.year,
                quarter=args.quarter,
            )
        finally:
            adapter.http.close()
        return {
            "status": result.status,
            "evidence_ids": [e.record.evidence_id for e in result.evidence],
            "detail": result.detail,
        }
    raise ValueError("unknown command")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, help="owned SQLite evidence database path")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("import-archive")
    p.add_argument("--manifest", required=True)
    p = commands.add_parser("query")
    p.add_argument("--as-of", required=True)
    p.add_argument("--kind")
    p.add_argument("--instrument-id")
    p = commands.add_parser("document")
    p.add_argument("--as-of", required=True)
    p.add_argument("--evidence-id", action="append", required=True)
    p.add_argument("--query", default="")
    p.add_argument("--max-chars", type=int, default=6000)
    p = commands.add_parser("snapshot")
    p.add_argument("--context", required=True)
    p.add_argument("--instruments", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--market-type", choices=["spot", "equity", "perpetual"])
    for command in ("train", "predict"):
        p = commands.add_parser(command)
        p.add_argument("--snapshot", required=True)
        p.add_argument("--context", required=True)
        p.add_argument("--features", required=True)
        p.add_argument(
            "--model",
            default="LightGBM",
            choices=["LightGBM", "CatBoost", "XGBoost", "Linear", "Ensemble"],
        )
        if command == "train":
            p.add_argument("--output", required=True)
            p.add_argument("--horizon", type=int, default=1)
        else:
            p.add_argument("--artifact", required=True)
    p = commands.add_parser("qlib-export")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("qlib-alpha158")
    p.add_argument("--scenario", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("fmp-transcript")
    p.add_argument("--instrument", required=True)
    p.add_argument("--context", required=True)
    p.add_argument("--year", required=True, type=int)
    p.add_argument("--quarter", required=True, type=int, choices=[1, 2, 3, 4])
    args = parser.parse_args(argv)
    try:
        print(json.dumps(execute(args), indent=2, allow_nan=False))
    except Exception as exc:
        # Provider adapters sanitize errors before they cross this boundary.
        print(
            json.dumps({"status": "failed", "error_type": type(exc).__name__, "detail": str(exc)})
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
