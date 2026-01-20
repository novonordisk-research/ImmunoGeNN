from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import run

from .runtime import ensure_data_record, package_working_directory

_BOOL_TO_STR = {True: "true", False: "false"}


def _coerce_cli_value(value) -> str:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__fspath__"):
        return str(Path(value))
    if isinstance(value, bool):
        return _BOOL_TO_STR[value]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return ",".join(str(item) for item in value)
    return str(value)


def predict_immunogenicity(
    fasta_file: str | Path = "data/input.fasta",
    *,
    model_names: Sequence[str] | None = None,
    model_names_str: str | Sequence[str] | None = None,
    human_references_pkl: str | Path | None = None,
    extra_references: str | Path = "",
    skip_plots: bool = False,
    threshold: float = 0.040,
    mode: str | None = None,
    variants_to_generate: int = 30,
    filter_variant_esm_rank: float = 60.0,
    filter_variant_pirs_rank: float = 80.0,
    ranges: str | Sequence[str] = "",
    plot_first_n: int = 1,
    top_n: int = 20,
    esm_model: str = "esm2_t6_8M_UR50D",
    tsv_file: str | Path = "data/_sequences/lyzl4/input.fasta",
    outdir: str | Path = "output",
    verbose: int = 0,
    advanced: str | None = None,
    extra_args: Mapping[str, object] | None = None,
):
    fasta_path = Path(fasta_file).expanduser().resolve()
    outdir_path = Path(outdir).expanduser().resolve()
    outdir_path.mkdir(parents=True, exist_ok=True)

    if model_names is None and model_names_str is not None:
        if isinstance(model_names_str, str):
            model_names = tuple(
                part.strip() for part in model_names_str.split(",") if part.strip()
            )
        else:
            model_names = tuple(model_names_str)

    model_names = tuple(model_names) if model_names is not None else ("DRB1",)

    ensure_data_record()

    cli_args = [
        "--fasta_file",
        str(fasta_path),
        "--outdir",
        str(outdir_path),
        "--model_names_str",
        ",".join(model_names),
        "--skip_plots",
        _coerce_cli_value(skip_plots),
        "--threshold",
        str(threshold),
        "--variants_to_generate",
        str(variants_to_generate),
        "--filter_variant_esm_rank",
        str(filter_variant_esm_rank),
        "--filter_variant_pirs_rank",
        str(filter_variant_pirs_rank),
        "--plot_first_n",
        str(plot_first_n),
        "--top_n",
        str(top_n),
        "--esm_model",
        esm_model,
        "--tsv_file",
        str(Path(tsv_file).expanduser().resolve()),
        "--verbose",
        str(verbose),
    ]

    if human_references_pkl is not None:
        cli_args.extend([
            "--human_references_pkl",
            str(Path(human_references_pkl).expanduser().resolve()),
        ])

    if extra_references:
        cli_args.extend([
            "--extra_references",
            str(Path(extra_references).expanduser().resolve()),
        ])

    if mode is not None:
        cli_args.extend(["--mode", mode])

    if ranges:
        if isinstance(ranges, Iterable) and not isinstance(ranges, (str, bytes)):
            ranges_arg = ",".join(str(item) for item in ranges)
        else:
            ranges_arg = str(ranges)
        cli_args.extend(["--ranges_str", ranges_arg])

    if advanced is not None:
        cli_args.extend(["--advanced", advanced])

    if extra_args:
        for key, value in extra_args.items():
            if value is None:
                continue
            cli_args.extend([f"--{key}", _coerce_cli_value(value)])

    with package_working_directory():
        args = run.parse_args(cli_args)
        run.main(args)

    return outdir_path


def main_cli() -> None:
    args = run.parse_args()

    args.fasta_file = str(Path(args.fasta_file).expanduser().resolve())
    args.outdir = str(Path(args.outdir).expanduser().resolve())

    if args.extra_references:
        args.extra_references = str(Path(args.extra_references).expanduser().resolve())

    if args.tsv_file:
        args.tsv_file = str(Path(args.tsv_file).expanduser().resolve())

    data_dir = ensure_data_record()
    if args.human_references_pkl:
        args.human_references_pkl = str(
            Path(args.human_references_pkl).expanduser().resolve()
        )

    with package_working_directory():
        run.main(args)


__all__ = ["predict_immunogenicity", "main_cli"]
