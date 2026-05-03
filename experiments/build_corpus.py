"""End-to-end corpus build: download, convert, build dataset.jsonl."""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments import download_data, tptp_fof_to_course, build_dataset


def cmd_download(args) -> int:
    return download_data.main(dry_run=args.dry_run)


def cmd_convert(args) -> int:
    return tptp_fof_to_course.main(args.source, args.check)


def cmd_build(args) -> int:
    return build_dataset.main(args.check)


def cmd_all(args) -> int:
    rc = download_data.main(dry_run=False)
    if rc != 0:
        return rc
    rc = tptp_fof_to_course.main("all", do_check=False)
    if rc != 0:
        return rc
    return build_dataset.main(do_check=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the formula corpus end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sd = sub.add_parser("download", help="fetch TPTP / ILTP, generate hand-crafted formulas")
    sd.add_argument("--dry-run", action="store_true",
                    help="describe actions without touching disk")
    sd.set_defaults(func=cmd_download)

    sc = sub.add_parser("convert", help="convert TPTP-FOF .p files to course syntax")
    sc.add_argument("--source", choices=["pelletier", "iltp", "all"], default="all")
    sc.add_argument("--check", action="store_true",
                    help="round-trip without writing files")
    sc.set_defaults(func=cmd_convert)

    sb = sub.add_parser("build", help="merge category bundles into data/dataset.jsonl")
    sb.add_argument("--check", action="store_true",
                    help="report counts without writing files")
    sb.set_defaults(func=cmd_build)

    sa = sub.add_parser("all", help="run download + convert + build in order")
    sa.set_defaults(func=cmd_all)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
