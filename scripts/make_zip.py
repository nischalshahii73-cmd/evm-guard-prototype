from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    out_path = project_root.parent / "evm_guard_submission_updated.zip"
    if out_path.exists():
        out_path.unlink()

    shutil.make_archive(str(out_path).replace(".zip", ""), "zip", root_dir=str(project_root))
    print(f"Created: {out_path}")


if __name__ == "__main__":
    main()
