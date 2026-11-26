"""Write the OpenAPI schema for the web app: uv run python -m chess_trainer.api.export

Output: web/src/api/openapi.json (a shared file; say so before changing the contract).
Building the schema needs no data, checkpoint or engine.
"""

import json

from chess_trainer import config
from chess_trainer.api.app import create_app

TARGET = config.REPO_ROOT / "web" / "src" / "api" / "openapi.json"


def main() -> None:
    schema = create_app().openapi()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"{TARGET.relative_to(config.REPO_ROOT)}: {len(schema['paths'])} paths")


if __name__ == "__main__":
    main()
