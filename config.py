from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent

DATA_DIR = PROJECT_ROOT / "data"
CARDS_CONFIG = DATA_DIR / "cards.json"
COMMANDERS_CONFIG = DATA_DIR / "commanders.json"
ASSETS_DIR = DATA_DIR / "assets"

MIN_PLAYERS = 2
MAX_PLAYERS = 8

IDENTITY_CONFIG: Dict[int, List[str]] = {
	2: ["主公", "反贼"],
	3: ["主公", "忠臣", "反贼"],
	4: ["主公", "忠臣", "反贼", "反贼"],
	5: ["主公", "忠臣", "反贼", "反贼", "内奸"],
	6: ["主公", "忠臣", "反贼", "反贼", "反贼", "内奸"],
	7: ["主公", "忠臣", "忠臣", "反贼", "反贼", "反贼", "内奸"],
	8: ["主公", "忠臣", "忠臣", "反贼", "反贼", "反贼", "反贼", "内奸"],
}

WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
FPS = 60

CARD_WIDTH = 120
CARD_HEIGHT = 168
CARD_SPACING = 10
