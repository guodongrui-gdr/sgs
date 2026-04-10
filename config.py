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

# 训练时关闭详细日志输出
VERBOSE = False  # 设置为True可开启技能输出调试


def verbose_print(*args, **kwargs):
    """条件print，仅在VERBOSE=True时输出"""
    if VERBOSE:
        print(*args, **kwargs)


WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
FPS = 60

CARD_WIDTH = 120
CARD_HEIGHT = 168
CARD_SPACING = 10
