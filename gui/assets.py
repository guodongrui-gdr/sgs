import pygame
from pathlib import Path
from typing import Dict, Tuple, Optional

PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "素材"

WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
FPS = 60

CARD_WIDTH = 100
CARD_HEIGHT = 140
CARD_SPACING = 15
CARD_SELECTED_OFFSET = -20

PLAYER_AREA_WIDTH = 200
PLAYER_AREA_HEIGHT = 120

HAND_AREA_HEIGHT = 200

COLORS = {
    "background": (45, 80, 45),
    "white": (255, 255, 255),
    "black": (30, 30, 30),
    "red": (220, 50, 50),
    "heart": (220, 50, 50),
    "diamond": (220, 100, 50),
    "spade": (30, 30, 30),
    "club": (30, 30, 30),
    "card_bg": (250, 250, 245),
    "card_hover": (255, 255, 200),
    "card_selected": (200, 255, 200),
    "button_normal": (80, 120, 80),
    "button_hover": (100, 150, 100),
    "button_text": (255, 255, 255),
    "panel_bg": (40, 60, 40),
    "text_normal": (240, 240, 240),
    "text_highlight": (255, 220, 100),
    "hp_full": (220, 50, 50),
    "hp_empty": (100, 100, 100),
    "hp_lost": (60, 60, 60),
    "wei": (80, 120, 180),
    "shu": (180, 80, 80),
    "wu": (80, 160, 80),
    "qun": (140, 140, 140),
    "lord": (255, 215, 0),
    "loyalist": (100, 180, 100),
    "rebel": (180, 100, 100),
    "spy": (100, 100, 180),
    "basic_card": (250, 250, 245),
    "trick_card": (180, 200, 230),
    "equip_card": (200, 200, 180),
    "delayed_trick": (230, 200, 180),
    "fire": (255, 100, 50),
    "thunder": (150, 100, 255),
}

SUITS = {
    "红桃": "♥",
    "方块": "♦",
    "黑桃": "♠",
    "梅花": "♣",
}

SUIT_COLORS = {
    "红桃": COLORS["heart"],
    "方块": COLORS["diamond"],
    "黑桃": COLORS["spade"],
    "梅花": COLORS["club"],
}

NATION_COLORS = {
    "魏": COLORS["wei"],
    "蜀": COLORS["shu"],
    "吴": COLORS["wu"],
    "群": COLORS["qun"],
}

IDENTITY_COLORS = {
    "主公": COLORS["lord"],
    "忠臣": COLORS["loyalist"],
    "反贼": COLORS["rebel"],
    "内奸": COLORS["spy"],
}

CARD_TYPE_COLORS = {
    "BasicCard": COLORS["basic_card"],
    "JinnangCard": COLORS["trick_card"],
    "CommonJinnangCard": COLORS["trick_card"],
    "YanshiJinnangCard": COLORS["delayed_trick"],
    "WeaponCard": COLORS["equip_card"],
    "ArmourCard": COLORS["equip_card"],
    "AttackHorseCard": COLORS["equip_card"],
    "DefenseHorseCard": COLORS["equip_card"],
    "TreasureCard": COLORS["equip_card"],
    "EquipmentCard": COLORS["equip_card"],
}

POINT_NAMES = {1: "A", 11: "J", 12: "Q", 13: "K"}


class FontManager:
    _instance = None
    _fonts: Dict[str, pygame.font.Font] = {}
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def init(self):
        if self._initialized:
            return
        pygame.font.init()

        font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            None,
        ]

        font_path = None
        for path in font_paths:
            if path and Path(path).exists():
                font_path = path
                break

        sizes = {
            "large": 32,
            "medium": 24,
            "small": 18,
            "card_title": 20,
            "card_point": 16,
            "player_name": 18,
            "hp": 14,
            "button": 22,
        }

        for name, size in sizes.items():
            try:
                if font_path:
                    self._fonts[name] = pygame.font.Font(font_path, size)
                else:
                    self._fonts[name] = pygame.font.SysFont("arial", size)
            except:
                self._fonts[name] = pygame.font.Font(None, size)

        self._initialized = True

    def get(self, name: str) -> pygame.font.Font:
        if not self._initialized:
            self.init()
        return self._fonts.get(name, self._fonts.get("medium"))


class AssetManager:
    _instance = None
    _images: Dict[str, pygame.Surface] = {}
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def init(self):
        if self._initialized:
            return

        if ASSETS_DIR.exists():
            for img_file in ASSETS_DIR.iterdir():
                if img_file.suffix.lower() in [".jpg", ".png", ".bmp"]:
                    try:
                        img = pygame.image.load(str(img_file))
                        self._images[img_file.stem] = img
                    except:
                        pass

        self._initialized = True

    def get(self, name: str) -> Optional[pygame.Surface]:
        if not self._initialized:
            self.init()
        return self._images.get(name)

    def get_background(self) -> Optional[pygame.Surface]:
        bg = self.get("背景")
        if bg:
            return pygame.transform.scale(bg, (WINDOW_WIDTH, WINDOW_HEIGHT))
        return None


fonts = FontManager()
assets = AssetManager()
