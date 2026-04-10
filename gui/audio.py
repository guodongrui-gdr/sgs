import pygame
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "素材"
SOUNDS_DIR = PROJECT_ROOT / "sounds"


class AudioManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.enabled = True
        self.volume = 0.7
        self.sounds: Dict[str, pygame.mixer.Sound] = {}
        self._initialized = True

    def init(self):
        pygame.mixer.init()

        self._generate_sounds()

    def _generate_sounds(self):
        try:
            sample_rate = 44100

            self.sounds["card_use"] = self._create_beep(sample_rate, 800, 100)
            self.sounds["damage"] = self._create_beep(sample_rate, 400, 200)
            self.sounds["heal"] = self._create_beep(sample_rate, 1200, 150)
            self.sounds["draw"] = self._create_beep(sample_rate, 600, 80)
            self.sounds["click"] = self._create_beep(sample_rate, 1000, 50)
            self.sounds["victory"] = self._create_beep(sample_rate, 880, 500)
            self.sounds["defeat"] = self._create_beep(sample_rate, 330, 500)
            self.sounds["equip"] = self._create_beep(sample_rate, 900, 120)

            for sound in self.sounds.values():
                sound.set_volume(self.volume)

        except Exception as e:
            print(f"Sound initialization failed: {e}")

    def _create_beep(
        self, sample_rate: int, frequency: int, duration_ms: int
    ) -> pygame.mixer.Sound:
        import math

        n_samples = int(sample_rate * duration_ms / 1000)
        buf = bytes(
            [
                int(
                    128
                    + 127
                    * math.sin(2 * math.pi * frequency * t / sample_rate)
                    * (1 - t / n_samples) ** 2
                )
                for t in range(n_samples)
            ]
        )

        return pygame.mixer.Sound(buffer=buf)

    def play(self, sound_name: str):
        if not self.enabled:
            return

        if sound_name in self.sounds:
            try:
                self.sounds[sound_name].play()
            except:
                pass

    def set_volume(self, volume: float):
        self.volume = max(0, min(1, volume))
        for sound in self.sounds.values():
            sound.set_volume(self.volume)

    def toggle(self):
        self.enabled = not self.enabled

    def is_enabled(self) -> bool:
        return self.enabled


audio = AudioManager()
