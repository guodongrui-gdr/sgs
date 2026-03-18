from typing import Dict, Type, List, Optional
from pathlib import Path
import importlib
import inspect

from .base import Skill


class SkillRegistry:
    _skills: Dict[str, Type[Skill]] = {}
    _instances: Dict[str, Skill] = {}

    @classmethod
    def register(cls, skill_class: Type[Skill]):
        instance = skill_class()
        cls._skills[instance.name] = skill_class
        cls._instances[instance.name] = instance
        return skill_class

    @classmethod
    def get(cls, name: str) -> Optional[Skill]:
        if name in cls._instances:
            return cls._instances[name]
        return None

    @classmethod
    def get_class(cls, name: str) -> Optional[Type[Skill]]:
        return cls._skills.get(name)

    @classmethod
    def has_skill(cls, name: str) -> bool:
        return name in cls._skills

    @classmethod
    def all_skills(cls) -> List[str]:
        return list(cls._skills.keys())

    @classmethod
    def load_from_directory(cls, directory: Path):
        for file_path in directory.glob("*.py"):
            if file_path.name.startswith("_"):
                continue

            module_name = file_path.stem
            module = importlib.import_module(f"skills.{module_name}")

            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, Skill)
                    and obj is not Skill
                    and not inspect.isabstract(obj)
                ):
                    cls.register(obj)

    @classmethod
    def create_instance(cls, name: str, player=None) -> Optional[Skill]:
        skill_class = cls._skills.get(name)
        if skill_class:
            instance = skill_class()
            if player:
                instance.bind_player(player)
            return instance
        return None

    @classmethod
    def create_skills_for_commander(cls, commander_id: str, player=None) -> List[Skill]:
        import json

        config_path = Path(__file__).parent.parent / "data" / "commanders.json"
        with open(config_path, encoding="utf-8") as f:
            configs = json.load(f)

        if commander_id not in configs:
            return []

        config = configs[commander_id]
        skill_names = config.get("skills", [])

        skills = []
        for name in skill_names:
            skill = cls.create_instance(name, player)
            if skill:
                skills.append(skill)

        return skills


def skill_decorator(cls):
    SkillRegistry.register(cls)
    return cls
