# Bug修复总结 - 2026-03-23 更新

## 最新修复

### 游戏规则与官方规则不一致问题 ✅ 已修复 (2026-03-23)

根据 RULES.md 官方规则文档，修复了多项与规则不一致的问题：

#### P0 严重问题

##### 1. 白银狮子伤害计算错误 ✅ 已修复

**问题描述**：
- 规则：受到**大于1点**伤害时，将伤害值**改为1点**
- 代码：`actual_damage = max(1, actual_damage - 1)` (错误：减1而非改为1)

**修复方案**：
```python
# engine/response.py:279-285
if (
    target.equipment.get("防具")
    and target.equipment["防具"].name == "白银狮子"
    and actual_damage > 1
):
    actual_damage = 1
```

##### 2. 白银狮子失去回血效果缺失 ✅ 已修复

**问题描述**：
- 规则：每当失去装备区里的【白银狮子】后，回复1点体力
- 代码：未实现此效果

**修复方案**：
```python
# engine/game_engine.py:520-529
def _equip_armour(self, player: "Player", card: "Card"):
    old_armour = player.equipment.get("防具")
    if old_armour:
        if old_armour.name == "白银狮子":
            if player.current_hp < player.max_hp:
                player.current_hp += 1
                self.log(f"{player.commander_name} 失去白银狮子，回复1点体力")
        self.discard_pile.append(old_armour)
    player.equipment["防具"] = card
    self.tmp_cards.remove(card)
```

##### 3. 判定返回值类型错误 ✅ 已修复

**问题描述**：
- `JudgeCard.check()` 方法返回 bool，但声明返回 JudgeResult

**修复方案**：
```python
# engine/judge.py:22-28
def check(self, card: "Card") -> JudgeResult:
    result = self.success_condition(card)
    if result:
        return JudgeResult.SUCCESS
    return JudgeResult.FAIL
```

#### P1 高优先级问题

##### 4. 藤甲对雷电伤害错误+1 ✅ 已修复

**问题描述**：
- 规则：受到**火焰伤害**时伤害+1
- 代码：`is_elemental` 包括火焰和雷电，导致雷电伤害也+1

**修复方案**：
```python
# engine/response.py:271-278
is_fire = getattr(card, "is_fire", False) if card else False
if (
    target.equipment.get("防具")
    and target.equipment["防具"].name == "藤甲"
    and is_fire  # 只检查火焰，不是所有属性伤害
):
    actual_damage += 1
```

##### 5. 铁索连环传递伤害丢失属性 ✅ 已修复

**问题描述**：
- 铁索传递属性伤害时，`is_elemental=False`，导致藤甲+1效果不触发

**修复方案**：
```python
# engine/game_engine.py:552-649
def deal_damage(
    self,
    source: "Player",
    target: "Player",
    card: Optional["Card"],
    damage: int,
    is_elemental: bool = False,
    is_fire: bool = False,  # 新增
    is_thunder: bool = False,  # 新增
):
    ...
    if is_elemental:
        self._propagate_chain_damage(source, target, card, actual_damage, is_fire, is_thunder)

def _propagate_chain_damage(..., is_fire: bool = False, is_thunder: bool = False):
    ...
    self.deal_damage(source, current, card, damage, True, is_fire, is_thunder)
```

##### 6. 距离计算未取双向最小值 ✅ 已修复

**问题描述**：
- 规则：距离 = min(顺时针距离, 逆时针距离)
- 部分代码只计算单向距离

**修复方案**：
```python
# engine/response.py:522-545
def _calculate_distance(self, source: "Player", target: "Player") -> int:
    ...
    dist = 1
    current = source.next_player
    while current and current != target:
        dist += 1
        current = current.next_player

    reverse_dist = 1
    current = source.prev_player
    while current and current != target:
        reverse_dist += 1
        current = current.prev_player

    dist = min(dist, reverse_dist)  # 取最小值
    ...
```

#### P2 中优先级问题

##### 7. 缺少银月枪卡牌 ✅ 已修复

**问题描述**：
- 规则中有【银月枪】（方块Q，攻击范围3）
- 代码未实现

**修复方案**：
- 添加到 `card.py` 和 `data/cards.json`

##### 8. 武器技能未实现 ✅ 已修复

**实现的武器技能**：
| 武器 | 技能 |
|------|------|
| 青釭剑 | 无视防具 |
| 雌雄双股剑 | 杀异性目标后：弃对方牌/自己摸牌 |
| 寒冰剑 | 伤害改为弃置对方两张牌 |
| 古锭刀 | 对无手牌角色伤害+1 |
| 贯石斧 | 杀被闪后弃两牌强命 |
| 青龙偃月刀 | 杀被闪后追杀 |
| 麒麟弓 | 造成伤害弃置坐骑 |

##### 9. 铁索连环重铸 ✅ 已存在

铁索连环重铸功能已在 `new_main.py:308-344` 实现。

#### P3 低优先级问题

##### 10. 回合阶段不完整 ✅ 已修复

**修复方案**：
- 添加 `PREPARE_PHASE`（准备阶段）事件
- 添加 `END_PHASE`（结束阶段）事件
- 更新 `GamePhase` 枚举
- 添加 `end_turn()` 方法

**修改的文件**：
- `engine/event.py` - EventType 枚举
- `engine/state.py` - GamePhase 枚举
- `engine/game_engine.py` - next_turn 和 end_turn 方法

---

### 过河拆桥/顺手牵羊目标选择问题 ✅ 已修复 (2026-03-23)

**问题描述**：
- 玩家使用过河拆桥或顺手牵羊时，没有显示选择对手牌/装备的提示
- AI使用这些牌时也缺乏对坐骑和宝物的处理

**根本原因**：
`_resolve_chaiqiao` 和 `_resolve_shunshou` 方法检查的是 `target.is_human`（目标是否是人类），
但应该检查 `source.is_human`（使用卡牌的是否是人类玩家）。

**修复方案**：
1. 将 `target.is_human` 改为 `source.is_human`
2. 添加对进攻坐骑、防御坐骑、宝物的处理

**修改的文件**：
- `engine/game_engine.py` - `_resolve_chaiqiao` 和 `_resolve_shunshou` 方法

---

### TURN_START 事件触发问题 ✅ 已修复 (2026-03-23)

**问题描述**：
- 第一个玩家的回合开始时，`TURN_START` 事件没有被触发
- 观星、咆哮、洛神等技能不生效

**根本原因**：
`engine.setup_game()` 中将 `phase` 设置为 `TURN_START`，但 `game_loop` 中的检查会导致跳过事件触发。

**修复方案**：
1. 移除 `setup_game()` 中设置 `phase = TURN_START` 的代码
2. 在 `game_loop()` 中添加正确的事件触发逻辑

**修改的文件**：
- `engine/game_engine.py` - `setup_game` 方法
- `new_main.py` - `game_loop` 函数

---

### 装备系统无法生效 ✅ 已修复 (2026-03-23)

**问题描述**：
- 使用装备牌后，装备没有实际装备到玩家的装备槽位
- 武器、防具、坐骑都无法生效
- 诸葛连弩的无限杀效果不生效
- 距离计算因缺少坐骑而不正确
- 防具效果（仁王盾、藤甲、白银狮子等）无法触发

**根本原因**：
`engine/game_engine.py` 的 `_resolve_card` 方法缺少对装备牌的处理逻辑。
装备牌被使用后，只是被移到 `tmp_cards`，但没有调用 `_equip_*` 方法。

**修复方案**：
在 `_resolve_card` 方法中添加装备牌处理：

```python
elif card.card_type == "WeaponCard":
    self._equip_weapon(player, card)
    self.log(f"{player.commander_name} 装备了 {card.name}")
    if card.name == "诸葛连弩":
        player.unlimited_sha = True

elif card.card_type == "ArmourCard":
    self._equip_armour(player, card)
    self.log(f"{player.commander_name} 装备了 {card.name}")

elif card.card_type == "AttackHorseCard":
    self._equip_attack_horse(player, card)
    self.log(f"{player.commander_name} 装备了 {card.name}")

elif card.card_type == "DefenseHorseCard":
    self._equip_defense_horse(player, card)
    self.log(f"{player.commander_name} 装备了 {card.name}")

elif card.card_type == "TreasureCard":
    self._equip_treasure(player, card)
    self.log(f"{player.commander_name} 装备了 {card.name}")
```

**测试验证**：
- ✅ 武器装备正确，攻击范围正确更新
- ✅ 诸葛连弩无限杀效果生效
- ✅ 仁王盾抵挡黑杀
- ✅ 藤甲抵挡普通杀，火焰伤害+1
- ✅ 白银狮子伤害-1
- ✅ 进攻马/防御马距离计算正确

**修改的文件**：
- `engine/game_engine.py` - `_resolve_card` 方法

---

## Bug修复总结 - 2026-03-20

## 重要架构改进

### 卡牌类型系统重构 ✅ 完成

**问题**：
- 之前使用字符串匹配判断杀（`"杀" in card.name`）
- 火杀、雷杀需要特殊判断
- 代码可维护性差

**解决方案**：
创建清晰的卡牌继承体系：
```
Card
└── BasicCard
    └── ShaCard (杀基类)
        ├── FireSha (火杀)
        └── ThunderSha (雷杀)
```

**修改的文件**：
1. `card/base.py` - 创建 ShaCard 基类
   - 添加 `is_sha()` 方法
   - 添加 `is_elemental` 属性
   - FireSha 和 ThunderSha 继承自 ShaCard

2. `card/factory.py` - 智能卡牌类型选择
   - 根据卡牌名称自动选择正确类型
   - 普通杀自动使用 ShaCard 类型

3. 添加辅助函数 `is_sha_card(card)` - 类型安全的判断

**测试结果**：
```python
isinstance(ShaCard(), ShaCard)      # True
isinstance(FireSha(), ShaCard)      # True  
isinstance(ThunderSha(), ShaCard)   # True
```

## 修复的问题

### 1. 无双技能未生效 ✅ 已修复

**问题描述**：
- 吕布使用火杀或雷杀时，无双技能未触发
- 目标只需要出一张闪

**根本原因**：
无双技能判断使用字符串匹配：
```python
# 修复前 - 错误
return event.card.name in ["杀", "决斗"]  # 不匹配"雷杀"、"火杀"
```

**修复方案**：
使用类型判断替代字符串匹配：
```python
# 修复后 - 正确
from card.base import is_sha_card
return is_sha_card(event.card) or event.card.name == "决斗"
```

**修改的文件**：
- `skills/qun.py` - Wushuang 类
- `engine/game_engine.py` - use_card 和 _resolve_card 方法
- `engine/response.py` - resolve_sha 方法

### 2. 距离计算问题 ✅ 已验证

距离计算逻辑正确，RL AI 会自动过滤超出攻击范围的目标。

### 3. 卡牌使用提示信息 ✅ 已修复

启用了所有被注释的提示信息：
- 五谷丰登：显示翻开的卡牌
- 桃园结义：显示恢复体力的玩家
- 借刀杀人：显示可选目标
- 仁王盾、藤甲：显示生效信息

### 4. 响应系统提示信息 ✅ 已修复

玩家响应时显示：
- 可用卡牌列表
- 跳过选项
- 当前需要出多少张闪

### 5. 座位随机化 ✅ 已修复

添加 `random.shuffle(self.players)` 随机化座位顺序。

## 影响范围

### 训练环境

**是的，训练时也会有同样的问题！**

修复同时影响：
1. ✅ 主程序游戏 (`new_main.py`)
2. ✅ 训练环境 (`SGSEnv`)
3. ✅ RL AI 决策
4. ✅ 所有技能判断

### 建议

由于无双技能之前没有正确生效，建议：
1. 重新训练模型以学习正确的无双机制
2. 或在现有模型基础上继续训练微调

## 代码改进

### 使用类型判断替代字符串匹配

**修复前**：
```python
if "杀" in card.name:
    # 处理杀
if "火" in card.name or "雷" in card.name:
    is_elemental = True
```

**修复后**：
```python
from card.base import is_sha_card

if is_sha_card(card):
    # 处理杀
is_elemental = getattr(card, 'is_elemental', False)
```

### 优势

1. **类型安全** - 编译时检查，减少运行时错误
2. **可扩展** - 添加新类型的杀只需要继承 ShaCard
3. **可维护** - 判断逻辑集中在类层次结构中
4. **性能** - isinstance 比字符串匹配更快

## 测试验证

```bash
# 运行测试
python -c "
from card.base import is_sha_card, ShaCard, FireSha, ThunderSha

assert is_sha_card(ShaCard())
assert is_sha_card(FireSha())
assert is_sha_card(ThunderSha())
print('所有测试通过!')
"
```