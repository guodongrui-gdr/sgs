import pygame


# 初始化 Pygame
pygame.init()

# 设置窗口大小
screen_width, screen_height = 1920, 1080
screen = pygame.display.set_mode((screen_width, screen_height))

# 设置时钟
clock = pygame.time.Clock()


class Button:
    def __init__(self, image_path, x, y, width, height, active_image=None):
        # 加载图像
        self.image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.image, (width, height))
        if active_image:
            self.active_image = pygame.image.load(active_image)
            self.active_image = pygame.transform.scale(self.active_image, (width, height))
        # 设置按钮的大小和位置
        self.rect = pygame.Rect(x, y, width, height)
        # 默认情况下按钮不激活
        self.active = False

    def draw(self, screen):
        # 如果按钮激活，则绘制激活的图像
        if self.active:
            screen.blit(self.active_image, self.rect)
        else:
            screen.blit(self.image, self.rect)


# 创建手牌类
class DrawCard:
    def __init__(self, image_path, x, y, card_width, card_height):
        # 加载图像
        self.image = pygame.image.load(image_path)
        # 缩放图像
        self.image = pygame.transform.scale(self.image, (card_width, card_height))
        # 创建矩形
        self.rect = pygame.Rect(x, y, card_width, card_height)


# 加载背景图
background_image = pygame.image.load(r"素材\背景.jpg")
background_image = pygame.transform.scale(background_image, (screen_width, screen_height))
# 绘制背景图
screen.blit(background_image, (0, 0))

# 加载手牌图像
card_images = []
for i in range(2):
    card_image = fr"素材\{i}.jpg"
    card_images.append(card_image)

# 设置手牌的大小和间距
card_width, card_height = 200, 250
card_spacing = 40

# 计算手牌的数量
num_cards = len(card_images)

# 计算手牌的位置
hand_x = 120
hand_y = 800

# 创建手牌对象
cards = []
for i, card_image in enumerate(card_images):
    x = hand_x + i * (card_width + card_spacing)
    y = hand_y
    card = DrawCard(card_image, x, y, card_width, card_height)
    cards.append(card)

# 设置按钮的位置
buttons = []
queding_x, queding_y = 710, 660
jieshu_x, jieshu_y = 1190, 670

queding_width, queding_height = 370, 100
jieshu_width, jieshu_height = 220, 80

queding = Button(r'素材\确定.png', queding_x, queding_y, queding_width, queding_height, active_image=r'素材\确定2.png')
jieshu = Button(r'素材\结束.jpg', jieshu_x, jieshu_y, jieshu_width, jieshu_height)

buttons.append(queding)
buttons.append(jieshu)

# 循环处理事件
running = True
while running:
    # 处理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # 处理鼠标事件
        if event.type == pygame.MOUSEBUTTONDOWN:
            # 获取鼠标坐标
            mouse_x, mouse_y = pygame.mouse.get_pos()
            # 判断鼠标是否在手牌上
            for card in cards:
                if card.rect.collidepoint(mouse_x, mouse_y):
                    queding.active = True
            # 判断鼠标是否在按钮上
            if jieshu.rect.collidepoint(mouse_x, mouse_y):
                # 如果在，则结束出牌阶段
                pass
    # 绘制手牌
    for card in cards:
        screen.blit(card.image, (card.rect.x, card.rect.y))
    # 绘制按钮
    for button in buttons:
        button.draw(screen)

    # 更新屏幕
    pygame.display.update()
    # 设置帧率
    clock.tick(60)
