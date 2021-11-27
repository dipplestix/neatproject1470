import pygame


def get_hitmask(image):
    mask = []
    for x in range(image.get_width()):
        mask.append([])
    for y in range(image.get_height()):
        mask[x].append(bool(image.get_at((x, y))[3]))
    return mask


def load():
    IMAGES, HITMASKS = {}, {}
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    IMAGES['base'] = pygame.image.load(
        'assets/sprites/base.png').convert_alpha()
    IMAGES['background'] = pygame.image.load(
        'assets/sprites/background-day.png').convert()
    IMAGES['player'] = (
        pygame.image.load(
            'assets/sprites/yellowbird-upflap.png').convert_alpha(),
        pygame.image.load(
            'assets/sprites/yellowbird-midflap.png').convert_alpha(),
        pygame.image.load(
            'assets/sprites/yellowbird-downflap.png').convert_alpha(),
    )
    IMAGES['pipe'] = (
        pygame.transform.rotate(
            pygame.image.load('assets/sprites/pipe-green.png').convert_alpha(), 180),
        pygame.image.load('assets/sprites/pipe-green.png').convert_alpha(),
    )

    # HITMASKS

    HITMASKS['pipe'] = (
        get_hitmask(IMAGES['pipe'][0]),
        get_hitmask(IMAGES['pipe'][1]),
    )

    HITMASKS['player'] = (
        get_hitmask(IMAGES['player'][0]),
        get_hitmask(IMAGES['player'][1]),
        get_hitmask(IMAGES['player'][2]),
    )

    return IMAGES, HITMASKS
