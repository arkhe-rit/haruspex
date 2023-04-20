import cv2
import os
from pipeline import isolate_title_area, make_square

def card_path(card_name):
    path = os.path.join(os.getcwd(), 'cards', card_name)
    return path

cards_paths = {
    'fool': 'fool.jpg',
    'magician': 'magician.jpg',
    'high_priestess': 'highPriestess.jpg',
    'empress': 'empress.jpg',
    'emperor': 'emperor.jpg',
    'hierophant': 'hierophant.jpg',
    'lovers': 'lovers.jpg',
    'chariot': 'chariot.jpg',
    'strength': 'strength.jpg',
    'hermit': 'hermit.jpg',
    'wheel_of_fortune': 'wheelOfFortune.jpg',
    'justice': 'justice.jpg',
    'hanged_man': 'hangedMan.jpg',
    'death': 'death.jpg',
    'temperance': 'temperance.jpg',
    'devil': 'devil.jpg',
    'tower': 'tower.jpg',
    'star': 'star.jpg',
    'moon': 'moon.jpg',
    'sun': 'sun.jpg',
    'judgement': 'judgement.jpg',
    'world': 'world.jpg'
}

cards_paths_small = {
    'fool': 'fool_233x400.jpg',
    'magician': 'magician_233x400.jpg',
    'high_priestess': 'highPriestess_233x400.jpg',
    'empress': 'empress_233x400.jpg',
    'emperor': 'emperor_233x400.jpg',
    'hierophant': 'hierophant_233x400.jpg',
    'lovers': 'lovers_233x400.jpg',
    'chariot': 'chariot_233x400.jpg',
    'strength': 'strength_233x400.jpg',
    'hermit': 'hermit_233x400.jpg',
    'wheel_of_fortune': 'wheelOfFortune_233x400.jpg',
    'justice': 'justice_233x400.jpg',
    'hanged_man': 'hangedMan_233x400.jpg',
    'death': 'death_233x400.jpg',
    'temperance': 'temperance_233x400.jpg',
    'devil': 'devil_233x400.jpg',
    'tower': 'tower_233x400.jpg',
    'star': 'star_233x400.jpg',
    'moon': 'moon_233x400.jpg',
    'sun': 'sun_233x400.jpg',
    'judgement': 'judgement_233x400.jpg',
    'world': 'world_233x400.jpg'
}

to_invert = [
    'fool', 'magician', 'hermit', 'wheel_of_fortune', 'hanged_man', 'temperance', 'star', 'moon'
]

cards_text_full = {
    'fool': 'THE FOOL',
    'magician': 'THE MAGICIAN',
    'high_priestess': 'THE HIGH PRIESTESS',
    'empress': 'THE EMPRESS',
    'emperor': 'THE EMPEROR',
    'hierophant': 'THE HIEROPHANT',
    'lovers': 'THE LOVERS',
    'chariot': 'THE CHARIOT',
    'strength': 'STRENGTH',
    'hermit': 'THE HERMIT',
    'wheel_of_fortune': 'THE WHEEL OF FORTUNE',
    'justice': 'JUSTICE',
    'hanged_man': 'THE HANGED MAN',
    'death': 'DEATH',
    'temperance': 'TEMPERANCE',
    'devil': 'THE DEVIL',
    'tower': 'THE TOWER',
    'star': 'THE STAR',
    'moon': 'THE MOON',
    'sun': 'THE SUN',
    'judgement': 'JUDGEMENT',
    'world': 'THE WORLD'
}

cards_text_short = {
    'fool': 'FOOL',
    'magician': 'MAGICIAN',
    'high_priestess': 'HIGH PRIESTESS',
    'empress': 'EMPRESS',
    'emperor': 'EMPEROR',
    'hierophant': 'HIEROPHANT',
    'lovers': 'LOVERS',
    'chariot': 'CHARIOT',
    'strength': 'STRENGTH',
    'hermit': 'HERMIT',
    'wheel_of_fortune': 'WHEEL OF FORTUNE',
    'justice': 'JUSTICE',
    'hanged_man': 'HANGED MAN',
    'death': 'DEATH',
    'temperance': 'TEMPERANCE',
    'devil': 'DEVIL',
    'tower': 'TOWER',
    'star': 'STAR',
    'moon': 'MOON',
    'sun': 'SUN',
    'judgement': 'JUDGEMENT',
    'world': 'WORLD'
}

cards_keypoints = {}
for card_name, card_image in cards_paths_small.items():
    image = cv2.imread(card_path(card_image), cv2.IMREAD_GRAYSCALE)

    image = make_square(image)

    # invert image if in to_invert
    if card_name in to_invert:
        image = cv2.bitwise_not(image)

    title = isolate_title_area(image)

    cards_keypoints[card_name] = {
        'image': image,
        'text': cards_text_full[card_name],
        'text_short': cards_text_short[card_name]
    }