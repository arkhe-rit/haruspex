from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import os
from tarot_cards import cards_keypoints

print('Loading CLIP Model...')
model = SentenceTransformer('clip-ViT-B-32')

# model.encode all tarot card images from the dictionary cards_keypoints
card_images = [Image.fromarray(cards_keypoints[card_name]['image']) for card_name in cards_keypoints.keys()]
card_names = list(cards_keypoints.keys())
card_encodings = model.encode(card_images, batch_size=128, convert_to_tensor=True, show_progress_bar=True)

def find_best_sentence_transformer_match(cam_image):
    # model.encode cam_image
    cam_encoding = model.encode([Image.fromarray(cam_image)], batch_size=128, convert_to_tensor=True, show_progress_bar=True)

    # combine card_encodings and cam_encoding into a single list for paraphrase_mining_embeddings
    NUM_SIMILAR_IMAGES = 5 

    similarity_scores = util.pytorch_cos_sim(cam_encoding, card_encodings)[0]
    top_indices = similarity_scores.argsort(descending=True)[:NUM_SIMILAR_IMAGES]

    # return top entries matches in top_indices
    return [card_names[idx] for idx in top_indices]