import fs from 'fs';

import { listVoices, vocalize_rod } from "../apis/elevenLabs.js";
import { choose_videos, choose_videos_and_audio, generate_twilightZone } from "../generation/index.js";
import { publish } from "../subscribe.js";
import { State, Machine } from "./index.js";
import { arrayContainsAll, arrayMode } from '../util.js';
import { files as video_files } from '../../media/videos/index.js';
import { files as audio_files } from '../../media/sfx/index.js';
import { generating } from './generating.js';
import { eavesdropping } from './eavesdropping.js';

let voices;
try {
  voices = await listVoices();
} catch (err) {
  console.error(err);
}

const HISTORY_LENGTH = 3;
const cardsHistory = [];
const saveToHistory = (cards) => {
  cardsHistory.unshift(cards);
  if (cardsHistory.length > HISTORY_LENGTH) cardsHistory.splice(HISTORY_LENGTH);
}
const denoisedCards = () => {
  if (cardsHistory.length < HISTORY_LENGTH) return [];

  return arrayMode(cardsHistory);
};

let lastCards = [];

const waitingForCards = () => State.empty()
  .on('haruspex-cards-observed', async (message, emit) => {
    // console.log('received message');
    emit('haruspex-cards-observed-ack', message ? message.reverse() : 'bloop');

    saveToHistory(message);
    const theseCards = denoisedCards();

    if (theseCards.length === 0 || 
        JSON.stringify(lastCards) === JSON.stringify(theseCards)) {
      // console.log('Cards unchanged');
      return;
    }

    if (!(theseCards.length === 3 && lastCards.length < theseCards.length)) {
      // console.log('Spread incomplete...', lastCards, theseCards);
      lastCards = theseCards;
      return;
    }

    console.log('Spread finished!', theseCards);
    lastCards = theseCards;

    emit('haruspex-spread-complete', theseCards);
  })
  .on('haruspex-spread-complete', async (complete_spread, emit) => {
    return generating(complete_spread);
  })
  .and(eavesdropping({
    expirationTime: 30 * 1000, 
    maxWordsToKeep: 100
  }))

export {
  waitingForCards
}