import fs from 'fs';

import { listVoices, vocalize, vocalize_missCleo, vocalize_rod } from "../apis/elevenLabs.js";
import { choose_videos, choose_videos_and_audio, generate_missCleo, generate_twilightZone } from "../generation/index.js";
import { publish } from "../subscribe.js";
import { State, Machine } from "./index.js";
import { arrayContainsAll, arrayMode, pick } from '../util.js';
import { files as video_files } from '../../media/videos/index.js';
import { files as audio_files } from '../../media/sfx/index.js';
import { waitingForCards } from './waiting_for_cards.js';

let voices;
try {
  voices = await listVoices();
} catch (err) {
  console.error(err);
}

let mediaPlayTimer;
const generating = (card_spread) => State.empty()
  .enter(async (_, emit) => {
    // const twilightZone = await generate_twilightZone({ numWords: 70 })(theseCards);
    // console.log('Generated text:', twilightZone);
    await Promise.all([
      generateMedia(card_spread, emit),
      generateFortune(card_spread, emit)
    ]);
    // return streamingAudio(twilightZone);
  })
  .on('fortune-generated', async ({ voice, fortune }, emit) => { 
     const vocalize = {
      'rod': vocalize_rod({ voices }),
      'missCleo': vocalize_missCleo({ voices })
     }[voice] || vocalize({ stability: 0.5, similarity_boost: 0.5})

    const audioFile = await vocalize(fortune, card_spread);

    emit('fortune-file-generated', audioFile);
  })
  .on('media-play-completed', async (_, emit) => {
    if (mediaPlayTimer) clearTimeout(mediaPlayTimer);
    return waitingForCards();
  });

async function generateMedia(cards, emit) {
  emit('media-generating', '');

  try {
    const { videos, sounds } = await choose_videos_and_audio({
      videos: video_files,
      sounds: audio_files,
      numSounds: 2
    })(cards);

    emit('media-generated', { videos, sounds });
    mediaPlayTimer = setTimeout(() => {
      emit('media-play-completed', '');
    }, 40 * 1000);
  } catch (e) {
    console.error(e);
    console.error('^^ Error generating media');
    // if e is an axios error
    if (e.isAxiosError) {
      console.error(e.response.status);
      console.error(e.response.data);
    }

    emit('media-play-completed', '');
    if (mediaPlayTimer) clearTimeout(mediaPlayTimer);
  }
}

// Ask openai for a fortune
async function generateFortune(cards, emit) {
  emit('fortune-generating', '');

  const [voice, generate] = pick([
    ['rod', generate_twilightZone({ numWords: 60 })],
    ['missCleo', generate_missCleo({ numWords: 60 })]
  ]);

  try {
    const fortune = await generate(cards);
    emit('fortune-generated', { voice, fortune });
  } catch (e) {
    console.error('Error generating fortune', e);

    emit('fortune-generated', { voice, fortune: 'Error generating fortune' });
  }
}

export { generating };