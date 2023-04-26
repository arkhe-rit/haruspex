import fs from 'fs';

import { listVoices, vocalize_rod } from "../apis/elevenLabs.js";
import { choose_videos, choose_videos_and_audio, generate_twilightZone } from "../generation/index.js";
import { publish } from "../subscribe.js";
import { State, Machine } from "./index.js";
import { arrayContainsAll, arrayMode } from '../util.js';
import { files as video_files } from '../../media/videos/index.js';
import { files as audio_files } from '../../media/sfx/index.js';

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
  });

let mediaPlayTimer;
const generating = (card_spread) => State.empty()
  .enter(async (_, emit) => {
    // const twilightZone = await generate_twilightZone({ numWords: 70 })(theseCards);
    // console.log('Generated text:', twilightZone);
    emit('media-generating', '');

    try {
      const {videos, sounds} = await choose_videos_and_audio({
        videos: video_files,
        sounds: audio_files,
        numSounds: 2
      })(card_spread);
  
      emit('media-generated', {videos, sounds});
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
    // return streamingAudio(twilightZone);
  })
  .on('media-play-completed', async (_, emit) => {
    if (mediaPlayTimer) clearTimeout(mediaPlayTimer);
    return waitingForCards();
  });


const streamingAudio = (text) => State.empty()
  .enter(async (_, emit) => {
    console.log('Generating audio...');
    if (!voices) voices = await listVoices();
    const audioFile = await vocalize_rod({ voices })(text);
    console.log('Audio generation begun');

    publish('audio-begin');
    const fileStream = fs.createReadStream(filePath);

    // Wait for the readable stream to be ready before piping the data
    fileStream.on('readable', () => {
      let chunk;
      while ((chunk = fileStream.read()) !== null) {
        publish('audio-chunk', chunk);
      }
    });

    // Error handling for the file stream
    fileStream.on('error', (err) => {
      console.error('Error while reading the audio file:', err);
      fileStream.destroy();
      socket.emit('streamError', 'Error while reading the audio file');
    });

    fileStream.on('end', () => {
      console.log('Audio stream ended');
      publish('audio-end', audioFile);
    });
  })
  .on('audio-end', async (audioFile, emit) => {
    console.log('In audio-end');

    return waitingForCards();
  });

const machine = new Machine(waitingForCards());

export {
  machine
}