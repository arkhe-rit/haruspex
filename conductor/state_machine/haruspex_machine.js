import fs from 'fs';

import { listVoices, vocalize_rod } from "../apis/elevenLabs.js";
import { choose_videos, generate_twilightZone } from "../generation/index.js";
import { publish } from "../subscribe.js";
import { State, Machine } from "./index.js";
import { arrayContainsAll, arrayMode } from '../util.js';
import { video_files } from '../../videos/index.js';

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
// const denoisedCards = () => {
//   if (cardsHistory.length < HISTORY_LENGTH) return [];

//   const counts = {};
//   cardsHistory.forEach(cards => {
//     const key = JSON.stringify(cards);
//     counts[key] = (counts[key] || 0) + 1;
//   });
//   const mostCommon = Object.entries(counts).reduce((acc, [key, count]) => {
//     if (count > acc.count) {
//       return { key, count };
//     }
//     return acc;
//   }, { key: null, count: 0 });
//   return JSON.parse(mostCommon.key);
// };
let lastCards = [];

const waitingForCards = () => State.empty()
  .on('haruspex-cards-observed', async (message, emit) => {
    // console.log('received message');
    emit('haruspex-cards-observed-ack', message ? message.reverse() : 'bloop');

    saveToHistory(message);
    const theseCards = denoisedCards();
    console.log('---------------', theseCards);
    if (theseCards.length === 0 || 
        JSON.stringify(lastCards) === JSON.stringify(theseCards)) {
      console.log('Cards unchanged');
      return;
    }

    if (!(theseCards.length === 3 && lastCards.length < theseCards.length)) {
      console.log('Spread incomplete...', lastCards, theseCards);
      lastCards = theseCards;
      return;
    }

    console.log('Spread finished!');
    lastCards = theseCards;

    console.log('Generating text...');
    
    // const twilightZone = await generate_twilightZone({ numWords: 70 })(theseCards);
    // console.log('Generated text:', twilightZone);
    const videos = await choose_videos({videos: video_files, numVideos: 4})(theseCards);

    emit('videos-generated', videos);

    // return streamingAudio(twilightZone);
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