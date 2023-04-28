import fs from 'fs';

import { listVoices, vocalize_rod } from "../apis/elevenLabs.js";
import { choose_videos, choose_videos_and_audio, generate_twilightZone } from "../generation/index.js";
import { publish } from "../subscribe.js";
import { State, Machine } from "./index.js";
import { arrayContainsAll, arrayMode } from '../util.js';
import { files as video_files } from '../../media/videos/index.js';
import { files as audio_files } from '../../media/sfx/index.js';
import { waitingForCards } from './waiting_for_cards.js';
import { generating } from './generating.js';

let voices;
try {
  voices = await listVoices();
} catch (err) {
  console.error(err);
}

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