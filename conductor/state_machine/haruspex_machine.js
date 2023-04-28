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


const machine = new Machine(waitingForCards());

export {
  machine
}