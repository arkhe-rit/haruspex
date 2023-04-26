import { subscribe, publish } from "./subscribe.js";
import { chat_gpt35 } from "./apis/openai.js";
import { vocalize, listVoices, findVoiceID, vocalize_rod } from "./apis/elevenLabs.js";
import sound from 'sound-play';
import Speaker from 'speaker';
import { Machine, State } from "./state_machine/index.js";
import { generate_twilightZone } from "./generation/index.js";
const speaker = new Speaker({
  channels: 2,          // 2 channels
  bitDepth: 16,         // 16-bit samples
  sampleRate: 44100     // 44,100 Hz sample rate
});

const voices = await listVoices();

const waitingForCards = () => State.empty()
  .on('haruspex-cards-observed', async (message, emit) => {
    console.log('received message');
    emit('haruspex-cards-observed-ack', message ? message.reverse() : 'bloop');
    
    console.log('Generating text...');
    const cards = message && message.length > 0
      ? message
      : ['The High Priestess', 'The Sun', 'Strength'];
    const twilightZone = await generate_twilightZone({numWords: 100})(cards);
    console.log('Generated text:', twilightZone);

    return waitingForAudio(twilightZone);
  });

const waitingForAudio = (text) => State.empty()
  .enter(async (_, emit) => {
    console.log('Generating audio...');
    const audioFile = await vocalize_rod({voices})(text);
    console.log('Audio generated');
    emit('audio-ready', audioFile);
  })
  .on('audio-ready', async (audioFile, emit) => {
    console.log('Playing...');
    sound.play(audioFile);
    console.log('Played');
    return waitingForCards();
  });



  
// subscribe('haruspex-cards-observed', async (message, channel) => {
//   console.log('received message');
//   await publish('haruspex-cards-observed-ack', message ? message.reverse() : 'bloop');

//   console.log('Generating text...');
//   const twilightZone = await generate_twilightZone({numWords: 100})([
//     'The High Priestess',
//     'The Sun',
//     'Strength'
//   ]);
//   console.log('Generated text:', twilightZone);
//   console.log('Generating audio...');
//   const audioFile = await vocalize_rod({voices})(twilightZone);
//   console.log('Playing...');
//   // sound.play(audioFile);
//   console.log('Played');
// });
// console.log('after subscribe');

// const chat = chat_gpt35()
//   .system('You are a fortune teller. You speak mysteriously and with great authority. You give oddly specific details. You speak in the voice of an old gypsy lady.')
//   .user('You are a fortune teller. You speak mysteriously and with great authority. You give oddly specific details. You speak in the voice of an old gypsy lady.')
//   .user('Read my fortune based on these tarot cards I have drawn: The High Priestess, The Sun, and Strength.')
//   .user('Respond in 100 words or less.');

// const res = await chat();
// console.log('--- From openai:');
// console.log(res);
// console.log('vocalizing...');
// debugger;
// const voices = await listVoices();
// await vocalize(await findVoiceID('missCleo'), {stability: 0.5, similarity_boost: 0.85})(res);
