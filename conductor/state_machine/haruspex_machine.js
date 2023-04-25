import { listVoices } from "../apis/elevenLabs.js";
import { State, Machine } from "./index.js";

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

const machine = new Machine(waitingForCards());

export {
  machine
}