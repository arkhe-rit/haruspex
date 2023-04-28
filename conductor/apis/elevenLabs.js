import fetch from 'node-fetch';
import { createWriteStream } from 'fs';
import { nanoid } from 'nanoid';

const apiKey = process.env.ELEVENLABS_API_KEY;

const vocalizationStreams = {};

const vocalize = (
  {
    voice = {voice_id: 'EXAVITQu4vr4xnSDxMaL', 
            name: 'Bella'}, 
    stability = 0, 
    similarity_boost = 0
  } = {}
) => (text, cards = null) => {

  const url = `https://api.elevenlabs.io/v1/text-to-speech/${voice.voice_id}`;

  const data = {
    text,
    voice_settings: {
      stability,
      similarity_boost
    }
  };

  return fetch(url, {
    method: "POST",
    headers: {
      "Accept": "audio/mpeg",
      "xi-api-key": apiKey,
      "Content-Type": "application/json"
    },
    body: JSON.stringify(data)
  })
  .then(async (response) => {
    debugger;
    if (response.ok) {
      const makeFileName = () => {
        const currentDate = new Date();
        // voice.name + current time (with human readable month/day/hour/minute down to the ms
        const uniqueId = `${voice.name}${cards ? `-[${cards.join(',')}]-` : '-'}${currentDate.getMonth() + 1}-${currentDate.getDate()}_${currentDate.getHours()}${currentDate.getMinutes()}_${currentDate.getSeconds()}_${currentDate.getMilliseconds()}`;        
        const outputFile = `${uniqueId}.mp3`;
        return outputFile;
      };
      
      const filename = makeFileName();
      const fileStream = createWriteStream(`media/generated/${filename}`);

      response.body.pipe(fileStream);

      vocalizationStreams[filename] = fileStream;

      let resolve;
      const retPromise = new Promise(_resolve => {resolve = _resolve;});
      fileStream.on('finish', () => {
        console.log(`Audio saved to ${filename}`);
        resolve(filename);
      });

      return retPromise;
    } else {
      // Log the error details
      console.error(response.status, response.statusText);
      throw new Error("Error occurred while fetching the audio data");
    }
  })
  .catch((error) => {
    console.error("Error in vocalize:", error);
  });
}

const listVoices = () => {
  const url = `https://api.elevenlabs.io/v1/voices`;

  return fetch(url, {
    method: "GET",
    headers: {
      "Accept": "application/json",
      "xi-api-key": apiKey
    }
  })
  .then(async (response) => {
    if (response.ok) {
      let data = await response.json();
      return data.voices;
    } else {
      // Log the error details
      console.error(response.status, response.statusText);
      throw new Error("Error occurred while listing voices");
    }
  })
  .catch((error) => {
    console.error("Error in listVoices:", error);
  });
};

const findVoiceID = async (name) => {
  const voices = await listVoices();
  const voice = voices.find((voice) => voice.name === name);
  return voice.voice_id;
};

const vocalize_rod = ({voices, stability = .5, similarity_boost = .8} = {}) => {
  const voice = voices.find((voice) => voice.name === 'rod');
  return vocalize({voice, stability, similarity_boost});
};

const vocalize_missCleo = ({voices, stability = .5, similarity_boost = .8} = {}) => {
  const voice = voices.find((voice) => voice.name === 'missCleo');
  return vocalize({voice, stability, similarity_boost});
};

export {
  vocalizationStreams,
  vocalize,
  listVoices,
  findVoiceID,
  vocalize_rod,
  vocalize_missCleo
}