import { chat_gpt35, chat_gpt4 } from "../apis/openai.js"


const generate_twilightZone = ({numWords = 50, chatModel = chat_gpt4} = {}) => (tarotCards, detailObj = {}) => (
  chatModel()
    .system('You are Rod Serling, the narrator of the Twilight Zone.')
    .user('You are Rod Serling, the narrator of the Twilight Zone.')
    .user('Write an intro for the Twilight Zone based on the following tarot cards and details. This intro will function as a fortune given to a user who has drawn the state tarot cards.')
    .user('Do not mention "the Twilight Zone" until your last sentence.')
    .user(`Respond in ${numWords} words or less`)
    .user(`Tarot cards: ${JSON.stringify(tarotCards)}`)
    .user(`Details: ${JSON.stringify(detailObj)}`)
    ()
)

const generate_missCleo = ({numWords = 50, chatModel = chat_gpt4} = {}) => (tarotCards, detailObj = {}) => (
  chat_gpt4()
    .system('You are Miss Cleo, famous psychic for Psychic Readers Network infomercials.')
    .user('You are Miss Cleo, famous psychic for Psychic Readers Network infomercials.')
    .user('Tell me my fortune based on the following tarot cards and details.')
    .user(`I have drawn these cards: ${tarotCards.join(', ')}.`)
    .user(`Respond in ${numWords} words or less`)
    .user(`Tarot cards: ${JSON.stringify(tarotCards)}`)
    .user(`Details: ${JSON.stringify(detailObj)}`)
    ()
)

const choose_videos = ({chatModel = chat_gpt4, videos, numVideos = 3} = {}) => async (tarotCards) => {
  const systemMsg = `
    Respond with ${numVideos} strings, strictly adhering to the following JSON schema.  Begin and end with square brackets.
    \`\`\`
    {
      "$schema": "http://json-schema.org/draft-07/schema#",
      "type": "array",
      "items": {
        "type": "string",
        "enum": [
          ${videos.map(s => `"${s}"`).join(',\n')}
        ]
      },
      "minItems": ${numVideos}, 
      "maxItems": ${numVideos},
      "uniqueItems": true
    }
    \`\`\`
  `.trim();

  const userMsg = `
    Choose ${numVideos} videos that are most appropriate for the following tarot card spread. Choose videos that match the emotions conveyed by these cards. Choose videos that most resemble the future predicted by these cards.

    Tarot spread: ${tarotCards.join(', ')}.
  `.trim();
  
  const response = await chatModel()
    .system(systemMsg)
    .user(userMsg)
    ();

  // Find first '[' in response
  // If it exists, return the substring starting there and ending at the next ']'
  const firstBracket = response.indexOf('[');
  if (firstBracket !== -1) {
    const lastBracket = response.indexOf(']', firstBracket);
    if (lastBracket !== -1) {
      const substring = response.substring(firstBracket, lastBracket + 1);
      return JSON.parse(substring);
    }
  }
  // Otherwise, find the first '"' and return the substring starting there and ending at the last '"'
  const firstDoubleQuote = response.indexOf('"');
  if (firstDoubleQuote !== -1) {
    const lastDoubleQuote = response.lastIndexOf('"');
    if (lastDoubleQuote !== -1) {
      const substring = response.substring(firstDoubleQuote, lastDoubleQuote + 1);
      return JSON.parse('[' + substring + ']');
    }
  }
  // Otherwise, find the first "'" and return the substring starting there and ending at the last "'"
  const firstSingleQuote = response.indexOf("'");
  if (firstSingleQuote !== -1) {
    const lastSingleQuote = response.lastIndexOf("'");
    if (lastSingleQuote !== -1) {
      const substring = response.substring(firstSingleQuote, lastSingleQuote + 1);
      return JSON.parse('[' + substring + ']');
    }
  }
  // Otherwise, find numVideos strings that end with ".mp4" and return those
  const mp4Regex = /\.mp4$/;
  const mp4Matches = response.match(mp4Regex);
  if (mp4Matches) {
    return mp4Matches.slice(0, numVideos);
  }

  console.info("Could not parse choose_videos response from chat model. Returning default videos.")
  console.info("Response: ", response);
  return ["abstract_geometric.mp4", "cigarette_smoke.mp4", "blooming_flowers.mp4"];
}

const list_events = ({chatModel = chat_gpt35, numEvents = 5} = {}) => async (tarotCards) => {
  const systemMsg = `
    Respond with ${numEvents} strings, strictly adhering to the following JSON schema.  Just list the events, respond in JSON not in full sentences.
    \`\`\`
    {
      "$schema": "http://json-schema.org/draft-07/schema#",
      "type": "array",
      "items": {
        "type": "string"
      },
      "minItems": ${numEvents},
      "maxItems": ${numEvents}
    }
    \`\`\`
  `.trim();

  const userMsg = `
    List ${numEvents} specific events that might happen in the future predicted by the following tarot card spread.

    Tarot spread: ${tarotCards.join(', ')}.
  `.trim();

  const response = await chatModel()
  .system(systemMsg)
  .user(systemMsg)
  .user(userMsg)
  ();

  // TODO

}

export {
  generate_twilightZone,
  generate_missCleo,
  choose_videos,
  list_events
}