import { Configuration, OpenAIApi } from "openai";
import { encode, decode } from 'gpt-3-encoder';

const MAX_RETRIES = 0;

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);

const max_content_length = {
  'gpt-3.5-turbo': 4000,
  'gpt-4': 8000
}

const countChatTokens = (model, messages) => {
  let tokensPerMessage;
  let tokensPerName;

  if (model === "gpt-3.5-turbo") {
    tokensPerMessage = 4;
    tokensPerName = -1;
  } else if (model === "gpt-4") {
    tokensPerMessage = 3;
    tokensPerName = 1;
  }

  let numTokens = 0;

  for (const message of messages) {
    numTokens += tokensPerMessage;
    for (const [key, value] of Object.entries(message)) {
      numTokens += encode(value).length;
      if (key === "name") {
        numTokens += tokensPerName;
      }
    }
  }

  numTokens += 3; // every reply is primed with <|start|>assistant<|message|>
  return numTokens;
};

const chat = (model) => (messages = []) => {
  const go = async (chances = MAX_RETRIES) => {
    const messagesTokenCount = countChatTokens(model, messages);
    try {
      const response = await openai.createChatCompletion({
        model: model,
        messages,
        max_tokens: max_content_length[model] - messagesTokenCount,
        stream: false
      }, {
        timeout: 30 * 1000 // TODO
      });

      return response.data.choices[0].message.content;
    } catch (e) {
      console.error(e);
      if (chances > 0) {
        const delay_t = Math.pow(2, 12 * (MAX_RETRIES - chances + 1) / MAX_RETRIES);
        await delay(delay_t);
        return go(chances - 1);
      } else {
        throw e;
      }
    }
  };

  const chat_again = chat(model);

  Object.assign(go, {
    system: (msg) => chat_again([...messages, { role: "system", content: msg }]),
    user: (msg) => chat_again([...messages, { role: "user", content: msg }]),
    assistant: (msg) => chat_again([...messages, { role: "assistant", content: msg }]),
  });

  return go;
};

const chat_gpt35 = chat('gpt-3.5-turbo');
const chat_gpt4 = chat('gpt-4');

// As `chat` above, but made to work with the response as a stream
// const chatStream = (model) => (messages = []) => {
//   const go = async (chances = MAX_RETRIES) => {
//     const messagesTokenCount = countChatTokens(model, messages);
//     try {
//       const response = await openai.createChatCompletion({
//         model: model,
//         messages,
//         max_tokens: max_content_length[model] - messagesTokenCount,
//         stream: true
//       }, {
//         timeout: 30 * 1000,
//         responseType: 'stream'
//       });

//       for await (const chunk of response.data) {
//         const lines = chunk
//           .toString('utf8')
//           .split('\n')
//           .filter((line) => line.trim().startsWith('data: '))

//         for (const line of lines) {
//           const message = line.replace(/^data: /, '')
//           if (message === '[DONE]') {
//             return
//           }
//           const json = JSON.parse(message)
//           const token = json.choices[0].delta.content
//           if (token) {
//             console.log(token);
//           }
//         }
//       }


//     } catch (e) {
//       console.error(e);
//     }
//   };

//   const chat_again = chatStream(model);

//   Object.assign(go, {
//     system: (msg) => chat_again([...messages, { role: "system", content: msg }]),
//     user: (msg) => chat_again([...messages, { role: "user", content: msg }]),
//     assistant: (msg) => chat_again([...messages, { role: "assistant", content: msg }]),
//   });

//   return go;
// };

// const chat_gpt4_stream = chatStream('gpt-4');

// console.time();
// let res = chat_gpt4_stream()
//   .system('You are Rod Serling, the narrator of the Twilight Zone.')
//   .user('You are Rod Serling, the narrator of the Twilight Zone.')
//   .user('Write an intro for the Twilight Zone based on the following tarot cards and details. This intro will function as a fortune given to a user who has drawn the state tarot cards.')
//   .user('Do not mention "the Twilight Zone" until your last sentence.')
//   .user(`Respond in 75 words or less`)
//   .user(`Tarot cards: The Sun, Temperance, The High Priestess.`);
// console.log('----------------------------');
// res = await res();
// console.timeEnd();

export {
  chat_gpt35,
  chat_gpt4,
  countChatTokens
}