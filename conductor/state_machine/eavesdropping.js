import { State } from "./index.js";



const eavesdropping = ({expirationTime = 30 * 1000, maxWordsToKeep = 100} = {}) => {
  const saveLine = (store, line) => {
    let lines = store['overheard-conversation'] || [];
    // keep lines that haven't expired yet
    lines = lines.filter(([ln, time]) => time > Date.now() - expirationTime);
    lines.push([line, Date.now()]);

    // keep only the last maxWordsToKeep words
    lines = lines
      .toReversed()
      .reduce(({lines, numWords}, [ln, time]) => {
        const words = ln.split(' ');
        numWords += words.length;

        if (numWords > maxWordsToKeep) {
          const numToRemove = numWords - maxWordsToKeep;
          // Keep only the last numToRemove words
          const trimmedLine = words.slice(words.length - numToRemove).join(' ');
          return {numWords: maxWordsToKeep, lines: [...lines, [trimmedLine, time]]};
        }

        return {numWords, lines: [...lines, [ln, time]]};
      }, {lines: [], numWords: 0})
      .lines
      .toReversed();

    store['overheard-conversation'] = lines;
  }

  return State.empty()
    .on('overheard-conversation-line', async (line, emit, store) => {
      console.log('In overheard-conversation-line', line);

      saveLine(store, line);
    });
}

export {
  eavesdropping
}