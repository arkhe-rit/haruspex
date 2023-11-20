import { Machine } from "./index.js";
import { waitingForCards } from './waiting_for_cards.js';

const machine = new Machine(
  waitingForCards()
);

export {
  machine
}