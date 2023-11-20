import { subscribe, unsubscribe, publish } from "../subscribe.js";

class State {
  constructor(listeners) {
    this.listeners = listeners;
  }

  eventsOfInterest() {
    return Object.keys(this.listeners);
  }

  equals(otherState) {
    return otherState && otherState.listeners && this.listeners === otherState.listeners;
  }

  async transition(event, data, emit, store) {
    const listener = this.listeners[event];
    if (!listener) {
      console.info(`No listener found for event ${event}`);
      return this;
    }

    const nextState = await listener(data, emit, store);

    if (!nextState) {
      return this;
    }

    return nextState;
  }

  on(event, listener) {
    return new State({
      ...this.listeners,
      [event]: listener
    });
  }

  enter(listener) {
    return this.on('enter', listener);
  }

  exit(listener) {
    return this.on('exit', listener);
  }

  static empty() {
    return new State({});
  }

  and(state) {
    return new State({
      ...this.listeners,
      ...state.listeners
    });
  }
};

class Machine {
  constructor(arg1, arg2) {
    let state;
    let name;
    // if arg1 is a string and arg2 is a State
    if (typeof arg1 === 'string' && arg2 instanceof State) {
      name = arg1;
      state = arg2;
    } 
    // else if arg1 is a State and arg2 wasn't provided
    else if (arg1 instanceof State && !arg2) {
      state = arg1;
      name = 'initial';
    }

    this.state = state;
    this.stopped = false;
    this.registeredStates = {
      [name]: state
    };
    this.namesByState = new Map([[state, name]]);
  }

  register(name, state) {
    this.registeredStates[name] = state;
    this.namesByState.set(state, name);
    return this;
  }

  async run() {
    const emit = (event, data) => publish(event, data);

    let unsubs = [];
    let store = {};

    while (!this.stopped) {
      let eventsOfInterest = this.state.eventsOfInterest();

      let newStateResolve;
      let newState = new Promise(resolve => {
        newStateResolve = resolve;
      });

      unsubs = await Promise.all(eventsOfInterest.map(event => {
        return subscribe(event, async (message, channel) => {
          let nextState = await this.state.transition(event, message, emit, store) || {};

          // If nextState is a string, set nextState to the registered state with that name
          if (typeof nextState === 'string') {
            nextState = this.registeredStates[nextState];
            if (!nextState) {
              console.info(`No registered state found with name '${nextState}'`);
            }
          }

          if (!nextState) {
            return;
          }

          if (!this.state.equals(nextState)) {
            if (!unsubs) {
              await new Promise(resolve => setTimeout(resolve, 10));
            }
            await Promise.all(unsubs.map(unsub => unsub()));
            newStateResolve(nextState);
          }
        });
      }));

      await this.state.transition('enter', {}, emit, store);

      const stateToTransitionTo = await newState;
      const fromName = this.namesByState.get(this.state) || 'unknown';
      const toName = this.namesByState.get(stateToTransitionTo) || 'unknown';
      
      console.info(`Transitioning from state "${fromName}" to state "${toName}"`);
      await this.state.transition('exit', {}, emit, store);
      this.state = stateToTransitionTo;
    }
  }
}

export {
  State,
  Machine
}