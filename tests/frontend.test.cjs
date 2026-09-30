const {test} = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');

function setup() {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      listeners: {}, disabled: false, submissions: 0,
      addEventListener(name, fn) { this.listeners[name] = fn; },
      requestSubmit() { this.submissions++; }
    });
    return elements.get(id);
  };
  const context = vm.createContext({document: {getElementById: element},
    crypto: require('node:crypto').webcrypto});
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8'), context);
  vm.runInContext("apiKey = 'test';", context);
  vm.runInContext("sendMessage = () => document.getElementById('chat-form').requestSubmit();", context);
  return {element, context};
}

test('Enter sends; Shift+Enter and IME composition do not', () => {
  const {element} = setup();
  let prevented = 0;
  const press = overrides => element('question').listeners.keydown({
    key: 'Enter', preventDefault() { prevented++; }, ...overrides});
  press({});
  assert.equal(element('chat-form').submissions, 1);
  press({shiftKey:true}); press({isComposing:true}); press({key:'a'});
  assert.equal(element('chat-form').submissions, 1);
  assert.equal(prevented, 1);
});

test('No duplicate or disconnected keyboard submissions', () => {
  const {element, context} = setup();
  const press = overrides => element('question').listeners.keydown({
    key:'Enter', preventDefault() {}, ...overrides});
  press({repeat:true});
  vm.runInContext('busy = true;', context); press({});
  vm.runInContext("busy = false; apiKey = '';", context); press({});
  assert.equal(element('chat-form').submissions, 0);
});

test('Send button and Enter call the same send function', () => {
  const {element} = setup();
  element('chat-form').listeners.submit({preventDefault() {}});
  element('question').listeners.keydown({key:'Enter', preventDefault() {}});
  assert.equal(element('chat-form').submissions, 2);
});
