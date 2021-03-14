#include <util/atomic.h>

// 1 / 119.96
const unsigned long periodUs = 8336;
// 0.5ms after expected end of frame is when we detect missing frame
const unsigned long frameDelayDetectionUs = 500 + periodUs;

enum State {waitingForExpStart, waitingFirstClock, experimentActive};
State state = waitingForExpStart;

// Teensy 4.x have the LED on pin 13
const int ledPin = 13;

// the number of ms in each LED cycle outside experiments
const int ledPeriodMsInactive = 2500;
// the number of ms in each LED cycle waiting for exp to start
const int ledPeriodMsWaiting= 800;
// the number of ms in each LED cycle during experiment
const int ledPeriodMsActive = 250;
// how many ms have elapsed since start of current cycle
volatile int ledElapsed = 0;
// the number of ms in each LED cycle for current state
volatile int ledPeriodMs = ledPeriodMsInactive;
// timer controlling the LED blinking
IntervalTimer ledTimer;

const int clockPin = 14;
volatile unsigned long clockChangeTimeUs = 0;
volatile byte skippedFramesCurrent = 0;
int skippedFrames = 0;
// if skipped more than maxSkippedFrames, we end exp
const byte maxSkippedFrames = 20;
elapsedMillis elapsedPacketMs;

// buffer header magic number
const byte magicHeader[4] = {0xAB, 0xBC, 0xCD, 0xDF};

// RawHID packets are always 64 bytes
byte readBuffer[64];
const byte expStartPacketFlag = 0x01;
const byte expEndPacketFlag = 0x02;

// RawHID packets are always 64 bytes
byte sendBuffer[64] = {0xAB, 0xBC, 0xCD, 0xDF};
const byte pingPacketFlag = 0x01;
const byte countPacketFlag = 0x02;


void setup() {
  pinMode(clockPin, INPUT);
  attachInterrupt(digitalPinToInterrupt(clockPin), clockChange, CHANGE);
  
  pinMode(ledPin, OUTPUT);
  // it starts high in the cycle
  digitalWrite(ledPin, HIGH);

  // do this last so we can see if anything failed
  // Call interrupt every 1ms
  if (!ledTimer.begin(blinkLed, 1000))
    return;
}

void loop() {
  switch (state) {
    case waitingForExpStart:
      waitForExpStart();
      break;
    case waitingFirstClock:
      waitFirstClock();
      break;
    case experimentActive:
      doExperimentActive();
      break;
  }

}


void clockChange(){
  clockChangeTimeUs = micros();
  skippedFramesCurrent = 0;
}

void blinkLed(){
  ledElapsed++;
  if (ledElapsed == 1){
    // it has been ON for 1ms
    digitalWrite(ledPin, LOW);
  } else if (ledElapsed == ledPeriodMs) {
    // it has been OFF for desired duration
    ledElapsed = 0;
    digitalWrite(ledPin, HIGH);
  }
}

inline bool resetTimer(const int period){
  ledTimer.end();

  digitalWrite(ledPin, HIGH);

  ledElapsed = 0;
  ledPeriodMs = period;
  return ledTimer.begin(blinkLed, 1000);
}

inline bool matchMagicHeader(){
  return (readBuffer[0] == magicHeader[0] && readBuffer[1] == magicHeader[1] 
    && readBuffer[2] == magicHeader[2] && readBuffer[3] == magicHeader[3]);
}


void waitForExpStart() {
  // wait until we get message that experiment started, then go into waiting
  int n;
  // 0 timeout = do not wait in case of no message
  n = RawHID.recv(readBuffer, 0);
  if (n <= 0)
    return;

  if (!matchMagicHeader() || readBuffer[4] != expStartPacketFlag)
    return;

  if (!resetTimer(ledPeriodMsWaiting))
    return;

  state = waitingFirstClock;
  // reset
  elapsedPacketMs = 0;
  sendBuffer[4] = pingPacketFlag;
}

void waitFirstClock() {
  // we got message that experiment started so now we just wait for clock pin to go high
  int n;
  int value;
  value = digitalRead(clockPin);

  // have we started experiment with high clock?
  if (value == HIGH){
    if (!resetTimer(ledPeriodMsActive)){
      state = waitingForExpStart;
    } else {
      state = experimentActive;
      skippedFrames = 0;
      sendBuffer[4] = countPacketFlag;
    }
    return;
  }

  // send ping
  if (elapsedPacketMs >= 1){
    elapsedPacketMs -= 1;
    RawHID.send(sendBuffer, 0);
  }

  // check if asked to end. 0 timeout = do not wait in case of no message
  n = RawHID.recv(readBuffer, 0);
  if (n <= 0)
    return;

  // not a stop message?
  if (!matchMagicHeader() || readBuffer[4] != expEndPacketFlag)
    return;

  resetTimer(ledPeriodMsInactive);
  state = waitingForExpStart;
}


void doExperimentActive() {
  byte numCurrentSkipped;
  bool skipped = false;
  int n;

  ATOMIC_BLOCK(ATOMIC_RESTORESTATE)
  {
    if (micros() - clockChangeTimeUs >= frameDelayDetectionUs){
      // a frame is being dropped
      clockChangeTimeUs += periodUs;
      skippedFramesCurrent += 1;
      skippedFrames += 1;
      skipped = true;
    }

    numCurrentSkipped = skippedFramesCurrent;
  }

  if (numCurrentSkipped >= maxSkippedFrames){
    resetTimer(ledPeriodMsInactive);
    state = waitingForExpStart;
    return;
  }

  // send skipped count
  if (elapsedPacketMs >= 1 || skipped){
    if (elapsedPacketMs >= 1)
      elapsedPacketMs -= 1;

    *(int *)(&sendBuffer[5]) = skippedFrames;
    RawHID.send(sendBuffer, 0);
  }

  // check if asked to end. 0 timeout = do not wait in case of no message
  n = RawHID.recv(readBuffer, 0);
  if (n <= 0)
    return;

  // not a stop message?
  if (!matchMagicHeader() || readBuffer[4] != expEndPacketFlag)
    return;

  resetTimer(ledPeriodMsInactive);
  state = waitingForExpStart;
}
