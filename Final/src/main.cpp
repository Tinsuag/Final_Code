#include "IMU_Sensors.h"
#include "RMDX8.h"
#include <Wire.h>


// =================== Hardware Objects ===================
RMDX8Motor Ankle(53);        // motor driver object (uses SPI CS pin 53)
AnkleSensors _sensors;       // your sensor abstraction
unsigned long lastStatusTime = 0;

// NOTE: we assume `bno` (BNO055) is declared/owned in IMU_Sensors.*
// and that _sensors.readIMUData() fills accel/gyro/quats/etc.

// =================== System State Machine ===================
enum State {
  START,
  IDLE,
  CONTROL,
  CALIBRATE_SENSORS,
  EMERGENCY_STOP,
  INITIALIZING,
  READING,
  SPEED,
  OFFSETTING
};

float   init_Status = 0.0;
State   currentState = START;

// (Optional) auto-exit calibration once everything is fully calibrated
static const bool AUTO_EXIT_WHEN_FULLY_CAL = true;

// =================== Runtime Variables ===================
// last commanded / sensed values
int16_t targetTorque      = 0;      // actual torque applied to motor
float   requestedTorqueF  = 0;      // last requested torque from PC
float   requestedSpeedF   = 0;      // last requested speed magnitude from PC
int8_t  requestedDir      = 1;      // +1 forward / -1 reverse from PC

// live sensor channels
float angleTheta = 0.0;   // encoder angle (deg)
float angleAlpha = 0.0;   // IMU pitch (deg)
float speedVal   = 0.0;   // using gyroX as angular velocity proxy (deg/s-ish)
float tempVal    = 25.0;  // motor temp placeholder

float accelX = 0.0;
float accelY = 0.0;
float accelZ = 0.0;
float gyroX  = 0.0;
float gyroY  = 0.0;
float gyroZ  = 0.0;
float quatW  = 1.0;
float quatX  = 0.0;
float quatY  = 0.0;
float quatZ  = 0.0;

uint8_t imuFully = 0; // IMU calibrated? (1=yes)
uint8_t encOK    = 0; // encoder healthy? (1=yes)

// Keep the latest per-subsystem IMU calib levels
uint8_t cSys_level = 0, cG_level = 0, cA_level = 0, cM_level = 0;

// =================== COMMAND IDs (MUST match GUI) ===================
static const uint8_t CMD_IDLE            = 0x01;
static const uint8_t CMD_INITIALIZING    = 0x02;
static const uint8_t CMD_CALIBRATE       = 0x03;
static const uint8_t CMD_READING         = 0x04;
static const uint8_t CMD_OFFSETTING      = 0x05;
static const uint8_t CMD_CONTROL         = 0x06;
static const uint8_t CMD_STOP            = 0x07;
static const uint8_t CMD_SPEED           = 0x08;
// =================== PROTOCOL CONSTANTS ===================
//
// PC -> Arduino (command packet):
//   [0] START_BYTE        (0xAA)
//   [1] PKT_CMD_ID        (0x20)
//   [2] LEN (N)
//   [3..3+N-1] PAYLOAD
//   [3+N] CHECKSUM
//
// CHECKSUM = (PKT_CMD_ID + LEN + sum(PAYLOAD)) % 256
//
// CONTROL PAYLOAD FORMAT (NEW: no safe_flag):
//   byte0    = cmd_id (0x06)
//   byte1..4 = torque float32 LE
//   byte5..8 = speed  float32 LE
//   byte9    = direction_flag (0=reverse,1=forward)
//
// Other simple commands just send 1 byte payload = [cmd_id].
//
// Arduino -> PC (telemetry packet):
//   [0] START_BYTE        (0xAA)
//   [1] PKT_TELEM_ID      (0x30)
//   [2] LEN (payload size)
//   [3..] PAYLOAD
//   [last] CHECKSUM
//
// Telemetry PAYLOAD layout we send back:
//   byte0 = state_code  (we use CMD_* value that represents current mode)
//   byte1 = NUM_FLOATS  (13)
//   then 13 float32 LE in this order:
//        [0] angleTheta
//        [1] speedVal
//        [2] (float)targetTorque
//        [3] tempVal
//        [4] accelX
//        [5] accelY
//        [6] accelZ
//        [7] gyroX
//        [8] gyroZ
//        [9] quatW
//        [10] quatX
//        [11] quatY
//        [12] quatZ
//   then imuFully (1 byte)
//   then encOK    (1 byte)
//   WHILE CALIBRATING ONLY (extra 5 bytes appended):
//        cSys (1B), cG (1B), cA (1B), cM (1B), as5600Cal (1B)
// =========================================================

static const uint8_t START_BYTE   = 0xAA;
static const uint8_t PKT_CMD_ID   = 0x20; // incoming commands from PC
static const uint8_t PKT_TELEM_ID = 0x30; // outgoing telemetry to PC

// =================== Motor/Safety Helpers ===================
void applyTorqueCommand(int16_t torq) {
  // send a closed-loop torque command to the motor
  Ankle.torqueClosedLoopCommand(torq);
}

void applySpeedCommand(float signedSpeed) {
  // Stub for velocity mode if your motor lib supports it
  // Example:
  Ankle.targetSpeed=signedSpeed;
  Ankle.sendSpeedCommand();
  (void)signedSpeed;
}

// Helper: decide if AS5600 should be considered "calibrated" for the GUI detail
static inline uint8_t getAS5600CalFlag() {
  // If your AnkleSensors has an explicit flag like `_sensors.as5600_zeroed` or `as5600_calibrated`,
  // prefer that. Otherwise, we fall back to encoder OK as a proxy.
  // return _sensors.as5600_calibrated ? 1 : 0;
  return (encOK ? 1 : 0);
}

// =================== Sensor Read ===================
void updateSensorsFromHardware() {
  // Update sensor class
  _sensors.AS5600_angle = _sensors.readAS5600Angle();
  _sensors.readIMUData();

  // Angle from encoder
  angleTheta = _sensors.AS5600_angle;

  // Pitch (deg) from IMU
  angleAlpha = _sensors.roll;

  // Use gyroX as joint angular velocity proxy
  speedVal   = _sensors.gyroX;

  // You can replace with actual motor temp if you have it
  // tempVal  = Ankle.getMotorTemperature();

  accelX = _sensors.accelX;    
  accelY = _sensors.accelY;
  accelZ = _sensors.accelZ;
  gyroX  = _sensors.gyroX;
  gyroY  = _sensors.gyroY;
  gyroZ  = _sensors.gyroZ;
  quatW  = _sensors.quat_w;
  quatX  = _sensors.quat_x;
  quatY  = _sensors.quat_y;
  quatZ  = _sensors.quat_z;

  // IMU calibration check
  uint8_t cSys=0, cG=0, cA=0, cM=0;
  bno.getCalibration(&cSys, &cG, &cA, &cM);
  cSys_level = cSys; cG_level = cG; cA_level = cA; cM_level = cM;   // keep latest levels in globals variable type uint8_t
  imuFully = (cSys==3 && cG==3 && cA==3 && cM==3) ? 1 : 0;

  // Encoder health flag from your class
  encOK    = _sensors.enc_ok ? 1 : 0;
}

// =================== Report State Code ===================
// We send back one byte (state_code). Python maps it to human text.
uint8_t stateToProtoCode(State s) {
  switch (s) {
    case IDLE:              return CMD_IDLE;
    case INITIALIZING:      return CMD_INITIALIZING;
    case CALIBRATE_SENSORS: return CMD_CALIBRATE;
    case READING:           return CMD_READING;
    case OFFSETTING:        return CMD_OFFSETTING;
    case CONTROL:           return CMD_CONTROL;
    case EMERGENCY_STOP:    return CMD_STOP;
    case START:             return CMD_INITIALIZING; // treat boot like init
    case SPEED:              return CMD_SPEED;      // treat SPEED like reading
    default:                return CMD_READING;

  }
}

// =================== Telemetry TX -> PC ===================
void sendStatusToPC() {
  uint8_t currentStateCode = stateToProtoCode(currentState);

  const uint8_t NUM_FLOATS = 13;
  float fvals[NUM_FLOATS] = {
    angleTheta,          // 0
    speedVal,            // 1
    (float)targetTorque, // 2
    tempVal,             // 3
    accelX,              // 4
    accelY,              // 5
    accelZ,              // 6
    gyroX,               // 7
    gyroZ,               // 8
    quatW,               // 9
    quatX,               //10
    quatY,               //11
    quatZ                //12
  };

  // Build payload buffer
  uint8_t payload[128];
  uint8_t p = 0;

  // state code at 
  payload[p++] = currentStateCode;

  // number of floats at 
  payload[p++] = NUM_FLOATS;

  // helper to write float32 LE
  auto putF32 = [&](float v) {
    union { float f; uint8_t b[4]; } u;
    u.f = v;
    payload[p++] = u.b[0];
    payload[p++] = u.b[1];
    payload[p++] = u.b[2];
    payload[p++] = u.b[3];
  };

  // write floats
  for (uint8_t i = 0; i < NUM_FLOATS; i++) {
    putF32(fvals[i]);
  }

  // final flags
  payload[p++] = imuFully;
  payload[p++] = encOK;

  // While CALIBRATING, append detailed calib bytes expected by the Python (optional section)
  if (currentState == CALIBRATE_SENSORS) {
    payload[p++] = cSys_level;             // cSys
    payload[p++] = cG_level;               // cG
    payload[p++] = cA_level;               // cA
    payload[p++] = cM_level;               // cM
    payload[p++] = getAS5600CalFlag();     // AS5600 calibrated flag (1/0)
  }

  uint8_t lengthByte = p;

  // checksum rule:
  // checksum = (PKT_TELEM_ID + length + sum(payload)) % 256
  uint16_t calc = 0;
  calc += PKT_TELEM_ID;
  calc += lengthByte;
  for (uint8_t i=0; i<p; i++) {
    calc += payload[i];
  }
  uint8_t checksumByte = (uint8_t)(calc % 256);

  // Write the frame
  Serial.write(START_BYTE);      // [0]
  Serial.write(PKT_TELEM_ID);    // [1]
  Serial.write(lengthByte);      // [2]
  Serial.write(payload, p);      // [3..]
  Serial.write(checksumByte);    // [last]
}

// =================== APPLY COMMAND FROM PC ===================
// NOTE: no more safe_flag here.
// We trust the PC GUI to only send CONTROL if "safety" is enabled.
// =================== APPLY COMMAND FROM PC (PATCHED) ===================
//
// CONTROL  = pure torque mode (no speed command here)
// SPEED    = pure speed mode (no torque command here)
// Other states unchanged.
//
void applyCommandFromPC(
  uint8_t cmd_id,
  float   torqueReq,
  float   speedReq,
  int8_t  dirFlag
) {
  // log last request
  requestedTorqueF = torqueReq;
  requestedSpeedF  = speedReq;
  requestedDir     = (dirFlag == 0) ? -1 : +1;   // 0 = reverse, 1 = forward

  // Map cmd_id -> requestedState
  State requestedState = currentState;
  switch (cmd_id) {
    case CMD_IDLE:         requestedState = IDLE;              break;
    case CMD_INITIALIZING: requestedState = INITIALIZING;      break;
    case CMD_CALIBRATE:    requestedState = CALIBRATE_SENSORS; break;
    case CMD_READING:      requestedState = READING;           break;
    case CMD_OFFSETTING:   requestedState = OFFSETTING;        break;
    case CMD_CONTROL:      requestedState = CONTROL;           break;
    case CMD_STOP:         requestedState = EMERGENCY_STOP;    break;
    case CMD_SPEED:        requestedState = SPEED;             break;
    default:               requestedState = IDLE;              break;
  }

  // -------- EMERGENCY STOP (one-shot hard kill) --------
  if (requestedState == EMERGENCY_STOP) {
    Ankle.torqueClosedLoopCommand(0);
    Ankle.motorStopCommand();
    targetTorque = 0;
    currentState = IDLE;
    return;
  }

  // -------- INITIALIZING (hardware bring-up) --------
  if (requestedState == INITIALIZING) {
    init_Status = 1.0;

    Wire.begin();
    Wire.setClock(400000); // 400kHz I2C for BNO055 + AS5600

    Ankle.begin();

    if (!bno.begin()) {
      // IMU init failed; you may want to print or blink here
      // Serial.println("BNO055 not detected");
      while (1) {
        // trap if you prefer
      }
    }
    bno.setExtCrystalUse(true);

    currentState = IDLE;
    return;
  }

  // -------- OFFSETTING (zero AS5600) --------
  if (requestedState == OFFSETTING) {
    _sensors.offSetting_AS5600();
    currentState = IDLE;
    return;
  }

  // -------- CALIBRATE SENSORS --------
  if (requestedState == CALIBRATE_SENSORS) {
    // PC will keep you in this state while calibration is ongoing.
    // runStateMachine() will do the work & send telemetry.
    currentState = CALIBRATE_SENSORS;
    return;
  }

  // -------- READING (one-shot sensor refresh) --------
  if (requestedState == READING) {
    _sensors.readIMUData();
    _sensors.readAS5600Angle();
    currentState = IDLE;
    return;
  }

  // -------- IDLE --------
  if (requestedState == IDLE) {
    // No active motion command; motor can be commanded zero torque here if you want:
    // Ankle.torqueClosedLoopCommand(0);
    currentState = IDLE;
    return;
  }

  // -------- CONTROL = PURE TORQUE MODE --------
  if (requestedState == CONTROL) {
    // map float torqueReq (from PC) to int16_t for motor
    int16_t tau_cmd = (int16_t)(torqueReq);

    // apply direction: dirFlag = 0 -> reverse, 1 -> forward
    if (requestedDir == -1) {
      tau_cmd = -tau_cmd;
    }

    targetTorque       = tau_cmd;       // for telemetry
    Ankle.targetTorque = targetTorque;

    // PURE torque mode: no speed command here
    Ankle.torqueClosedLoopCommand(Ankle.targetTorque);

    // stay in CONTROL so GUI sees we are in torque mode
    currentState = CONTROL;
    return;
  }

  // -------- SPEED = PURE SPEED MODE --------
  if (requestedState == SPEED) {
    // signed speed in whatever units your RMDX8 library expects
    float signedSpeedCmd = speedReq * (float)requestedDir;

    // if your motor expects e.g. deg/s * 100, scale here:
    Ankle.targetSpeed = (int32_t)(signedSpeedCmd);

    // call your library's speed command (already using it)
    Ankle.sendSpeedCommand();

    // you can either stay in SPEED or go back to IDLE after one-shot
    currentState = SPEED;    // or IDLE; pick what you prefer
    return;
  }

  // -------- default --------
  currentState = IDLE;
}

// =================== SERIAL RX PARSER ===================
//
// We parse frames coming from the PC GUI.
// Layout:
//   [0] START_BYTE (0xAA)
//   [1] packet_id  (0x20)
//   [2] LEN (N)
//   [3..3+N-1] PAYLOAD
//   [3+N] CHECKSUM
//
// CHECKSUM = (packet_id + LEN + sum(PAYLOAD)) % 256
//
// NEW CONTROL PAYLOAD FORMAT (no safe_flag):
//   byte0    = cmd_id
//   byte1..4 = torque float32 LE
//   byte5..8 = speed  float32 LE
//   byte9    = dir_flag (0 or 1)
//
// Simple cmds like IDLE / INIT may just be [cmd_id] only (N=1).
//
void handleSerialInput() {
  // tiny state machine
  static enum {
    RX_WAIT_START,
    RX_GOT_ID,
    RX_GOT_LEN,
    RX_GOT_PAYLOAD,
    RX_GOT_CHECKSUM
  } rxState = RX_WAIT_START;

  static uint8_t rxPacketID = 0;
  static uint8_t rxLen      = 0;
  static uint8_t rxBuf[64];
  static uint8_t rxPos      = 0;

  while (Serial.available() > 0) {
    uint8_t b = (uint8_t)Serial.read();

    switch (rxState) {

      case RX_WAIT_START:
        if (b == START_BYTE) {
          rxState = RX_GOT_ID;
        }
        break;

      case RX_GOT_ID:
        rxPacketID = b;
        rxState = RX_GOT_LEN;
        break;

      case RX_GOT_LEN:
        rxLen = b;
        rxPos = 0;
        rxState = RX_GOT_PAYLOAD;
        break;

      case RX_GOT_PAYLOAD:
        rxBuf[rxPos++] = b;
        if (rxPos >= rxLen) {
          rxState = RX_GOT_CHECKSUM;
        }
        break;

      case RX_GOT_CHECKSUM: {
        uint8_t receivedChecksum = b;

        // calculate checksum
        uint16_t calc = 0;
        calc += rxPacketID;
        calc += rxLen;
        for (uint8_t i = 0; i < rxLen; i++) {
          calc += rxBuf[i];
        }
        uint8_t calcChecksum = (uint8_t)(calc % 256); // we devide the sum by 256 and keep the remainder 

        if (calcChecksum == receivedChecksum) {
          // valid frame -> decode
          if (rxLen >= 1) {
            uint8_t cmd_id = rxBuf[0];

            float torqueReq = 0.0f;
            float speedReq  = 0.0f;
            int8_t dirFlag  = 1;

            // torque at bytes 1..4
            if (rxLen >= 5) {
              union { float f; uint8_t b[4]; } ut;
              ut.b[0] = rxBuf[1];
              ut.b[1] = rxBuf[2];
              ut.b[2] = rxBuf[3];
              ut.b[3] = rxBuf[4];
              torqueReq = ut.f;
            }

            // speed at bytes 5..8
            if (rxLen >= 9) {
              union { float f; uint8_t b[4]; } us;
              us.b[0] = rxBuf[5];
              us.b[1] = rxBuf[6];
              us.b[2] = rxBuf[7];
              us.b[3] = rxBuf[8];
              speedReq = us.f;
            }

            // dirFlag at byte 9
            if (rxLen >= 10) {
              dirFlag = (int8_t)rxBuf[9];
            }

            applyCommandFromPC(cmd_id, torqueReq, speedReq, dirFlag);
          }
        }

        rxState = RX_WAIT_START;
      } break;
    }
  }
}

// =================== Tiny State Machine Hook ===================
//
// Most actions are immediate in applyCommandFromPC().
// This is here in case you want background behaviors later.
void runStateMachine() {
  switch (currentState) {
    case IDLE:
      // idle does nothing, motor should be at 0 torque
      break;

    case INITIALIZING:
      // we already handle init in applyCommandFromPC()
      break;

    case CONTROL:
      // we immediately handled then forced IDLE
      
      break;

    case CALIBRATE_SENSORS: {
      // Run calibration routines (can be non-blocking internally)
      _sensors.calibrateAS5600();
      _sensors.calibrateBNO055();

      // We already call updateSensorsFromHardware() in loop()
      // so calibration levels (cSys_level, cG_level, etc.) get refreshed.
      updateSensorsFromHardware();
      sendStatusToPC();

      const bool imu_full = (cSys_level == 3 && cG_level == 3 && cA_level == 3 && cM_level == 3);
      const bool as_ok    = (getAS5600CalFlag() == 1);

      

      if (AUTO_EXIT_WHEN_FULLY_CAL && imu_full && as_ok) {
        currentState = IDLE;
      }
    }break;


    case EMERGENCY_STOP:
      // handled in applyCommandFromPC()
      break;

    case START:
      // startup goes to IDLE
      currentState = IDLE;
      break;

    case READING:
      // handled in applyCommandFromPC()
      break;

    case OFFSETTING:
      // handled in applyCommandFromPC()
      break;

    default:
      currentState = IDLE;
      break;
  }
}

// =================== setup() / loop() ===================
void setup() {
  Serial.begin(115200);          // match your Python BAUD = 115200

  init_Status = 1.0;
  Wire.begin();
  Wire.setClock(400000);         // speed up I2C for BNO055/AS5600 if wiring is short & solid

  Ankle.begin();

  if (!bno.begin()) {
    // handle IMU missing if you care
  }
  bno.setExtCrystalUse(true);

  _sensors.readIMUData();
  _sensors.readAS5600Angle();

  currentState = START;
  lastStatusTime = millis();
}

void loop() {
  // 1. receive & act on PC commands
  handleSerialInput();

  // 2. optional background state logic
  runStateMachine();

  // 3. refresh sensor snapshot
  updateSensorsFromHardware();

  // 4. send telemetry periodically
  unsigned long now = millis();
  if (now - lastStatusTime >= 10) { // ~100 Hz target
    lastStatusTime = now;
    sendStatusToPC();
  }
}
