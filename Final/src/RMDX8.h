#ifndef _RMDX8MOTOR_H_
#define _RMDX8MOTOR_H_

#include <SPI.h>
#include <mcp2515.h>
#include <Arduino.h>
#include <Wire.h>

class RMDX8Motor {
public:
    // ====== PUBLIC VARIABLES ======
    int32_t gearRatio;
    int32_t desiredSpeed;
    int32_t targetSpeed;
    int16_t torque;
    int16_t targetTorque;
    int32_t position_deg100; // Position in degrees * 100

    // ====== CONSTRUCTOR ======
    RMDX8Motor(uint8_t csPin)
        : mcp2515(csPin)
    {
        gearRatio = 6;
        desiredSpeed = 0; // convert to dps * 100
        targetSpeed = desiredSpeed * gearRatio *100; // convert to dps * 100
        targetTorque = 0;
        torque = 0;
        position_deg100 = 0; // Initialize position
    }

    // ====== INIT ======
    void begin() {
        SPI.begin();
        mcp2515.reset();
        mcp2515.setBitrate(CAN_1000KBPS, MCP_8MHZ);
        mcp2515.setNormalMode();
        //Serial.println("MCP2515 Initialized");
    }

    // ====== SPEED COMMAND ======
    void sendSpeedCommand() {
        struct can_frame msg;
        msg.can_id  = 0x141;
        msg.can_dlc = 8;
        msg.data[0] = 0xA2;
        msg.data[1] = 0x00;
        msg.data[2] = 0x00;
        msg.data[3] = 0x00;
        msg.data[4] = (uint8_t)(targetSpeed & 0xFF);
        msg.data[5] = (uint8_t)((targetSpeed >> 8) & 0xFF);
        msg.data[6] = (uint8_t)((targetSpeed >> 16) & 0xFF);
        msg.data[7] = (uint8_t)((targetSpeed >> 24) & 0xFF);

        mcp2515.sendMessage(&msg);
        //Serial.println("Sent speed command.");
    }

    // ====== READ STATUS ======
    void readMotorStatus() {
        struct can_frame msg;
        msg.can_id  = 0x141;
        msg.can_dlc = 8;
        msg.data[0] = 0x9C;
        for (int i = 1; i < 8; i++) msg.data[i] = 0x00;

        mcp2515.sendMessage(&msg);
        delay(10);

        struct can_frame response;
        if (mcp2515.readMessage(&response) == MCP2515::ERROR_OK) {
            if (response.can_id == 0x141 && response.data[0] == 0x9C) {
                int8_t temperature = response.data[1];
                int16_t torque_raw = (response.data[2] | (response.data[3] << 8));
                int16_t speed_dps  = (response.data[4] | (response.data[5] << 8));
                uint16_t encoder_counts = (response.data[6] | (response.data[7] << 8));

                float torque_A = (torque_raw * 33.0) / 2048.0;
                float angle_deg = encoder_counts * 360.0 / 65535.0;

                //Serial.println("---- Motor Status ----");
                //Serial.print("Temperature: "); Serial.println(temperature);
                //Serial.print("Torque Current (A): "); Serial.println(torque_A, 2);
                //Serial.print("Speed (deg/sec): "); Serial.println(speed_dps / gearRatio);
                //Serial.print("Angle (degrees): "); Serial.println(angle_deg, 2);
                //Serial.println("-----------------------");
            }
        } else {
            //Serial.println("No response received");
        }

        //delay(500);
    }

    // ====== TORQUE COMMAND ======
    void torqueClosedLoopCommand(int16_t torque) {
        struct can_frame msg;
        msg.can_id  = 0x141;
        msg.can_dlc = 8;
        msg.data[0] = 0xA1;
        msg.data[1] = 0x00;
        msg.data[2] = 0x00;
        msg.data[3] = 0x00;
        msg.data[4] = (uint8_t)(torque & 0xFF);
        msg.data[5] = (uint8_t)((torque >> 8) & 0xFF);
        msg.data[6] = 0x00;
        msg.data[7] = 0x00;

        mcp2515.sendMessage(&msg);
        //Serial.println("Torque command sent.");

        delay(10);
        //readMotorStatus();
        struct can_frame response;
        if (mcp2515.readMessage(&response) == MCP2515::ERROR_OK) {
            if (response.can_id == 0x141 && response.data[0] == 0xA1) {
                int8_t temperature = response.data[1];
                int16_t torque_raw = (response.data[2] | (response.data[3] << 8));
                int16_t speed_dps  = (response.data[4] | (response.data[5] << 8));
                uint16_t encoder_counts = (response.data[6] | (response.data[7] << 8));

                float torque_A = (torque_raw * 33.0) / 2048.0;
                float angle_deg = encoder_counts * 360.0 / 65535.0;

                //Serial.println("---- Motor Status ----");
                //Serial.print("Temperature: "); Serial.println(temperature);
                //Serial.print("Torque Current (A): "); Serial.println(torque_A, 2);
                //Serial.print("Speed (deg/sec): "); Serial.println(speed_dps / gearRatio);
                //Serial.print("Angle (degrees): "); Serial.println(angle_deg, 2);
                //Serial.println("-----------------------");
            }
        } else {
            //Serial.println("No response received");
        }

        
    }

    // ====== MOTOR CONTROL COMMANDS ======
    void motorOffCommand() {
        sendSimpleCommand(0x80, "Motor Off");
    }

    void motorStopCommand() {
        sendSimpleCommand(0x81, "Motor Stop");
    }

    void motorResumeCommand() {
        sendSimpleCommand(0x88, "Motor Resume");
    }

    // ====== READ PID ======
    void readPIDDataCommand() {
        struct can_frame msg;
        msg.can_id  = 0x141;
        msg.can_dlc = 8;
        msg.data[0] = 0x30;
        for (int i = 1; i < 8; i++) msg.data[i] = 0x00;

        mcp2515.sendMessage(&msg);
        //Serial.println("Read PID command sent.");

        delay(10);

        struct can_frame response;
        if (mcp2515.readMessage(&response) == MCP2515::ERROR_OK) {
            if (response.can_id == 0x141 && response.data[0] == 0x31) {
                uint8_t Kp = response.data[1];
                uint8_t Ki = response.data[2];
                uint8_t Kd = response.data[3];

                //.Serial.println("---- PID Data ----");
                //Serial.print("Kp: "); Serial.println(Kp);
                //Serial.print("Ki: "); Serial.println(Ki);
                //Serial.print("Kd: "); Serial.println(Kd);
                //Serial.println("------------------");
            } else {
                //Serial.println("Unexpected PID response");
            }
        } else {
            //Serial.println("No PID response received");
        }

        delay(500);
    }

    void setMotorPosition( int32_t position_deg100) {
        struct can_frame msg;
        msg.can_id  = 0x141;           // Motor CAN ID
        msg.can_dlc = 8;

        msg.data[0] = 0xA3;            // Position control command
        msg.data[1] = 0x00;
        msg.data[2] = 0x00;
        msg.data[3] = 0x00;

        msg.data[4] = (uint8_t)(position_deg100 & 0xFF);
        msg.data[5] = (uint8_t)((position_deg100 >> 8) & 0xFF);
        msg.data[6] = (uint8_t)((position_deg100 >> 16) & 0xFF);
        msg.data[7] = (uint8_t)((position_deg100 >> 24) & 0xFF);

        mcp2515.sendMessage(&msg);    // Assuming you wrap mcp2515.sendMessage in RMDX8Motor
    }


private:
    MCP2515 mcp2515;

    void sendSimpleCommand(uint8_t command, const char* description) {
        struct can_frame msg;
        msg.can_id  = 0x141;
        msg.can_dlc = 8;
        for (int i = 0; i < 8; i++) msg.data[i] = 0x00;
        msg.data[0] = command;

        mcp2515.sendMessage(&msg);

        Serial.print(description);
        //Serial.println(" command sent.");
        delay(500);
    }
};

#endif // _RMDX8MOTOR_H_
