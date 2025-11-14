#ifndef IMU_SENSORS_H
#define IMU_SENSORS_H

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>


// Sensor object for BNO055
extern Adafruit_BNO055 bno;

// I2C address and register for AS5600
#define AS5600_ADDR 0x36
#define RAW_ANGLE_MSB 0x0C

class AS5600 {
private:
    const int AS5600_ADDRESS = 0x36;
    const int RAW_ANGLE_REGISTER = 0x0C;
    float offset = 0.0;

public:
    void begin() {
        Wire.begin();
    }

    // Read raw angle from AS5600
    uint16_t readRawAngle() {
        Wire.beginTransmission(AS5600_ADDRESS);
        Wire.write(RAW_ANGLE_REGISTER);
        Wire.endTransmission();
        
        Wire.requestFrom(AS5600_ADDRESS, 2);
        
        if(Wire.available() <= 2) {
            uint16_t rawAngle = (Wire.read() << 8) | Wire.read();
            return rawAngle & 0x0FFF; // 12-bit resolution
        }
        return 0;
    }

    // Convert raw angle to degrees
    float convertToDegrees(uint16_t rawAngle) {
        return (rawAngle * 360.0) / 4096.0;
    }

    // Set zero position (offset)
    void setZeroPosition() {
        uint16_t currentRawAngle = readRawAngle();
        offset = convertToDegrees(currentRawAngle);
    }

    // Get current angle with offset compensation
    float getCurrentAngle() {
        uint16_t rawAngle = readRawAngle();
        float currentAngle = convertToDegrees(rawAngle);
        
        // Apply offset
        float compensatedAngle = currentAngle - offset;
        
        
        // Normalize angle to 0-360 range
        if (compensatedAngle < 0) {
            compensatedAngle += 360.0;
        } else if (compensatedAngle >= 360.0) {
            compensatedAngle -= 360.0;
        }
        
        return compensatedAngle;
    }
};

class AnkleSensors{
public:


    bool enc_ok = false;
    //public variables

    float quat_x, quat_y, quat_z, quat_w;
    float yaw, pitch, roll;
    float accelX, accelY, accelZ;
    float gyroX, gyroY, gyroZ;

    float AS5600_angle;
    float prev_AS5600_angle;

    /**
     * @brief Creates an instance of the Adafruit_BNO055 IMU sensor.
     *
     * This object allows communication with the BNO055 sensor, providing access to
     * orientation, acceleration, and other sensor data. The constructor takes an
     * optional sensor ID parameter, which can be used to identify the sensor in
     * multi-sensor setups.
     *
     * @param sensorID The unique identifier for the sensor instance (default is 55).
     */
    


    // ===== BNO055 CALIBRATION FUNCTION =====
    void calibrateBNO055() {
        uint8_t sys, gyro, accel, mag;
        bno.getCalibration(&sys, &gyro, &accel, &mag);
            //Serial.print("Calib - SYS: "); Serial.print(sys);
            //Serial.print(" | GYRO: "); Serial.print(gyro);
            //Serial.print(" | ACCEL: "); Serial.print(accel);
            //Serial.print(" | MAG: "); Serial.println(mag);
        delay(1000);
    }
    // ===== AS5600 "CALIBRATION" CHECK =====
    void calibrateAS5600() {
        Wire.beginTransmission(AS5600_ADDR);
        if (Wire.endTransmission() == 0) {
            Serial.println("AS5600 Detected and Ready.");
        } else {
            Serial.println("AS5600 Not Detected! Check wiring.");
            while (1); // Halt if not found
        }
    }
    uint16_t zeroOffset = 0;

    uint16_t readAS5600RawAngle() {
    Wire.beginTransmission(AS5600_ADDR);
    Wire.write(RAW_ANGLE_MSB);
    Wire.endTransmission();
    Wire.requestFrom(AS5600_ADDR, 2);
    uint16_t angle = Wire.read() << 8 | Wire.read();
    return angle & 0x0FFF; // 12-bit mask
    }
    
    
    // ===== READ IMU DATA =====
    void readIMUData() {
        imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
        imu::Vector<3> accel = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
        imu::Vector<3> gyro = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
        imu::Quaternion quat = bno.getQuat();
        // Assign quaternion values to variables
        quat_w = quat.w();
        quat_x = quat.x();
        quat_y = quat.y();
        quat_z = quat.z();
        // Assign linear acceleration values to variables
        accelX = accel.x();
        accelY = accel.y();
        accelZ = accel.z();
        // Assign gyroscope values to variables
        gyroX = gyro.x();
        gyroY = gyro.y();
        gyroZ = gyro.z();

        // Assign Euler angles to variables (in degrees)

        /*
        // Print quaternion values to Serial Monitor
        Serial.print("Assigned Quaternion - W: "); Serial.print(quat_w);
        Serial.print(" | X: "); Serial.print(quat_x);
        Serial.print(" | Y: "); Serial.print(quat_y);
        Serial.print(" | Z: "); Serial.println(quat_z);    

        Serial.print("Orientation - Yaw: "); Serial.print(euler.x());
        Serial.print(" | Pitch: "); Serial.print(euler.y());
        Serial.print(" | Roll: "); Serial.println(euler.z());

        Serial.print("Accel [m/s^2] - X: "); Serial.print(accel.x());
        Serial.print(" | Y: "); Serial.print(accel.y());
        Serial.print(" | Z: "); Serial.println(accel.z());

        Serial.print("Gyro [dps] - X: "); Serial.print(gyro.x());
        Serial.print(" | Y: "); Serial.print(gyro.y());
        Serial.print(" | Z: "); Serial.println(gyro.z());
        Serial.println();
        */

    }

    // ===== AS5600 OFFSET SETTING =====
    AS5600 encoder; 
    void offSetting_AS5600() {
        encoder.begin();
        encoder.setZeroPosition();
        Serial.println("AS5600 Zero Position Set.");
    }
    float readAS5600Angle() {
        return encoder.getCurrentAngle();
    }

};

#endif
