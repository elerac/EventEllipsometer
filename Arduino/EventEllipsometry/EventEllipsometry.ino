#include <PID_v1.h>
#define DEBUG 1

// Baud rate
unsigned long baudRate = 115200;

// Pin assignment
pin_size_t pinIn_x1 = 2;
pin_size_t pinOut_x1 = 5;
pin_size_t pinIn_x5 = 3;
pin_size_t pinOut_x5 = 6;
pin_size_t pinOut_trig = 12;

// PID parameters
double setpoint_x1 = 24.0;
double setpoint_x5 = setpoint_x1 * 5.0;
double input_x1, output_x1;
double input_x5, output_x5;
PID myPID_x1(&input_x1, &output_x1, &setpoint_x1, 50, 150, 0.5, DIRECT);
PID myPID_x5(&input_x5, &output_x5, &setpoint_x5, 50, 150, 0.5, DIRECT);

// Initialize
unsigned long period_x1 = 1000000000; // [us/rot]
unsigned long period_x5 = 1000000000; // [us/rot]
unsigned int count_x1 = 0;
unsigned int count_x5 = 0;
unsigned long t_prev_x1 = micros();
unsigned long t_prev_x5 = micros();
int trigState = LOW;

void updatePeriod_x1()
{
    static bool flag = true;
    if (flag)
    {
        unsigned long t_now = micros();
        period_x1 = (t_now - t_prev_x1);
        t_prev_x1 = t_now;
        count_x1++;
    }
    flag = !flag;

    // Trigger
    if (trigState == LOW)
    {
        trigState = HIGH;
        digitalWrite(pinOut_trig, trigState);
    }
}

void updatePeriod_x5()
{
    static bool flag = true;
    if (flag)
    {
        unsigned long t_now = micros();
        period_x5 = (t_now - t_prev_x5);
        t_prev_x5 = t_now;
        count_x5++;
    }
    flag = !flag;

    // Trigger
    if (trigState == HIGH)
    {
        trigState = LOW;
        digitalWrite(pinOut_trig, trigState);
    }
}

void setup()
{
    // Digital out settings
    pinMode(pinOut_trig, OUTPUT);
    digitalWrite(pinOut_trig, trigState);

    // Analog out settings
    analogWriteResolution(12);
    pinMode(pinOut_x1, OUTPUT);
    pinMode(pinOut_x5, OUTPUT);

    // Interrupt settings
    pinMode(pinIn_x1, INPUT);
    attachInterrupt(digitalPinToInterrupt(pinIn_x1), updatePeriod_x1, RISING);
    pinMode(pinIn_x5, INPUT);
    attachInterrupt(digitalPinToInterrupt(pinIn_x5), updatePeriod_x5, RISING);

    // PID settings
    myPID_x1.SetMode(AUTOMATIC);
    myPID_x5.SetMode(AUTOMATIC);
    myPID_x1.SetOutputLimits(0, 4095);
    myPID_x5.SetOutputLimits(0, 4095);

    // Calibration
    double coff = 1.002154632459789;
    coff = 1.0;
    setpoint_x1 = setpoint_x1 * coff;
    setpoint_x5 = setpoint_x5 * coff;

#if DEBUG
    Serial.begin(baudRate);
#endif
}

void loop()
{
    delay(8);
    unsigned long t_now = micros();

    double freq_x1 = 1000000.0 / (double)(period_x1);
    double freq_x5 = 1000000.0 / (double)(period_x5);

    input_x1 = freq_x1;
    input_x5 = freq_x5;

    myPID_x1.Compute();
    myPID_x5.Compute();

    analogWrite(pinOut_x1, (int)output_x1);
    analogWrite(pinOut_x5, (int)output_x5);

    unsigned long t_end_pid = micros();

#if DEBUG
    Serial.print(t_now);
    Serial.print(", ");
    Serial.print(freq_x1, 1);
    Serial.print(", ");
    Serial.print(freq_x5, 1);
    Serial.print(", ");
    Serial.print(output_x1, 0);
    Serial.print(", ");
    Serial.print(output_x5, 0);
    Serial.print(", ");
    Serial.print(count_x1);
    Serial.print(", ");
    Serial.print(count_x5);
    Serial.print(", ");
    Serial.print(t_end_pid - t_now);
    Serial.print(", ");
    unsigned long t_end_serial = micros();
    Serial.println(t_end_serial - t_end_pid);
#endif
}
