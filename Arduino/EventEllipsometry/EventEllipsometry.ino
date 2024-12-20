#include <PID_v1.h>
#define DEBUG 0
#define DEBUG_LITE 1

// Baud rate
unsigned long baudRate = 115200;

// Pin assignment
pin_size_t pinIn_x1 = 2;
pin_size_t pinOut_x1 = 5;
pin_size_t pinIn_x5 = 3;
pin_size_t pinOut_x5 = 6;
pin_size_t pinOut_trig = 12;

// PID parameters
double setpoint_x1 = 15.0;
double setpoint_x5 = setpoint_x1 * 5.0;
double input_x1, output_x1;
double input_x5, output_x5;
PID myPID_x1(&input_x1, &output_x1, &setpoint_x1, 160, 900, 0.1, DIRECT);
PID myPID_x5(&input_x5, &output_x5, &setpoint_x5, 160, 900, 0.1, DIRECT);

// Initialize
unsigned long period_x1 = 1000000000; // [us/rot]
unsigned long period_x5 = 1000000000; // [us/rot]
unsigned int count_x1 = 0;
unsigned int count_x5 = 0;
unsigned long t_prev_x1 = micros();
unsigned long t_prev_x5 = micros();
int trigState = LOW;

unsigned long t_trig_x1 = 0;
unsigned long t_trig_x5 = 0;

void updatePeriod_x1()
{
    static bool flag = true;

    // Trigger
    if (trigState == LOW)
    {
        trigState = HIGH;
        digitalWrite(pinOut_trig, trigState);
        t_trig_x1 = micros();
    }

    if (flag)
    {
        unsigned long t_now = micros();
        period_x1 = (t_now - t_prev_x1);
        t_prev_x1 = t_now;
        count_x1++;
    }
    flag = !flag;
}

void updatePeriod_x5()
{
    static bool flag = true;

    // Trigger
    if (trigState == HIGH)
    {
        trigState = LOW;
        digitalWrite(pinOut_trig, trigState);
        t_trig_x5 = micros();
    }

    if (flag)
    {
        unsigned long t_now = micros();
        period_x5 = (t_now - t_prev_x5);
        t_prev_x5 = t_now;
        count_x5++;
    }
    flag = !flag;
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

    Serial.begin(baudRate);

    delay(1000);
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

#if DEBUG_LITE
    Serial.print(t_now);
    Serial.print(", ");
    Serial.print(freq_x1, 2);
    Serial.print(", ");
    Serial.print(freq_x5, 2);
    Serial.print(", ");
    Serial.print(output_x1, 0);
    Serial.print(", ");
    Serial.print(output_x5, 0);
    Serial.println();
#endif

#if DEBUG
    Serial.print(t_now);
    Serial.print(", ");
    Serial.print(freq_x1, 2);
    Serial.print(", ");
    Serial.print(freq_x5, 2);
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
