#include <PID_v1.h>

double setpoint1, input1, output1;
PID myPID1(&input1, &output1, &setpoint1, 700, 2000, 10, DIRECT);

double setpoint2, input2, output2;
PID myPID2(&input2, &output2, &setpoint2, 50, 150, 2, DIRECT);

double setpoint3, input3, output3;
PID myPID3(&input3, &output3, &setpoint3, 0.005, 0.01, 0.0001, DIRECT);

// Initialize
unsigned long period1 = 1000000000; // [us/rot]
unsigned long period2 = 1000000000; // [us/rot]

unsigned long t_prev1 = micros();
unsigned long t_prev2 = micros();

double diff = 1.0;
long t_diff = 0;

unsigned int count1 = 0;
unsigned int count2 = 0;

double residual1 = 0.0f;
double residual2 = 0.0f;

void measurePeriod1()
{
    unsigned long t_now = micros();
    period1 = (t_now - t_prev1);
    t_prev1 = t_now;

    t_diff = t_prev1 - t_prev2; // min(t_prev1 - t_prev2, (t_prev2 + 41667) - t_prev1);

    double freq1 = 1000000.0 / (double)(period1);
    residual1 += (freq1 - 24);

    count1++;
}

void measurePeriod2()
{
    static bool flag = true;
    if (flag)
    {
        unsigned long t_now = micros();
        period2 = (t_now - t_prev2);
        t_prev2 = t_now;

        double freq2 = 1000000.0 / (double)(period2);

        residual2 += (freq2 - 120);
        count2++;
    }
    flag = !flag;
}

void setup()
{
    Serial.begin(9600);
    pinMode(LED_BUILTIN, OUTPUT);

    analogWriteResolution(12);

    pinMode(5, OUTPUT);
    // pinMode(6, OUTPUT);
    analogWrite(5, 1000);
    // digitalWrite(6, 0);
    pinMode(2, INPUT);
    attachInterrupt(digitalPinToInterrupt(2), measurePeriod1, RISING);

    delay(1);

    pinMode(6, OUTPUT);
    // pinMode(10, OUTPUT);
    analogWrite(6, 1000);
    // digitalWrite(10, 0);
    pinMode(3, INPUT);
    attachInterrupt(digitalPinToInterrupt(3), measurePeriod2, RISING);

    double coff = 1.002154632459789;
    coff = 1.0;
    setpoint1 = 24 * coff;
    setpoint2 = 120 * coff;
    setpoint3 = 0;

    myPID1.SetMode(AUTOMATIC);
    myPID2.SetMode(AUTOMATIC);
    myPID3.SetMode(AUTOMATIC);

    myPID1.SetOutputLimits(0, 4095);
    myPID2.SetOutputLimits(0, 4095);
    myPID3.SetOutputLimits(-4095, 4095);

    delay(2000);
}

void loop()
{
    delay(8);

    double freq1 = 1000000.0 / (double)(period1);
    double freq2 = 1000000.0 / (double)(period2);

    input1 = freq1;
    input2 = freq2;

    myPID1.Compute();
    myPID2.Compute();

    analogWrite(5, (int)output1);
    analogWrite(6, (int)output2);

    Serial.print(freq1);
    Serial.print(", ");
    Serial.print(freq2);
    Serial.print(", ");
    Serial.print(output1);
    Serial.print(", ");
    Serial.print(output2);
    Serial.print(", ");
    Serial.print(count1);
    Serial.print(", ");
    Serial.print(count2);
    Serial.print(", ");
    Serial.print(count1 - count2);
    Serial.print(", ");
    Serial.print(t_diff);
    // Serial.print(", ");
    // Serial.print((double) t_diff / 8333.333 * 36);

    // Serial.print(residual1);
    // Serial.print(", ");
    // Serial.print(residual2);
    Serial.println();
}
