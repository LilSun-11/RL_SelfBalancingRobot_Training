#include <Arduino.h>
#include <Wire.h>
#include <math.h>

// =============================================================================
//  Balancing_Robot_V2_PID — dieu khien bang PID kinh dien (thay cho policy MLP)
// =============================================================================
//  Ban goc (thu muc Balancing_Robot_V2) dung mang no-ron (policy_weights.h) de
//  suy ra hanh dong tu quan sat. Ban nay THAY THE hoan toan phan do bang 1 vong
//  PID don gian:
//     setpoint = 0 do (PID_SETPOINT_DEG)  -> robot can bang thang dung
//     input    = sai so goc nghieng: error = setpoint - pitchDeg
//     output   = 1 gia tri duty (KHONG PHAN BIET 2 banh) ap CHUNG cho ca 2 dong co
//  Phan cam bien (LSM6DS3), bo loc goc, driver dong co (PWM + pulse-density
//  shaper chong stiction o duty thap) duoc giu y nguyen tu ban policy vi day la
//  phan cung, khong lien quan thuat toan dieu khien.
// =============================================================================

struct Shaper { float charge=0, remain=0, coast=0; int lastDir=0; };
Shaper shpL, shpR;

// =============================== USER CONFIG =================================
static const int SDA_PIN = 21, SCL_PIN = 22, LED_PIN = 27;   // LED_PIN: chan RGB (neopixel) bao trang thai, doi neu da dung GPIO27 viec khac
static const int ENA = 15, IN1 = 2, IN2 = 4;      // banh A (TRAI)
static const int ENB = 5, IN3 = 19, IN4 = 18;     // banh B (PHAI)

static const int PWM_FREQ = 20000, PWM_RES = 10;
static const uint32_t PWM_MAX_LEDC = (1u << PWM_RES) - 1u;
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  #define PWM_A ENA
  #define PWM_B ENB
#else
  static const int CH_ENA = 0, CH_ENB = 1;
  #define PWM_A CH_ENA
  #define PWM_B CH_ENB
#endif

float MIN_DUTY = 0.0f, MAX_DUTY = 0.9f, U_DEADBAND = 0.01f;
float LINEAR_START_U     = 0.10f;
float NEAR_BAL_MAX_DUTY  = 0.42f;
float NEAR_BAL_PITCH_DEG = 1.5f;
float NEAR_BAL_RATE_DPS  = 18.0f;
float U_SCALE     = 1.0f;
int   MOTOR_SIGN   = +1;
int   GYRO_SIGN    = +1;    // dao dau rieng toc do goc (gyro) so voi goc nghieng (accel)
int   PITCH_SIGN   = +1;    // dao dau CA goc nghieng lan toc do goc (dinh huong IMU nguoc)

// ---- PID ----
float PID_SETPOINT_DEG = 0.0f;   // goc dat: 0 do = than robot thang dung
float PID_KP = 0.4f;    // duty / do lech goc            (P)
float PID_KI = 1.00f;    // duty / (do*giay) tich luy sai so (I) -- mac dinh TAT (0), bat sau khi Kp/Kd on
float PID_KD = 0.01f;  // duty / (do/giay) nhan voi TOC DO GOC (rate) lam so hang D, xem giai thich duoi day
// Vi setpoint co dinh nen d(error)/dt = -d(pitch)/dt = -rateDps. Dung thang rateDps
// (co san tu gyro, it nhieu hon dao ham so cua error) lam so hang D la ky thuat
// chuan ("derivative on measurement") de tranh derivative-kick va khuech dai nhieu.
float pidIntegral = 0.0f;
float PID_I_MAX = 0.6f;   // gioi han |Ki*integral| (anti-windup, tinh theo duty)

const int CONTROL_HZ = 100;
const unsigned long LOOP_US = 1000000UL / CONTROL_HZ;
const float MAX_SAFE_TILT_DEG = 45.0f;
const float COMP_ALPHA = 0.98f;
// Khop nhip 1 tick loop (10ms @ 100Hz): gia tri nho hon LOOP_US se bi "lam tron len" thanh 1 tick
// vi motor chi duoc cap nhat 1 lan/tick, nen dat dung bang 1 tick de tranh sai lech (~2x) khong chu dinh.
const float PULSE_WIDTH_S = (float)LOOP_US*1e-6f, REVERSE_COAST_S = (float)LOOP_US*1e-6f;

// ------- LSM6DS3 -------
static const uint8_t REG_WHO_AM_I=0x0F, REG_CTRL1_XL=0x10, REG_CTRL2_G=0x11, REG_CTRL3_C=0x12;
static const uint8_t REG_OUTX_L_G=0x22, REG_OUTX_L_XL=0x28;
static const float ACC_SENS_G=0.000061f, GYRO_SENS_DPS=0.00875f;  // ODR=104Hz, accel +-2g, gyro 250dps
uint8_t IMU_ADDR = 0x6B;

// ------- state -------
float pitchDeg=0, rateDps=0, pitchFiltered=0, pitchOffsetDeg=0, gyroBiasX=0;
bool  filterSeeded=false, enabled=true;
float gAx,gAy,gAz,gGx;
float lastU=0, lastDutyFrac=0;
unsigned long lastLoopUs=0;

// ============================ LSM6DS3 I/O =====================================
void bmiWrite(uint8_t r,uint8_t v){ Wire.beginTransmission(IMU_ADDR); Wire.write(r); Wire.write(v); Wire.endTransmission(); }
bool bmiRead(uint8_t r,uint8_t*b,uint8_t n){
  Wire.beginTransmission(IMU_ADDR); Wire.write(r);
  if(Wire.endTransmission(false)!=0) return false;
  uint8_t g=Wire.requestFrom((int)IMU_ADDR,(int)n);
  for(uint8_t i=0;i<n && Wire.available();i++) b[i]=Wire.read();
  return g==n;
}
uint8_t bmiChipId(uint8_t a){
  Wire.beginTransmission(a); Wire.write(REG_WHO_AM_I);
  if(Wire.endTransmission(false)!=0) return 0;
  if(Wire.requestFrom((int)a,1)!=1) return 0;
  return Wire.read();
}
bool bmiInit(){
  uint8_t cand[2]={0x6B,0x6A}; bool found=false;   // SA0=1 -> 0x6B, SA0=0 -> 0x6A
  for(int i=0;i<2;i++){ if(bmiChipId(cand[i])==0x69){ IMU_ADDR=cand[i]; found=true; break; } }
  if(!found){ for(int i=0;i<2;i++){ Wire.beginTransmission(cand[i]); if(Wire.endTransmission()==0){ IMU_ADDR=cand[i]; found=true; break; } } }
  if(!found) return false;
  bmiWrite(REG_CTRL1_XL, 0x40);  // accel: 104 Hz, +-2g
  bmiWrite(REG_CTRL2_G,  0x40);  // gyro:  104 Hz, 250 dps
  bmiWrite(REG_CTRL3_C,  0x44);  // BDU=1, IF_INC=1 (auto tang dia chi thanh ghi)
  delay(50);
  return true;
}
void readSensors(){
  uint8_t ab[6], gb[6];
  if(!bmiRead(REG_OUTX_L_XL, ab, 6)) return;
  if(!bmiRead(REG_OUTX_L_G,  gb, 6)) return;
  int16_t rax=(int16_t)(((uint16_t)ab[1]<<8)|ab[0]);
  int16_t ray=(int16_t)(((uint16_t)ab[3]<<8)|ab[2]);
  int16_t raz=(int16_t)(((uint16_t)ab[5]<<8)|ab[4]);
  int16_t rgx=(int16_t)(((uint16_t)gb[1]<<8)|gb[0]);
  gAx=rax*ACC_SENS_G; gAy=ray*ACC_SENS_G; gAz=raz*ACC_SENS_G;
  gGx=rgx*GYRO_SENS_DPS;
}

// ====================== ANGLE =================================================
void updateAngle(float dt){
  readSensors();
  float accA = PITCH_SIGN * atan2f(gAy, gAz) * RAD_TO_DEG;
  float rate = PITCH_SIGN * GYRO_SIGN * (gGx - gyroBiasX);
  if(!filterSeeded){ pitchFiltered=accA; filterSeeded=true; }
  else pitchFiltered = COMP_ALPHA*(pitchFiltered + rate*dt) + (1.0f-COMP_ALPHA)*accA;
  pitchDeg = pitchFiltered - pitchOffsetDeg;
  while(pitchDeg >  180.0f) pitchDeg -= 360.0f;
  while(pitchDeg < -180.0f) pitchDeg += 360.0f;
  rateDps  = rate;
}
void calibrate(){
  Serial.println("# CALIB: GIU ROBOT DUNG THANG va YEN ~2s...");
  const int N=1000; double sA=0,sG=0;
  for(int i=0;i<N;i++){ readSensors(); sA+=PITCH_SIGN*atan2f(gAy,gAz)*RAD_TO_DEG; sG+=gGx; delay(2); }
  pitchOffsetDeg=(float)(sA/N); gyroBiasX=(float)(sG/N);
  filterSeeded=false; pidIntegral=0.0f; lastLoopUs=micros();
  Serial.printf("# CALIB offset=%.2f gyroBiasX=%.3f\n", pitchOffsetDeg, gyroBiasX);
}

// ====================== PID (setpoint=0, input=sai so goc, output=1 duty) ====
float pidUpdate(float dt){
  float error = PID_SETPOINT_DEG - pitchDeg;              // sai so goc nghieng (do)
  if(PID_KI > 1e-6f){
    pidIntegral += error*dt;
    float iLim = PID_I_MAX / PID_KI;                       // anti-windup: gioi han truoc khi nhan KI
    pidIntegral = constrain(pidIntegral, -iLim, iLim);
  } else {
    pidIntegral = 0.0f;                                    // Ki tat -> khong tich luy, tranh giat khi bat lai
  }
  float pTerm = PID_KP * error;
  float iTerm = PID_KI * pidIntegral;
  float dTerm = -PID_KD * rateDps;                         // xem giai thich o khai bao PID_KD phia tren
  return constrain(pTerm + iTerm + dTerm, -1.0f, 1.0f);
}

// ====================== MOTOR SHAPER (pulse-density) =========================
float shaperUpdate(Shaper &s, float signedU, bool near, float dt){
  dt = constrain(dt, 0.0001f, 0.020f);
  signedU = constrain(signedU, -1.0f, 1.0f);
  float absU = fabsf(signedU);
  float linStart = constrain(LINEAR_START_U, U_DEADBAND+0.001f, 1.0f);
  if(absU < U_DEADBAND){ s = Shaper{}; return 0.0f; }
  int reqDir = (signedU>0.0f) ? 1 : -1;
  if(s.lastDir!=0 && reqDir!=s.lastDir){
    s.charge=0; s.remain=0; s.coast=fmaxf(0.0f, REVERSE_COAST_S-dt); s.lastDir=reqDir; return 0.0f;
  }
  s.lastDir=reqDir;
  if(s.coast>0.0f){ s.coast=fmaxf(0.0f,s.coast-dt); return 0.0f; }
  float reqDuty;
  if(absU<linStart && MIN_DUTY>0.0f) reqDuty = MIN_DUTY*(absU/linStart);
  else { float nrm=(absU-linStart)/fmaxf(0.001f,1.0f-linStart);
         reqDuty = MIN_DUTY + constrain(nrm,0.0f,1.0f)*(MAX_DUTY-MIN_DUTY); }
  if(near) reqDuty = fminf(reqDuty, constrain(NEAR_BAL_MAX_DUTY, MIN_DUTY, MAX_DUTY));
  reqDuty = constrain(reqDuty, 0.0f, MAX_DUTY);
  float applied = reqDuty;
  if(MIN_DUTY>0.0f && reqDuty<MIN_DUTY){
    s.charge += (reqDuty/MIN_DUTY)*dt;
    if(s.remain<=0.0f && s.charge>=PULSE_WIDTH_S){ s.charge-=PULSE_WIDTH_S; s.remain=PULSE_WIDTH_S; }
    if(s.remain>0.0f){ applied=MIN_DUTY; s.remain=fmaxf(0.0f,s.remain-dt); }
    else applied=0.0f;
  } else { s.charge=0; s.remain=0; }
  if(applied<=0.0f) return 0.0f;
  return (reqDir>0) ? applied : -applied;
}
void writeMotorPins(int inA,int inB,int pwmHandle,float signedDuty){
  if(signedDuty==0.0f){ ledcWrite(pwmHandle,0); digitalWrite(inA,LOW); digitalWrite(inB,LOW); return; }
  bool fwd = signedDuty>0.0f;
  uint32_t d=(uint32_t)(fabsf(signedDuty)*PWM_MAX_LEDC+0.5f);
  digitalWrite(inA, fwd?HIGH:LOW); digitalWrite(inB, fwd?LOW:HIGH); ledcWrite(pwmHandle,d);
}
void motorCoast(){
  lastDutyFrac=0; shpL=Shaper{}; shpR=Shaper{};
  ledcWrite(PWM_A,0); ledcWrite(PWM_B,0);
  digitalWrite(IN1,LOW); digitalWrite(IN2,LOW);
  digitalWrite(IN3,LOW); digitalWrite(IN4,LOW);
}
// Ap CUNG 1 gia tri duty (tu PID) cho ca 2 banh. Da nhan U_SCALE truoc khi goi.
void driveWheels(float u, float dt){
  float cmd = constrain((float)MOTOR_SIGN*u, -1.0f, 1.0f);
  bool near = (fabsf(pitchDeg-PID_SETPOINT_DEG) <= NEAR_BAL_PITCH_DEG) &&
              (fabsf(rateDps) <= NEAR_BAL_RATE_DPS);
  float sdL = shaperUpdate(shpL, cmd, near, dt);
  float sdR = shaperUpdate(shpR, cmd, near, dt);
  writeMotorPins(IN1,IN2,PWM_A, sdL);
  writeMotorPins(IN3,IN4,PWM_B, sdR);
  lastDutyFrac = 0.5f*(fabsf(sdL)+fabsf(sdR));
}
void motorTest(){
  const uint32_t d=(uint32_t)(0.5f*PWM_MAX_LEDC);
  Serial.println("# TEST 2 banh THUAN 50% 1.5s (phai lan CUNG huong)...");
  digitalWrite(IN1,HIGH); digitalWrite(IN2,LOW); ledcWrite(PWM_A,d);
  digitalWrite(IN3,HIGH); digitalWrite(IN4,LOW); ledcWrite(PWM_B,d);
  delay(1500); motorCoast(); delay(400);
  Serial.println("# TEST 2 banh NGHICH 50% 1.5s...");
  digitalWrite(IN1,LOW); digitalWrite(IN2,HIGH); ledcWrite(PWM_A,d);
  digitalWrite(IN3,LOW); digitalWrite(IN4,HIGH); ledcWrite(PWM_B,d);
  delay(1500); motorCoast();
  Serial.println("# TEST xong. 2 banh nguoc nhau -> doi day dong co, hoac 'm -1'.");
  lastLoopUs=micros();
}

// ============================ SERIAL TUNER ===================================
void printParams(){
  Serial.printf("# PARAM scale=%.2f setpoint=%.2f MIN=%.2f MAX=%.2f dead=%.3f nearCap=%.2f "
                "Kp=%.4f Ki=%.4f Kd=%.4f MOTOR_SIGN=%d GYRO=%d PITCH=%d en=%d | %dHz PID\n",
                U_SCALE, PID_SETPOINT_DEG, MIN_DUTY, MAX_DUTY, U_DEADBAND, NEAR_BAL_MAX_DUTY,
                PID_KP, PID_KI, PID_KD, MOTOR_SIGN, GYRO_SIGN, PITCH_SIGN, enabled, CONTROL_HZ);
}
void parseCmd(char*s){
  char c=s[0]; float v=atof(s+1);
  switch(c){
    case 'a': U_SCALE=constrain(v,0.0f,1.0f); break;
    case 's': PID_SETPOINT_DEG=constrain(v,-20.0f,20.0f); break;    // goc dat (mac dinh 0)
    case 'n': MIN_DUTY=constrain(v,0.0f,MAX_DUTY); break;
    case 'l': MAX_DUTY=constrain(v,MIN_DUTY,1.0f); break;
    case 'b': U_DEADBAND=constrain(v,0.0f,0.95f); break;
    case 'p': PID_KP=fmaxf(v,0.0f); break;
    case 'k': PID_KI=fmaxf(v,0.0f); pidIntegral=0.0f; break;        // doi Ki -> reset tich phan cho an toan
    case 'd': PID_KD=fmaxf(v,0.0f); break;
    case 'm': MOTOR_SIGN  =(v>=0)?+1:-1; break;
    case 'g': GYRO_SIGN   =(v>=0)?+1:-1; filterSeeded=false; break;
    case 'i': PITCH_SIGN  =(v>=0)?+1:-1; filterSeeded=false; break; // dao dau goc nghieng (IMU nguoc chieu)
    case 'c': motorCoast(); calibrate(); break;
    case 't': motorCoast(); motorTest(); return;
    case 'x': enabled=false; motorCoast(); Serial.println("# STOP motor OFF"); break;
    case 'o': enabled=true; pidIntegral=0.0f; Serial.println("# GO motor ON"); break;
    case '?': default: printParams(); return;
  }
  printParams();
}
// ---- Ngat nhan UART (Serial.onReceive) ----
// Callback chay trong task su kien UART (khong phai vong loop), nen chi GOM dong va
// PHAN TICH; viec ap gia tri/chay lenh duoc day ve loop() qua co bao (applySerial) de
// bo 3 Kp,Ki,Kd doi CUNG luc giua 2 chu ky PID va cac lenh co delay (c, t) khong
// chay trong task UART.
//   "kp,ki,kd"  vd: 0.06,0,0.0015  -> dat ca 3 he so PID
//   con lai    -> lenh 1 ky tu cu (p/k/d/a/s/.../?) nhu truoc
portMUX_TYPE serialMux = portMUX_INITIALIZER_UNLOCKED;
volatile bool gainsPending=false, cmdPending=false;
float pendKp=0, pendKi=0, pendKd=0;
char  pendCmd[32];

void processLine(char*buf, uint8_t len){
  if(len==0) return;
  Serial.printf("# RX '%s'\n", buf);                      // echo de biet chac da nhan duoc gi
  float kp,ki,kd;
  if(strchr(buf,',')){
    if(sscanf(buf,"%f ,%f ,%f",&kp,&ki,&kd)==3 && kp>=0 && ki>=0 && kd>=0){
      portENTER_CRITICAL(&serialMux);
      pendKp=kp; pendKi=ki; pendKd=kd; gainsPending=true;
      portEXIT_CRITICAL(&serialMux);
    } else {
      Serial.println("# RX sai dinh dang, can: kp,ki,kd (vd 0.1,0.01,0.0015)");
    }
  } else if(!cmdPending){
    memcpy(pendCmd, buf, len+1); cmdPending=true;
  }
}
// Dang ky voi onlyOnTimeout=true: callback chi goi khi UART ngung nhan (~het 1 tin nhan),
// nen ket thuc callback cung duoc coi la het dong -> nhan duoc ca khi Line ending = None.
void onUartRx(){
  static char buf[32]; static uint8_t idx=0;
  while(Serial.available()){
    char c=Serial.read();
    if(c!='\n' && c!='\r'){ if(idx<sizeof(buf)-1) buf[idx++]=c; continue; }
    buf[idx]=0; processLine(buf, idx); idx=0;
  }
  buf[idx]=0; processLine(buf, idx); idx=0;
}
void applySerial(){
  if(gainsPending){
    portENTER_CRITICAL(&serialMux);
    PID_KP=pendKp; PID_KI=pendKi; PID_KD=pendKd; gainsPending=false;
    portEXIT_CRITICAL(&serialMux);
    pidIntegral=0.0f;                                      // doi Ki -> reset tich phan cho an toan
    printParams();
  }
  if(cmdPending){ parseCmd(pendCmd); cmdPending=false; }
}
void telemetry(){
  static unsigned long last=0;
  if(millis()-last < 100) return;
  last=millis();
  float error = PID_SETPOINT_DEG - pitchDeg;
  // Dong debug day du
  Serial.printf("# pitch=%6.2f rate=%7.1f err=%6.2f u=%+.3f duty=%4.0f%% Kp=%.4f Ki=%.4f Kd=%.4f %s\n",
                pitchDeg, rateDps, error, lastU, lastDutyFrac*100.0f,
                PID_KP, PID_KI, PID_KD, enabled ? "" : "[OFF]");
  // Dong so lieu "ten:gia_tri" rieng, dung format PlatformIO/Arduino Serial Plotter can de ve do thi
  // Serial.printf("pitch:%.2f,rate:%.1f,err:%.2f,u:%.3f,duty:%.1f\n",
  //               pitchDeg, rateDps, error, lastU, lastDutyFrac*100.0f);
}

// ================================ SETUP ======================================
void setup(){
  // Serial.begin(921600);   // baud cao de telemetry() day xong nhanh, tranh block vong lap dieu khien
  Serial.begin(115200);
  #if ESP_ARDUINO_VERSION_MAJOR >= 3
  Serial.setTxTimeoutMs(2);   // buffer TX day -> bo bot du lieu thay vi treo ca vong lap (xem ban policy)
#endif
  Serial.onReceive(onUartRx, true);   // ngat nhan UART: gui "kp,ki,kd" de chinh PID
  delay(500);
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(ENA, PWM_FREQ, PWM_RES);
  ledcAttach(ENB, PWM_FREQ, PWM_RES);
#else
  ledcSetup(CH_ENA, PWM_FREQ, PWM_RES); ledcAttachPin(ENA, CH_ENA);
  ledcSetup(CH_ENB, PWM_FREQ, PWM_RES); ledcAttachPin(ENB, CH_ENB);
#endif
  pinMode(IN1,OUTPUT); pinMode(IN2,OUTPUT);
  pinMode(IN3,OUTPUT); pinMode(IN4,OUTPUT);
  motorCoast();

  Wire.begin(SDA_PIN,SCL_PIN); Wire.setClock(400000);
  Wire.setTimeOut(2);   // gioi han cho toi da 2ms/giao dich I2C, tranh 1 lan NACK/bus loi treo ca chu ky dieu khien
  if(!bmiInit()){
    Serial.println("# LOI: khong thay LSM6DS3 (SDA=21/SCL=22, 3V3, GND)");
    while(true){ neopixelWrite(LED_PIN,80,0,0); delay(150); neopixelWrite(LED_PIN,0,0,0); delay(150); }
  }
  calibrate();

  Serial.printf("# READY PID setpoint=%.1f Kp=%.4f Ki=%.4f Kd=%.4f, %dHz. Go '?' xem lenh, 'kp,ki,kd' de chinh PID.\n",
                PID_SETPOINT_DEG, PID_KP, PID_KI, PID_KD, CONTROL_HZ);
  Serial.println("# KIEM CHIEU: (1) 'm' chong nga, (2) neu vua nga vua tang toc -> doi dau 'i' (PITCH_SIGN).");
  lastLoopUs=micros();
}

// ================================= LOOP ======================================
void loop(){
  applySerial();
  unsigned long now=micros();
  if((now-lastLoopUs) < LOOP_US) return;
  float dt=(now-lastLoopUs)*1e-6f; lastLoopUs=now;

  updateAngle(dt);

  if(!enabled){ motorCoast(); neopixelWrite(LED_PIN,20,0,0); telemetry(); return; }
  if(fabsf(pitchDeg-PID_SETPOINT_DEG) > MAX_SAFE_TILT_DEG){
    filterSeeded=false; pidIntegral=0.0f; lastU=0; motorCoast(); neopixelWrite(LED_PIN,60,0,0); telemetry(); return;
  }

  float u = pidUpdate(dt);
  u = constrain(u*U_SCALE, -1.0f, 1.0f);
  lastU = u;

  driveWheels(u, dt);
  neopixelWrite(LED_PIN,0,40,0);
  telemetry();
}
