#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include <ESP32Encoder.h>
#include "policy_weights.h"

#if POLICY_OBS_DIM != 7
#error "Firmware lap obs 7 chieu [pitch,rate,vel1,vel2,lastAct1,lastAct2,velCmd] (khop selfbalancing_env_cfg.py). Model khac chieu -> sua assembleObs()."
#endif
#if POLICY_ACT_DIM != 2
#error "Firmware lap obs co 2 o last_action (1 o/banh) -> can dung dung 2 action. Model khac -> sua assembleObs()/driveWheels()/loop()."
#endif

struct Shaper { float charge=0, remain=0, coast=0; int lastDir=0; };
Shaper shpL, shpR;

// =============================== USER CONFIG =================================
static const int SDA_PIN = 21, SCL_PIN = 22, LED_PIN = 27;   // LED_PIN: chan RGB (neopixel) bao trang thai, doi neu da dung GPIO27 viec khac
static const int ENA = 15, IN1 = 2, IN2 = 4;      // banh A (TRAI)
static const int ENB = 5, IN3 = 19, IN4 = 18;     // banh B (PHAI)
static const int ENC_L_A = 14, ENC_L_B = 26;
static const int ENC_R_A = 33, ENC_R_B = 32;      // GPIO32/33 la chan ADC1 binh thuong, co dien tro keo len noi -> khong can tro ngoai
const float ENC_TICKS_PER_REV = 1320.0f;   // 11 PPR x hop so 30:1 x quadrature x4 (do bang ESP32Encoder::attachFullQuad)
const float WHEEL_RADIUS_M    = 0.0325f;   // duong kinh banh 65mm -> ban kinh 32.5mm


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

float MIN_DUTY = 0.49f, MAX_DUTY = 1.0f, U_DEADBAND = 0.01f;
float LINEAR_START_U     = 0.10f;
float NEAR_BAL_MAX_DUTY  = 0.42f;
float NEAR_BAL_PITCH_DEG = 1.5f;
float NEAR_BAL_RATE_DPS  = 18.0f;
float U_SCALE     = 1.0f;
float PITCH_TRIM_DEG = 0.0f;
float VEL_CMD_MPS = 0.0f;   // velocity_command (m/s) obs cuoi cung -- LUON = 0 vi khi train file
                            // selfbalancing_env_cfg.py da TAT ca curriculum lan reward lien quan toi
                            // target_velocity (xem RewardsCfg/CurriculumsCfg: velocity_tracking* va
                            // velocity_range/resampling_time deu bi comment). Dat khac 0 la NGOAI
                            // phan phoi da train, policy chua hoc lai theo huong nay.
int   MOTOR_SIGN   = +1;
int   GYRO_SIGN    = +1;    // dao dau rieng toc do goc (gyro) so voi goc nghieng (accel)
int   PITCH_SIGN   = +1;    // dao dau CA goc nghieng lan toc do goc (dinh huong IMU nguoc)
int   ENC_L_SIGN   = +1;    // dao chieu dem encoder banh A (khong anh huong chieu motor)
int   ENC_R_SIGN   = -1;    // dao chieu dem encoder banh B
int   WHEEL_SWAP   = 0;     // POLICY_ACT_DIM==2: doi thu tu act[0]/act[1] neu bi lap nguoc 2 banh


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
float lastUL=0, lastUR=0, lastDutyFrac=0;
unsigned long lastLoopUs=0, inferUs=0;

ESP32Encoder encL, encR;
int64_t encLCount=0, encRCount=0;
int64_t encLPrev=0, encRPrev=0;
float   wheelLVel=0, wheelRVel=0;

// hanh dong RAW (truoc U_SCALE/MOTOR_SIGN) cua policy o tick TRUOC -- dung lam obs last_action1/2
// (mdp.last_action_index trong selfbalancing_env_cfg.py doc dung gia tri nay, khong phai gia tri
// da nhan scale/MOTOR_SIGN). Cap nhat o cuoi loop() ngay sau policyForward().
float prevAct[POLICY_ACT_DIM] = {0.0f, 0.0f};

// ===================== MLP FORWARD (obs -> POLICY_ACT_DIM output) =============
// Mang: (obs-mean)/std -> [FC + act]xN. Tra ra CA POLICY_ACT_DIM gia tri.
static float bufA[POLICY_MAX_WIDTH], bufB[POLICY_MAX_WIDTH];

void policyForward(const float obs[POLICY_OBS_DIM], float out[POLICY_ACT_DIM]){
  float *in = bufA, *o = bufB;
  for(int i=0;i<POLICY_OBS_DIM;i++)
    in[i] = (obs[i] - POLICY_OBS_MEAN[i]) / POLICY_OBS_STD[i];   // chuan hoa (0/1 => giu nguyen)
  for(int l=0; l<POLICY_N_LAYERS; l++){
    const float *W = POLICY_W[l], *B = POLICY_B[l];
    const int nin = POLICY_LAYER_IN[l], nout = POLICY_LAYER_OUT[l];
    for(int j=0;j<nout;j++){
      const float *w = W + (long)j*nin;
      float s = B[j];
      for(int i=0;i<nin;i++) s += w[i]*in[i];
      switch(POLICY_LAYER_ACT[l]){
        case ACT_ELU:     s = (s>0.0f) ? s : expf(s)-1.0f; break;
        case ACT_RELU:    if(s<0.0f) s=0.0f;               break;
        case ACT_TANH:    s = tanhf(s);                    break;
        case ACT_SIGMOID: s = 1.0f/(1.0f+expf(-s));        break;
        // ACT_NONE: giu nguyen
      }
      o[j]=s;
    }
    float *t=in; in=o; o=t;
  }
  for(int i=0;i<POLICY_ACT_DIM;i++) 
    out[i] = in[i];   // lay CA cac output
}

// ============================ ENCODER (quadrature x4, ESP32Encoder/PCNT) =====
void updateEncoders(float dt){
  encLCount = (int64_t)ENC_L_SIGN * encL.getCount();
  encRCount = (int64_t)ENC_R_SIGN * encR.getCount();
  float k = (dt>1e-6f) ? (TWO_PI/ENC_TICKS_PER_REV/dt) : 0.0f;
  wheelLVel = (float)(encLCount-encLPrev)*k;  encLPrev=encLCount;
  wheelRVel = (float)(encRCount-encRPrev)*k;  encRPrev=encRCount;
}
void resetOdometry(){
  encL.clearCount(); encR.clearCount();
  encLCount=0; encRCount=0; encLPrev=0; encRPrev=0; wheelLVel=0; wheelRVel=0;
}

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

// ====================== ANGLE / OBS ==========================================
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
  filterSeeded=false; resetOdometry(); prevAct[0]=prevAct[1]=0.0f; lastLoopUs=micros();
  Serial.printf("# CALIB offset=%.2f gyroBiasX=%.3f\n", pitchOffsetDeg, gyroBiasX);
}
void assembleObs(float obs[POLICY_OBS_DIM]){
  // Obs 7 chieu, khop DUNG THU TU khai bao trong ObservationsCfg.PolicyCfg cua
  // selfbalancing_env_cfg.py (concatenate_terms=True -> noi theo thu tu khai bao):
  //   [pitch_angle, pitch_rate, wheel1_vel, wheel2_vel, last_action1, last_action2, velocity_command]
  // wheel1 = WHEEL_JOINT_NAMES[0] = banh TRAI (IN1/IN2/ENC_L), wheel2 = banh PHAI (IN3/IN4/ENC_R).
  obs[0] = (pitchDeg - PITCH_TRIM_DEG) * DEG_TO_RAD;   // pitch_angle (rad)
  // pitch_rate (rad/s): DAU AM co chu dich. Trong sim, obs pitch_rate (root_ang_vel_b[:,1]) NGUOC
  // dau voi d(pitch_angle)/dt (do bang scripts/pid_balance.py: corr = -0.97), con rateDps o day la
  // +d(pitch)/dt (bo loc complementary can vay). Dao dau de khop voi phan phoi policy da train.
  obs[1] = -rateDps * DEG_TO_RAD;
  obs[2] = wheelLVel;                                  // wheel1_vel (rad/s) - banh TRAI
  obs[3] = wheelRVel;                                  // wheel2_vel (rad/s) - banh PHAI
  obs[4] = prevAct[0];                                 // last_action1 - hanh dong policy tick truoc, banh TRAI
  obs[5] = prevAct[1];                                 // last_action2 - banh PHAI
  obs[6] = VEL_CMD_MPS;                                // velocity_command (m/s) - hien luon = 0
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
// Nhan lenh RIENG cho 2 banh (uL cho banh A, uR cho banh B). Da nhan U_SCALE truoc khi goi.
void driveWheels(float uL, float uR, float dt){
  float cmdL = constrain((float)MOTOR_SIGN*uL, -1.0f, 1.0f);
  float cmdR = constrain((float)MOTOR_SIGN*uR, -1.0f, 1.0f);
  bool near = (fabsf(pitchDeg-PITCH_TRIM_DEG) <= NEAR_BAL_PITCH_DEG) &&
              (fabsf(rateDps) <= NEAR_BAL_RATE_DPS);
  float sdL = shaperUpdate(shpL, cmdL, near, dt);
  float sdR = shaperUpdate(shpR, cmdR, near, dt);
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
  Serial.printf("# PARAM scale=%.2f trim=%.2f MIN=%.2f MAX=%.2f dead=%.3f nearCap=%.2f "
                "MOTOR_SIGN=%d GYRO=%d PITCH=%d ENC_L=%d ENC_R=%d swap=%d en=%d | %dHz, MLP %d lop obs=%d act=%d infer=%luus\n",
                U_SCALE, PITCH_TRIM_DEG, MIN_DUTY, MAX_DUTY, U_DEADBAND, NEAR_BAL_MAX_DUTY,
                MOTOR_SIGN, GYRO_SIGN, PITCH_SIGN, ENC_L_SIGN, ENC_R_SIGN, WHEEL_SWAP, enabled,
                CONTROL_HZ, POLICY_N_LAYERS, POLICY_OBS_DIM, POLICY_ACT_DIM, inferUs);
}
void parseCmd(char*s){
  char c=s[0]; float v=atof(s+1);
  switch(c){
    case 'a': U_SCALE=constrain(v,0.0f,1.0f); break;
    case 's': PITCH_TRIM_DEG=constrain(v,-20.0f,20.0f); break;
    case 'n': MIN_DUTY=constrain(v,0.0f,MAX_DUTY); break;
    case 'l': MAX_DUTY=constrain(v,MIN_DUTY,1.0f); break;
    case 'b': U_DEADBAND=constrain(v,0.0f,0.95f); break;
    case 'm': MOTOR_SIGN  =(v>=0)?+1:-1; break;
    case 'g': GYRO_SIGN   =(v>=0)?+1:-1; filterSeeded=false; break;
    case 'i': PITCH_SIGN  =(v>=0)?+1:-1; filterSeeded=false; break;   // dao dau goc nghieng (IMU nguoc chieu)
    case 'e': ENC_L_SIGN  =(v>=0)?+1:-1; resetOdometry(); break;      // dao chieu dem encoder banh A
    case 'r': ENC_R_SIGN  =(v>=0)?+1:-1; resetOdometry(); break;      // dao chieu dem encoder banh B
    case 'w': WHEEL_SWAP  =(v>=0.5f)?1:0; break;                      // doi thu tu 2 action (chi co tac dung khi POLICY_ACT_DIM==2)
    case 'c': motorCoast(); calibrate(); break;
    case 't': motorCoast(); motorTest(); return;
    case 'x': enabled=false; motorCoast(); Serial.println("# STOP motor OFF"); break;
    case 'o': enabled=true; prevAct[0]=prevAct[1]=0.0f; Serial.println("# GO motor ON"); break;
    case '?': default: printParams(); return;
  }
  printParams();
}
void handleSerial(){
  static char buf[32]; static uint8_t idx=0;
  while(Serial.available()){
    char c=Serial.read();
    if(c=='\n'||c=='\r'){ buf[idx]=0; if(idx>0) parseCmd(buf); idx=0; }
    else if(idx<sizeof(buf)-1) buf[idx++]=c;
  }
}
void telemetry(){
  static unsigned long last=0;
  if(millis()-last < 100) return;
  last=millis();
  // Dong debug day du (dang comment, plotter se bo qua vi khong dung format so lieu)
  Serial.printf("# pitch=%6.2f rate=%7.1f uL=%+.3f uR=%+.3f duty=%4.0f%% | encL=%lld encR=%lld velL=%5.1f velR=%5.1f | infer=%luus %s\n",
                -pitchDeg, rateDps, lastUL, lastUR, lastDutyFrac*100.0f,
                (long long)encLCount, (long long)encRCount, wheelLVel, wheelRVel, inferUs,
                enabled ? "" : "[OFF]");
  // Dong so lieu "ten:gia_tri" rieng, dung format PlatformIO/Arduino Serial Plotter can de ve do thi
  Serial.printf("pitch:%.2f,rate:%.1f,uL:%.3f,uR:%.3f,duty:%.1f,velL:%.1f,velR:%.1f\n",
                -pitchDeg, rateDps, lastUL, lastUR, lastDutyFrac*100.0f, wheelLVel, wheelRVel);
}

// ================================ SETUP ======================================
void setup(){
  Serial.begin(921600);   // baud cao de telemetry() day xong trong <2ms thay vi ~14ms @115200
                          // (doi monitor_speed trong platformio.ini + toc do may do/logic analyzer cho khop)
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  // QUAN TRONG: setTxTimeoutMs(0) = cho VO HAN neu buffer TX day (khong phai "tat timeout" an toan).
  // Neu may nhan (USB-serial/monitor) doc cham, Serial.printf() trong telemetry() se bi block toi
  // vai ms -> gay tre chu ky dieu khien (day la nghi pham chinh cho hien tuong chu ky >13ms doi luc).
  // Dat 2ms: qua thoi gian nay thi bo bot du lieu con lai thay vi treo ca vong lap.
  Serial.setTxTimeoutMs(2);
#endif
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
  ESP32Encoder::useInternalWeakPullResistors = puType::up;
  encL.attachFullQuad(ENC_L_A, ENC_L_B);
  encR.attachFullQuad(ENC_R_A, ENC_R_B);
  encL.clearCount(); encR.clearCount();

  Wire.begin(SDA_PIN,SCL_PIN); Wire.setClock(400000);
  Wire.setTimeOut(2);   // gioi han cho toi da 2ms/giao dich I2C, tranh 1 lan NACK/bus loi treo ca chu ky dieu khien
  if(!bmiInit()){
    Serial.println("# LOI: khong thay LSM6DS3 (SDA=21/SCL=22, 3V3, GND)");
    while(true){ neopixelWrite(LED_PIN,80,0,0); delay(150); neopixelWrite(LED_PIN,0,0,0); delay(150); }
  }
  calibrate();

  // Self-test MLP: obs = 0 het (dung tra danh gia dung/sai, chi de kiem tra day nao chay duoc)
  {
    float t0v[POLICY_OBS_DIM]={0}; float o0[POLICY_ACT_DIM];
    unsigned long t0=micros(); policyForward(t0v,o0); inferUs=micros()-t0;
    Serial.print("# SELF-TEST obs=0 -> [");
    for(int i=0;i<POLICY_ACT_DIM;i++) Serial.printf("%+.4f%s", o0[i], i<POLICY_ACT_DIM-1?", ":"");
    Serial.printf("]  infer=%luus\n", inferUs);
  }
  Serial.printf("# READY MLP %d lop, obs=%d act=%d, %dHz. Go '?' xem lenh.\n",
                POLICY_N_LAYERS, POLICY_OBS_DIM, POLICY_ACT_DIM, CONTROL_HZ);
  Serial.println("# KIEM CHIEU: (1) 'm' chong nga, (2) neu xoay -> kiem tra dau day IN1-4 mot ben, hoac 'w 1' neu act=2.");
  lastLoopUs=micros();
}

// ================================= LOOP ======================================
void loop(){
  handleSerial();
  unsigned long now=micros();
  if((now-lastLoopUs) < LOOP_US) return;
  float dt=(now-lastLoopUs)*1e-6f; lastLoopUs=now;

  updateAngle(dt);
  updateEncoders(dt);

  if(!enabled){ motorCoast(); neopixelWrite(LED_PIN,20,0,0); telemetry(); return; }
  if(fabsf(pitchDeg-PITCH_TRIM_DEG) > MAX_SAFE_TILT_DEG){
    filterSeeded=false; lastUL=lastUR=0; prevAct[0]=prevAct[1]=0.0f;
    motorCoast(); neopixelWrite(LED_PIN,60,0,0); telemetry(); return;
  }

  float obs[POLICY_OBS_DIM]; assembleObs(obs);
  float act[POLICY_ACT_DIM];
  unsigned long t0=micros();
  policyForward(obs, act);              // <-- tra ra POLICY_ACT_DIM gia tri
  inferUs = micros()-t0;
  prevAct[0]=act[0]; prevAct[1]=act[1];  // luu lai (RAW, truoc scale) cho obs last_action1/2 tick sau

  float uL, uR;
  if(WHEEL_SWAP){ uL = act[1]; uR = act[0]; }
  else          { uL = act[0]; uR = act[1]; }
  uL = constrain(uL*U_SCALE, -1.0f, 1.0f);
  uR = constrain(uR*U_SCALE, -1.0f, 1.0f);
  lastUL=uL; lastUR=uR;

  driveWheels(uL, uR, dt);
  neopixelWrite(LED_PIN,0,40,0);
  telemetry();

}