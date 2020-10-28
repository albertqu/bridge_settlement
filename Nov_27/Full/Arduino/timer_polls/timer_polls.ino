// Global Variables
int timer = 0;
int minInterval = 5;
int syncInterval = 1;
int flag_sync = 0;
//int timer_sync = 0;

int CurrentStatePin2 = LOW;
int PrevStatePin2 = LOW;
int CurrentStatePin3 = LOW;
int PrevStatePin3 = LOW;

void setup() {
  // Setup pins for pin activated interrupts: p8, p7
  pinMode(3,INPUT);//digitalWrite(8,HIGH);
  pinMode(2,INPUT);//digitalWrite(7,HIGH);
  
  // Setup pin for timer activated interrupts: p13
  pinMode(13,OUTPUT);//`******digitalWrite(13,LOW);
  pinMode(12,OUTPUT); // Open the relay to shut of RPi

  // Turn on the RPi pin to start the flow of energy to the RPi so it 
  // can be configured to run the system and eventually activate p8 and
  // p9 interrupt ISR's which will begin the total system operation.
  digitalWrite(13,HIGH); // this flips a relay energizing the RPi
  digitalWrite(12,LOW);

  Serial.begin(9600);
  Serial.println("Running...");
}

// timer1 interrupt service routine
ISR(TIMER1_COMPA_vect){
  // turn on the RPi, as the time interval has elapsed
  if(timer == 60*minInterval && flag_sync == 0){
    digitalWrite(12,LOW);
    digitalWrite(13,HIGH);
    timer = 1;
    Serial.println(timer);
  } else if (timer == 60*syncInterval && flag_sync == 1){
    digitalWrite(12,LOW);
    digitalWrite(13,HIGH);
    timer = 1;
    Serial.println(timer);
    flag_sync = 0;
    Serial.println("flag_sync off");
  //} else if  (timer_sync == 1) {
  //  timer = 1;
  //  timer_sync = 0;
  //  Serial.println("timer_sync off");
  } else {
    timer += 1;
    Serial.println(timer);
  }
}

// pin2 interrupt service routine
void ISR_p2(){
  // turn off RPi, as the signal pin has lifted saying it is shutdown
  delay(1000);
  digitalWrite(13,LOW);
  digitalWrite(12,HIGH);
  Serial.println("pin 2 lifted");
}

// pin3 interrup service routine
void ISR_p3(){
  // start the timer1, as the signal pin has been lifted saying the clocks are synchronized
  cli();
  Serial.println("l1");
  TCCR1A = 0;// set entire TCCR1A register to 0
  TCCR1B = 0;// same for TCCR1B
  TCNT1  = 0;//initialize counter value to 0
  Serial.println("l2");
  // set compare match register for 1hz increments
  OCR1A = 15624;// = (16*10^6) / (1*1024) - 1 (must be <65536)
  //OCR1A = 15614;// for Laser Module
  Serial.println("l3");
  // Set CS12 and CS10 bits for 1024 prescaler
  TCCR1B |= (1 << CS12) | (1 << CS10);  
  // enable timer compare interrupt
  TIMSK1 |= (1 << OCIE1A);
  TCCR1B |= (1 << WGM12); 
  Serial.println("l4");
  sei();
  Serial.println("l5");
  timer = 0; // initialize or reset timer (first execution, or re-sync)
  Serial.println("pin 3 lifted");
}


void loop() {
  
  PrevStatePin2 = CurrentStatePin2;
  PrevStatePin3 = CurrentStatePin3;

  delay (10);
  
  CurrentStatePin2 = digitalRead(2);
  CurrentStatePin3 = digitalRead(3);

  if ((PrevStatePin2 == HIGH) && (CurrentStatePin2 == LOW)){
    ISR_p2();
  }
  if ((PrevStatePin3 == LOW) && (CurrentStatePin3 == HIGH)){
    ISR_p3();
    flag_sync = 1;
    //timer_sync = 1;
    Serial.println("flag_sync on");
    //Serial.println("timer_sync on");
  }
}
