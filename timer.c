// Global Variables
int timer = 0;
int on = 0;
int minInterval = 2;


void setup() {
  // Setup pins for pin activated interrupts: p8, p7
  pinMode(3,INPUT);//digitalWrite(8,HIGH);
  pinMode(2,INPUT);//digitalWrite(7,HIGH);
  
  // Setup pin for timer activated interrupts: p13
  pinMode(13,OUTPUT);//`******digitalWrite(13,LOW);
  pinMode(12,OUTPUT); // Open the relay to shut of RPi

  
  // Setup the timer interrupt configuration for 1 Hz
  /*
  cli();
  TCCR1A = 0;// set entire TCCR1A register to 0
  TCCR1B = 0;// same for TCCR1B
  TCNT1  = 0;//initialize counter value to 0
  // set compare match register for 1hz increments
  OCR1A = 15624;// = (16*10^6) / (1*1024) - 1 (must be <65536)
  // Set CS12 and CS10 bits for 1024 prescaler
  TCCR1B |= (1 << CS12) | (1 << CS10);  
  // enable timer compare interrupt
  TIMSK1 |= (1 << OCIE1A);
  // Normally would turn on, but this is left for the p8 interrupt to do:
  // turn on CTC mode
  // TCCR1B |= (1 << WGM12);
  //
  sei();
  */

  // Setup p3 and p2 interrupt code
  attachInterrupt(digitalPinToInterrupt(2),ISR_p2,FALLING);
  attachInterrupt(digitalPinToInterrupt(3),ISR_p3,RISING);

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
  if(timer == 60*minInterval){
  //if(timer == 10){
    digitalWrite(12,LOW);
    digitalWrite(13,HIGH);
    timer = 1;
    Serial.println(timer);
  } else {
    timer += 1;
    Serial.println(timer);
  }
}

// pin2 interrupt service routine
void ISR_p2(){
  // turn off RPi, as the signal pin has lifted saying it is shutdown
  if (timer > 5){
       digitalWrite(13,LOW);
       digitalWrite(12,HIGH);
       Serial.println("pin 2 lifted");
  }
}

// pin3 interrup service routine
void ISR_p3(){
  // start the timer1, as the signal pin has been lifted saying the clocks are synchronized
  cli();
  TCCR1A = 0;// set entire TCCR1A register to 0
  TCCR1B = 0;// same for TCCR1B
  TCNT1  = 0;//initialize counter value to 0
  // set compare match register for 1hz increments
  OCR1A = 15624;// = (16*10^6) / (1*1024) - 1 (must be <65536)
  // Set CS12 and CS10 bits for 1024 prescaler
  TCCR1B |= (1 << CS12) | (1 << CS10);  
  // enable timer compare interrupt
  TIMSK1 |= (1 << OCIE1A);
  TCCR1B |= (1 << WGM12); 
  sei();
  Serial.println("pin 3 lifted");
  detachInterrupt(3);
}



void loop() {
  // put your main code here, to run repeatedly:
  //delay(1000);
  //Serial.println("yup, again");

}
