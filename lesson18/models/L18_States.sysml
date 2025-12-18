package L18_States {
    private import ScalarValues::*;
    
    item def StartMission;
    item def EndMission;
    item def Approach;
    item def Next;

    item def Command {
        attribute cmd : String;
    }
    item def Status {
        attribute status : String;
    }

    state def ReindeerController {
        attribute reindeerCount : Integer;
        attribute totalPower subsets ISQ::power;
        attribute averageEnergy : Real;
        ref port eyelet;
        ref port hmi;
        entry;
            then idle;
        state idle;
            accept StartMission via hmi
            if reindeerCount == 9 and totalPower >= 36 [SI::kW]
            do send new Command("Gee-up!") via eyelet
            then mission;
        state mission parallel {
            state monitoringReindeer {
                entry;
                    then nominal;
                state nominal;
                state lowEnergy {
                    entry send new Status("low energy") via hmi;
                    do action blinkLed;
                    exit send new Status("recharged") via hmi;
                }

                transition nominal2low
                    first nominal
                    accept when averageEnergy < 50
                    then lowEnergy;
                transition low2nomnal
                    first lowEnergy
                    accept when averageEnergy >= 50
                    then nominal;
            }
            state controllingReindeer {
                entry;
                    then takingOff;
                state takingOff;
                    accept after 60 [SI::s]
                    then flying;
                state flying;
                    accept a : Approach via hmi
                    do send new Command("Whoa!") via eyelet
                    then landing;
                state landing;
                    accept after 5 [SI::minute]
                    then flying;
                state waiting;
                    accept Next via hmi
                    do send new Command("Gee-up!") via eyelet
                    then takingOff;
            }
        }
            accept EndMission via hmi
            then idle;
    }
}