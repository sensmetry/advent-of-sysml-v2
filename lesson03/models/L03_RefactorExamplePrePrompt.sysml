package L03_RefactorExamplePrePrompt {
    private import ScalarValues::*;

    part def SantaSleigh {
        ref part dasher : Reindeer;
        ref part dancer { attribute energyLevel : Real := 100.0; }
        ref part prancer { attribute energyLevel : Real := 100.0; }
        ref part vixen { attribute energyLevel : Real := 100.0; }
        ref part comet { attribute energyLevel : Real := 100.0; }
        ref part cupid { attribute energyLevel : Real := 100.0; }
        ref part donner { attribute energyLevel : Real := 100.0; }
        ref part blitzen { attribute energyLevel : Real := 100.0; }
        ref part rudolph { attribute energyLevel : Real := 200.0; }
    }

    part def Reindeer { attribute energyLevel : Real default := 100.0; }
}
