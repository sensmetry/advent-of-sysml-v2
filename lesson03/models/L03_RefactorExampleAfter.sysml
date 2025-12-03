package L03_RefactorExampleAfter {
    private import ScalarValues::*;

    part def SantaSleigh {
        ref part dasher : Reindeer;
        ref part dancer : Reindeer;
        ref part prancer : Reindeer;
        ref part vixen : Reindeer;
        ref part comet : Reindeer;
        ref part cupid : Reindeer;
        ref part donner : Reindeer;
        ref part blitzen : Reindeer;
        ref part rudolph : Reindeer { redefines energyLevel := 200.0; }
    }

    part def Reindeer { attribute energyLevel : Real default := 100.0; }
}
