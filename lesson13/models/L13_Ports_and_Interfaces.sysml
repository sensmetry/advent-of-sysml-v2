package L13_Ports_and_Interfaces {
    // Components definitions
    part def Sleigh {
        abstract ref part reindeer : Reindeer [9];
        port eyelet : HarnessConnectionPoint;
    }

    part def Reindeer {
        ref part pulling : Sleigh;
        port harnessLoop : ~HarnessConnectionPoint;
    }

    // Port definitions
    port def HarnessConnectionPoint {
        // ...
    }

    // Interface definitions connecting ports, composing flows
    interface def HarnessInterface {
        end port sleighPort : HarnessConnectionPoint;
        end port reindeerPort : ~HarnessConnectionPoint;
    }

    // ...

    // Connection definitions connecting components, composing interface
    connection def Harness {
        end part sleigh : Sleigh crosses reindeer.pulling;
        end part reindeer : Reindeer crosses sleigh.reindeer;

        interface : HarnessInterface connect sleigh.eyelet to reindeer.harnessLoop;
    }
}