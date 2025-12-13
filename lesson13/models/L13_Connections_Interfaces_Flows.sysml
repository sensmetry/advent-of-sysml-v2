package L13_Connections_Interfaces_Flows {
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
        out item command : Command;
        in item feedback;
    }

    // Item definitions
    item def Command;
    item def Feedback;

    // Connection definitions connecting components, composing interface
    connection def Harness {
        end part sleigh : Sleigh crosses reindeer.pulling;
        end part reindeer : Reindeer crosses sleigh.reindeer;

        interface : HarnessInterface connect sleigh.eyelet to reindeer.harnessLoop;
    }

    // Interface definitions connecting ports, composing flows
    interface def HarnessInterface {
        end port sleighPort : HarnessConnectionPoint;
        end port reindeerPort : ~HarnessConnectionPoint;

        flow of Command from sleighPort.command to reindeerPort.command;
        flow of Feedback from reindeerPort.feedback to sleighPort.feedback;
    }

    // Connecting in a context
    part def ReindeerDrawnSleighAssembly{
        part sleigh : Sleigh;
        abstract part reindeer : Reindeer [9];
        part rudolph [1] : Reindeer subsets reindeer;

        // Internal structure, including the interface and the flows in that
        // are automatically set up when the connection is established.
        connection harnesses : Harness connect [1] sleigh to [9] reindeer;
        connection subsets harnesses {
            end [1] part :>> sleigh references ReindeerDrawnSleighAssembly::sleigh;
            end [1] part :>> reindeer references rudolph;
        }
    }
}