package L13_Items_and_Flows {
    // ...

    // Port definitions
    port def HarnessConnectionPoint {
        out item command : Command;
        in item feedback : Feedback;
    }

    // Item definitions
    item def Command;
    item def Feedback;

    // Interface definitions connecting ports, composing flows
    interface def HarnessInterface {
        end port eyelet : HarnessConnectionPoint;
        end port harnessLoop : ~HarnessConnectionPoint;

        flow of Command from eyelet.command to harnessLoop.command;
        flow of Feedback from harnessLoop.feedback to eyelet.feedback;
    }
}