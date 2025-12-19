package L19_State_Simulation {
    private import ScalarValues::*;
    private import SI::*;

    package SleighFlightStates {
        // Events for state transitions
        item def UpdateFlightData;
        item def StartFlight;
        item def AbortTakeoff;
        item def StartLanding;
        item def AbortLanding;
        item def StartDelivery;
        item def DeliveryComplete;

        state def SleighFlightMonitor {
            doc
            /* Monitors the flight state of Santa's sleigh.
             * Helicopter-style flight model with command-driven takeoff/landing protocols.
             */

            // Flight parameters
            attribute altitude subsets ISQ::length default 0 [m];
            attribute groundSpeed subsets ISQ::speed default 0 [m / s];
            attribute verticalSpeed subsets ISQ::speed default 0 [m / s];

            entry;
                then grounded;

            state grounded {
                doc
                /* On the ground. Ready for takeoff. */
            }
                accept StartFlight then liftOff;

            state liftOff {
                doc
                /* Transient state: Automated takeoff protocol.
                 * Climbing to cruise altitude. */
            }
                accept UpdateFlightData if altitude >= 20 [m] then flying;
                accept AbortTakeoff if altitude >= 1 [m] then descending;
                accept AbortTakeoff if altitude < 1 [m] then grounded;

            state flying {
                doc
                /* Cruising flight. Normal forward flight operations. */
            }
                accept UpdateFlightData if groundSpeed < 2 [m / s] then hovering;
                accept StartLanding then descending;

            state hovering {
                doc
                /* Hovering in place. Composite state for delivery operations. */

                entry;
                    then idle;

                state idle {
                    doc
                    /* Hovering idle, waiting for delivery command. */
                }
                    accept StartDelivery then giftMode;

                state giftMode {
                    doc
                    /* Actively delivering gifts. */
                }
                    accept DeliveryComplete then idle;
            }
                accept UpdateFlightData if groundSpeed > 5 [m / s] then flying;
                accept StartLanding then descending;

            state descending {
                doc
                /* Transient state: Automated landing protocol.
                 * Controlled descent to ground. */
            }
                accept UpdateFlightData if altitude <= 0 [m] then grounded;
                accept AbortLanding if altitude > 10 [m] then flying;
        }
    }
}