package L18_SantaSleighCruiseControl_Solution {
    private import ScalarValues::*;
    private import SI::*;

    // Events for cruise control transitions
    item def PowerOn;
    item def PowerOff;
    item def SetSpeed;
    item def Cancel;
    item def Resume;
    item def Brake;

    state def SantaSleighCruiseControl {
        doc
        /* Santa's Sleigh Cruise Control System.
         * Maintains a constant flying speed during long gift delivery routes.
         * Similar to automotive cruise control but adapted for magical sleigh flight.
         *
         * In cruising mode, system regulates speed and monitors weather conditions in parallel.
         *
         * State Transition Summary (initial state: off):
         *
         * | From      | Event    | Guard                    | To        |
         * |-----------|----------|--------------------------|-----------|
         * | off       | PowerOn  | —                        | standby   |
         * | standby   | SetSpeed | currentSpeed >= minSpeed | cruising  |
         * | standby   | PowerOff | —                        | off       |
         * | cruising  | Brake    | —                        | suspended |
         * | cruising  | Cancel   | —                        | standby   |
         * | cruising  | PowerOff | —                        | off       |
         * | suspended | Resume   | currentSpeed >= minSpeed | cruising  |
         * | suspended | Cancel   | —                        | standby   |
         * | suspended | PowerOff | —                        | off       |
         */

        // Cruise control parameters
        attribute currentSpeed subsets ISQ::speed default 0 ['m/s'];
        attribute targetSpeed subsets ISQ::speed default 0 ['m/s'];
        attribute minSpeed subsets ISQ::speed default 10 ['m/s'];
        attribute tolerance subsets ISQ::speed default 2 ['m/s'];
        
        // Weather parameters
        attribute visibility subsets ISQ::length default 1000 [m];
        attribute windSpeed subsets ISQ::speed default 0 [m / s];
        attribute lowVisibility subsets ISQ::length default 500 [m];
        attribute highWindSpeed subsets ISQ::speed default 20 [m / s];

        entry;
            then off;

        state off;
            accept PowerOn then standby;

        state standby;
            accept SetSpeed if currentSpeed >= minSpeed then cruising;
            accept PowerOff then off;

        state cruising parallel {
            doc
            /* Active cruise control with parallel monitoring.
             * Speed regulation and weather monitoring operate independently.
             */

            state speedRegulation {
                doc
                /* Manages reindeer power to maintain target speed (initial state: maintaining).
                 *
                 * | From         | Trigger (when)                         | To           |
                 * |--------------|----------------------------------------|--------------|
                 * | maintaining  | currentSpeed < targetSpeed - tolerance | accelerating |
                 * | maintaining  | currentSpeed > targetSpeed + tolerance | decelerating |
                 * | accelerating | currentSpeed >= targetSpeed            | maintaining  |
                 * | decelerating | currentSpeed <= targetSpeed            | maintaining  |
                 */

                entry;
                    then maintaining;

                state maintaining;
                    accept when currentSpeed < targetSpeed - tolerance then accelerating;
                    accept when currentSpeed > targetSpeed + tolerance then decelerating;

                state accelerating;
                    accept when currentSpeed >= targetSpeed then maintaining;

                state decelerating;
                    accept when currentSpeed <= targetSpeed then maintaining;
            }

            state weatherMonitoring {
                doc
                /* Monitors weather conditions during cruise (initial state: clear).
                 *
                 * | From          | Trigger (when)                                            | To            |
                 * |---------------|-----------------------------------------------------------|---------------|
                 * | clear         | visibility < lowVisibility AND windSpeed > highWindSpeed  | foggyAndWindy |
                 * | clear         | visibility < lowVisibility                                | foggy         |
                 * | clear         | windSpeed > highWindSpeed                                 | windy         |
                 * | foggy         | windSpeed > highWindSpeed                                 | foggyAndWindy |
                 * | foggy         | visibility >= lowVisibility                               | clear         |
                 * | windy         | visibility < lowVisibility                                | foggyAndWindy |
                 * | windy         | windSpeed <= highWindSpeed                                | clear         |
                 * | foggyAndWindy | visibility >= lowVisibility AND windSpeed <= highWindSpeed| clear         |
                 * | foggyAndWindy | visibility >= lowVisibility                               | windy         |
                 * | foggyAndWindy | windSpeed <= highWindSpeed                                | foggy         |
                 */

                entry;
                    then clear;

                state clear;
                    accept when visibility < lowVisibility and windSpeed > highWindSpeed then foggyAndWindy;
                    accept when visibility < lowVisibility then foggy;
                    accept when windSpeed > highWindSpeed then windy;

                state foggy;
                    accept when windSpeed > highWindSpeed then foggyAndWindy;
                    accept when visibility >= lowVisibility then clear;

                state windy;
                    accept when visibility < lowVisibility then foggyAndWindy;
                    accept when windSpeed <= highWindSpeed then clear;

                state foggyAndWindy;
                    accept when visibility >= lowVisibility and windSpeed <= highWindSpeed then clear;
                    accept when visibility >= lowVisibility then windy;
                    accept when windSpeed <= highWindSpeed then foggy;
            }
        }
            accept Brake then suspended;
            accept Cancel then standby;
            accept PowerOff then off;

        state suspended;
            accept Resume if currentSpeed >= minSpeed then cruising;
            accept Cancel then standby;
            accept PowerOff then off;
    }
}
