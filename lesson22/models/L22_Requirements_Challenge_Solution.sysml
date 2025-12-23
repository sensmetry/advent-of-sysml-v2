package L22_Requirements_Challenge_Solution {
    package SantaSleighSpecification {
        part def SantaSleigh;
    }
    package UserRequirements {
        requirement def <UR2> StealthMode {
            subject sleigh : SantaSleighSpecification::SantaSleigh;
            doc
            /* Santa's sleigh must be able to fly in stealth mode. */
        }
    }
    package SantaSleighDesign {
        part def SantaSleigh specializes SantaSleighSpecification::SantaSleigh {
            part fieldGenerator : FieldGenerator;
            part powerSubsystem;
        }

        part def FieldGenerator;
    }

    package RequirementBreakdown {
        private import SantaSleighDesign::*;

        requirement def <TR2> StealthMode specializes UserRequirements::StealthMode {
            subject redefines sleigh : SantaSleigh;

            requirement : DistortionField {
                // subject can also be a subcomponent
                subject :>> fieldGenerator = sleigh.fieldGenerator;
            }
            requirement : SufficientPower; // subject is by default bound to parent requirement's subject
        }

        requirement def <'TR2.1'> DistortionField {
            subject fieldGenerator : FieldGenerator;
            doc
            /* The field generator subsystem shall generate a distortion field 
             * that bends light around the sleigh.
             */
        }

        requirement def <'TR2.2'> SufficientPower {
            subject sleigh : SantaSleigh;
            doc
            /* The power subsystem must be able to power stealth mode for the entire trip. */
        }
    }
}
