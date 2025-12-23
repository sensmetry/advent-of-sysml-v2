package L23_AdvancedRequirements {
    package SantaSleighDesign {
        part def SantaSleigh {
            part cockpit {
                attribute volume :> ISQ::volume = 500 [SI::L];
            }
            part cargoBay : CargoBay;
            ref part santa {
                attribute weight :> ISQ::mass = 150 [SI::kg];
            }
        }

        abstract part def CargoBay {
            item payload : Bag [*];
        }
        part def OrdinaryCaroBay specializes CargoBay {
            item redefines payload [0..1] : BottomlessGiftBag;
        }
        part def MagicalCargoBay specializes CargoBay {
            item redefines payload : OrdinaryBag;
        }

        abstract item def Bag {
            item gifts [*];
        }
        item def OrdinaryBag specializes Bag {
            item redefines gifts [100];
        }
        item def BottomlessGiftBag specializes Bag {
            item redefines gifts;
        }
    }

    package UserRequirements {
        requirement def <UR1> DeliverGifts {
            subject sleigh : SantaSleighSpecification::SantaSleigh;
            doc
            /* Santa's sleigh shall be able to deliver gifts to all nice children. */
        }
    }

    package RequirementTemplates {
        requirement def SizingRequirement {
            doc
            /* The value of {toSize} must be at least {minimumScale} times as large as {toFit}. */
            attribute toSize;
            attribute toFit;
            attribute minimumScale;
            assume constraint {
                toFit >= 0
            }
            require constraint {
                toSize >= toFit * minimumScale
            }
        }
    }

    package RequirementBreakdown {
        private import SantaSleighDesign::*;

        requirement def <TR1> DeliverGifts specializes UserRequirements::DeliverGifts {
            subject redefines sleigh : SantaSleigh;

            requirement : CockpitForSanta; // subject is by default bound to parent requirement's subject
            requirement : CargoCapacity {
                // subject can also be a subcomponent
                subject :>> cargoBay = sleigh.cargoBay;
            }
        }

        requirement def <'TR1.1'> CockpitForSanta {
            subject sleigh : SantaSleigh;
            require constraint {
                language "English"
                /* The sleigh must have a cockpit for Santa. */
            }

            requirement : CockpitSize; // subject is by default bound to parent requirement's subject
        }

        requirement def <'TR1.1.1'> CockpitSize specializes RequirementTemplates::SizingRequirement {
            subject sleigh : SantaSleigh; // subject is by default bound to parent requirement's subject
            doc
            /* The volume of the cockpit must be at least 3 times as much as the volume of Santa. */
            attribute :>> toSize = sleigh.cockpit.volume;
            attribute :>> toFit = sleigh.santa.weight / 1.03 [SI::kg / SI::L];
            attribute :>> minimumScale = 3;
            // constraint is inherited from requirement definition
        }

        requirement def <'TR1.2'> CargoCapacity {
            subject cargoBay : CargoBay;
            doc
            /* The cargo bay must be able to accommodate an arbitrary number of gifts. */
        }
    }

    package SantaSleighSpecification {
        part def SantaSleigh;
    }
}
