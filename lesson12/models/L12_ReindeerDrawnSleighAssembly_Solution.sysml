package L12_ReindeerDrawnSleighAssembly_Solution {
    // Abstract components
    abstract part def Vehicle {
        ref part driver : Person;
    }

    part def Person {
        ref part driving : Vehicle;
    }

    // Components definitions
    part def Sleigh specializes Vehicle {
        abstract ref part reindeer : Reindeer [9];
    }

    part def Reindeer {
        ref part pulling : Sleigh;
    }

    // Connection definitions connecting components
    connection def Harness {
        end part sleigh : Sleigh crosses reindeer.pulling;
        end part reindeer : Reindeer crosses sleigh.reindeer;
    }

    connection def Drives {
        end part driver : Person crosses driven.driver;
        end part driven : Vehicle crosses driver.driving;
    }

    // Connecting in a context
    part def ReindeerDrawnSleighAssembly {
        part sleigh : Sleigh;
        part santa : Person;

        abstract part reindeer : Reindeer [9];
        part rudolph : Reindeer [1] subsets reindeer;

        connection harnesses : Harness connect [1] sleigh to [9] reindeer;

        connection subsets harnesses {
            end [1] part :>> sleigh references ReindeerDrawnSleighAssembly::sleigh;
            end [1] part :>> reindeer references rudolph;
        }
        
        connection : Drives connect [1] santa to [1] sleigh;
    }
}
